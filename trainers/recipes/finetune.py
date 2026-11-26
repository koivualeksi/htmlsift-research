"""Recipe finetune: fine-tune an encoder+head on one board, select the epoch on val, then
optionally score test (export_test) and post-hoc int8 variants (quant_eval).
"""
import copy

import torch
from transformers import AutoTokenizer

from core.model import build_encoder, build_head, infer_page
from core.quant import emb_int8_fakequant, patch_qat_w8a8, ptq_dynamic
from core.training import feature_dim
from trainers.loop import train_encoder
from trainers.recipes import bench, cached
from vendors.shared import bench_ops
from vendors.shared.arms import WINDOW, build_arms


def _arm(cfg):
    """cfg -> the arm dict the keeper loader reads back from a ckpt's meta."""
    (arm,) = build_arms([cfg["model"]], str(cfg["layers"]), [cfg["head"]], [cfg["cap"]],
                        cfg["hidden"], [cfg["feats"]])
    if cfg["qat_w8a8"]:
        arm["name"] += "-qat"
        arm["qat"] = True
    return arm


def name(cfg):
    """The keeper name tools/promote.py gives a kept run: <board>/ckpt/<name>_s<seed>.pt."""
    return _arm(cfg)["name"]


def build_model(arm, dev):
    """One arm's fresh encoder+head on dev; dims from the encoder config plus any
    structural feature columns concatenated at the head input. fp32 (build_encoder
    default) -- the §4 reference dtype."""
    encoder = build_encoder(arm["model"], arm["layers"], device=dev)
    head = build_head(arm.get("head", "bigru"),
                      d_in=encoder.config.hidden_size + feature_dim(arm.get("feats")),
                      hidden=arm.get("hidden", 256)).to(dev)
    return encoder, head


def _pages(ctx, cfg, arm):
    """Tokenizer, (train, val, val scorer) and z-scored feats -- the expensive step,
    shared by every run with the same board, holdout, model, cap, feats, train_limit."""
    b = bench(ctx, cfg)

    def build():
        tok = AutoTokenizer.from_pretrained(arm["model"])
        fz = b.load_features(arm["feats"])[1] if arm["feats"] else None
        ids = b.train_subset(cfg["train_limit"]) if cfg["train_limit"] is not None else None
        train_p = b.build_pages(tok, "train", WINDOW, arm["cap"], fz, ctx.limit, ids=ids)
        val_p = b.build_pages(tok, "val", WINDOW, arm["cap"], fz, ctx.limit)
        return tok, (train_p, val_p, b.build_val_scorer(val_p, ctx.eval_workers)), fz

    key = (cfg["board"], cfg["holdout"], cfg["model"], cfg["cap"], cfg["feats"],
           cfg["train_limit"])
    return cached(ctx, "pages", key, build)


def _test(ctx, b, model, tok, fz, arm, result):
    """Run the trained model on val+test: export the probs, tune the threshold on val,
    score test at 0.5 (headline) and the val-tuned threshold (upside), fold into result."""
    enc, head = model
    infer = lambda page: infer_page(enc, head, page, ctx.dev, ctx.dev.type == "cuda")
    export = lambda fold: b.export_probs(fold, tok, infer, ctx.probs / f"{fold}.jsonl",
                                         WINDOW, arm["cap"], fz, ctx.limit)
    tp, vp = export("test"), export("val")
    grid = [round(0.30 + 0.05 * i, 2) for i in range(9)]           # 0.30..0.70
    vscan = {t: b.score_fold(vp, "val", t)["f1"] for t in grid}
    vthr = max(vscan, key=vscan.get)
    t05, topt = b.score_fold(tp, "test", 0.5), b.score_fold(tp, "test", vthr)
    result.update(test_f1=round(t05["f1"], 6), test_prec=round(t05["prec"], 6),
                  test_rec=round(t05["rec"], 6), test_n=t05["n"],
                  val_thr=vthr, val_f1_opt=round(vscan[vthr], 6),
                  test_f1_opt=round(topt["f1"], 6))
    for name, ss in bench_ops.score_subsets(b, tp, "test", 0.5).items():
        result.update({f"test_{name}_f1": round(ss["f1"], 6), f"test_{name}_n": ss["n"]})
    print(f"test@0.5 {t05['f1']:.4f} | val-opt thr {vthr:.2f} -> test {topt['f1']:.4f}",
          flush=True)


def _quant_eval(ctx, b, model, tok, fz, arm, result, export_test):
    """Post-hoc int8 variants of the trained fp32 model, scored at 0.5: PTQ-dynamic (the
    dead foil, CPU-only int8) and encoder emb-int8 (size lever, ~0 fold). Adds ptq_ /
    emb_int8_ f1 fields (val, + test with export_test) plus emb_int8_dw (max|dw|)."""
    enc, head = model
    cpu = torch.device("cpu")
    enc_ptq = ptq_dynamic(copy.deepcopy(enc).to(cpu))   # dynamic int8 -> CPU only
    head_cpu = copy.deepcopy(head).to(cpu)
    enc_emb = copy.deepcopy(enc)
    result["emb_int8_dw"] = emb_int8_fakequant(enc_emb)    # in place; returns max|dw|
    variants = {"ptq": (enc_ptq, head_cpu, cpu), "emb_int8": (enc_emb, head, ctx.dev)}
    folds = ["val"] + (["test"] if export_test else [])
    for tag, (e, h, d) in variants.items():
        infer = lambda page, e=e, h=h, d=d: infer_page(e, h, page, d, d.type == "cuda")
        for fold in folds:
            pr = b.export_probs(fold, tok, infer, ctx.probs / f"{tag}_{fold}.jsonl",
                                WINDOW, arm["cap"], fz, ctx.limit)
            result[f"{tag}_{fold}_f1"] = round(b.score_fold(pr, fold, 0.5)["f1"], 6)
    print(f"quant-eval val: fp32 {result['best_val']:.4f} | ptq {result['ptq_val_f1']:.4f} "
          f"| emb-int8 {result['emb_int8_val_f1']:.4f}", flush=True)


def run(cfg, ctx):
    arm = _arm(cfg)
    b = bench(ctx, cfg)
    tok, pages, fz = _pages(ctx, cfg, arm)
    torch.manual_seed(cfg["seed"])            # head init is part of the seed
    enc, head = build_model(arm, ctx.dev)
    if cfg["qat_w8a8"]:
        patch_qat_w8a8(enc)
    ckpt = ctx.folder / "ckpt.pt" if cfg["keep"] else None
    hp = {"head_lr": arm["head_lr"]} if "head_lr" in arm else {}
    s = train_encoder(enc, head, *pages, ctx.dev, seed=cfg["seed"], epochs=cfg["epochs"],
                      out_path=ckpt, meta={**arm, "seed": cfg["seed"]}, steps=cfg.get("steps"),
                      **hp)
    result = {"best_val": s["best_val"], "best_epoch": s["best_epoch"]}
    curves = [{"epoch": m["epoch"], "val": m["val"], "block_f1": m["block_f1"], "loss": m["loss"]}
              for m in s["history"]]
    ctx.probs.mkdir(parents=True, exist_ok=True)
    if cfg["export_test"]:
        _test(ctx, b, (enc, head), tok, fz, arm, result)
    if cfg["quant_eval"]:
        _quant_eval(ctx, b, (enc, head), tok, fz, arm, result, cfg["export_test"])
    del enc, head
    if ctx.dev.type == "cuda":
        torch.cuda.empty_cache()
    return result, curves
