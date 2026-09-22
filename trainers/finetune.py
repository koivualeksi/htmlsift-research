"""Fine-tune trainer: train each (arm x seed) encoder+head over a benchmark, record the
val score, resume/push through the ledger, and optionally export + score test
(--export-test) and post-hoc int8 variants (--quant-eval).

    python trainers/finetune.py --benchmark wmb --models 311m --layers 11 --seeds 3
    python trainers/finetune.py --benchmark wmb --models 311m --layers 11 --seeds 3 --export-test
    python trainers/finetune.py --benchmark wmb --models 97m --layers 6 --seeds 1 \
        --device cpu --limit 20 --epochs 1        # CPU smoke
"""
import argparse
import copy
from dataclasses import dataclass
from itertools import groupby

import torch
from transformers import AutoTokenizer

from core.features import feature_dim
from core.model import build_encoder, build_head, infer_page
from core.quant import emb_int8_fakequant, patch_qat_w8a8, ptq_dynamic
from trainers.train import train
from trainers import ledger
from trainers._args import base_parser, parse_feats
from trainers._common import Ctx, setup
from vendors.shared.arms import SHORT, WINDOW, build_arms
from vendors.shared.benchmark import load_benchmark


@dataclass
class FtCtx(Ctx):
    feats_z: dict
    keep: bool


def build_model(arm, dev):
    """One arm's fresh encoder+head on dev; dims from the encoder config plus any
    structural feature columns concatenated at the head input. fp32 (build_encoder
    default) -- the §4 reference dtype."""
    encoder = build_encoder(arm["model"], arm["layers"], device=dev)
    head = build_head(arm.get("head", "bigru"),
                      d_in=encoder.config.hidden_size + feature_dim(arm.get("feats")),
                      hidden=arm.get("hidden", 256)).to(dev)
    return encoder, head


def _key(arm, seed, train_limit=None, holdout=None):
    base = ledger.arm_key(SHORT[arm["model"]], arm["layers"], arm["head"],
                          arm["feats"], arm["cap"], seed)
    if train_limit is not None:
        base = base + (train_limit,)
    if holdout is not None:
        base = base + (holdout,)
    return base


def _record(arm, seed, s):
    """One (arm, seed) result: identity + val summary + per-epoch curves."""
    rec = {"model": SHORT[arm["model"]], "layers": arm["layers"], "head": arm["head"],
           "feats": arm["feats"], "cap": arm["cap"], "seed": seed,
           "best_val": s["best_val"], "best_epoch": s["best_epoch"],
           "val_by_epoch": [round(m["val"], 6) for m in s["history"]],
           "f1_by_epoch": [round(m["block_f1"], 6) for m in s["history"]],
           "loss_by_epoch": [round(m["loss"], 6) for m in s["history"]],
           "frozen": False}
    if "qat" in arm:
        rec["qat"] = True
    return rec


def _page_group(ctx, arm, train_limit):
    """Prepared train/val pages + val scorer + z-scored feats for a (model, cap, feats)
    group -- built once and shared by every arm×seed in the group (the expensive step)."""
    tok = AutoTokenizer.from_pretrained(arm["model"])
    fz = ctx.feats_z.get(arm["feats"])
    train_ids = None
    if train_limit is not None:
        train_ids = ctx.bench.train_subset(train_limit)
    train_p = ctx.bench.build_pages(tok, "train", WINDOW, arm["cap"], fz,
                                    ctx.args.limit, ids=train_ids)
    val_p = ctx.bench.build_pages(tok, "val", WINDOW, arm["cap"], fz, ctx.args.limit)
    return tok, (train_p, val_p, ctx.bench.build_val_scorer(val_p, ctx.args.eval_workers)), fz


def _train(ctx, arm, seed, pages):
    """Build one arm's model and fine-tune it on the group's pages; return (model, record).
    Writes a checkpoint when the arm is a keeper (--keep). Does not free the model -- the
    caller runs any post-hoc scoring first."""
    enc, head = build_model(arm, ctx.dev)
    if ctx.args.qat_w8a8:
        patch_qat_w8a8(enc)
    ckpt = ctx.out_dir / "ckpt" / f"{arm['name']}_s{seed}.pt" if ctx.keep else None
    if ckpt:
        ckpt.parent.mkdir(parents=True, exist_ok=True)
    hp = {"head_lr": arm["head_lr"]} if "head_lr" in arm else {}
    s = train(enc, head, *pages, ctx.dev, seed=seed, epochs=ctx.args.epochs,
              out_path=ckpt, meta={**arm, "seed": seed}, **hp)
    if ckpt:
        ctx.led.push_ckpt(ckpt, f"{ctx.bench.name}/ckpt/{ckpt.name}", f"keeper {arm['name']} s{seed}")
    return (enc, head), _record(arm, seed, s)


def _test(ctx, model, tok, fz, arm, seed, rec):
    """Run the trained model on val+test: export the probs, tune the threshold on val,
    score test at 0.5 (headline) and the val-tuned threshold (upside), fold into rec."""
    enc, head = model
    pb = ctx.out_dir / "probs"
    pb.mkdir(parents=True, exist_ok=True)
    infer = lambda page: infer_page(enc, head, page, ctx.dev, ctx.dev.type == "cuda")
    export = lambda fold: ctx.bench.export_probs(
        fold, tok, infer, pb / f"{arm['name']}_{fold}_s{seed}.jsonl",
        WINDOW, arm["cap"], fz, ctx.args.limit)
    tp, vp = export("test"), export("val")
    grid = [round(0.30 + 0.05 * i, 2) for i in range(9)]           # 0.30..0.70
    vscan = {t: ctx.bench.score_fold(vp, "val", t)["f1"] for t in grid}
    vthr = max(vscan, key=vscan.get)
    t05, topt = ctx.bench.score_fold(tp, "test", 0.5), ctx.bench.score_fold(tp, "test", vthr)
    rec.update(test_f1=round(t05["f1"], 6), test_prec=round(t05["prec"], 6),
               test_rec=round(t05["rec"], 6), test_n=t05["n"],
               val_thr=vthr, val_f1_opt=round(vscan[vthr], 6),
               test_f1_opt=round(topt["f1"], 6))
    print(f"[{arm['name']} s{seed}] test@0.5 {t05['f1']:.4f} "
          f"| val-opt thr {vthr:.2f} -> test {topt['f1']:.4f}", flush=True)


def _quant_eval(ctx, model, tok, fz, arm, seed, rec):
    """Post-hoc int8 variants of the trained fp32 model, scored at 0.5: PTQ-dynamic (the
    dead foil, CPU-only int8) and encoder emb-int8 (size lever, ~0 fold). Adds ptq_ /
    emb_int8_ f1 fields (val, + test with --export-test) plus emb_int8_dw (max|dw|)."""
    enc, head = model
    pb = ctx.out_dir / "probs"
    pb.mkdir(parents=True, exist_ok=True)
    cpu = torch.device("cpu")
    enc_ptq = ptq_dynamic(copy.deepcopy(enc).to(cpu))   # dynamic int8 -> CPU only
    head_cpu = copy.deepcopy(head).to(cpu)
    enc_emb = copy.deepcopy(enc)
    rec["emb_int8_dw"] = emb_int8_fakequant(enc_emb)    # in place; returns max|dw|
    variants = {"ptq": (enc_ptq, head_cpu, cpu), "emb_int8": (enc_emb, head, ctx.dev)}
    folds = ["val"] + (["test"] if ctx.args.export_test else [])
    for tag, (e, h, d) in variants.items():
        infer = lambda page, e=e, h=h, d=d: infer_page(e, h, page, d, d.type == "cuda")
        for fold in folds:
            pr = ctx.bench.export_probs(fold, tok, infer,
                                        pb / f"{arm['name']}_{tag}{fold}_s{seed}.jsonl",
                                        WINDOW, arm["cap"], fz, ctx.args.limit)
            rec[f"{tag}_{fold}_f1"] = round(ctx.bench.score_fold(pr, fold, 0.5)["f1"], 6)
    print(f"[{arm['name']} s{seed}] quant-eval val: fp32 {rec['best_val']:.4f} "
          f"| ptq {rec['ptq_val_f1']:.4f} | emb-int8 {rec['emb_int8_val_f1']:.4f}", flush=True)


def _run_board(args, tag, arms, seeds, train_limits, holdout):
    """One benchmark instance (its own ledger namespace) trained over the arm x seed x
    train_limit grid. holdout is the DAnIEL held-out language (None otherwise); it selects
    the benchmark, is part of the ledger key, and rides into each record. Its train set
    fixes the feature z-scores, so feats_z is (re)built here per run."""
    args.holdout = holdout
    base = setup(args, tag)
    feats_z = {g: base["bench"].load_features(g)[1]
               for g in {a["feats"] for a in arms if a["feats"]}}
    ctx = FtCtx(**base, feats_z=feats_z, keep=args.keep)
    print(f"{len(arms)} arms x seeds {seeds}"
          + (f" holdout={holdout}" if holdout else "") + f" on {ctx.dev}", flush=True)

    page_key = lambda a: (a["model"], a["cap"], a["feats"] or "")
    for tl in train_limits:
        for _, group in groupby(sorted(arms, key=page_key), page_key):
            group = list(group)
            if ctx.led.all_done(_key(arm, seed, tl, holdout) for arm in group for seed in seeds):
                print(f"[skip {group[0]['model']} group tl={tl} holdout={holdout}: all arms done]", flush=True)
                continue
            tok, pages, fz = _page_group(ctx, group[0], tl)
            for arm in group:
                for seed in seeds:
                    if ctx.led.is_done(_key(arm, seed, tl, holdout)):
                        print(f"[skip {arm['name']} s{seed} tl={tl} holdout={holdout}: done]", flush=True)
                        continue
                    model, rec = _train(ctx, arm, seed, pages)
                    if tl is not None:
                        rec["train_limit"] = tl
                    if holdout:
                        rec["holdout"] = holdout
                    if args.export_test:
                        _test(ctx, model, tok, fz, arm, seed, rec)
                    if args.quant_eval:
                        _quant_eval(ctx, model, tok, fz, arm, seed, rec)
                    ctx.led.append(rec)
                    ctx.led.merge_push(f"finetune {arm['name']} s{seed} tl={tl} holdout={holdout}")
                    del model
                    if ctx.dev.type == "cuda":
                        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(parents=[base_parser()])
    ap.add_argument("--layers", default="full",
                    help="'full', 'all' (1..max), or a list/range like 7,11,22 or 1-22")
    ap.add_argument("--caps", default="0")
    ap.add_argument("--seed-ids", default="",
                    help="run exactly these seeds (comma list) instead of 0..seeds-1")
    ap.add_argument("--keep", action="store_true",
                    help="checkpoint the run's single arm (all its seeds); requires exactly one arm")
    ap.add_argument("--qat-w8a8", action="store_true",
                    help="quantization-aware training: W8A8 fake-quant the encoder body")
    ap.add_argument("--export-test", action="store_true",
                    help="also export test + val probs per arm and score test (@0.5 and "
                         "the val-tuned threshold) into the record")
    ap.add_argument("--quant-eval", action="store_true",
                    help="after training, score post-hoc int8 variants of the fp32 model: "
                         "PTQ-dynamic (the dead foil) + encoder emb-int8 (size lever). "
                         "Adds ptq_/emb_int8_ f1 fields (val, + test with --export-test)")
    ap.add_argument("--train-limit", default=None,
                    help="subsample training pool to N pages (or a comma list N1,N2,... "
                         "run in sequence), stratified by level to match val (nested subsets)")
    ap.add_argument("--lolo", action="store_true",
                    help="DAnIEL leave-one-language-out: train all five held-out languages "
                         "in one run; results only, no keepers")
    args = ap.parse_args()
    bench = load_benchmark(args.benchmark)
    if args.lolo:
        if not bench.holdout_groups:
            raise SystemExit(f"--lolo needs a board with holdout groups; {bench.name} has none")
        if args.holdout or args.keep:
            raise SystemExit("--lolo iterates all groups, results-only: drop --holdout/--keep")
    if args.holdout and not bench.holdout_groups:
        raise SystemExit(f"--holdout needs a board with holdout groups; {bench.name} has none")
    if bench.holdout_groups and not args.holdout and not args.lolo:
        raise SystemExit(f"{bench.name} needs --holdout <group> or --lolo")
    if args.train_limit is not None and not bench.supports_train_limit:
        raise SystemExit(f"--train-limit not supported for {bench.name}")
    if args.quant_eval and args.qat_w8a8:
        raise SystemExit("--quant-eval scores int8 variants against the fp32 model; "
                         "incompatible with --qat-w8a8")

    feats = parse_feats(args.feats)
    arms = build_arms(args.models.split(","), args.layers, args.heads.split(","),
                      [int(c) for c in args.caps.split(",")], args.hidden, feats)
    if args.qat_w8a8:
        for a in arms:
            a["name"] += "-qat"
            a["qat"] = True
    if args.keep and len(arms) != 1:
        raise SystemExit(f"--keep requires exactly one arm (got {len(arms)})")
    seeds = ([int(s) for s in args.seed_ids.split(",")] if args.seed_ids
             else list(range(args.seeds)))
    train_limits = ([int(x) for x in args.train_limit.split(",")]
                    if args.train_limit else [None])
    if args.keep and len(train_limits) > 1:
        raise SystemExit("--keep is incompatible with a multi-value --train-limit "
                         "(one ckpt name per arm/seed would be overwritten)")

    holdouts = [args.holdout]
    if args.lolo:
        holdouts = bench.holdout_groups
    tag = "finetune_qat_results" if args.qat_w8a8 else "finetune_results"
    for holdout in holdouts:
        _run_board(args, tag, arms, seeds, train_limits, holdout)


if __name__ == "__main__":
    main()
