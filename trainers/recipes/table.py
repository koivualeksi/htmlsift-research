"""Recipe table: the no-encoder model -- pool the backbone's frozen token-embedding table
per block and train a head on it, no transformer layers, no window. emb fp32 or int8
(fake-quantised table). keep saves the head + config + this board's z-score stats as
ckpt.pt; export_test reloads it through the keeper loader and scores val + test.

extra: custom adds vendors/custom's pages to the train fold only, repeated extra_repeat
times (0 is the same run without them).
"""
import numpy as np
import torch
from transformers import AutoTokenizer

from htmlsift.core.features import apply_zscore
from htmlsift.extractor import THRESHOLD
from core.loader import build_infer
from core.model import build_head, build_table_embedder, pool_table_pages
from core.training import zscore_stats
from trainers.loop import train_head
from trainers.objective import Objective
from trainers.recipes import bench, cached
from vendors.shared import bench_ops
from vendors.shared.arms import MODELS, WINDOW


def name(cfg):
    """The keeper name tools/promote.py gives a kept run: <board>/ckpt/<name>_s<seed>.pt.
    The shipped mini is 311m-table-bigru-fABC."""
    return (f"{cfg['model']}-table" + ("-int8" if cfg["emb"] == "int8" else "")
            + (f"-h{cfg['hidden']}" if cfg["hidden"] != 256 else "")
            + (f"+{cfg['extra']}{cfg['extra_repeat']}" if cfg["extra"] else "")
            + f"-{cfg['head']}-f{cfg['feats'] or 'none'}"
            + (f"-len{cfg['len_cap']}f{cfg['len_floor']:g}"
               + ("" if cfg.get("len_side", "both") == "both" else cfg["len_side"])
               if cfg.get("len_cap") is not None else "")
            + ("-pn" if cfg.get("page_norm") else "")
            + (f"-fp{cfg['fp_weight']:g}" if cfg.get("fp_weight", 1.0) != 1.0 else "")
            + (f"-box{cfg['fp_box']}" if cfg.get("fp_box") is not None else "")
            + (f"-td{cfg['text_drop']:g}" if cfg.get("text_drop", 0.0) else "")
            + (f"-fd{cfg['feat_drop']:g}" if cfg.get("feat_drop", 0.0) else "")
            + ("-plain" if cfg.get("plain") else ""))


def _extra_src(cfg):
    if cfg["extra"] == "custom":
        from vendors.custom.adapter import extra
        return extra
    return None


def _pages(ctx, cfg):
    """Tokenizer, train/val pages, val scorer and the extra pages, per board x model."""
    def build():
        b = bench(ctx, cfg)
        tok = AutoTokenizer.from_pretrained(MODELS[cfg["model"]][0])
        train_p = b.build_pages(tok, "train", WINDOW, limit=ctx.limit)
        val_p = b.build_pages(tok, "val", WINDOW, limit=ctx.limit)
        src = _extra_src(cfg)
        ext_p = src.build_pages(tok, WINDOW, ctx.limit) if src else []
        clash = {p["tid"] for p in ext_p} & {p["tid"] for p in train_p + val_p}
        if clash:
            raise RuntimeError(f"{len(clash)} extra track_ids collide with {b.name}")
        return train_p, val_p, b.build_val_scorer(val_p, ctx.eval_workers), ext_p

    return cached(ctx, "pages", (cfg["board"], cfg["model"], cfg["extra"], cfg.get("plain")), build)


def _pooled(ctx, cfg, train_p, val_p, ext_p):
    """Per-page pooled table embeddings (train, val, extra) for one model x emb."""
    def build():
        embedder = build_table_embedder(MODELS[cfg["model"]][0], int8=(cfg["emb"] == "int8"),
                                        device=ctx.dev)
        out = tuple(pool_table_pages(embedder, p, ctx.dev) if p else []
                    for p in (train_p, val_p, ext_p))
        del embedder
        return out

    key = (cfg["board"], cfg["model"], cfg["extra"], cfg["emb"], cfg.get("plain"))
    return cached(ctx, "pooled", key, build)


def _feats(ctx, cfg, b, train_p_s):
    """z-scored features {track_id: [n, F]} for this run, or None for text-only. A
    train_limit refits the z-score on those N pages; extra pages use the host board's
    stats, never refit."""
    g = cfg["feats"]
    if not g:
        return None
    raw, z = cached(ctx, "feats", (cfg["board"], cfg.get("holdout"), g),
                    lambda: b.load_features(g))
    if cfg["train_limit"] is not None:
        sub = {p["tid"] for p in train_p_s}
        st = zscore_stats([raw[t] for t in sub if t in raw], g)
        z = {t: apply_zscore(a, st) for t, a in raw.items()}
    src = _extra_src(cfg)
    if src:
        xr, xz = src.load_features(b.feature_stats(), g)
        z = {**z, **xz}
    return z


def _save(ctx, cfg, b, head):
    ck = ctx.folder / "ckpt.pt"
    torch.save({"kind": "table", "model": MODELS[cfg["model"]][0], "int8": cfg["emb"] == "int8",
                "head": cfg["head"], "feats": cfg["feats"], "hidden": cfg["hidden"],
                "head_state": head.state_dict(),
                "zscore": b.feature_stats() if cfg["feats"] else None}, ck)
    print(f"  saved {ck}", flush=True)
    return ck


def _export_test(ctx, b, ck_path, result):
    """Reload the saved ckpt through the keeper loader -- the path bench/accuracy/predict.py
    --ckpt scores -- and fold val + test F1 at the threshold into result. An fp32-table run
    is also scored with its table int8 at inference (predict's --table-int8, the shipped mini)."""
    ck = torch.load(ck_path, map_location=ctx.dev, weights_only=False)
    feats = bench_ops.carried_features(b, ck)
    variants = [("", ck["int8"])] + ([("_int8", True)] if not ck["int8"] else [])
    for sfx, int8 in variants:
        tok, infer, _kind, cap = build_infer({**ck, "int8": int8}, ctx.dev)
        for fold in ("val", "test"):        # val probs ship with the run (errors.py); test stays on the pod
            out = (ctx.folder if fold == "val" else ctx.probs) / f"{fold}{sfx}.jsonl"
            pr = b.export_probs(fold, tok, infer, out, WINDOW, cap, feats, ctx.limit)
            s = b.score_fold(pr, fold, THRESHOLD)
            result.update({f"{fold}{sfx}_f1": round(s["f1"], 6), f"{fold}{sfx}_n": s["n"]})
        print(f"  ckpt{sfx}: val {result[f'val{sfx}_f1']:.4f} "
              f"test {result[f'test{sfx}_f1']:.4f} (n={result[f'test{sfx}_n']})", flush=True)


def run(cfg, ctx):
    b = bench(ctx, cfg)
    train_p, val_p, score_val, ext_p = _pages(ctx, cfg)
    etr, eva, eext = _pooled(ctx, cfg, train_p, val_p, ext_p)
    idx = list(range(len(train_p)))
    if cfg["train_limit"] is not None:
        keep = b.train_subset(cfg["train_limit"])
        idx = [i for i, p in enumerate(train_p) if p["tid"] in keep]
    n = cfg["extra_repeat"] or 0
    tp = [train_p[i] for i in idx] + ext_p * n
    et = [etr[i] for i in idx] + eext * n
    yt = [train_p[i]["y"] for i in idx] + [p["y"] for p in ext_p] * n
    z = _feats(ctx, cfg, b, [train_p[i] for i in idx])
    print(f"[table {cfg['model']} {cfg['emb']} {cfg['head']} {cfg['feats'] or 'none'} "
          f"s{cfg['seed']}: {len(tp)} train pages]", flush=True)

    def cat(pages, emb):
        return [np.concatenate([e, z[p["tid"]]], 1) if z is not None else e
                for p, e in zip(pages, emb)]
    tx, vx = cat(tp, et), cat(val_p, eva)
    torch.manual_seed(cfg["seed"])            # head init is part of the seed
    head = build_head(cfg["head"], d_in=tx[0].shape[1], hidden=cfg["hidden"]).to(ctx.dev)
    s = train_head(head, tx, yt, vx, val_p, score_val, ctx.dev,
                   Objective(cfg, yt, [p["chars"] for p in tp], ctx.dev,
                             [p["box"] for p in tp]),
                   epochs=cfg["epochs"], seed=cfg["seed"], batch_size=cfg["batch_size"],
                   head_lr=cfg["head_lr"])
    result = {"best_val": s["best_val"], "best_epoch": s["best_epoch"]}
    if cfg["extra"]:
        result["extra_pages"] = len(ext_p)
    curves = [{"epoch": i, **m} for i, m in enumerate(s["history"], 1)]
    if cfg["keep"]:
        ck = _save(ctx, cfg, b, head)
        if cfg["export_test"]:
            ctx.probs.mkdir(parents=True, exist_ok=True)
            _export_test(ctx, b, ck, result)
    return result, curves
