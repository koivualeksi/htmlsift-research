"""Features-only trainer: heads on the structural features alone, no encoder -- the
model- and depth-independent floor, the "why not just XGBoost, skip the encoder"
baseline. Features come off the render, never the backbone, so this is one pass over
groups x heads x seeds (no per-(model, layer) loop). Val-only; records
model="features"/layers=0 to feature_results.jsonl.

    python trainers/features_only.py --benchmark wmb --heads xgboost,bigru \
        --feats A,ABC --seeds 1 --device cpu --limit 20 --epochs 1     # CPU smoke
"""
import argparse

import numpy as np
from transformers import AutoTokenizer

from trainers.frozen_screen import fit_head
from trainers import ledger
from trainers._args import base_parser, head_parser
from trainers._common import Ctx, setup
from vendors.shared.arms import MODELS, WINDOW


def _record(head, g, seed, best):
    return {"model": "features", "layers": 0, "head": head, "feats": g, "seed": seed,
            "frozen": None, "best_val": best["val"], "best_epoch": best["epoch"]}


def main():
    ap = argparse.ArgumentParser(parents=[base_parser(), head_parser()])
    args = ap.parse_args()

    heads = args.heads.split(",")
    groups = [g for g in args.feats.split(",") if g not in ("", "none")]
    if not groups:
        raise SystemExit("features-only needs at least one feature group (--feats)")
    base = setup(args, "feature_results")
    feats = {g: base["bench"].load_features(g) for g in groups}
    ctx = Ctx(**base)
    print(f"features: {heads} x {groups} on {ctx.dev}", flush=True)

    # Any tokenizer: block/label/feature alignment is tokenizer-independent (the
    # encoder never runs here). Hand the heads zero-width embeddings so fit_torch_head's
    # concat yields features only and fit_xgb_head (use_emb=False) ignores them.
    tok = AutoTokenizer.from_pretrained(MODELS["97m"][0])
    train_p = ctx.bench.build_pages(tok, "train", WINDOW, limit=args.limit)
    val_p = ctx.bench.build_pages(tok, "val", WINDOW, limit=args.limit)
    ytr = [p["y"] for p in train_p]
    score_val = ctx.bench.build_val_scorer(val_p, args.eval_workers)
    etr = [np.zeros((len(p["y"]), 0), np.float32) for p in train_p]
    eva = [np.zeros((len(p["y"]), 0), np.float32) for p in val_p]

    for g in groups:
        raw, z = feats[g]
        for head in heads:
            for seed in range(args.seeds):
                if ctx.led.is_done(ledger.arm_key("features", 0, head, g, 0, seed)):
                    continue
                print(f"[features {head} {g} s{seed}]", flush=True)
                best = fit_head(head, raw, z, train_p, val_p, etr, eva, ytr, score_val,
                                ctx.dev, hidden=args.hidden, epochs=args.epochs, seed=seed,
                                batch_size=args.batch_size, head_lr=args.head_lr,
                                use_emb=False)[0]
                ctx.led.append(_record(head, g, seed, best))
        ctx.led.merge_push(f"features {g}")


if __name__ == "__main__":
    main()
