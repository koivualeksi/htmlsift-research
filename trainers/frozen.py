"""Frozen screen: encode once per (model, layer), then train every head x feature-set on
the cached block embeddings -- the cheap layers x heads x feats array. Val-only (the
benchmark's val metric); one record per (model, layer, head, feats, seed) to
frozen_results.jsonl. WMB is the selection board.

    python trainers/frozen.py --benchmark wmb --models 311m,97m --layers all \
        --heads bigru,linear,transformer,xgboost --feats none,A,AB,ABC,BC,C --seeds 1
    python trainers/frozen.py --benchmark wmb --models 97m --layers 6 --heads bigru,xgboost \
        --feats none,ABC --seeds 1 --device cpu --limit 20 --epochs 1     # CPU smoke
"""
import argparse
from dataclasses import dataclass

import torch
from transformers import AutoTokenizer

from trainers.frozen_screen import fit_head, pool_pages
from core.model import build_encoder
from trainers import ledger
from trainers._args import base_parser, head_parser, parse_feats
from trainers._common import Ctx, setup
from vendors.shared.arms import MODELS, WINDOW, parse_layers


@dataclass
class FzCtx(Ctx):
    feats: dict     # {group: (raw, z)} per feature group; text-only groups aren't loaded


def _model_pages(ctx, model):
    """Prepared train/val pages (no cap, no per-page feats -- frozen attaches features at
    head training) + train labels + the val scorer, for one model."""
    tok = AutoTokenizer.from_pretrained(MODELS[model][0])
    train_p = ctx.bench.build_pages(tok, "train", WINDOW, limit=ctx.args.limit)
    val_p = ctx.bench.build_pages(tok, "val", WINDOW, limit=ctx.args.limit)
    return (train_p, val_p, [p["y"] for p in train_p],
            ctx.bench.build_val_scorer(val_p, ctx.args.eval_workers))


def _encode(ctx, model, L, train_p, val_p):
    """Cached block embeddings for one (model, layer): one frozen encoder forward per
    page, encoder freed after -- the only expensive step, shared by every head x feats."""
    enc = build_encoder(MODELS[model][0], L, device=ctx.dev)
    enc.requires_grad_(False), enc.eval()
    ac = ctx.dev.type == "cuda"
    etr, eva = pool_pages(enc, train_p, ctx.dev, ac), pool_pages(enc, val_p, ctx.dev, ac)
    del enc
    if ctx.dev.type == "cuda":
        torch.cuda.empty_cache()
    return etr, eva


def _head(ctx, head, g, train_p, val_p, etr, eva, ytr, score_val, seed):
    """Train one head x feature-group on the cached embeddings. xgboost takes raw
    features (scale-free); torch heads take z-scored. g is None for text-only."""
    raw, z = ctx.feats[g] if g else (None, None)
    return fit_head(head, raw, z, train_p, val_p, etr, eva, ytr, score_val, ctx.dev,
                    hidden=ctx.args.hidden, epochs=ctx.args.epochs, seed=seed,
                    batch_size=ctx.args.batch_size, head_lr=ctx.args.head_lr)[0]


def _record(model, L, head, g, seed, best):
    return {"model": model, "layers": L, "head": head, "feats": g, "seed": seed,
            "frozen": True, "best_val": best["val"], "best_epoch": best["epoch"]}


def main():
    ap = argparse.ArgumentParser(parents=[base_parser(), head_parser()])
    ap.add_argument("--layers", default="full",
                    help="'full', 'all' (1..max), or a list/range like 7,11,22 or 1-22")
    args = ap.parse_args()

    heads = args.heads.split(",")
    groups = parse_feats(args.feats)
    base = setup(args, "frozen_results")
    feats = {g: base["bench"].load_features(g) for g in groups if g}
    ctx = FzCtx(**base, feats=feats)
    print(f"frozen: {args.models} x layers {args.layers} x {heads} x "
          f"{[g or 'none' for g in groups]} on {ctx.dev}", flush=True)

    for model in args.models.split(","):
        train_p, val_p, ytr, score_val = _model_pages(ctx, model)
        for L in parse_layers(args.layers, MODELS[model][1]):
            if ctx.led.all_done(ledger.arm_key(model, L, head, g, 0, seed)
                                for g in groups for head in heads for seed in range(args.seeds)):
                print(f"[skip {model}-{L}: all done]", flush=True)
                continue
            etr, eva = _encode(ctx, model, L, train_p, val_p)
            for g in groups:
                for head in heads:
                    for seed in range(args.seeds):
                        if ctx.led.is_done(ledger.arm_key(model, L, head, g, 0, seed)):
                            continue
                        print(f"[frozen {model}-{L} {head} {g or 'none'} s{seed}]", flush=True)
                        best = _head(ctx, head, g, train_p, val_p, etr, eva, ytr, score_val, seed)
                        ctx.led.append(_record(model, L, head, g, seed, best))
            del etr, eva
            if ctx.dev.type == "cuda":
                torch.cuda.empty_cache()
            ctx.led.merge_push(f"frozen {model}-{L}")


if __name__ == "__main__":
    main()
