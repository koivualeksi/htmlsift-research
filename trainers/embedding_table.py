"""Embedding-table trainer: the no-encoder arm -- pool the backbone's frozen
token-embedding table per block and train heads on it, no transformer layers, no window.
Board-agnostic. The fp32 and int8 tables (--emb) are separate arms, tagged
model="<m>-table"/"<m>-table-int8", layers=0; --keep checkpoints the run's single arm (weights
+ config + this board's z-score stats) for cross-board inference. Val-only, to
table_results.jsonl.

    python trainers/embedding_table.py --benchmark wmb --models 97m --emb fp32 \
        --heads xgboost,bigru --feats none,ABC --seeds 1 --device cpu --limit 20 --epochs 1
"""
import argparse

import torch
from transformers import AutoTokenizer

from core.features import apply_zscore, zscore_stats
from trainers.frozen_screen import fit_head
from core.model import build_table_embedder, pool_table_pages
from trainers import ledger
from trainers._args import base_parser, head_parser, parse_feats
from trainers._common import Ctx, setup
from vendors.shared.arms import MODELS, WINDOW


def _tag(model, emb, hidden):
    return (f"{model}-table" + ("-int8" if emb == "int8" else "")
            + (f"-h{hidden}" if hidden != 256 else ""))


def _akey(tag, head, g, seed, tl):
    """Resume/skip key: arm_key plus train_limit when set (mirrors ledger.record_key)."""
    k = ledger.arm_key(tag, 0, head, g, 0, seed)
    return (k + (tl,)) if tl is not None else k


def _record(tag, head, g, seed, best, train_limit):
    rec = {"model": tag, "layers": 0, "head": head, "feats": g, "seed": seed,
           "frozen": True, "best_val": best["val"], "best_epoch": best["epoch"]}
    if train_limit is not None:
        rec["train_limit"] = train_limit
    return rec


def _save(ctx, hf, emb, tag, head_kind, g, seed, head):
    stats = ctx.bench.feature_stats() if g else None
    ck = ctx.out_dir / "ckpt" / f"{tag}-{head_kind}-f{g or 'none'}_s{seed}.pt"
    ck.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"kind": "table", "model": hf, "int8": emb == "int8", "head": head_kind,
                "feats": g, "hidden": ctx.args.hidden, "head_state": head.state_dict(),
                "zscore": stats}, ck)
    print(f"  saved {ck.name}", flush=True)
    ctx.led.push_ckpt(ck, f"{ctx.bench.name}/ckpt/{ck.name}", f"table ckpt {ck.stem}")


def main():
    ap = argparse.ArgumentParser(parents=[base_parser(), head_parser()])
    ap.add_argument("--emb", default="fp32,int8",
                    help="table variants to run (fp32, int8); int8 fake-quants the table")
    ap.add_argument("--keep", action="store_true",
                    help="checkpoint the run's single arm (all its seeds); requires exactly one arm")
    ap.add_argument("--train-limit", default=None,
                    help="subsample training pool to N pages (or a comma list N1,N2,... run in "
                         "sequence) for the data-efficiency ladder; per-N structural z-score refit")
    args = ap.parse_args()

    heads = args.heads.split(",")
    groups = parse_feats(args.feats)
    embs = [e for e in args.emb.split(",") if e]
    train_limits = ([int(x) for x in args.train_limit.split(",")]
                    if args.train_limit else [None])
    if args.keep and len(args.models.split(",")) * len(embs) * len(groups) * len(heads) != 1:
        raise SystemExit("--keep requires exactly one arm (one model/emb/head/feats)")
    if args.keep and len(train_limits) > 1:
        raise SystemExit("--keep is incompatible with a multi-value --train-limit "
                         "(one ckpt name per arm/seed would be overwritten)")
    base = setup(args, "table_results")
    if args.train_limit is not None and not base["bench"].supports_train_limit:
        raise SystemExit(f"--train-limit not supported for {base['bench'].name}")
    feats = {g: base["bench"].load_features(g) for g in groups if g}
    ctx = Ctx(**base)
    print(f"table: {args.models} x {embs} x {heads} x "
          f"{[g or 'none' for g in groups]} on {ctx.dev}", flush=True)

    for model in args.models.split(","):
        hf = MODELS[model][0]
        tok = AutoTokenizer.from_pretrained(hf)
        train_p = ctx.bench.build_pages(tok, "train", WINDOW, limit=args.limit)
        val_p = ctx.bench.build_pages(tok, "val", WINDOW, limit=args.limit)
        ytr = [p["y"] for p in train_p]
        score_val = ctx.bench.build_val_scorer(val_p, args.eval_workers)
        for emb in embs:
            tag = _tag(model, emb, args.hidden)
            if ctx.led.all_done(_akey(tag, head, g, seed, tl)
                                for tl in train_limits for g in groups
                                for head in heads for seed in range(args.seeds)):
                print(f"[skip {tag}: all arms done]", flush=True)
                continue
            embedder = build_table_embedder(hf, int8=(emb == "int8"), device=ctx.dev)
            etr = pool_table_pages(embedder, train_p, ctx.dev)
            eva = pool_table_pages(embedder, val_p, ctx.dev)
            for tl in train_limits:
                keep = None
                if tl is not None:
                    keep = ctx.bench.train_subset(tl)
                idx = ([i for i, p in enumerate(train_p) if p["tid"] in keep]
                       if keep is not None else list(range(len(train_p))))
                train_p_s = [train_p[i] for i in idx]
                etr_s, ytr_s = [etr[i] for i in idx], [ytr[i] for i in idx]
                for g in groups:
                    raw, z = feats[g] if g else (None, None)
                    if g and tl is not None:              # refit z-score on these N pages only
                        sub = {p["tid"] for p in train_p_s}
                        st = zscore_stats([raw[t] for t in sub if t in raw], g)
                        z = {t: apply_zscore(a, st) for t, a in raw.items()}
                    for head in heads:
                        for seed in range(args.seeds):
                            if ctx.led.is_done(_akey(tag, head, g, seed, tl)):
                                continue
                            print(f"[table {tag} {head} {g or 'none'} tl={tl} s{seed}]", flush=True)
                            best, hd = fit_head(head, raw, z, train_p_s, val_p, etr_s, eva, ytr_s,
                                                score_val, ctx.dev, hidden=args.hidden,
                                                epochs=args.epochs, seed=seed,
                                                batch_size=args.batch_size, head_lr=args.head_lr)
                            if hd is not None and args.keep:
                                _save(ctx, hf, emb, tag, head, g, seed, hd)
                            ctx.led.append(_record(tag, head, g, seed, best, tl))
            del embedder, etr, eva
            if ctx.dev.type == "cuda":
                torch.cuda.empty_cache()
            ctx.led.merge_push(f"table {tag}")


if __name__ == "__main__":
    main()
