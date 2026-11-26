"""Shared Benchmark method bodies: the parts of adapter/benchmark.py that are identical
across every board, differing only in two board primitives -- a fold's training ids
(_train_ids) and a fold's {tid: blocks} accessor (_fold_blocks). A board binds those and
delegates the rest here, so the copy is written once. Kept out of shared/benchmark.py to
leave the Protocol a load-time leaf (it imports only typing; these need core).
"""
import json

import numpy as np

from htmlsift.core.features import apply_zscore
from core.training import group_columns, train_page, zscore_stats


def raw_features(feats_path):
    """{track_id: [n_blocks, K]} over the full A/B/C layout from feats.jsonl, empty
    renders (no rows) dropped. The caller owns the cache -- this only reads."""
    raw = {}
    with open(feats_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            a = np.asarray(r["feats"], dtype=np.float32)
            if a.ndim == 2 and a.shape[0]:
                raw[r["track_id"]] = a
    return raw


def load_features(raw_full, train_ids, group):
    cols = group_columns(group)
    raw = {t: a[:, cols] for t, a in raw_full.items()}
    stats = zscore_stats([raw[t] for t in train_ids if t in raw], group)
    return raw, {t: apply_zscore(a, stats) for t, a in raw.items()}


def feature_stats(raw_full, train_ids, group="ABC"):
    return zscore_stats([raw_full[t] for t in train_ids if t in raw_full], group, full=True)


def score_subsets(bench, probs, fold, thr):
    """{name: score_fold dict} for each id subset the board names for this fold; most boards
    name none."""
    subs = bench.subsets(fold) if hasattr(bench, "subsets") else {}
    return {name: bench.score_fold({t: p for t, p in probs.items() if t in ids}, fold, thr)
            for name, ids in subs.items()}


def carried_features(bench, ck):
    """Target board's features for a ckpt trained with them, z-scored by the ckpt's
    CARRIED (training-board) stats then sliced to the group -- a foreign board normalized
    by the training yardstick, never its own (as bench/speed/frontend.py:build_feats_fn).
    apply_zscore runs on the FULL layout before the slice, since ck['zscore'] indexes it.
    Only table ckpts carry stats; the encoder keeper is text-only, so an encoder ckpt with
    features cross-board is unsupported."""
    group = ck.get("feats")
    if not group:
        return None
    stats = ck.get("zscore")
    if stats is None:
        raise SystemExit("ckpt has features but no carried z-score stats "
                         "(encoder+features cross-board unsupported; keepers are text-only)")
    cols = group_columns(group)
    return {tid: apply_zscore(a, stats)[:, cols] for tid, a in bench.raw_features().items()}


def export_probs(fold_blocks, fold, tok, infer_fn, out_path, window, cap=0, feats=None, limit=None):
    """{track_id: probs} for every fold id (empties -> []), also written to out_path as
    {track_id, probs} jsonl. fold_blocks(fold) -> ordered {tid: blocks}; infer_fn(page) ->
    per-block probs wraps the trainer's model, so this stays model-free."""
    out = {}
    with open(out_path, "w", encoding="utf-8") as f:
        for tid, blocks in list(fold_blocks(fold).items())[:limit]:
            if blocks:
                page = train_page(tok, tid, blocks, None, window, cap=cap,
                                  feats=(feats[tid] if feats else None))
                probs = [round(float(x), 6) for x in infer_fn(page)]
            else:
                probs = []
            f.write(json.dumps({"track_id": tid, "probs": probs}) + "\n")
            out[tid] = probs
    return out
