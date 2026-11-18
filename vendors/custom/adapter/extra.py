"""Custom pages as extra training pages for another board's train fold: prepared pages
and features in the shapes the table trainer consumes. Train-only -- they carry no
eval references, so they can never select or score a model.

Features are z-scored by the host board's train-fold stats, the yardstick the saved
ckpt carries, so adding pages never moves the host's own normalization.
"""
import json
from pathlib import Path

from htmlsift.core.features import apply_zscore

from core.training import group_columns, train_page
from vendors.shared import bench_ops

DATA = Path(__file__).resolve().parents[1] / "data"
BLOCKS = DATA / "blocks.jsonl"
FEATS = DATA / "feats.jsonl"


def build_pages(tok, window, limit=None):
    with open(BLOCKS, encoding="utf-8") as f:
        recs = [json.loads(line) for line in f]
    return [train_page(tok, r["track_id"], r["blocks"], r["labels"], window)
            for r in recs if r["blocks"]][:limit]


def load_features(stats, group):
    """(raw, z-scored) for a group, z-scored on the FULL layout by the host board's
    stats (idx, mean, std) and then sliced -- as bench_ops.load_features shapes them."""
    cols = group_columns(group)
    full = bench_ops.raw_features(FEATS)
    return ({t: a[:, cols] for t, a in full.items()},
            {t: apply_zscore(a, stats)[:, cols] for t, a in full.items()})
