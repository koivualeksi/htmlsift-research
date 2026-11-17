"""Training-only helpers on top of htmlsift's inference path: labels and training
samples on the page dict, and train-fold feature stats."""

import numpy as np
from htmlsift.core.features import _ORDER, GROUPS, _zscore_idx
from htmlsift.core.prep_page import prep_page, window_starts


def make_samples(input_ids, members, window, stride=None):
    """Training samples (tok_lo, tok_hi, blk_lo, blk_hi). A page within the
    window is one sample over all blocks. A longer page yields one sample per
    window, covering the blocks whose every member token lies inside the window
    -- so the BiGRU trains on the window's block subsequence, matching what the
    inference stitch will hand it."""
    n = len(input_ids)
    if n <= window:
        return [(0, n, 0, len(members))]
    first = [m[0] if m else -1 for m in members]
    last = [m[-1] if m else -1 for m in members]
    for i in range(len(members)):
        if first[i] < 0:                      # token-less block rides along at
            first[i] = last[i] = last[i - 1] if i else 0   # the previous position
    out = []
    for s in window_starts(n, window, stride):
        e = s + window
        inside = [i for i in range(len(members)) if s <= first[i] and last[i] < e]
        if inside:
            out.append((s, e, inside[0], inside[-1] + 1))
    return out


def train_page(tok, tid, blocks, labels, window, stride=None, cap=0, feats=None):
    """htmlsift's prep_page plus y (labels as float32, None without labels) and the
    training samples."""
    page = prep_page(tok, blocks, window, stride=stride, cap=cap, feats=feats, tid=tid)
    page["y"] = None if labels is None else np.array(labels, dtype=np.float32)
    page["samples"] = make_samples(page["ids"], page["members"], window, stride)
    return page


def group_columns(groups):
    """Column indices a group subset occupies within the FULL A/B/C layout that
    feats.jsonl is written in -- so the frozen sweep slices a subset (e.g. 'BC')
    from the one file instead of re-rendering per group."""
    full = [c for g in _ORDER for c in GROUPS[g]]
    want = {c for g in _ORDER if g in groups for c in GROUPS[g]}
    return [i for i, c in enumerate(full) if c in want]


def zscore_stats(rows, groups):
    """(idx, mean, std) for the z-scored columns over stacked TRAIN-fold feature
    rows -- rows is a list of per-page [n_blocks, K] arrays. Computed on train only
    and applied to every fold (apply_zscore), so no val/test statistics leak into
    the normalization. std is floored so a constant column maps to 0, not NaN."""
    idx = _zscore_idx(groups)
    col = np.vstack(rows)[:, idx]
    std = col.std(0)
    std[std < 1e-6] = 1.0
    return idx, col.mean(0), std
