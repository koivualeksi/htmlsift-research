"""Training-only helpers on top of htmlsift's inference path: labels and training
samples on the page dict, and train-fold feature stats."""

import numpy as np
from htmlsift.core.features import _ORDER, GROUPS, ZSCORE_COLS
from htmlsift.core.prep_page import prep_page, window_starts

# Research-only groups, screened here before any enters the package: appended after the
# package's, so a package selection keeps its column positions in a file that also
# carries them. Built by vendors/wmb/adapter/blocks.py: D (hint_features), E
# (container_features). E's two unbounded scalars join ZSCORE_COLS; the package's own
# z-score positions are unchanged because research columns only ever follow its columns.
GROUPS = {**GROUPS, "D": ["hidden_hint", "button_like"],
          "E": ["cont_log_chars", "cont_link_density", "cont_rows_log", "row_pos", "cont_pos"]}
_ORDER = (*_ORDER, "D", "E")
ZSCORE_COLS = [*ZSCORE_COLS, "cont_log_chars", "cont_rows_log"]


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


def train_page(tok, tid, blocks, labels, window, stride=None, cap=0, feats=None, box=None):
    """htmlsift's prep_page plus y (labels as float32, None without labels), chars (each
    block's length, for length-weighted losses), box (each block's gold-free box size,
    for fp_box; None on boards that do not build it) and the training samples."""
    page = prep_page(tok, blocks, window, stride=stride, cap=cap, feats=feats, tid=tid)
    page["y"] = None if labels is None else np.array(labels, dtype=np.float32)
    page["chars"] = np.array([len(b) for b in blocks], dtype=np.int64)
    page["box"] = None if box is None else np.array(box, dtype=np.int64)
    page["samples"] = make_samples(page["ids"], page["members"], window, stride)
    return page


def feature_dim(groups):
    """htmlsift.core.features.feature_dim over the research groups too."""
    return sum(len(GROUPS[g]) for g in _ORDER if groups and g in groups)


def group_columns(groups):
    """Column indices a group subset occupies within the FULL A/B/C(/D) layout that
    feats.jsonl is written in -- so the frozen sweep slices a subset (e.g. 'BC')
    from the one file instead of re-rendering per group."""
    full = [c for g in _ORDER for c in GROUPS[g]]
    want = {c for g in _ORDER if g in groups for c in GROUPS[g]}
    return [i for i, c in enumerate(full) if c in want]


def zscore_stats(rows, groups, full=False):
    """(idx, mean, std) for the z-scored columns over stacked TRAIN-fold feature
    rows -- rows is a list of per-page [n_blocks, K] arrays. Computed on train only
    and applied to every fold (apply_zscore), so no val/test statistics leak into
    the normalization. std is floored so a constant column maps to 0, not NaN.
    full: rows are in the full feats.jsonl layout, not the group's own, so idx points
    there (the stats a ckpt carries)."""
    cols = [c for g in _ORDER if g in groups for c in GROUPS[g]]
    idx = [i for i, c in enumerate(cols) if c in ZSCORE_COLS]
    if full:
        idx = [group_columns(groups)[i] for i in idx]
    col = np.vstack(rows)[:, idx]
    std = col.std(0)
    std[std < 1e-6] = 1.0
    return idx, col.mean(0), std
