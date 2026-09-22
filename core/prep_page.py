"""
Grad-free page layout: tokenize a page's blocks once and map every token back
to the block it came from, then plan the training windows. No encoder, no
gradients -- deterministic data shaping consumed by the trainer (the samples),
the inference path (the members) and the cap/feature stages.
"""

import numpy as np


def block_spans(blocks):
    """Char span [start, end) of each block within "\\n".join(blocks)."""
    spans, pos = [], 0
    for b in blocks:
        spans.append((pos, pos + len(b)))
        pos += len(b) + 1                     # +1 for the joining newline
    return spans


def page_members(enc, blocks):
    """Per block, the indices of the tokens whose characters fall inside it.

    enc is a fast-tokenizer output carrying offset_mapping and
    special_tokens_mask. One cursor (ti) advances through the non-special tokens
    as the blocks are consumed left to right, so the whole map is a single linear
    pass -- not a search per block. A token joins a block when it overlaps the
    block's characters: the cursor skips tokens that end at or before the block
    start, and the `offset end > bs` guard inside the loop drops zero-width
    tokens (e.g. (0,0) offsets) while keeping a subword that straddles the
    joining newline with the block it reaches into. A block with no tokens of its
    own gets [] and will pool to a zero vector, so members stays aligned with
    blocks and labels.
    """
    offsets = enc["offset_mapping"]
    special = enc["special_tokens_mask"]
    tok_idx = [i for i in range(len(offsets)) if not special[i]]
    members = []
    ti = 0
    for bs, be in block_spans(blocks):
        while ti < len(tok_idx) and offsets[tok_idx[ti]][1] <= bs:
            ti += 1
        j, mem = ti, []
        while j < len(tok_idx) and offsets[tok_idx[j]][0] < be:
            if offsets[tok_idx[j]][1] > bs:
                mem.append(tok_idx[j])
            j += 1
        members.append(mem)
    return members


def window_starts(n, window, stride=None):
    """Start offsets of the windows tiling n tokens (n > window). stride defaults
    to window//2 = 50%% overlap; the last window is snapped back to n-window so
    the tail is always fully covered. Shared with the encoder stitch in model.py,
    so window boundaries are defined in exactly one place (root CLAUDE.md §4)."""
    stride = stride or window // 2
    starts = list(range(0, n - window + 1, stride))
    if not starts:
        starts = [0]
    if starts[-1] != n - window:
        starts.append(n - window)
    return starts


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


def apply_cap(ids, members, k):
    """Keep each block's first k member tokens, dropping the rest from the
    sequence -- a per-block WIDTH cap. Tokens belonging to no block (specials,
    the newline joins) are always kept. Members are renumbered onto the
    compacted ids."""
    keep = np.ones(len(ids), dtype=bool)
    for mem in members:
        if len(mem) > k:
            keep[np.asarray(mem[k:], dtype=np.int64)] = False
    newpos = np.cumsum(keep) - 1
    return ids[keep], [[int(newpos[t]) for t in mem[:k]] for mem in members]


def prep_page(tok, tid, blocks, labels, window, stride=None, cap=0, feats=None):
    """Blocks (+ optional 0/1 labels) -> the page dict the forward consumes:
    token ids, per-block members, labels, and the training-window plan. The page
    carries its own window/stride so training samples and inference cannot drift.
    truncation=False -- the page is windowed here, never truncated by the
    tokenizer. cap>0 applies the per-block width cap before windowing.
    labels=None (inference) leaves y None; samples still cover it.

    feats, when given, is an [n_blocks, K] array (already z-scored on train-fold
    stats, aligned with blocks) carried through to the head concat. The width cap
    drops tokens, not blocks, so feats stays aligned and is never reindexed.
    feats=None runs the arm text-only."""
    enc = tok("\n".join(blocks), add_special_tokens=True,
              return_offsets_mapping=True, return_special_tokens_mask=True,
              return_tensors=None, truncation=False)
    ids = np.array(enc["input_ids"], dtype=np.int64)
    members = page_members(enc, blocks)
    if cap:
        ids, members = apply_cap(ids, members, cap)
    y = None if labels is None else np.array(labels, dtype=np.float32)
    if feats is not None:
        feats = np.asarray(feats, dtype=np.float32)
        assert feats.shape[0] == len(members), (feats.shape[0], len(members))
    return {"tid": tid, "ids": ids, "members": members, "y": y, "feats": feats,
            "window": window, "stride": stride,
            "samples": make_samples(ids, members, window, stride)}
