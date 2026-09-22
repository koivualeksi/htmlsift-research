"""
Structural block features for the table-embedding arm: the signal a mean-pooled
bag of token embeddings cannot carry, and nothing the rendered text already shows.

Computed post-hoc from core.render_tree's line_src (the DOM elements behind each
block) -- no second parse, no marker injection, no change to the renderer.
Booleans for presence; only the two genuine scalars (depth, log_chars) are
z-scored, on train-fold stats.

A column earns its place under one rule: it is absent from the tokenized block
text AND not recoverable by the BiGRU. Tag structure is flattened by markdown, and
any marker that survives is diluted by the per-block mean-pool; DOM depth and
link-ancestry never appear in the rendered text at all (the renderer prints an <a>
as bare text); block length is destroyed the moment the tokens are averaged.

Dropped, with reasons in docs/PROTOCOL.md: class/id name heuristics (expensive
per-ancestor string scan, and the head learns boilerplate indirectly from A across
pages), the marker flags (heading_level, is_bullet, ... -- already in A and in the
text as markdown markers), and position (the BiGRU carries order). The three kept
groups are relettered A/B/C for a contiguous read; group C (log_chars) was the
archive's group D, and the archive's C (the dropped class heuristics) has no
counterpart here, so the letters never collide.

This module is the manifest -- column order and group map live here and nowhere
else. The collector and the trainer's --feats selector both read from it.
"""

import math

import numpy as np

# Group A -- ancestor-or-self tag presence, 30 booleans. Semantic containers that
# markdown flattens away, plus block tags whose surviving text marker the
# mean-pool dilutes.
TAG_VOCAB = [
    "main", "article", "section", "nav", "header", "footer", "aside",
    "h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "ul", "ol", "dl",
    "table", "tr", "td", "th", "blockquote", "pre", "code",
    "figure", "figcaption", "form", "button", "div",
]

GROUPS = {
    "A": [f"tag_{t}" for t in TAG_VOCAB],   # ancestor/self tag presence (boolean)
    "B": ["depth", "has_link"],             # DOM nesting depth (scalar); an <a> ancestor (boolean)
    "C": ["log_chars"],                     # block length -- what the mean-pool averages away
}

FEATURE_NAMES = [name for g in ("A", "B", "C") for name in GROUPS[g]]

# The only unbounded columns; z-scored with train-fold stats. Booleans and
# already-bounded columns stay raw.
ZSCORE_COLS = ["depth", "log_chars"]

_ORDER = ("A", "B", "C")
_VOCAB = set(TAG_VOCAB)


def collect_features(lines, line_src, groups=_ORDER):
    """(lines, line_src) from render_tree -> [n_blocks, K] float32, one row per
    non-empty line (the block unit -- same non-empty-line rule as the labels). Columns are the
    requested groups in manifest order; only those groups are computed, so a
    text-only arm passes groups=() and pays nothing. A, B and C come off the DOM
    render already recorded in line_src; the ancestor chain of each element is
    cached and shared across blocks (below)."""
    groups = [g for g in _ORDER if g in groups]
    cols = [c for g in groups for c in GROUPS[g]]
    col = {c: i for i, c in enumerate(cols)}
    blocks = [(ln, src) for ln, src in zip(lines, line_src) if ln.strip()]
    out = np.zeros((len(blocks), len(cols)), dtype=np.float32)
    need_a, need_b, need_c = "A" in groups, "B" in groups, "C" in groups

    # element -> (vocab tags, depth, link) for its ancestor-or-self chain, computed
    # once and reused. Every block re-reaches the same container chain to root, so
    # caching collapses that repetition: the walk stops at the first cached
    # ancestor. Keyed by the element object -- the cache holds the reference, which
    # keeps the lxml proxy (and thus its identity) stable for the call.
    chain = {}

    def chain_info(e):
        stack = []
        node = e
        while node is not None and isinstance(node.tag, str) and node not in chain:
            stack.append(node)
            node = node.getparent()
        if node is not None and isinstance(node.tag, str):
            tags, depth, link = chain[node]
        else:
            tags, depth, link = frozenset(), 0, False
        for nd in reversed(stack):                     # fold from cached base to e
            t = nd.tag
            if need_a and t in _VOCAB and t not in tags:
                tags = tags | {t}
            depth += 1
            link = link or t == "a"
            chain[nd] = (tags, depth, link)
        return tags, depth, link

    for bi, (text, src) in enumerate(blocks):
        if need_a or need_b:
            btags, depth_sum, n_src, has_link = set(), 0, 0, False
            for e in src:
                et, ed, el = chain_info(e)
                btags |= et
                depth_sum += ed
                n_src += 1
                has_link = has_link or el
            if need_a:
                for t in btags:
                    out[bi, col[f"tag_{t}"]] = 1.0
            if need_b:
                out[bi, col["depth"]] = depth_sum / n_src if n_src else 0.0
                out[bi, col["has_link"]] = float(has_link)
        if need_c:
            out[bi, col["log_chars"]] = math.log1p(len(text))
    return out


def feature_dim(groups):
    """Number of feature columns a group selection produces (None/"" -> 0). Sizes
    the head's d_in without hardcoding K."""
    return sum(len(GROUPS[g]) for g in _ORDER if groups and g in groups)


def group_columns(groups):
    """Column indices a group subset occupies within the FULL A/B/C layout that
    feats.jsonl is written in -- so the frozen sweep slices a subset (e.g. 'BC')
    from the one file instead of re-rendering per group."""
    full = [c for g in _ORDER for c in GROUPS[g]]
    want = {c for g in _ORDER if g in groups for c in GROUPS[g]}
    return [i for i, c in enumerate(full) if c in want]


def _zscore_idx(groups):
    """Column positions of the unbounded scalars (ZSCORE_COLS) within a group
    selection's layout."""
    cols = [c for g in _ORDER if g in groups for c in GROUPS[g]]
    return [i for i, c in enumerate(cols) if c in ZSCORE_COLS]


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


def apply_zscore(fa, stats):
    """A copy of a page's [n_blocks, K] array with the z-scored columns normalized
    by train-fold stats; booleans and bounded columns untouched."""
    idx, mean, std = stats
    out = fa.copy()
    out[:, idx] = (out[:, idx] - mean) / std
    return out
