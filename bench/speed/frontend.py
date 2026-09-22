"""The model's live-render front-end for the speed harness -- the one render every model
path does identically. Torch-free: parse -> sanitize -> render_tree -> blocks(+DOM source
+features), the inference render (not the labeling render; compute_main_set and html2text
are F1-only and would inflate a speed number). WMB-flavored (wmb sanitize), like the rest
of bench/speed.

`htmlsift` uses all four returns (root/block_src feed serialize); `stages` and
`gpu_throughput` take blocks (+feats) and ignore the rest. Everything after the render --
prep, inference, serialize, batching, timing -- is caller-specific and stays there.
"""
from core.features import apply_zscore, collect_features, group_columns
from core.render import parse, render_tree
from vendors.wmb.adapter.sanitize import sanitize_tree


def build_feats_fn(ck):
    """Live per-page feature builder from a ckpt, or None for a text-only ckpt. Full A/B/C
    is computed on the render, z-scored by the ckpt's carried train-fold stats, then sliced
    to the ckpt's group -- identical to the frozen feats.jsonl path, rendered live."""
    groups = ck.get("feats")
    if not groups:
        return None
    stats, cols = ck["zscore"], group_columns(groups)
    return lambda lines, line_src: apply_zscore(collect_features(lines, line_src), stats)[:, cols]


def render_blocks(html, feats_fn=None):
    """Inference render: html -> (root, blocks, block_src, feats). One render feeds the
    model and (via root/block_src) serialize; feats is None for a text-only ckpt."""
    root = parse(html)
    sanitize_tree(root)
    lines, line_src = render_tree(root)
    blocks, block_src = [], []
    for i, ln in enumerate(lines):
        if ln.strip():
            blocks.append(ln.rstrip())
            block_src.append(line_src[i])
    return root, blocks, block_src, (feats_fn(lines, line_src) if feats_fn else None)
