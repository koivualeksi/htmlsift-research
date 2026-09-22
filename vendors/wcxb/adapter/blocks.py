"""
Block segmentation for WCXB, on the core lxml renderer.

WCXB has no DOM annotations (no cc-select), so unlike WMB this stage produces
blocks only -- no labels. Each page is rendered with core's renderer and the
non-empty lines are the blocks, the exact input zero-shot inference sees. Labels,
if an experiment trains on WCXB-dev, are a separate step that aligns the
plain-text reference to these blocks.

No sanitize: WMB's sanitizer strips cc-select and Immersive-Translate residue,
artifacts of the annotators' browser that wild pages do not carry (a verified
no-op there) and that live in WMB's adapter. Rendering wild pages through core
with no fallback also stress-tests the renderer -- a page it cannot render is a
core bug, and raises.

Reads data/wcxb.jsonl, writes data/blocks.jsonl, one record per page:
    {track_id, blocks}

--feats <groups> also gathers the structural features (core.features) on the same
render and writes them to data/feats.jsonl ({track_id, feats [n_blocks, K]}),
parallel to blocks.jsonl -- the same file and format WMB's blocks.py --feats
produces. Omit it and no feature pass runs.

    python vendors/wcxb/adapter/blocks.py [--feats ABC]
"""
import argparse
import json
from pathlib import Path

from core import render as cr
from core.features import collect_features

DATA = Path(__file__).resolve().parents[1] / "data"
COMBINED = DATA / "wcxb.jsonl"
BLOCKS = DATA / "blocks.jsonl"
FEATS = DATA / "feats.jsonl"


def render_page(html, feats=None):
    """Raw page HTML -> blocks (the non-empty rendered lines). render_hidden is on:
    WCXB's reference is captured from a hydrated DOM, so display:none-until-hydration
    content is main content here. feats is a feature-group string (e.g. "ABC"): when
    given, features are gathered on the SAME render and returned as (blocks, feat_array
    [n_blocks, K]); None skips the pass. collect_features filters non-empty lines with
    the same rule as `keep`, so its rows align 1:1 with blocks -- asserted."""
    if not html or not html.strip():
        lines, line_src = [], []
    else:
        lines, line_src = cr.render_tree(cr.parse(html), render_hidden=True)
    keep = [i for i, ln in enumerate(lines) if ln.strip()]
    blocks = [lines[i].rstrip() for i in keep]
    if feats is None:
        return blocks
    fa = collect_features(lines, line_src, feats)
    assert fa.shape[0] == len(blocks), (fa.shape[0], len(blocks))
    return blocks, fa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default=None,
                    help="feature groups to gather on the same render (e.g. ABC); "
                         "writes feats.jsonl. Omit to skip the feature pass.")
    args = ap.parse_args()

    n = n_empty = total_blocks = 0
    ffeat = open(FEATS, "w", encoding="utf-8") if args.feats else None
    with open(COMBINED, encoding="utf-8") as fin, \
            open(BLOCKS, "w", encoding="utf-8") as fout:
        for line in fin:
            r = json.loads(line)
            out = render_page(r["html"], args.feats)
            blocks = out[0] if args.feats else out
            fout.write(json.dumps({"track_id": r["track_id"],
                                   "blocks": blocks}, ensure_ascii=False) + "\n")
            if ffeat is not None:
                ffeat.write(json.dumps({"track_id": r["track_id"],
                                        "feats": out[1].round(4).tolist()}) + "\n")
            n += 1
            n_empty += not blocks
            total_blocks += len(blocks)
    if ffeat is not None:
        ffeat.close()

    assert n == 2008, f"expected 2008 records, got {n}"
    print(f"{BLOCKS.name}: {n} pages, {total_blocks} blocks "
          f"({total_blocks / n:.1f}/page), {n_empty} empty renders")
    if args.feats:
        print(f"  {FEATS.name}: features [{args.feats}] on the same render")


if __name__ == "__main__":
    main()
