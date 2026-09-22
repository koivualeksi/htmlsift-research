"""
Block segmentation for DAnIEL, on the core lxml renderer.

Like WCXB, DAnIEL has no DOM annotations -- its gold is external <p>-derived text
-- so this stage produces blocks only; labels are a separate align-to-reference
step. Each page is rendered with core's renderer and the non-empty lines are the
blocks, the exact input a block classifier sees.

render_hidden=False, the opposite of WCXB: DAnIEL is ~2012 static news with no JS
hydration, so there is no display:none-until-hydration main content to recover; a
hidden subtree here is boilerplate. Rendering with no fallback also stress-tests
the renderer -- a page it cannot render is a core bug, and raises.

--feats <groups> also gathers the structural features (core.features) on the same
render and writes them to data/feats.jsonl ({track_id, feats [n_blocks, K]}),
parallel to blocks.jsonl -- the same file and format WMB's and WCXB's blocks.py
--feats produce (needed only for the table-abc arm). Omit it and no feature pass runs.

Reads data/daniel.jsonl, writes data/blocks.jsonl, one record per page:
    {track_id, blocks}

    python vendors/daniel/adapter/blocks.py [--feats ABC]
"""
import argparse
import json
from pathlib import Path

from core import render as cr
from core.features import collect_features

DATA = Path(__file__).resolve().parents[1] / "data"
COMBINED = DATA / "daniel.jsonl"
BLOCKS = DATA / "blocks.jsonl"
FEATS = DATA / "feats.jsonl"


def render_page(html, feats=None):
    """Raw page HTML -> blocks (the non-empty rendered lines). render_hidden is off:
    DAnIEL is static 2012 news, so display:none is boilerplate. feats is a feature-
    group string (e.g. "ABC"): when given, features are gathered on the SAME render
    and returned as (blocks, feat_array [n_blocks, K]); None skips the pass.
    collect_features filters non-empty lines with the same rule as blocks, so its
    rows align 1:1 with blocks -- asserted."""
    if not html or not html.strip():
        lines, line_src = [], []
    else:
        lines, line_src = cr.render_tree(cr.parse(html), render_hidden=False)
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

    assert n == 1689, f"expected 1689 records, got {n}"
    print(f"{BLOCKS.name}: {n} pages, {total_blocks} blocks "
          f"({total_blocks / n:.1f}/page), {n_empty} empty renders")
    if args.feats:
        print(f"  {FEATS.name}: features [{args.feats}] on the same render")


if __name__ == "__main__":
    main()
