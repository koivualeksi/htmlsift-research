"""
Block segmentation and provenance labels for WMB, on the core lxml renderer.

label_page parses the raw page once, marks which elements are the cc-selected
main content, sanitizes the tree in place, and renders it with core's provenance
renderer. A line is main iff one of its source elements is in the main set. The
element objects survive the in-place sanitize, so the mapping from rendered line
to cc-select decision is exact -- no marker injection, no alignment gate, no
matcher fallback.

Reads data/wmb.jsonl, writes data/blocks.jsonl, one record per page:
    {track_id, blocks, labels}
labels is a 0/1 list parallel to blocks. A page with no cc-select (no annotated
main content) gets all-zero labels.

--feats <groups> also gathers the structural features (core.features) on the same
render and writes them to data/feats.jsonl ({track_id, feats [n_blocks, K]}),
parallel to blocks.jsonl. Omit it and no feature pass runs.

    python vendors/wmb/adapter/blocks.py [--feats ABC]
"""
import argparse
import json
from pathlib import Path

from core import render as cr
from core.features import collect_features
from vendors.wmb.adapter.sanitize import sanitize_tree

WMB = Path(__file__).resolve().parents[1]
DATA = WMB / "data"
COMBINED = DATA / "wmb.jsonl"
BLOCKS = DATA / "blocks.jsonl"
FEATS = DATA / "feats.jsonl"

SELECT_ATTR = "cc-select"                         # upstream main_html.SELECT_ATTR


def compute_main_set(root):
    """The set of element objects kept by upstream extract_main_html -- computed
    on the raw tree, before sanitize strips cc-select. Reimplemented (not called)
    because extract_main_html returns pruned HTML and loses element identity,
    which the per-line labeling needs. Mirrors upstream exactly; validated by the
    main-block reconstruction of convert_main_content."""
    remained = set()

    def walk_add(el):
        style = el.get("style", "") or ""
        if "display: none" in style or "display:none" in style:
            return
        if el.get(SELECT_ATTR) == "true":
            for item in el.iter():
                remained.add(item)
        else:
            for item in el.iterchildren():
                if isinstance(item.tag, str):
                    walk_add(item)

    walk_add(root)
    full = remained.copy()
    for el in remained:                           # keep ancestors of kept elements
        for anc in el.iterancestors():
            if anc not in full:
                full.add(anc)
            else:
                break
    last = None                                   # recall a <br> adjacent to kept content
    for el in root.iter():
        if last is not None:
            if el.tag == "br" and last in full and last.tag != "br":
                full.add(el)
            if getattr(last, "tag", None) == "br" and el in full and el.tag != "br":
                full.add(last)
        last = el
    return full


def labels_from_src(line_src, main_set):
    """main iff a source element is in main_set; structural lines (empty src) take
    the sandwich rule (main iff both labeled neighbours are main)."""
    lab = [None] * len(line_src)
    for i, src in enumerate(line_src):
        if src:
            lab[i] = 1 if any(e in main_set for e in src) else 0
    known = [(i, l) for i, l in enumerate(lab) if l is not None]
    for i, l in enumerate(lab):
        if l is None:
            prev = next((kl for ki, kl in reversed(known) if ki < i), None)
            nxt = next((kl for ki, kl in known if ki > i), None)
            lab[i] = 1 if (prev == 1 and nxt == 1) else 0
    return lab


def label_page(html, feats=None):
    """Raw page HTML -> (blocks, labels) with exact DOM provenance. No fallback.

    feats is a feature-group string (e.g. "ABC"): when given, structural features
    are gathered on the SAME render and returned as (blocks, labels, feat_array
    [n_blocks, K]); None (default) skips the feature pass entirely, so a run that
    does not want them pays nothing. collect_features filters non-empty lines with
    the same rule as `keep`, so its rows align 1:1 with blocks -- asserted."""
    root = cr.parse(html)
    main_set = compute_main_set(root)
    sanitize_tree(root)
    lines, line_src = cr.render_tree(root)
    labels_full = labels_from_src(line_src, main_set)
    keep = [i for i, ln in enumerate(lines) if ln.strip()]
    blocks = [lines[i].rstrip() for i in keep]
    labels = [labels_full[i] for i in keep]
    if feats is None:
        return blocks, labels
    fa = collect_features(lines, line_src, feats)
    assert fa.shape[0] == len(blocks), (fa.shape[0], len(blocks))
    return blocks, labels, fa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default=None,
                    help="feature groups to gather on the same render (e.g. ABC); "
                         "writes feats.jsonl. Omit to skip the feature pass.")
    args = ap.parse_args()

    n = n_main = total_blocks = total_main = 0
    ffeat = open(FEATS, "w", encoding="utf-8") if args.feats else None
    with open(COMBINED, encoding="utf-8") as fin, \
            open(BLOCKS, "w", encoding="utf-8") as fout:
        for line in fin:
            r = json.loads(line)
            out = label_page(r["html"], args.feats)
            blocks, labels = out[0], out[1]
            fout.write(json.dumps({"track_id": r["track_id"], "blocks": blocks,
                                   "labels": labels}, ensure_ascii=False) + "\n")
            if ffeat is not None:
                ffeat.write(json.dumps({"track_id": r["track_id"],
                                        "feats": out[2].round(4).tolist()}) + "\n")
            n += 1
            n_pos = sum(labels)
            n_main += n_pos > 0
            total_blocks += len(blocks); total_main += n_pos
    if ffeat is not None:
        ffeat.close()

    assert n == 7825, f"expected 7825 records, got {n}"
    print(f"{BLOCKS.name}: {n} pages, every page labeled (no fallback)")
    print(f"  pages with main content: {n_main}/{n}")
    print(f"  blocks: {total_blocks} total, {total_main} main "
          f"({total_main / total_blocks:.1%})")
    if args.feats:
        print(f"  {FEATS.name}: features [{args.feats}] on the same render")


if __name__ == "__main__":
    main()
