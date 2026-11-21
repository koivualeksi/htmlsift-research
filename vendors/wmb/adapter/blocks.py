"""
Block segmentation and provenance labels for WMB, on the core lxml renderer.

label_page parses the raw page once, marks which elements are the cc-selected
main content, sanitizes the tree in place, and renders it with core's provenance
renderer. A line is main iff one of its source elements is in the main set. The
element objects survive the in-place sanitize, so the mapping from rendered line
to cc-select decision is exact -- no marker injection, no alignment gate, no
matcher fallback.

Reads data/wmb.jsonl, writes data/blocks.jsonl, one record per page:
    {track_id, blocks, labels, box, plain}
labels is a 0/1 list parallel to blocks. A page with no cc-select (no annotated
main content) gets all-zero labels. box is parallel too: see box_sizes. plain is
blocks without markdown markers, for the table recipe's plain arm.

--feats <groups> also gathers the structural features (htmlsift.core.features) on the same
render and writes them to data/feats.jsonl ({track_id, feats [n_blocks, K]}),
parallel to blocks.jsonl. Omit it and no feature pass runs. Groups D (hint_features) and
E (container_features) are research screens, not in the package: their columns follow the
package's, D before E.

    python vendors/wmb/adapter/blocks.py [--feats ABC]
"""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from htmlsift.core import render as cr
from htmlsift.core.features import collect_features
from vendors.wmb.adapter.sanitize import sanitize_tree

WMB = Path(__file__).resolve().parents[1]
DATA = WMB / "data"
COMBINED = DATA / "wmb.jsonl"
BLOCKS = DATA / "blocks.jsonl"
FEATS = DATA / "feats.jsonl"

SELECT_ATTR = "cc-select"                         # upstream main_html.SELECT_ATTR

# Group D: hiding and button-like markers that the renderer prints as ordinary text (it drops
# only inline display:none). Class tokens match exactly, so "overflow-hidden" is not hidden.
HIDDEN_CLASSES = {"sr-only", "visually-hidden", "visuallyhidden", "screen-reader-text",
                  "screen-reader-only", "d-none", "hidden", "invisible", "hide", "is-hidden",
                  "u-hidden", "a11y-hidden", "offscreen"}
BUTTON_CLASSES = {"btn", "button", "cta"}
BUTTON_PREFIXES = ("btn-", "button-", "cta-")
BUTTON_CLASS_TAGS = {"a", "span", "div", "input"}
BUTTON_TYPES = {"button", "submit", "reset"}


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


def box_sizes(line_src, labels):
    """Per line, the line count of the largest element that contains it and no gold
    line, or 0: the gold-free box the fp_box loss reads. A line is inside an element
    iff all of its source elements are, the annotator's rule (htmlsift-data
    annotate), so a box here is the box shown there. Counts are over the given lines."""
    inside = [set.intersection(*({e, *e.iterancestors()} for e in src)) if src else set()
              for src in line_src]
    n, gold = Counter(), Counter()
    for els, y in zip(inside, labels):
        for e in els:
            n[e] += 1
            gold[e] += y
    return [max((n[e] for e in els if not gold[e]), default=0) for els in inside]


def _hidden(e):
    style = "".join((e.get("style") or "").lower().split())
    return (e.get("hidden") is not None
            or (e.get("aria-hidden") or "").strip().lower() == "true"
            or "visibility:hidden" in style
            or any(t in HIDDEN_CLASSES for t in (e.get("class") or "").lower().split()))


def _button_like(e):
    return (e.tag == "button"
            or (e.get("role") or "").strip().lower() == "button"
            or (e.tag == "input" and (e.get("type") or "").strip().lower() in BUTTON_TYPES)
            or (e.tag in BUTTON_CLASS_TAGS
                and any(t in BUTTON_CLASSES or t.startswith(BUTTON_PREFIXES)
                        for t in (e.get("class") or "").lower().split())))


def hint_features(lines, line_src):
    """Group D [n_blocks, 2]: (hidden_hint, button_like), ancestor-or-self over each
    block's source elements, rows filtered like collect_features. Chains are cached as
    there, for the same reason (every block re-walks the same containers)."""
    memo = {}

    def chain(e):
        stack = []
        while e is not None and isinstance(e.tag, str) and e not in memo:
            stack.append(e)
            e = e.getparent()
        h, b = memo.get(e, (False, False))
        for nd in reversed(stack):
            h, b = h or _hidden(nd), b or _button_like(nd)
            memo[nd] = (h, b)
        return h, b

    srcs = [src for ln, src in zip(lines, line_src, strict=True) if ln.strip()]
    out = np.zeros((len(srcs), 2), dtype=np.float32)
    for bi, src in enumerate(srcs):
        for e in src:
            out[bi] = np.maximum(out[bi], chain(e))
    return out


def container_features(lines, line_src):
    """Group E [n_blocks, 5]: (cont_log_chars, cont_link_density, cont_rows_log, row_pos,
    cont_pos), rows filtered like collect_features. A row's container is the nearest
    element holding all of its source elements (box_sizes' rule) and at least 2 rows, else
    the page (None, which every row's chain ends in, so a row without sources lands there).
    The ABC columns see only the row's own chain; references, bios and teaser lists look
    like content row by row and differ in the box they sit in and where it is on the page.
    Link density counts a row's chars as linked by has_link's rule (an <a> ancestor-or-self)."""
    rows = [(len(ln), src) for ln, src in zip(lines, line_src, strict=True) if ln.strip()]
    chains, n, chars, linked = [], Counter(), Counter(), Counter()
    for c, src in rows:
        anc = [{e, *e.iterancestors()} for e in src]
        # src is a set: walking up from any one source and keeping what every source
        # shares gives the same deepest-first chain whichever source comes out first.
        e = next(iter(src), None)
        walk = [e, *e.iterancestors()] if src else []
        chain = [a for a in walk if all(a in s for s in anc)] + [None]
        link = any(a.tag == "a" for s in anc for a in s)
        for e in chain:
            n[e] += 1
            chars[e] += c
            linked[e] += c * link
        chains.append(chain)
    seen = Counter()
    out = np.zeros((len(rows), 5), dtype=np.float32)
    for i, chain in enumerate(chains):
        k = next(e for e in chain if e is None or n[e] >= 2)
        out[i] = (math.log1p(chars[k]), linked[k] / chars[k], math.log1p(n[k]),
                  i / max(1, len(rows) - 1), seen[k] / max(1, n[k] - 1))
        for e in chain:
            seen[e] += 1
    return out


def label_page(html, feats=None):
    """Raw page HTML -> (blocks, labels, box, plain) with exact DOM provenance. No
    fallback.

    plain renders the same tree without markdown markers. Its walk emits the same
    lines, so it is filtered by the markdown lines' `keep`: a row stays aligned with
    blocks, labels and features even where plain is empty (table separator, rule,
    fence, quote spacer).

    feats is a feature-group string (e.g. "ABC"): when given, structural features
    are gathered on the SAME render and returned as (blocks, labels, box, plain,
    feat_array [n_blocks, K]); None (default) skips the feature pass entirely, so a
    run that does not want them pays nothing. collect_features filters non-empty lines with
    the same rule as `keep`, so its rows align 1:1 with blocks -- asserted."""
    root = cr.parse(html)
    main_set = compute_main_set(root)
    sanitize_tree(root)
    lines, line_src = cr.render_tree(root)
    labels_full = labels_from_src(line_src, main_set)
    keep = [i for i, ln in enumerate(lines) if ln.strip()]
    blocks = [lines[i].rstrip() for i in keep]
    labels = [labels_full[i] for i in keep]
    box = box_sizes([line_src[i] for i in keep], labels)
    plain_lines, _ = cr.render_tree(root, plain=True)
    assert len(plain_lines) == len(lines), (len(plain_lines), len(lines))
    plain = [plain_lines[i].rstrip() for i in keep]
    if feats is None:
        return blocks, labels, box, plain
    fa = collect_features(lines, line_src, feats)
    if "D" in feats:
        fa = np.hstack([fa, hint_features(lines, line_src)])
    if "E" in feats:
        fa = np.hstack([fa, container_features(lines, line_src)])
    assert fa.shape[0] == len(blocks), (fa.shape[0], len(blocks))
    return blocks, labels, box, plain, fa


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
                                   "labels": labels, "box": out[2], "plain": out[3]},
                                  ensure_ascii=False) + "\n")
            if ffeat is not None:
                ffeat.write(json.dumps({"track_id": r["track_id"],
                                        "feats": out[4].round(4).tolist()}) + "\n")
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
