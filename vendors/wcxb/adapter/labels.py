"""
Per-block training labels for WCXB-dev.

WCXB ships no DOM annotations, so labels are derived by aligning the plain-text
reference (ground_truth.main_content) to the rendered blocks, then a hill-climb
that maximizes the benchmark's own word-F1 to a fixpoint. word_f1 is a
bag-of-words multiset metric, so labeling is subset selection: the hill-climb is
the maximizer and its per-page result is the training ceiling. The align/revcont
seed does not move the ceiling -- it decides which of several token-equivalent
blocks carries the positive label, i.e. what the model learns.

Dev only. Test is never labeled. No page is dropped: every dev page is emitted
with its ceiling f1 and pass provenance, so a low-ceiling page can be
investigated rather than filtered.

The alignment and selection primitives (norm, adapt, unescape, propagate_labels,
PageScore, repair, label_ranges) live in vendors/shared/labeling.py, shared with
DAnIEL; only WCXB's tokenizer (its vendored word_f1) and snippet handling stay here.

    python vendors/wcxb/adapter/labels.py

Reads data/wcxb.jsonl (references) and data/blocks.jsonl (blocks), writes
data/labels.jsonl: {track_id, page_type, labels, ranges, f1, precision, recall,
with_rate, without_rate, ref_empty, n_blocks, n_pos, src_counts}.
"""
import json
from collections import Counter
from pathlib import Path

from vendors.shared.labeling import (
    norm,
    adapt,
    unescape,
    join_selected,
    propagate_labels,
    PageScore,
    repair,
    label_ranges
)
from vendors.shared.upstream import load

DATA = Path(__file__).resolve().parents[1] / "data"
COMBINED = DATA / "wcxb.jsonl"
BLOCKS = DATA / "blocks.jsonl"
LABELS = DATA / "labels.jsonl"

# word_f1 / tokenize / snippet_check, loaded from the verbatim vendored scorer
# (never reimplemented, CLAUDE.md 8/9).
_UPSTREAM = Path(__file__).resolve().parents[1] / "upstream" / "evaluate.py"
wcxb_eval = load(_UPSTREAM, "wcxb_evaluate")


def tokens(s: str) -> Counter:
    return Counter(wcxb_eval.tokenize(s))


# ---------- per-page labeling ----------

MIN_RC = 12  # min normalized chars for a reverse-containment hit (guards against short-string coincidence)


def label_page(blocks, ref_text, with_snips, without_snips):
    """Label one page's blocks against its plain-text reference. Seed by alignment
    and reverse-containment, nudge by the annotators' with/without snippets, then
    hill-climb to the word-F1 fixpoint. Returns
    (labels, precision, recall, f1, with_rate, without_rate, src_counts)."""
    adapted = [adapt(unescape(b)) for b in blocks]
    ref_lines = [ln.strip() for ln in ref_text.splitlines() if ln.strip()]
    src = Counter()

    if ref_lines:
        labels, unmatched = propagate_labels(adapted, ref_lines)
        src["align"] = sum(labels)
    else:
        labels, unmatched = [0] * len(blocks), []

    # revcont: a reference paragraph our render split (<br> etc.) sits inside a
    # still-unlabeled block as a substring of a still-unmatched reference line
    norm_unmatched = [norm(u) for u in unmatched]
    for i, b in enumerate(adapted):
        if labels[i]:
            continue
        nb = norm(b)
        if len(nb) < MIN_RC:
            continue
        if any(nb in nu for nu in norm_unmatched if nu):
            labels[i] = 1
            src["revcont"] += 1

    ref = tokens(ref_text)
    ps = PageScore(ref, [tokens(a) for a in adapted])
    for i, l in enumerate(labels):
        if l:
            ps.add(i)

    low = [a.lower() for a in adapted]
    for snip in with_snips or []:  # cover every required snippet not already selected
        s = snip.lower()
        holders = [i for i, b in enumerate(low) if s in b]
        if not holders or any(labels[i] for i in holders):
            continue
        best = max(holders, key=lambda i: ps.f1_if_flip(i, False))
        ps.add(best)
        labels[best] = 1
        src["with"] += 1

    for snip in without_snips or []:  # drop boilerplate the annotators marked out
        s = snip.lower()
        for i, b in enumerate(low):
            if labels[i] and s in b:
                ps.remove(i)
                labels[i] = 0
                src["without"] += 1

    src["repair"] = repair(ps, labels)  # runs last: F1 has final say over the nudges

    # cross-check the incremental bookkeeping against the vendored scorer on the
    # assembled prediction. Any drift here means adapt/concat or PageScore is wrong.
    pred = join_selected(blocks, labels)
    p, r, f1 = wcxb_eval.word_f1(pred, ref_text)
    assert abs(f1 - ps.f1()) < 1e-6, f"bookkeeping drift: {f1} vs {ps.f1()}"
    with_rate = wcxb_eval.snippet_check(pred, with_snips or [])
    without_rate = wcxb_eval.snippet_check(pred, without_snips or [])
    return labels, p, r, f1, with_rate, without_rate, dict(src)


# ---------- driver ----------

def load_refs() -> dict:
    """Dev references keyed by track_id: (main_content, with, without, page_type)."""
    refs = {}
    with open(COMBINED, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["split"] != "dev":
                continue
            refs[r["track_id"]] = (r["main_content"] or "", r["with"] or [],
                                   r["without"] or [], r["page_type"])
    return refs


def main():
    refs = load_refs()
    per_type: dict[str, list[float]] = {}
    scoreable: list[tuple[str, str, float]] = []  # (track_id, page_type, f1) for non-empty-ref rendered pages
    empty_render: list[tuple[str, str]] = []      # reference present but the page rendered to zero blocks
    src_total = Counter()
    n = 0
    with open(BLOCKS, encoding="utf-8") as fin, open(LABELS, "w", encoding="utf-8") as fout:
        for line in fin:
            b = json.loads(line)
            tid = b["track_id"]
            if tid not in refs:  # blocks.jsonl holds dev + test; test is never labeled
                continue
            ref_text, with_snips, without_snips, page_type = refs[tid]
            ref_empty = int(not ref_text.strip())
            labels, p, r, f1, wr, wor, src = label_page(
                b["blocks"], ref_text, with_snips, without_snips)
            src_total.update(src)
            fout.write(json.dumps({
                "track_id": tid, "page_type": page_type,
                "labels": labels, "ranges": label_ranges(labels),
                "f1": round(f1, 4), "precision": round(p, 4), "recall": round(r, 4),
                "with_rate": round(wr, 4), "without_rate": round(wor, 4),
                "ref_empty": ref_empty, "n_blocks": len(labels), "n_pos": sum(labels),
                "src_counts": src}) + "\n")
            n += 1
            if not ref_empty and b["blocks"]:
                scoreable.append((tid, page_type, f1))
                per_type.setdefault(page_type, []).append(f1)
            elif not ref_empty:  # reference exists but nothing rendered -- the SPA failure mode
                empty_render.append((tid, page_type))

    assert n == len(refs), f"labeled {n} pages, expected {len(refs)} dev pages"
    m = len(scoreable)
    ceiling = sum(f for _, _, f in scoreable) / m
    print(f"{n} dev pages labeled ({m} scoreable) -> {LABELS}")
    print(f"label ceiling (mean F1): {ceiling:.4f}")
    # frozen from this repo's run; trips on any render (option handler, render_hidden)
    # or labeler drift. dev n=1471 scoreable, render_hidden=True.
    assert round(ceiling, 4) == 0.9890, f"dev ceiling {ceiling:.4f} != 0.9890"
    for pt in sorted(per_type, key=lambda k: -len(per_type[k])):
        v = per_type[pt]
        print(f"  {pt:<14} {len(v):>5}  {sum(v) / len(v):.4f}")
    print(f"pass contributions: {dict(src_total)}")
    for edge in (0.95, 0.90, 0.85, 0.80):
        c = sum(f >= edge for _, _, f in scoreable)
        print(f"  F1 >= {edge:.2f}: {c} ({c / m:.1%})")
    low = sorted((x for x in scoreable if x[2] < 0.80), key=lambda x: x[2])
    if low:
        print(f"below 0.80 ({len(low)}) -- investigate, not drop:")
        for tid, pt, f1 in low[:20]:
            print(f"  {tid:<14} {pt:<12} {f1:.4f}")
    if empty_render:
        print(f"reference present, zero blocks rendered ({len(empty_render)}) -- content absent from shipped HTML:")
        for tid, pt in empty_render:
            print(f"  {tid:<14} {pt}")


if __name__ == "__main__":
    main()
