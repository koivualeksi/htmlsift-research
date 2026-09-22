"""
Per-block labels and the selection self-check for DAnIEL.

Like WCXB, DAnIEL's gold is external plain text (<p>-derived), so labels are
derived by aligning it to the rendered blocks and hill-climbing a subset to the
capped-overlap (word-F1) fixpoint -- the shared machinery in
vendors/shared/labeling.py. Two differences from WCXB: no with/without annotator
snippets (the corpus has none), and tokenization is per language -- jieba for
Chinese, \\w+ for the rest -- so the Chinese selection is not one glued token.

The word-F1 ceiling printed here is the *selection* self-check: how well the
labels can reproduce the gold at all, and a gate on labeler/render drift. The
board's ROUGE-L oracle ceiling is eval.py's, computed from these labels.

Reads data/daniel.jsonl (references, language) and data/blocks.jsonl (blocks),
writes data/labels.jsonl:
    {track_id, language, labels, ranges, f1, n_blocks, n_pos, src_counts}

    python vendors/daniel/adapter/labels.py
"""
import json
import re
from collections import Counter
from pathlib import Path

import jieba

from vendors.shared.labeling import (
    norm,
    adapt,
    unescape,
    propagate_labels,
    PageScore,
    repair,
    label_ranges
)

DATA = Path(__file__).resolve().parents[1] / "data"
COMBINED = DATA / "daniel.jsonl"
BLOCKS = DATA / "blocks.jsonl"
LABELS = DATA / "labels.jsonl"

LANGS = ["Greek", "Polish", "Russian", "English", "Chinese"]
ISO = {"Greek": "el", "Polish": "pl", "Russian": "ru", "English": "en", "Chinese": "zh"}
MIN_RC = 12  # min normalized chars for a reverse-containment hit
CEILING = 0.9870  # word-F1 selection ceiling, frozen from this repo's run; trips on labeler/render drift

_WORD = re.compile(r"\w+", re.UNICODE)


def counts(s: str, language: str) -> Counter:
    r"""Token multiset for the word-F1 hill-climb. Tokenizes as eval.py scores:
    jieba for Chinese (no spaces), \w+ lowercased for the rest."""
    if language == "Chinese":
        return Counter(t for t in jieba.lcut(s) if t.strip())
    return Counter(_WORD.findall(s.lower()))


def label_page(blocks, ref_text, language):
    """Align the reference to the blocks, recover render-split lines by
    containment, then hill-climb the selection to the word-F1 fixpoint. Returns
    (labels, f1, src)."""
    adapted = [adapt(unescape(b)) for b in blocks]
    ref_lines = [ln.strip() for ln in ref_text.splitlines() if ln.strip()]
    src = Counter()

    if ref_lines:
        labels, unmatched = propagate_labels(adapted, ref_lines)
        src["align"] = sum(labels)
    else:
        labels, unmatched = [0] * len(blocks), []

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

    ps = PageScore(counts(ref_text, language), [counts(a, language) for a in adapted])
    for i, l in enumerate(labels):
        if l:
            ps.add(i)
    src["repair"] = repair(ps, labels)
    return labels, ps.f1(), dict(src)


def load_refs() -> dict:
    """track_id -> (reference, language), from daniel.jsonl."""
    refs = {}
    with open(COMBINED, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            refs[r["track_id"]] = (r["reference"], r["language"])
    return refs


def main():
    refs = load_refs()
    per_lang: dict[str, list[float]] = {}
    scoreable = []
    src_total = Counter()
    n = 0
    with open(BLOCKS, encoding="utf-8") as fin, open(LABELS, "w", encoding="utf-8") as fout:
        for line in fin:
            b = json.loads(line)
            tid = b["track_id"]
            ref_text, language = refs[tid]
            labels, f1, src = label_page(b["blocks"], ref_text, language)
            src_total.update(src)
            fout.write(json.dumps({
                "track_id": tid, "language": language,
                "labels": labels, "ranges": label_ranges(labels),
                "f1": round(f1, 4), "n_blocks": len(labels), "n_pos": sum(labels),
                "src_counts": src}, ensure_ascii=False) + "\n")
            n += 1
            scoreable.append((tid, language, f1))
            per_lang.setdefault(language, []).append(f1)

    assert n == 1689, f"labeled {n} pages, expected 1689"
    m = len(scoreable)
    ceiling = sum(f for _, _, f in scoreable) / m
    print(f"{n} pages labeled -> {LABELS}")
    print(f"word-F1 selection ceiling: {ceiling:.4f}")
    if CEILING is not None:
        assert round(ceiling, 4) == CEILING, f"ceiling {ceiling:.4f} != {CEILING}"
    for l in LANGS:
        v = per_lang[l]
        print(f"  {ISO[l]:5} {len(v):>5}  {sum(v) / len(v):.4f}")
    print(f"pass contributions: {dict(src_total)}")
    for edge in (0.95, 0.90, 0.85, 0.80):
        c = sum(f >= edge for _, _, f in scoreable)
        print(f"  F1 >= {edge:.2f}: {c} ({c / m:.1%})")
    low = sorted((x for x in scoreable if x[2] < 0.80), key=lambda x: x[2])
    if low:
        print(f"below 0.80 ({len(low)}) -- investigate, not drop:")
        for tid, l, f1 in low[:20]:
            print(f"  {ISO[l]} {f1:.4f}  {tid}")


if __name__ == "__main__":
    main()
