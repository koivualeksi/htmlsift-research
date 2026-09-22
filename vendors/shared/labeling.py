"""
Board-agnostic labeling primitives, shared by the WCXB and DAnIEL adapters.

Both boards face the same problem: a benchmark ships main content as external
plain text with no DOM provenance, so per-block training labels must be derived
by aligning that reference to the core renderer's blocks. This module is that
alignment and subset-selection machinery, lifted verbatim out of the WCXB labeler
so a second board reuses it rather than copying it. Pure algorithm -- stdlib
only, no core, no benchmark: it names no board and knows no metric.

The metric stays with the caller. PageScore/repair maximize a capped-overlap
(bag-of-words) F1 over token *counts*, but they take pre-built Counters, so the
caller picks the tokenizer -- \\w+ for WCXB, jieba for DAnIEL's Chinese -- and
scores the resulting selection in whatever metric its board reports.

  norm / adapt / unescape        markdown-block text -> alignment / token space
  join_selected                  selected blocks -> prediction text (the one assembly)
  lcs_match / propagate_labels   align reference lines to blocks (seed labels)
  PageScore / repair             hill-climb a selection to the word-F1 fixpoint
  label_ranges                   labels -> contiguous [start, end) spans
"""
import bisect
import re
from collections import Counter

_LITERAL = "\x00"  # stand-in for an escaped underscore while emphasis markers are stripped


def adapt(md: str) -> str:
    r"""Markdown block -> word_f1 token space. \w+ already discards every markdown
    marker except '_', a word char: html2text writes emphasis as bare '_..._'
    (drop) but escapes literal underscores as '\_' (keep -- snake_case on
    documentation pages). Must stay identical between labeling here and
    inference-time prediction assembly, or the labels do not transfer."""
    s = md.replace("\\_", _LITERAL)
    s = re.sub(rf"(?<![\w{_LITERAL}])_+|_+(?![\w{_LITERAL}])", "", s)
    return s.replace(_LITERAL, "_")


_ESCAPE_RE = re.compile(r"\\([^\w\s])")


def unescape(s: str) -> str:
    r"""Undo html2text's backslash-escaping of punctuation. Invisible to \w+
    scoring, but the align/revcont passes match on substrings, where a stray
    backslash breaks containment."""
    return _ESCAPE_RE.sub(r"\1", s)


def join_selected(blocks, mask=None):
    r"""Selected blocks -> prediction text: adapt(unescape(b)) for each kept block,
    '\n'-joined. mask is a truthy-per-block list (0/1 labels or a prob>thr result),
    or None for select-all. THE single definition of inference-time prediction
    assembly -- labels transfer only because the labeler scored its selections this
    same way, so every scorer, oracle and select-all floor routes through here."""
    keep = mask if mask is not None else [True] * len(blocks)
    return "\n".join(adapt(unescape(b)) for b, k in zip(blocks, keep) if k)


_WS_RE = re.compile(r"\s+")
_LEAD_MARKER_RE = re.compile(r"^(?:\d+\.\s+|>\s*|#+\s+)+")
_PIPE_RUN_RE = re.compile(r"\s*\|[\s|]*")


def norm(s: str) -> str:
    """Alignment fingerprint: collapse whitespace, drop emphasis * and _, fold
    table-pipe runs, strip leading list/quote/heading markers. Seed-pass matching
    only; never scored."""
    s = _WS_RE.sub(" ", s.replace("*", "").replace("_", "")).strip()
    s = _PIPE_RUN_RE.sub("|", s)
    return _LEAD_MARKER_RE.sub("", s)


# ---------- exact incremental word-F1 ----------

class PageScore:
    """Capped-overlap bookkeeping for one page: overlap = sum_t min(sel[t], ref[t]).
    add / remove / flip-deltas are O(tokens in the block), exact -- so a full
    hill-climb sweep costs one pass over the page's tokens, not a re-score."""

    def __init__(self, ref: Counter, block_toks: list[Counter]):
        self.ref = ref
        self.ref_n = sum(ref.values())
        self.toks = block_toks
        self.sizes = [sum(t.values()) for t in block_toks]
        self.sel = Counter()
        self.n_sel = 0
        self.overlap = 0

    def gain_add(self, i: int) -> int:
        g = 0
        for t, c in self.toks[i].items():
            r = self.ref.get(t, 0)
            if r:
                s = self.sel.get(t, 0)
                g += min(s + c, r) - min(s, r)
        return g

    def loss_remove(self, i: int) -> int:
        lo = 0
        for t, c in self.toks[i].items():
            r = self.ref.get(t, 0)
            if r:
                s = self.sel[t]
                lo += min(s, r) - min(s - c, r)
        return lo

    def add(self, i: int):
        self.overlap += self.gain_add(i)
        self.sel += self.toks[i]
        self.n_sel += self.sizes[i]

    def remove(self, i: int):
        self.overlap -= self.loss_remove(i)
        self.sel -= self.toks[i]
        self.n_sel -= self.sizes[i]

    def f1(self, overlap=None, n_sel=None) -> float:
        o = self.overlap if overlap is None else overlap
        n = self.n_sel if n_sel is None else n_sel
        if not self.ref_n:
            return 1.0 if n == 0 else 0.0
        if not n:
            return 0.0
        p, r = o / n, o / self.ref_n
        return 2 * p * r / (p + r) if p + r else 0.0

    def f1_if_flip(self, i: int, selected: bool) -> float:
        if selected:
            return self.f1(self.overlap - self.loss_remove(i), self.n_sel - self.sizes[i])
        return self.f1(self.overlap + self.gain_add(i), self.n_sel + self.sizes[i])


def repair(ps: PageScore, labels: list[int], max_sweeps: int = 20) -> int:
    """Greedy hill-climb to fixpoint: flip any block whose flip raises page F1,
    sweep until none does. Selection can only gain F1, so the fixpoint is the
    per-page ceiling this seed reaches. Returns the flip count."""
    flips = 0
    for _ in range(max_sweeps):
        improved = False
        base = ps.f1()
        for i in range(len(labels)):
            if not ps.sizes[i]:
                continue
            cand = ps.f1_if_flip(i, labels[i] == 1)
            if cand > base + 1e-12:
                if labels[i] == 1:
                    ps.remove(i)
                    labels[i] = 0
                else:
                    ps.add(i)
                    labels[i] = 1
                base = ps.f1()
                improved = True
                flips += 1
        if not improved:
            break
    return flips


# ---------- alignment seed ----------

def lcs_match(page_blocks: list[str], main_blocks: list[str]) -> dict[int, int]:
    """Max-WEIGHT common subsequence of the two block lists (weight = text length),
    via max-weight increasing subsequence over match pairs with a Fenwick tree.
    Returns main index -> page index, strictly increasing in both. Weighting by
    length stops junk separators ('---') from winning ties against real content
    anchors when few pairs are order-compatible. Empty strings never match
    (normalized marker-only blocks fall to containment)."""
    page_pos = {}
    for i, b in enumerate(page_blocks):
        if b:
            page_pos.setdefault(b, []).append(i)
    pairs = []  # (main j asc, page i DESC within j) -- descending keeps one pick per j
    for j, text in enumerate(main_blocks):
        positions = page_pos.get(text) if text else None
        if positions:
            pairs.extend((j, i) for i in reversed(positions))

    # Fenwick tree over page index: prefix-max of chain weight + backpointer.
    # i-desc-within-j processing order guarantees a query at page index i never
    # sees a same-j pair (those sit at larger i), so chains strictly increase j.
    n = len(page_blocks)
    tree_w = [0] * (n + 1)
    tree_p = [-1] * (n + 1)
    back = [-1] * len(pairs)
    score = [0] * len(pairs)
    best_end = -1
    for pidx, (j, i) in enumerate(pairs):
        best_w, best_p = 0, -1
        k = i  # query prefix max over page indices [0, i)
        while k > 0:
            if tree_w[k] > best_w:
                best_w, best_p = tree_w[k], tree_p[k]
            k -= k & (-k)
        score[pidx] = best_w + len(main_blocks[j])
        back[pidx] = best_p
        k = i + 1  # update position i
        while k <= n:
            if score[pidx] > tree_w[k]:
                tree_w[k], tree_p[k] = score[pidx], pidx
            k += k & (-k)
        if best_end < 0 or score[pidx] > score[best_end]:
            best_end = pidx

    main_to_page = {}
    cur = best_end
    while cur != -1:
        j, i = pairs[cur]
        main_to_page[j] = i
        cur = back[cur]
    return main_to_page


def propagate_labels(page_blocks: list[str], main_blocks: list[str]):
    """Seed labels by aligning reference lines (main_blocks) to page blocks. LCS
    anchors matches in order; each unmatched reference line is then recovered by
    order-constrained containment -- searched only between its aligned neighbours,
    raw first then normalized, so an in-order copy wins over a coincidental one.
    Returns (labels, unmatched): reference lines that found no block."""
    labels = [0] * len(page_blocks)
    norm_page = [norm(b) for b in page_blocks]
    norm_main = [norm(b) for b in main_blocks]
    main_to_page = lcs_match(norm_page, norm_main)
    for i in main_to_page.values():
        labels[i] = 1

    if len(main_to_page) == len(main_blocks):
        return labels, []

    matched_js = sorted(main_to_page)
    unmatched = []
    for j, text in enumerate(main_blocks):
        if j in main_to_page:
            continue
        # order constraint: only between the page positions of the nearest
        # matched reference lines on either side
        pos = bisect.bisect_left(matched_js, j)
        lo = main_to_page[matched_js[pos - 1]] + 1 if pos > 0 else 0
        hi = main_to_page[matched_js[pos]] if pos < len(matched_js) else len(page_blocks)
        hit = next((i for i in range(lo, hi) if text in page_blocks[i]), None)
        if hit is None and norm_main[j]:
            hit = next((i for i in range(lo, hi) if norm_main[j] in norm_page[i]), None)
        if hit is None:
            unmatched.append(text)
        else:
            labels[hit] = 1
    return labels, unmatched


def label_ranges(labels: list[int]) -> list[list[int]]:
    out, start = [], None
    for i, l in enumerate(labels):
        if l == 1 and start is None:
            start = i
        elif l != 1 and start is not None:
            out.append([start, i])
            start = None
    if start is not None:
        out.append([start, len(labels)])
    return out
