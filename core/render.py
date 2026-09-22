"""
Readable-markdown renderer with block-level DOM provenance.

One walk of a parsed HTML tree emits markdown lines and, per line, the set of
source DOM elements that produced it (line_src). That provenance lets a caller
label or serialize each rendered line by its DOM origin with no marker injection,
no re-render, and no alignment step. The renderer is board-agnostic: it does not
sanitize and knows nothing of any benchmark's annotations.

Layout tables (page structure built from <table>) are flowed as ordinary blocks so
their regions stay independently labelable; data tables render as markdown tables.

    parse(html)             -> lxml tree
    render_tree(tree)       -> (lines, line_src)      # labeling / serialization
    render(html)            -> markdown text          # inference
"""

import re

from lxml import html as lxml_html

COLLAPSE_RE = re.compile(r"\s+")
SKIP_TAGS = {"head", "script", "style", "title", "meta", "link", "base"}
SIMPLE_BLOCK_TAGS = {"p", "div", "section", "article", "header", "footer", "main",
                     "aside", "figure", "figcaption", "fieldset"}
HEADING_TAGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
WRAP_TAGS = {"em": ("_", "_"), "i": ("_", "_"), "strong": ("**", "**"),
             "b": ("**", "**"), "code": ("`", "`"), "del": ("~~", "~~"),
             "s": ("~~", "~~"), "strike": ("~~", "~~")}
LIST_TAGS = {"ul", "ol"}
ROW_GROUP_TAGS = {"thead", "tbody", "tfoot"}

# A cell holding any of these is page-structure, not tabular data.
BLOCKISH_TAGS = (SIMPLE_BLOCK_TAGS | set(HEADING_TAGS) | LIST_TAGS |
                 {"table", "blockquote"})


def parse(html_str):
    """HTML string -> lxml tree. Bytes-encode when the string carries an encoding
    declaration, which lxml rejects on a str."""
    parser = lxml_html.HTMLParser(collect_ids=False, encoding="utf-8",
                                  remove_blank_text=True, remove_comments=True,
                                  remove_pis=True)
    if isinstance(html_str, str) and (
            "<?xml" in html_str or "<meta charset" in html_str or "encoding=" in html_str):
        html_str = html_str.encode("utf-8")
    return lxml_html.fromstring(html_str, parser=parser)


def _direct_cells(el):
    """Cells of this table's own rows (not nested tables')."""
    def rows(node):
        for ch in node:
            t = ch.tag if isinstance(ch.tag, str) else ""
            if t in ROW_GROUP_TAGS:
                yield from rows(ch)
            elif t == "tr":
                yield ch
    return [c for tr in rows(el) for c in tr
            if isinstance(c.tag, str) and c.tag.lower() in ("td", "th")]


def _is_layout_table(el):
    """A <table> used for page layout, not data: no <th> in its OWN cells, and at
    least one own cell holds block-level content (a nested <table>, <div>, …).
    Checks own cells only — a nested data table with <th> must not mask an outer
    layout table."""
    cells = _direct_cells(el)
    if any(c.tag.lower() == "th" for c in cells):
        return False
    for c in cells:
        if any(isinstance(d.tag, str) and d.tag.lower() in BLOCKISH_TAGS
               for d in c.iterdescendants()):
            return True
    return False


def _cell_text(el):
    """Flatten a table cell (incl. any nested table) to one readable inline string,
    dropping script/style subtrees; source = the cell's elements."""
    parts, src = [], set()

    def rec(node):
        if isinstance(node.tag, str):
            if node.tag.lower() in SKIP_TAGS:
                return
            src.add(node)
        if node.text:
            parts.append(node.text)
        for c in node:
            rec(c)
            if c.tail:
                parts.append(c.tail)

    rec(el)
    txt = COLLAPSE_RE.sub(" ", "".join(parts)).strip().replace("|", "\\|")
    return txt, src


class _Renderer:
    """Accumulates output lines and, per line, the set of source elements."""

    def __init__(self, render_hidden=False):
        self.render_hidden = render_hidden
        self.lines = []
        self.line_src = []
        self.buf = []
        self.buf_src = []
        self.linepfx = ""
        self.pending = None
        self.swallow = False
        self.keep_lead = False
        self.quote = 0
        self.list_stack = []

    def _qpfx(self):
        return "> " * self.quote

    def _append(self, line, src):
        self.swallow = False
        if self.pending is not None:
            self.lines.append(self.pending); self.line_src.append(set())
            self.pending = None
        self.lines.append(line); self.line_src.append(src)

    def flush(self):
        if not self.buf:
            return
        src = {s for s in self.buf_src if s is not None}
        self._append((self._qpfx() + self.linepfx + "".join(self.buf)).rstrip(), src)
        self.buf = []; self.buf_src = []; self.linepfx = ""

    def blank(self):
        self.flush()
        if not self.lines:
            return
        self.pending = self._qpfx().rstrip() if self.quote else ""

    def emit(self, line, src):
        """Emit a standalone line with an explicit source set."""
        self.flush()
        self._append(self._qpfx() + line, src if src else set())

    def text(self, s, src):
        if not s:
            return
        s = COLLAPSE_RE.sub(" ", s)
        if self.swallow:
            s = s.lstrip(" "); self.swallow = False
            if not s:
                return
        if not self.buf and not self.keep_lead:
            s = s.lstrip()
            if not s:
                return
        self.buf.append(s); self.buf_src.append(src)

    def raw(self, s, src):
        self.buf.append(s); self.buf_src.append(src)

    def children(self, el):
        self.text(el.text, el)
        for child in el:
            self.walk(child)

    def inner_keeping_lead(self, el):
        prev = self.keep_lead; self.keep_lead = True
        self.children(el); self.keep_lead = prev

    def walk(self, el):
        tag = el.tag
        if not isinstance(tag, str):
            return
        tag = tag.lower()
        style = el.get("style") or ""
        hidden = ("display: none" in style or "display:none" in style) and not self.render_hidden
        if tag in SKIP_TAGS or hidden:
            pass                              # skip subtree; tail still renders below
        elif tag == "br":
            self.flush()
        elif tag == "hr":
            self.blank(); self.emit("* * *", {el}); self.blank()
        elif tag in HEADING_TAGS:
            self.blank()
            self.linepfx = "#" * HEADING_TAGS[tag] + " "
            self.inner_keeping_lead(el)
            self.blank()
        elif tag in LIST_TAGS:
            self.list(el, tag)
        elif tag == "li":
            self.item(el)
        elif tag == "blockquote":
            self.blank(); self.quote += 1; self.children(el)
            self.flush(); self.quote -= 1; self.blank()
        elif tag == "pre":
            self.pre(el)
        elif tag == "table":
            self.table(el)
        elif tag == "dt":
            self.flush(); self.children(el); self.flush()
        elif tag == "option":                 # <select> options; own block each, else they glue
            self.flush(); self.children(el); self.flush()
        elif tag == "dd":
            self.flush(); self.linepfx = "    "
            self.inner_keeping_lead(el); self.flush()
        elif tag in SIMPLE_BLOCK_TAGS:
            self.blank(); self.children(el); self.blank()
        elif tag in WRAP_TAGS:
            open_s, close_s = WRAP_TAGS[tag]
            inner = _sub_inline(el, self.render_hidden)
            if inner:
                self.raw(open_s + inner + close_s, el)
            else:
                self.swallow = True
        else:
            self.children(el)                 # a, span, abbr, sectioning-inline, ...
        # el.tail is stored on el and survives pruning iff el is kept, so it is
        # el's provenance (not the parent's).
        self.text(el.tail, el)

    def list(self, el, kind):
        self.flush()
        try:
            start = int(el.get("start", "1"))
        except ValueError:
            start = 1
        self.list_stack.append([kind, start])
        self.text(el.text, el)
        for child in el:
            ctag = child.tag if isinstance(child.tag, str) else ""
            if ctag.lower() == "li":
                self.item(child)
            else:
                self.walk(child)
        self.list_stack.pop()
        self.flush(); self.blank()

    def item(self, el):
        self.flush()
        depth = max(len(self.list_stack), 1)
        indent = "  " * (depth - 1)
        if self.list_stack and self.list_stack[-1][0] == "ol":
            n = self.list_stack[-1][1]; self.list_stack[-1][1] = n + 1
            self.linepfx += f"{indent}{n}. "
        else:
            self.linepfx += f"{indent}- "
        self.children(el)
        self.flush()

    def pre(self, el):
        self.blank()
        self._append("```", {el})
        for line in el.text_content().split("\n"):
            self._append(line, {el})
        self._append("```", {el})
        self.blank()

    def table(self, el):
        def rows(node):
            for child in node:
                ctag = child.tag if isinstance(child.tag, str) else ""
                if ctag in ROW_GROUP_TAGS:
                    yield from rows(child)
                elif ctag == "tr":
                    yield child
        if _is_layout_table(el):
            self.blank()
            for tr in rows(el):
                for c in tr:
                    if isinstance(c.tag, str) and c.tag.lower() in ("td", "th"):
                        self.blank(); self.children(c); self.blank()
            self.blank()
            return
        self.blank()
        first = True
        for tr in rows(el):
            cells = [c for c in tr if isinstance(c.tag, str)
                     and c.tag.lower() in ("td", "th")]
            if not cells:
                continue
            texts, src = [], {tr}
            for c in cells:
                t, s = _cell_text(c)
                texts.append(t); src |= s
            if not any(texts):                 # skip fully-empty rows
                continue
            self.emit("| " + " | ".join(texts) + " |", src)
            if first:
                self.emit("| " + " | ".join(["---"] * len(cells)) + " |", src)
                first = False
        self.blank()


def _sub_inline(el, render_hidden=False):
    sub = _Renderer(render_hidden); sub.children(el); sub.flush()
    return " ".join(ln.strip() for ln in sub.lines if ln.strip())


def render_tree(root, render_hidden=False):
    """Walk a parsed tree -> (lines, line_src). line_src[i] is the set of source
    DOM elements that produced line i (empty for structural separators).

    render_hidden keeps display:none subtrees instead of skipping them. Default
    off — a reference built by a pruner that also skips display:none stays
    symmetric with the render. A reference captured from a hydrated DOM sets it
    on, so hydration-hidden main content is rendered."""
    r = _Renderer(render_hidden); r.walk(root); r.flush()
    return r.lines, r.line_src


def render(html_str, render_hidden=False):
    """HTML string -> readable markdown text (inference; provenance discarded).
    Does not sanitize — a caller that needs it sanitizes the tree first."""
    if not html_str or not html_str.strip():
        return ""
    lines, _ = render_tree(parse(html_str), render_hidden)
    return "\n".join(lines) + "\n"
