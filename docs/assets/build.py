"""Render every figure in proposed_assets/ from the numbers in results/ and docs/CLAIMS.md.

    python proposed_assets/build.py

Numbers are copied here by hand with their source; each figure's footer carries the
population, basis, n, seed count and the results file it was read from (CLAUDE.md §6).
"""
import math
import os
import textwrap

OUT = os.path.dirname(os.path.abspath(__file__))

# ---- palette (validated: dataviz six checks, light surface) ----
GROUND, CARD, LINE, GRID = "#f1f3f5", "#ffffff", "#d7dbe0", "#e9ecef"
INK, MUTED, FAINT = "#1b1f24", "#565d66", "#8a9199"
BLUE, AMBER, PURPLE = "#3d6fb4", "#c17d11", "#6a4c9c"     # base / mini / 97m (blue+amber also = WMB/WCXB in annotation-example)
GREEN, RED = "#15663c", "#b3261e"                          # keep / cost (status, never series)
BLUE_T, AMBER_T, PURPLE_T = "#e3ebf6", "#f7ecd9", "#ebe4f3"
GREEN_T, RED_T, GRAY_T = "#dcefe3", "#f6e3e1", "#e9ecef"
FONT = "'Segoe UI', system-ui, -apple-system, Roboto, Helvetica, Arial, sans-serif"
MONO = "'Cascadia Mono', 'SF Mono', 'Roboto Mono', Menlo, Consolas, monospace"


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def tw(s, size, bold=False, mono=False):
    """Rough text width for layout decisions (no font metrics in a static build)."""
    k = 0.6 if mono else (0.55 if bold else 0.5)
    return len(s) * size * k


class SVG:
    def __init__(self, h, w=960):
        self.w, self.h, self.b = w, h, []
        self.b.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
                      f'width="{w}" height="{h}" fill="{INK}" font-family="{FONT}">')
        self.b.append(f'<style>.m{{font-family:{MONO};}}</style>')
        self.rect(0, 0, w, h, GROUND)
        self.rect(0.5, 0.5, w - 1, h - 1, "none", stroke=LINE)

    def rect(self, x, y, w, h, fill, stroke=None, rx=0, sw=1, op=None):
        s = f'<rect x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" fill="{fill}"'
        if rx: s += f' rx="{rx}"'
        if stroke: s += f' stroke="{stroke}" stroke-width="{sw}"'
        if op is not None: s += f' opacity="{op}"'
        self.b.append(s + "/>")

    def card(self, x, y, w, h):
        self.rect(x, y, w, h, CARD, stroke=LINE, rx=10)

    def line(self, x1, y1, x2, y2, stroke=LINE, sw=1, dash=None, cap=None):
        s = f'<line x1="{x1:g}" y1="{y1:g}" x2="{x2:g}" y2="{y2:g}" stroke="{stroke}" stroke-width="{sw}"'
        if dash: s += f' stroke-dasharray="{dash}"'
        if cap: s += f' stroke-linecap="{cap}"'
        self.b.append(s + "/>")

    def circle(self, cx, cy, r, fill, stroke=CARD, sw=2):
        self.b.append(f'<circle cx="{cx:g}" cy="{cy:g}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

    def poly(self, pts, stroke, sw=2, fill="none", dash=None, op=None):
        p = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        s = f'<polyline points="{p}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" stroke-linejoin="round" stroke-linecap="round"'
        if dash: s += f' stroke-dasharray="{dash}"'
        if op is not None: s += f' opacity="{op}"'
        self.b.append(s + "/>")

    def polygon(self, pts, fill, op=1):
        p = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        self.b.append(f'<polygon points="{p}" fill="{fill}" opacity="{op}"/>')

    def path(self, d, stroke, sw=2, fill="none"):
        self.b.append(f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round"/>')

    def text(self, x, y, s, size=12, fill=INK, anchor="start", weight=None, mono=False, ls=None, italic=False):
        a = f'<text x="{x:g}" y="{y:g}" font-size="{size}" fill="{fill}"'
        if anchor != "start": a += f' text-anchor="{anchor}"'
        if weight: a += f' font-weight="{weight}"'
        if mono: a += ' class="m"'
        if ls: a += f' letter-spacing="{ls}"'
        if italic: a += ' font-style="italic"'
        self.b.append(a + f">{esc(s)}</text>")

    def rich(self, x, y, parts, size=12, anchor="start"):
        """parts: list of (text, fill, weight, mono)."""
        a = f'<text x="{x:g}" y="{y:g}" font-size="{size}"'
        if anchor != "start": a += f' text-anchor="{anchor}"'
        a += ">"
        for t, fill, weight, mono in parts:
            sp = f'<tspan fill="{fill}"'
            if weight: sp += f' font-weight="{weight}"'
            if mono: sp += ' class="m"'
            a += sp + f">{esc(t)}</tspan>"
        self.b.append(a + "</text>")

    def pill(self, x, y, s, fill, text, size=10.5, pad=9, h=20, stroke=None):
        w = tw(s, size, bold=True) + 2 * pad
        self.rect(x, y, w, h, fill, rx=h / 2, stroke=stroke)
        self.text(x + w / 2, y + h / 2 + size * 0.36, s, size, text, "middle", 700)
        return w

    def check(self, x, y, color=GREEN):
        self.path(f"M{x - 6},{y} l4,5 l9,-11", color, 2.4)

    def cross(self, x, y, color="#b9c0c7"):
        self.path(f"M{x - 5},{y - 5} l10,10 M{x + 5},{y - 5} l-10,10", color, 2.2)

    def arrow(self, x1, y1, x2, y2, color=MUTED, sw=1.5):
        self.line(x1, y1, x2, y2, color, sw)
        ang = math.atan2(y2 - y1, x2 - x1)
        for d in (0.5, -0.5):
            self.line(x2, y2, x2 - 8 * math.cos(ang + d), y2 - 8 * math.sin(ang + d), color, sw, cap="round")

    def frame(self, title, subtitle):
        self.text(40, 46, title, 22, INK, weight=700)
        lines = subtitle if isinstance(subtitle, (list, tuple)) else [subtitle]
        for i, ln in enumerate(lines):
            self.text(40, 70 + i * 16, ln, 13, MUTED)
        return 92 + (len(lines) - 1) * 16

    def footer(self, y, lead, rest, source):
        """Takeaway (bold lead + plain rest, wrapped). The source is kept as an XML comment,
        not rendered: the same files ship in the pip repo, where those paths do not exist.
        Records the last baseline so save() can trim the canvas to it."""
        words = (lead + " " + rest).split(" ")
        lines, cur = [], ""
        for w in words:
            t = (cur + " " + w).strip()
            if tw(t, 12.5) > 870 and cur:
                lines.append(cur); cur = w
            else:
                cur = t
        lines.append(cur)
        self.rect(40, y - 9, 10, 10, GREEN, rx=2)
        nlead = len(lead.split(" "))
        for i, ln in enumerate(lines):
            ws = ln.split(" ")
            if i == 0:
                b, r = " ".join(ws[:nlead]), " ".join(ws[nlead:])
                self.rich(58, y + i * 18, [(b + (" " if r else ""), INK, 700, False), (r, MUTED, None, False)], 12.5)
            else:
                self.text(58, y + i * 18, ln, 12.5, MUTED)
        if source:
            self.b.append(f"<!-- {esc(source)} -->")
        self.end_y = y + (len(lines) - 1) * 18
        return self.end_y

    def save(self, name):
        # trim the canvas to the footer: figures were sized with a source line that is no
        # longer drawn, so the declared height is patched in the three elements that carry it
        end = getattr(self, "end_y", None)
        if end is not None and end + 30 < self.h:
            h = end + 30
            self.b[0] = self.b[0].replace(f'viewBox="0 0 {self.w} {self.h}"', f'viewBox="0 0 {self.w} {h}"') \
                                 .replace(f'height="{self.h}"', f'height="{h}"')
            self.b[2] = self.b[2].replace(f'height="{self.h:g}"', f'height="{h:g}"')
            self.b[3] = self.b[3].replace(f'height="{self.h - 1:g}"', f'height="{h - 1:g}"')
            self.h = h
        self.b.append("</svg>")
        with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
            f.write("\n".join(self.b))
        print("wrote", name, f"{self.w}x{self.h}")


def legend(s, x, y, items, size=11, gap=18):
    """items: (label, color, kind) kind in dot/hollow/line/dash."""
    for label, color, kind in items:
        if kind == "dot":
            s.circle(x + 5, y - 4, 4.5, color)
        elif kind == "hollow":
            s.circle(x + 5, y - 4, 4.5, CARD, stroke=color, sw=2)
        elif kind == "line":
            s.line(x - 2, y - 4, x + 12, y - 4, color, 2.5)
        elif kind == "dash":
            s.line(x - 2, y - 4, x + 12, y - 4, color, 1.5, dash="4 3")
        s.text(x + 17, y, label, size, MUTED)
        x += 17 + tw(label, size) + gap
    return x


def yaxis(s, x0, x1, y_of, ticks, fmt="{:.2f}", label_x=None):
    for t in ticks:
        y = y_of(t)
        s.line(x0, y, x1, y, GRID, 1)
        s.text((label_x or x0) - 10, y + 4, fmt.format(t), 10.5, FAINT, "end", mono=True)


def section_rule(s, y, label):
    # the rule stops a fixed gutter short of the label on both sides, whatever its length
    half = (tw(label, 10.5, bold=True) + len(label) * 1.0) / 2 + 14
    s.line(40, y, 480 - half, y, LINE)
    s.text(480, y + 4, label, 10.5, FAINT, "middle", 700, ls=1)
    s.line(480 + half, y, 920, y, LINE)


# =====================================================================================
# 1. concept.svg  -- element labels vs line labels, and the trade-off
# =====================================================================================
def concept():
    s = SVG(640)
    y0 = s.frame("Element labels vs line labels",
                 "The same page labeled two ways. htmlsift renders it to markdown lines first, then keeps or drops each line.")
    section_rule(s, y0 + 4, "THE CONCEPT")
    top, hh = y0 + 20, 262

    # left: other extractors, element level
    s.card(40, top, 360, hh)
    s.text(56, top + 28, "OTHER EXTRACTORS", 12, INK, weight=700, ls=0.8)
    s.text(56, top + 46, "keep or drop each HTML element", 12, MUTED)
    tree = [(0, "<nav>", "Home · About · Login", "drop"),
            (0, "<h2>", "The 2026 field report", "keep"),
            (0, "<p>", "Downloads opened this week.", "keep"),
            (1, "<a>", "Subscribe now", "drop")]
    ry = top + 62
    for depth, tag, txt, lab in tree:
        h = 34
        x = 56 + depth * 18
        w = 328 - depth * 18
        fill, stroke = (GREEN_T, "#9fceb7") if lab == "keep" else (GRAY_T, "#ccd2d8")
        if depth:
            s.path(f"M{x - 10},{ry - 6} L{x - 10},{ry + h / 2} L{x - 2},{ry + h / 2}", "#ccd2d8", 1)
        s.rect(x, ry, w, h, fill, stroke=stroke, rx=7)
        s.text(x + 12, ry + 22, tag, 11, GREEN if lab == "keep" else FAINT, mono=True)
        s.text(x + 12 + tw(tag, 11, mono=True) + 8, ry + 22, txt, 11, INK if lab == "keep" else MUTED, mono=True)
        pw = 44 if lab == "keep" else 48
        s.pill(x + w - pw - 8, ry + 8, "IN" if lab == "keep" else "OUT", GREEN if lab == "keep" else CARD,
               CARD if lab == "keep" else MUTED, 9.5, 9, 18, stroke=None if lab == "keep" else "#b9c0c7")
        ry += h + 8
    s.text(56, top + hh - 14, "the annotation unit is the element", 10.5, MUTED, italic=True)

    # arrow
    s.arrow(404, top + hh / 2, 456, top + hh / 2)
    s.text(430, top + hh / 2 - 10, "render", 10.5, MUTED, "middle", 600)
    s.text(430, top + hh / 2 + 24, "lxml", 9.5, FAINT, "middle", mono=True)

    # right: htmlsift, line level
    lx, lw = 460, 460
    s.card(lx, top, lw, hh)
    s.text(lx + 16, top + 28, "HTMLSIFT", 12, INK, weight=700, ls=0.8)
    s.text(lx + 16, top + 46, "keep or drop each rendered line", 12, MUTED)
    rows = [("Home · About · Login", "drop", False),
            ("## The 2026 field report", "keep", False),
            ("Downloads opened this week. Subscribe now", "keep", True)]
    ry = top + 62
    for txt, lab, mixed in rows:
        h = 46 if mixed else 34
        fill, stroke = (GREEN_T, "#9fceb7") if lab == "keep" else (GRAY_T, "#ccd2d8")
        s.rect(lx + 16, ry, lw - 32, h, fill, stroke=stroke, rx=7)
        if mixed:
            s.text(lx + 28, ry + 15, "MIXED LINE: one verdict for both", 8.5, RED, ls=0.6, weight=700)
            s.rich(lx + 28, ry + 34, [("Downloads opened this week. ", INK, None, True), ("Subscribe now", RED, None, True)], 11)
        else:
            s.text(lx + 28, ry + 22, txt, 11, INK if lab == "keep" else MUTED, mono=True)
        pw = 44 if lab == "keep" else 48
        s.pill(lx + lw - 16 - pw - 8, ry + 8, "IN" if lab == "keep" else "OUT", GREEN if lab == "keep" else CARD,
               CARD if lab == "keep" else MUTED, 9.5, 9, 18, stroke=None if lab == "keep" else "#b9c0c7")
        ry += h + 8
    s.text(lx + 16, top + hh - 14, "the annotation unit is the line; every line knows its DOM elements", 10.5, MUTED, italic=True)

    # trade-off
    ty = top + hh + 22
    section_rule(s, ty, "THE TRADE-OFF")
    cy, ch = ty + 16, 150
    s.card(40, cy, 430, ch)
    s.rect(56, cy + 18, 10, 10, RED, rx=2)
    s.text(74, cy + 28, "Lower gold ceiling", 13.5, INK, weight=700)
    s.text(56, cy + 48, "WMB oracle: the best ROUGE-5 F1 any selection can reach.", 11, MUTED)
    s.line(56, cy + 60, 454, cy + 60, GRID)
    s.text(190, cy + 80, "test545", 10, FAINT, ls=0.5, mono=True)
    s.text(340, cy + 80, "val732", 10, FAINT, ls=0.5, mono=True)
    s.text(56, cy + 104, "element labels", 11.5, MUTED)
    s.text(190, cy + 104, "1.000", 12, MUTED, mono=True)
    s.text(340, cy + 104, "1.000", 12, MUTED, mono=True)
    s.text(56, cy + 130, "htmlsift lines", 11.5, INK, weight=600)
    s.rich(190, cy + 130, [("0.992", INK, 700, True), ("  -0.8 pt", RED, 700, True)], 12)
    s.rich(340, cy + 130, [("0.978", INK, 700, True), ("  -2.2 pt", RED, 700, True)], 12)

    s.card(490, cy, 430, ch)
    s.rect(506, cy + 18, 10, 10, GREEN, rx=2)
    s.text(524, cy + 28, "Fewer predictions to classify", 13.5, INK, weight=700)
    s.text(506, cy + 48, "Blocks per page: content-bearing elements vs rendered lines.", 11, MUTED)
    s.line(506, cy + 60, 904, cy + 60, GRID)
    s.text(640, cy + 80, "test545", 10, FAINT, ls=0.5, mono=True)
    s.text(790, cy + 80, "val732", 10, FAINT, ls=0.5, mono=True)
    s.text(506, cy + 104, "element labels", 11.5, MUTED)
    s.text(640, cy + 104, "369", 12, MUTED, mono=True)
    s.text(790, cy + 104, "366", 12, MUTED, mono=True)
    s.text(506, cy + 130, "htmlsift lines", 11.5, INK, weight=600)
    s.rich(640, cy + 130, [("284", INK, 700, True), ("  -23%", GREEN, 700, True)], 12)
    s.rich(790, cy + 130, [("211", INK, 700, True), ("  -42%", GREEN, 700, True)], 12)

    s.footer(cy + ch + 30, "One unit end to end.",
             "The line the model scores is the line the label names and the line the serializer writes back; a mixed line costs under 1 point of ceiling on test.",
             "source: docs/CLAIMS.md (WMB oracle ceilings, n=544 / 732); block counts per page as measured for README-outline (add the command to CLAIMS)")
    s.save("concept.svg")


# =====================================================================================
# 2. signals.svg -- what contextual extraction needs, and who has it
# =====================================================================================
def signals():
    s = SVG(604)
    y0 = s.frame("Three signals, one pass",
                 "Contextual boilerplate removal needs three signals. Heuristics carry one and a half; an LLM carries all three but generates token by token.")
    section_rule(s, y0 + 4, "THE THREE SIGNALS")
    top = y0 + 20
    cards = [("Element hierarchy", ["where the block sits in the DOM:", "tag, depth, link ancestry", "a footer link reads like body text"]),
             ("Text meaning", ["what the block says, in any", "language, as a vector", "not length or link density"]),
             ("Page-wide context", ["each block judged against the", "whole page; repeated boilerplate", "only shows up in contrast"])]
    for i, (title, body) in enumerate(cards):
        x = 40 + i * 300
        s.card(x, top, 280, 118)
        s.circle(x + 26, top + 28, 12, GREEN, stroke="none")
        s.text(x + 26, top + 32.5, str(i + 1), 12, CARD, "middle", 700)
        s.text(x + 48, top + 33, title, 14, INK, weight=700)
        for j, ln in enumerate(body):
            s.text(x + 16, top + 60 + j * 17, ln, 11.5, MUTED)

    my = top + 140
    section_rule(s, my, "WHO CARRIES THEM")
    mt = my + 16
    mh = 258
    s.card(40, mt, 880, mh)
    cols = [380, 560, 760]
    heads = ["HIERARCHY", "TEXT MEANING", "PAGE-WIDE CONTEXT"]
    for cx, h in zip(cols, heads):
        s.text(cx, mt + 26, h, 10, FAINT, "middle", 700, ls=0.8)
    s.line(56, mt + 36, 904, mt + 36, GRID)
    rows = [
        ("Heuristics", "trafilatura, resiliparse, readability", None,
         [("check", "tag rules"), ("partial", "text stats only"), ("cross", "fixed rules")]),
        ("LLM extractor", "generates the page", None,
         [("check", ""), ("check", ""), ("check", "")]),
        ("htmlsift base", "encoder, 10 layers", BLUE,
         [("check", "markdown markers"), ("check", "fine-tuned encoder"), ("check", "encoder attention + BiGRU")]),
        ("htmlsift mini", "embedding table, no encoder", AMBER,
         [("check", "33 structural features"), ("check", "frozen token table"), ("check", "BiGRU")]),
    ]
    rowh = 50
    for i, (name, sub, color, cells) in enumerate(rows):
        y = mt + 50 + i * rowh          # y = top edge of the row's content band
        if color:
            # the tint is the full row band minus a 2px surface gap to its neighbour
            s.rect(48, y - 6, 864, rowh - 2, BLUE_T if color == BLUE else AMBER_T, rx=7, op=0.7)
        s.text(64, y + 16, name, 12.5, INK, weight=700)
        s.text(64, y + 32, sub, 10, FAINT)
        for cx, (kind, note) in zip(cols, cells):
            if kind == "check":
                s.check(cx, y + 14, GREEN if color else "#1a8f4f")
            elif kind == "cross":
                s.cross(cx, y + 12)
            else:
                s.path(f"M{cx - 7},{y + 12} l14,0", "#c17d11", 3)
            if note:
                s.text(cx, y + 33, note, 9.5, MUTED if color else FAINT, "middle", mono=bool(color))
        if i < len(rows) - 1 and not (color and rows[i + 1][2]):
            s.line(56, y + rowh - 7, 904, y + rowh - 7, GRID)

    s.footer(mt + mh + 30, "All three, without generating a token.",
             "One encoder pass (base) or one table lookup (mini) plus one recurrent sweep, then a threshold. What that costs in time is a separate measured figure (cpu_speed_accuracy.svg).",
             "source: core/model.py, core/features.py, docs/PROTOCOL.md")
    s.save("signals.svg")


# =====================================================================================
# 3. architecture.svg -- the pipeline, two tiers, key points only
# =====================================================================================
def architecture():
    s = SVG(500)
    y0 = s.frame("One pass over the whole page",
                 "HTML in, selected lines out. Both tiers share the render, the pooling and the head; they differ in what embeds the tokens.")
    cols = [(40, 150), (206, 300), (522, 150), (688, 116), (820, 100)]
    names = ["RENDER", "EMBED TOKENS", "POOL PER LINE", "HEAD", "OUTPUT"]
    top = y0 + 8
    for (x, w), n in zip(cols, names):
        s.text(x + w / 2, top + 6, n, 10, FAINT, "middle", 700, ls=1)
    lane_y = {"base": top + 30, "mini": top + 180}
    lane_h = 130

    def bullets(x, y, items, size=11, lh=17, color=MUTED):
        for i, t in enumerate(items):
            c = color
            if isinstance(t, tuple):
                t, c = t
            s.circle(x + 3, y + i * lh - 4, 2, c, stroke="none")
            s.text(x + 12, y + i * lh, t, size, c)

    x, w = cols[0]
    s.card(x, lane_y["base"], w, lane_y["mini"] + lane_h - lane_y["base"])
    b0 = lane_y["base"]
    s.text(x + 14, b0 + 28, "lxml render", 13, INK, weight=700)
    bullets(x + 14, b0 + 50, ["markdown lines", "each line keeps its", "  DOM elements"], lh=16)
    s.line(x + 14, b0 + 98, x + w - 14, b0 + 98, GRID)
    s.text(x + 14, b0 + 122, "tokenize", 13, INK, weight=700)
    bullets(x + 14, b0 + 144, ["granite tokenizer", "offsets map tokens", "  to lines"], lh=16)
    s.line(x + 14, b0 + 192, x + w - 14, b0 + 192, GRID)
    s.text(x + 14, b0 + 216, "features", 13, INK, weight=700)
    bullets(x + 14, b0 + 238, ["33 columns: tag flags,", "  depth, link, length", ("mini only", AMBER)], lh=16)

    def lane(key, color, tint, title, sub, items):
        y = lane_y[key]
        x, w = cols[1]
        s.rect(x, y, w, lane_h, tint, rx=10)
        s.rect(x, y, 6, lane_h, color, rx=3)
        s.text(x + 20, y + 28, title, 13, INK, weight=700)
        s.text(x + 20 + tw(title, 13, True) + 8, y + 28, sub, 10.5, color, weight=600)
        bullets(x + 20, y + 52, items)

    lane("base", BLUE, BLUE_T, "base", "encoder",
         ["granite-311m, 10 of 22 layers, fp32", "whole page in one 8,192-token window", "tokens attend across the page", "fine-tuned on WMB, text only"])
    lane("mini", AMBER, AMBER_T, "mini", "embedding table",
         ["the backbone's token table, nothing else", "262,152 x 768, int8: 202 MB", "a lookup, no attention, frozen", "+ 33 structural features per line"])

    x, w = cols[2]
    for key in ("base", "mini"):
        y = lane_y[key]
        s.card(x, y, w, lane_h)
        s.text(x + 14, y + 28, "mean-pool", 13, INK, weight=700)
        bullets(x + 14, y + 52, ["one 768-d vector", "  per line"] +
                ([("+ 33 features", AMBER), ("= 801-d", AMBER)] if key == "mini" else ["no features"]))

    x, w = cols[3]
    for key in ("base", "mini"):
        y = lane_y[key]
        s.card(x, y, w, lane_h)
        s.text(x + 14, y + 28, "BiGRU", 13, INK, weight=700)
        bullets(x + 14, y + 52, ["over all lines", "hidden 256", "keep if p >= 0.5"])

    x, w = cols[4]
    for key in ("base", "mini"):
        y = lane_y[key]
        s.card(x, y, w, lane_h)
        s.text(x + 14, y + 28, "lines", 13, INK, weight=700)
        bullets(x + 14, y + 52, ["kept lines", "back to HTML", "or markdown"])

    for key in ("base", "mini"):
        ym = lane_y[key] + lane_h / 2
        for i in range(len(cols) - 1):
            s.arrow(cols[i][0] + cols[i][1] + 3, ym, cols[i + 1][0] - 3, ym)

    s.footer(lane_y["mini"] + lane_h + 30, "No token-by-token decode.",
             "One forward pass (or one table lookup) and one recurrent sweep per page, then a threshold.",
             "source: core/model.py, core/features.py, export/infer.py, docs/PROTOCOL.md; 202 MB from the export build")
    s.save("architecture.svg")


# =====================================================================================
# 4. tiers.svg -- what you install
# =====================================================================================
def tiers():
    s = SVG(516)
    y0 = s.frame("Two tiers of the same model family",
                 "mini runs without torch on any CPU; base is the accuracy flagship and runs on a GPU. Every number is this repo's own run.")
    top = y0 + 10
    W, H = 430, 316

    def tier(x, color, tint, name, sub, spec, nums, note):
        s.card(x, top, W, H)
        s.rect(x, top, W, 56, tint, rx=10)
        s.rect(x, top + 46, W, 10, tint)
        s.text(x + 20, top + 34, name, 18, INK, weight=700)
        s.text(x + 20 + tw(name, 18, True) + 10, top + 34, sub, 11.5, color, weight=600)
        yy = top + 84
        for k, v, v2 in spec:
            s.text(x + 20, yy, k, 10.5, FAINT, weight=700, ls=0.6)
            s.text(x + 110, yy, v, 11.5, INK, mono=k in ("SIZE", "SPEED"))
            if v2:
                yy += 15
                s.text(x + 110, yy, v2, 10, FAINT)
            yy += 22
        s.line(x + 20, yy - 6, x + W - 20, yy - 6, GRID)
        yy += 14
        s.text(x + 20, yy, "F1 ON THE TEST FOLDS", 9.5, FAINT, weight=700, ls=0.6)
        yy += 24
        for board, val, tag in nums:
            s.text(x + 20, yy, board, 11.5, MUTED)
            s.text(x + 218, yy, val, 12.5, INK, weight=700, mono=True)
            s.text(x + 292, yy, tag, 10, FAINT)
            yy += 21
        if note:
            s.text(x + 20, top + H - 14, note, 9.5, FAINT, italic=True)

    tier(40, AMBER, AMBER_T, "mini", "default install, torch-free",
         [("WHAT", "int8 embedding table + BiGRU head, ONNX", None),
          ("SIZE", "202 MB + 6.5 MB", None),
          ("RUNTIME", "onnxruntime, tokenizers, lxml, numpy", None),
          ("SPEED", "CPU 36.1 ms / page median", "Ryzen 5 3600, single thread, 544 pages")],
         [("WMB test545, ROUGE-5", "0.9010", "in-domain, int8, 3 seeds"),
          ("WCXB test511, word-F1", "0.8474", "zero-shot, int8, 3 seeds"),
          ("DAnIEL 1,689, ROUGE-L", "0.8797", "zero-shot, int8, 3 seeds")],
         "")
    tier(490, BLUE, BLUE_T, "base", "full install, torch",
         [("WHAT", "granite-311m encoder, 10 of 22 layers, fine-tuned", None),
          ("SIZE", "~1.0 GB fp32 checkpoint", None),
          ("RUNTIME", "+ torch, transformers", None),
          ("SPEED", "GPU 24.6 pages / s, full pipeline", "RTX 4090, band attention; 7.4 ms p50 batch-1, model only")],
         [("WMB test545, ROUGE-5", "0.9311", "± 0.0027, in-domain, 3 seeds"),
          ("WCXB test511, word-F1", "0.8633", "± 0.0016, zero-shot, 3 seeds"),
          ("DAnIEL 1,689, ROUGE-L", "0.9175", "± 0.0027, zero-shot, 3 seeds")],
         "")

    s.b.append("<!-- source: results/generalization.md, results/speed-wmb.md, results/speed-wmb-311m-10.md; GPU: results/throughput-wmb.md (311m-10, RTX 4090, 2026-09-18) -->")
    s.end_y = top + H
    s.save("tiers.svg")


# =====================================================================================
# 5. annotation-example.svg -- same page kind, opposite gold
# =====================================================================================
def annotation_example():
    s = SVG(560)
    y0 = s.frame("Same kind of page, opposite gold",
                 "Two shop category pages, each a heading and blurb above a product grid. WCXB kept blurb and grid; WMB kept the blurb and dropped the grid.")
    top = y0 + 10

    def page(x, color, tint, board, url, ptype, keep, kept_note):
        W, H = 430, 330
        s.card(x, top, W, H)
        for i in range(3):
            s.circle(x + 16 + i * 12, top + 14, 3.5, LINE, stroke="none")
        s.rect(x + 60, top + 8, W - 76, 12, GRID, rx=6)
        s.text(x + 66, top + 17, url, 8.5, FAINT, mono=True)
        s.rect(x + 16, top + 34, W - 32, 22, GRID, rx=4)
        for i in range(5):
            s.rect(x + 28 + i * 56, top + 42, 36, 6, "#cfd4d9", rx=3)
        by = top + 70
        s.rect(x + 16, by, 200, 12, "#b9c0c7", rx=3)
        for i, w in enumerate([390, 380, 360, 220]):
            s.rect(x + 16, by + 24 + i * 12, w, 6, "#cfd4d9", rx=3)
        blurb_box = (x + 10, by - 6, W - 20, 78)
        gy = by + 88
        for r in range(2):
            for c in range(4):
                cx, cy = x + 16 + c * 100, gy + r * 74
                s.rect(cx, cy, 90, 44, GRID, rx=4)
                s.rect(cx, cy + 50, 60, 6, "#cfd4d9", rx=3)
                s.rect(cx + 66, cy + 50, 24, 6, "#b9c0c7", rx=3)
        grid_box = (x + 10, gy - 6, W - 20, 160)
        s.rect(x + 16, top + H - 26, W - 32, 14, GRID, rx=4)
        both_box = (x + 10, by - 6, W - 20, gy - by + 160)
        box = {"blurb": blurb_box, "grid": grid_box, "both": both_box}[keep]
        s.rect(*box, tint, stroke=color, rx=6, sw=2, op=0.85)
        s.pill(box[0] + box[2] - 92, box[1] + 8, "kept as main", color, CARD, 9.5, 8, 18)
        s.pill(x + 16, top + H + 10, board, color, CARD, 10.5, 9, 20)
        s.text(x + 16 + (72 if board == "WCXB" else 66), top + H + 24, ptype, 11, MUTED)
        s.text(x + 16, top + H + 46, kept_note, 10.5, FAINT, mono=True)

    page(40, AMBER, AMBER_T, "WCXB", "upliftdesk.com/standing-desks", "type: collection", "both",
         "gold = blurb + all 30 desks with prices; 65 of 178 blocks")
    page(490, BLUE, BLUE_T, "WMB", "apeainthepod.com/maternity/tees-and-tanks.asp", "same page kind", "blurb",
         "gold = heading + blurb; 2 of 809 blocks, 241 priced tees dropped")

    s.footer(top + 330 + 72, "Two policies, not two difficulties.",
             "This is why every cross-board number is reported as its own cell, never blended, and why the harness never trains on a board it reports.",
             "source: gold of WCXB dev-0643 (vendors/wcxb/data/wcxb.jsonl) and WMB 09157708-8786-480f-ac55-145501dd5c1c (vendors/wmb/data/blocks.jsonl)")
    s.save("annotation-example.svg")


# =====================================================================================
# 6. benchmark-divergence.svg -- the per-type gap as one bar per type
# =====================================================================================
DIV = [  # type, n(test511), WMB-trained (mean, std), WCXB-trained (mean, std)  -- results/generalization.md, WCXB per type, 311m-10 encoder
    ("collection", 34, (0.5306, 0.0100), (0.8795, 0.0090)),
    ("product", 28, (0.8275, 0.0175), (0.9348, 0.0113)),
    ("listing", 40, (0.6667, 0.0303), (0.7587, 0.0070)),
    ("article", 257, (0.9342, 0.0010), (0.9723, 0.0008)),
    ("service", 59, (0.8274, 0.0076), (0.8561, 0.0068)),
    ("forum", 51, (0.8668, 0.0147), (0.8601, 0.0053)),
    ("documentation", 42, (0.9561, 0.0055), (0.9496, 0.0021)),
]


def divergence():
    s = SVG(600)
    y0 = s.frame("Where two benchmarks disagree on main content",
                 "Train on WCXB labels instead of WMB labels, score both on WCXB test511: word-F1 points gained per page type. 311m-10 encoder, 3 seeds each.")
    top, rowh = y0 + 16, 46
    s.card(40, top, 880, 7 * rowh + 76)
    px0, px1 = 290, 660
    lo, hi = -8, 42
    x_of = lambda v: px0 + (v - lo) / (hi - lo) * (px1 - px0)
    for t in [-5, 0, 5, 10, 15, 20, 25, 30, 35, 40]:
        s.line(x_of(t), top + 14, x_of(t), top + 7 * rowh + 22, GRID if t else MUTED, 1)
        s.text(x_of(t), top + 7 * rowh + 38, f"{t:+d}" if t else "0", 10.5, FAINT, "middle", mono=True)
    s.text(x_of(0), top + 7 * rowh + 56, "gain in word-F1 points when trained on WCXB labels", 10.5, FAINT, "middle")
    s.text(790, top + 26, "WMB-trained", 9.5, FAINT, "end", 700, ls=0.5)
    s.text(896, top + 26, "WCXB-trained", 9.5, FAINT, "end", 700, ls=0.5)
    for i, (t, n, a, b) in enumerate(DIV):
        y = top + 34 + i * rowh + rowh / 2
        gap = (b[0] - a[0]) * 100
        err = math.sqrt(a[1] ** 2 + b[1] ** 2) * 100
        sig = abs(gap) > err and abs(gap) >= 1  # sub-1-pt gaps are not decision-grade (CLAUDE.md section 6)
        loud = sig and gap > 3
        color = RED if loud else ("#6b7480" if sig else "#c3c9cf")
        s.text(56, y + 4, t, 12.5, INK if sig else MUTED, weight=700 if loud else None)
        s.text(56 + tw(t, 12.5, loud) + 8, y + 4, f"n={n}", 10, FAINT, mono=True)
        x0, x1 = x_of(0), x_of(gap)
        s.rect(min(x0, x1), y - 9, abs(x1 - x0), 18, color, rx=4)
        s.line(x_of(gap - err), y, x_of(gap + err), y, INK if sig else MUTED, 1.2)
        s.line(x_of(gap - err), y - 5, x_of(gap - err), y + 5, INK if sig else MUTED, 1.2)
        s.line(x_of(gap + err), y - 5, x_of(gap + err), y + 5, INK if sig else MUTED, 1.2)
        lab = f"{gap:+.1f}"
        if not sig:
            lab += "  noise"
        lx = x_of(gap + err) + 8 if gap >= 0 else x_of(gap - err) - 8
        s.text(lx, y + 4, lab, 11, INK if sig else FAINT, "start" if gap >= 0 else "end", 700 if loud else None, mono=True)
        s.text(790, y + 4, f"{a[0]:.3f}", 11, MUTED, "end", mono=True)
        s.text(896, y + 4, f"{b[0]:.3f}", 11, INK, "end", mono=True)
    ly = top + 7 * rowh + 56
    s.text(904, ly, "whisker: combined std of the two 3-seed means; a bar inside it, or under 1 pt, is noise", 10, FAINT, "end")

    s.footer(top + 7 * rowh + 76 + 30, "The gap lives on collection pages.",
             "Collection pages gain 35 points under WCXB's policy; product and listing gain 11 and 9, article 4, service 3; forum and documentation move inside seed noise. Same model, different idea of main content.",
             "source: results/generalization.md, WCXB per type (wmb-311m-10 vs wcxb-311m-10); n from vendors/wcxb/data/splits.json")
    s.save("benchmark-divergence.svg")


# =====================================================================================
# 7. layer-cut.svg -- frozen screen vs fine-tuned, one panel
# =====================================================================================
FROZEN_ABC = [0.8599, 0.8667, 0.8711, 0.8634, 0.8695, 0.8727, 0.8783, 0.8843, 0.8837, 0.8824, 0.8877,
              0.8912, 0.8881, 0.8862, 0.8804, 0.8855, 0.8819, 0.8796, 0.8705, 0.8646, 0.8678, 0.8631]  # layers 1..22, bigru/ABC, best-epoch, 1 seed
FROZEN_NONE = [0.8405, 0.8477, 0.8556, 0.8557, 0.8636, 0.8666, 0.8719, 0.8727, 0.8786, 0.8770, 0.8790,
               0.8835, 0.8833, 0.8778, 0.8793, 0.8750, 0.8742, 0.8711, 0.8634, 0.8556, 0.8577, 0.8450]  # layers 1..22, bigru/none, best-epoch, 1 seed
TABLE_ABC = (0.8597, 0.0024)    # 311m-table bigru/ABC, 3 seeds, best-epoch
TABLE_NONE = (0.8401, 0.0019)   # 311m-table bigru/none, 3 seeds, best-epoch
# fine-tuned val, 311m bigru/none, 8 ep (results/wmb-finetune.md): each depth is drawn as a
# best-epoch dot dropping to its epoch-mean (the §6 selection basis) with the mean's ±seed std.
# (layers, best-epoch, epoch-mean, epoch-mean seed std or None, n_seeds)
FT = [(10, 0.8986, 0.8940, 0.0011, 3), (11, 0.8971, 0.8915, 0.0019, 3),
      (12, 0.9016, 0.8941, 0.0009, 3), (13, 0.9029, 0.8965, 0.0009, 2),
      (21, 0.9020, 0.8963, None, 1), (22, 0.9008, 0.8945, 0.0027, 3)]


def layer_cut():
    s = SVG(720)
    y0 = s.frame("Frozen embeddings do not need the whole encoder",
                 ["WMB val732 ROUGE-5 F1 vs encoder layers kept (granite-311m).",
                  "Frozen screen (best-epoch) against the fine-tuned encoder (epoch-mean, §6)."])
    top, H = y0 + 12, 400
    s.card(40, top, 880, H)
    px0, px1, py0, py1 = 110, 880, top + 30, top + H - 56
    lo, hi = 0.835, 0.910
    y_of = lambda v: py1 - (v - lo) / (hi - lo) * (py1 - py0)
    x_of = lambda L: px0 + L / 22 * (px1 - px0)
    yaxis(s, px0, px1, y_of, [0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.90, 0.91])
    for L in range(0, 23, 2):
        s.text(x_of(L), py1 + 18, str(L), 10.5, FAINT, "middle")
    s.text((px0 + px1) / 2, py1 + 38,
           "encoder layers kept  (compute grows with layers; 0 = no encoder)", 11, MUTED, "middle")

    # legend rides the card's top band so it never sits among the marks
    legend(s, px0, top + 22,
           [("fine-tuned, 3 seeds", BLUE, "dot"),
            ("fine-tuned, <3 seeds", BLUE, "hollow"),
            ("frozen, with features", AMBER, "line"),
            ("frozen, no features", GREEN, "line")])

    # frozen screen, two readout heads: text embeddings only (green) and + structural features (amber).
    # the 0-layer point is each curve's left end, the embedding table with no encoder.
    def frozen_curve(series, table, color):
        pts = [(x_of(0), y_of(table[0]))] + [(x_of(L + 1), y_of(v)) for L, v in enumerate(series)]
        s.poly(pts, color, 2.1)
        for (x, y) in pts[1:]:
            s.circle(x, y, 2.8, color, sw=1.5)
        tx = x_of(0)
        s.line(tx, y_of(table[0] - table[1]), tx, y_of(table[0] + table[1]), color, 1.5)
        s.circle(tx, y_of(table[0]), 5.5, color)

    frozen_curve(FROZEN_NONE, TABLE_NONE, GREEN)
    frozen_curve(FROZEN_ABC, TABLE_ABC, AMBER)
    s.circle(x_of(12), y_of(0.8912), 4.5, AMBER)          # with-features peak (footer callout)
    s.circle(x_of(12), y_of(0.8835), 4.5, GREEN)          # no-features peak
    s.text(x_of(0) + 12, y_of(TABLE_ABC[0]) + 18, "0 layers  0.860", 10.5, INK, mono=True)
    s.text(x_of(0) + 12, y_of(TABLE_NONE[0]) + 18, "0.840", 10.5, INK, mono=True)
    s.text(x_of(22) + 5, y_of(0.8631) + 4, "0.863", 10.5, INK, mono=True)
    s.text(x_of(22) + 5, y_of(0.8450) + 4, "0.845", 10.5, INK, mono=True)

    # fine-tuned: best-epoch dot, stick down to the epoch-mean tick, ±seed std around the mean
    for L, best, mean, msd, n in FT:
        x, yb, ym = x_of(L), y_of(best), y_of(mean)
        top_e = y_of(mean + msd) if msd else ym
        s.line(x, yb, x, top_e, BLUE, 1.5)
        if msd:
            bot_e = y_of(mean - msd)
            s.line(x, top_e, x, bot_e, BLUE, 1.3)
            s.line(x - 3.5, top_e, x + 3.5, top_e, BLUE, 1.3)
            s.line(x - 3.5, bot_e, x + 3.5, bot_e, BLUE, 1.3)
        s.line(x - 6, ym, x + 6, ym, BLUE, 2.2)            # the epoch-mean tick = selection value
        if n < 3:
            s.circle(x, yb, 3.6, CARD, stroke=BLUE, sw=2)  # fewer than 3 seeds
        else:
            s.circle(x, yb, 3.6, BLUE, sw=1)
    kx, ky = x_of(10), y_of(0.8940)                        # keeper: label at its epoch-mean, no marker
    s.text(kx - 11, ky + 4, "keeper  0.894", 10.5, INK, "end", 700, mono=True)

    s.footer(top + H + 30, "Cut the encoder, take the cheapest depth on the plateau.",
             "Frozen quality peaks near 12 layers and falls off by 22. Structural features add about 1 to 2 "
             "points to the frozen readout (0.891 with, 0.884 without, at the peak) but nothing once the encoder "
             "fine-tunes (0.8940 vs 0.8942 at 10 layers). Fine-tuning lifts every depth above both frozen curves "
             "and flattens them, so the keeper takes the shallowest, 10 layers.",
             "source: results/wmb-frozen.md (311m bigru/none and bigru/ABC, best-epoch), "
             "results/wmb-finetune.md (311m bigru/none, 8 ep, both bases), "
             "docs/CLAIMS.md (features flat after fine-tuning), docs/LIMITS.md")
    s.save("layer-cut.svg")


# =====================================================================================
# 8. data-efficiency.svg -- three arms, best-epoch, with the value grid
# =====================================================================================
PAGES = [125, 250, 500, 1000, 2000, 4000, 6548]
LADDERS = [  # label, color, [(mean, std)] per PAGES, best-epoch val732, 3 seeds
    ("311m encoder, 10 layers", BLUE,
     [(0.8349, 0.0060), (0.8614, 0.0037), (0.8690, 0.0034), (0.8829, 0.0032), (0.8887, 0.0029), (0.8972, 0.0017), (0.9024, 0.0035)]),
    ("97m encoder, 6 layers, int8 QAT", PURPLE,
     [(0.8060, 0.0074), (0.8362, 0.0037), (0.8479, 0.0070), (0.8620, 0.0025), (0.8743, 0.0014), (0.8799, 0.0008), (0.8821, 0.0030)]),
    ("311m table + features (mini)", AMBER,
     [(0.7188, 0.0407), (0.7820, 0.0023), (0.8053, 0.0063), (0.8243, 0.0078), (0.8406, 0.0014), (0.8528, 0.0030), (0.8597, 0.0024)]),
]
FLOOR_VAL, CEIL_VAL = 0.6359, 0.9784
TRAFI_VAL = 0.6649   # trafilatura 2.2.0 ROUGE-5 on WMB val732 (NOT test545's 0.7525)


def data_efficiency():
    s = SVG(720)
    y0 = s.frame("The labeling policy is learned from a few hundred pages",
                 "WMB val732 ROUGE-5 F1 as the training set grows. Stratified nested prefixes of the 6,548-page train split; 3 seeds, best-epoch, mean ± std.")
    top, H = y0 + 12, 380
    s.card(40, top, 880, H)
    px0, px1, py0, py1 = 110, 880, top + 30, top + H - 56
    lo, hi = 0.60, 1.0
    y_of = lambda v: py1 - (v - lo) / (hi - lo) * (py1 - py0)
    yaxis(s, px0, px1, y_of, [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
    lx = lambda n: px0 + (math.log2(n) - math.log2(125)) / (math.log2(6548) - math.log2(125)) * (px1 - px0)
    for n in PAGES:
        s.text(lx(n), py1 + 18, f"{n:,}", 10.5, FAINT, "middle")
    s.text((px0 + px1) / 2, py1 + 38, "training pages (log scale); 6,548 = the full train split", 11, MUTED, "middle")
    s.line(px0, y_of(CEIL_VAL), px1, y_of(CEIL_VAL), GREEN, 1.2, dash="5 4")
    s.text(px1 - 4, y_of(CEIL_VAL) - 6, "DOM oracle ceiling  0.978", 10.5, GREEN, "end")
    # trafilatura and the floor are close; label trafilatura above its line and the floor below
    # its line so neither the two lines nor their labels collide
    s.line(px0, y_of(TRAFI_VAL), px1, y_of(TRAFI_VAL), RED, 1.2, dash="5 4")
    s.text(px1 - 4, y_of(TRAFI_VAL) - 6, "trafilatura (val732)  0.665", 10.5, RED, "end")
    s.line(px0, y_of(FLOOR_VAL), px1, y_of(FLOOR_VAL), MUTED, 1.2, dash="5 4")
    s.text(px1 - 4, y_of(FLOOR_VAL) + 15, "select-all floor  0.636", 10.5, MUTED, "end")
    for label, color, vals in LADDERS:
        up = [(lx(n), y_of(m + sd)) for n, (m, sd) in zip(PAGES, vals)]
        dn = [(lx(n), y_of(m - sd)) for n, (m, sd) in reversed(list(zip(PAGES, vals)))]
        s.polygon(up + dn, color, 0.12)
        s.poly([(lx(n), y_of(m)) for n, (m, sd) in zip(PAGES, vals)], color, 2.2)
        for n, (m, sd) in zip(PAGES, vals):
            s.circle(lx(n), y_of(m), 4.5, color)
        s.text(lx(125) + 8, y_of(vals[0][0]) + (-10 if color == BLUE else 20), f"{vals[0][0]:.3f}", 10.5, INK, "start", 700, mono=True)
        s.text(lx(6548) + 8, y_of(vals[-1][0]) + 4, f"{vals[-1][0]:.3f}", 10.5, INK, "start", 700, mono=True)
    legend(s, px0 + 4, y_of(0.94), [(l, c, "line") for l, c, _ in LADDERS])

    # value grid
    gy = top + H + 16
    gh = 26 * 4 + 14
    s.card(40, gy, 880, gh)
    cx0 = 300
    step = (900 - cx0) / len(PAGES)
    s.text(56, gy + 24, "best-epoch F1, mean over 3 seeds", 10, FAINT, weight=700, ls=0.5)
    for j, n in enumerate(PAGES):
        s.text(cx0 + j * step + step / 2, gy + 24, f"{n:,}", 10.5, FAINT, "middle", 700, mono=True)
    s.line(56, gy + 32, 904, gy + 32, GRID)
    for i, (label, color, vals) in enumerate(LADDERS):
        y = gy + 56 + i * 26
        s.circle(62, y - 4, 4.5, color)
        s.text(74, y, label, 11, INK)
        for j, (m, sd) in enumerate(vals):
            s.text(cx0 + j * step + step / 2, y, f"{m:.3f}", 11, INK, "middle", mono=True)

    s.footer(gy + gh + 30, "Most of the gain arrives in the first few hundred pages.",
             "With 125 pages the 311m encoder already clears the select-all floor by 20 points and 500 pages come within 3.3 of the full pool. The 97m needs about 4x the pages for the same score, and the no-encoder table needs the whole pool to reach what the 311m has at 250.",
             "source: results/wmb-finetune.md, results/wmb-finetune-qat.md, results/wmb-frozen.md (training-set size, val best-epoch); floor and ceiling docs/CLAIMS.md")
    s.save("data-efficiency.svg")


# =====================================================================================
# 9. cpu_speed_accuracy.svg -- one row per tool, two bars
# =====================================================================================
# htmlsift F1 = 3-seed test mean (results/generalization.md); the deterministic heuristics are one run (= mean)
SPEED = [("htmlsift mini", 36.1, 79.7, 0.9010, AMBER), ("readability", 19.3, 33.7, 0.8016, "#6b7480"),
         ("trafilatura", 31.2, 70.8, 0.7525, "#6b7480"), ("resiliparse", 1.9, 4.5, 0.7135, "#6b7480")]
SIZE_DEP = [("<1k", 140, 1.3), ("1k-4k", 248, 1.2), ("4k-16k", 120, 1.0), ("16k-32k", 21, 0.9), (">32k", 15, 0.6)]
FLOOR_TEST, CEIL_TEST = 0.7156, 0.9918


def cpu_speed_accuracy():
    s = SVG(616)
    y0 = s.frame("htmlsift mini vs heuristic extractors",
                 "WMB test545, 544 pages, CPU single thread (Ryzen 5 3600). Time one run, extraction only; htmlsift F1 = 3-seed test mean (seeds 0-2).")
    top = y0 + 12
    rowh = 58
    main_h = 4 * rowh + 70
    s.card(40, top, 880, main_h)
    # column geometry
    fx0, fx1 = 230, 540      # F1 bars
    tx0, tx1 = 640, 880      # time bars
    flo, fhi = 0.60, 1.0
    f_of = lambda v: fx0 + (v - flo) / (fhi - flo) * (fx1 - fx0)
    t_of = lambda ms: tx0 + ms / 40 * (tx1 - tx0)
    s.text(fx0, top + 26, "ROUGE-5 F1", 10, FAINT, weight=700, ls=0.6)
    s.text(fx1, top + 26, "higher is better", 9.5, FAINT, "end")
    s.text(tx0, top + 26, "MEDIAN MS PER PAGE", 10, FAINT, weight=700, ls=0.6)
    s.text(tx1, top + 26, "lower is better", 9.5, FAINT, "end")
    for i, (name, p50, mean, f1, color) in enumerate(SPEED):
        y = top + 44 + i * rowh + rowh / 2
        ours = name.startswith("htmlsift")
        s.text(56, y + 4, name, 12.5, INK, weight=700 if ours else None)
        # F1 bar
        s.rect(f_of(flo), y - 9, f_of(f1) - f_of(flo), 18, color, rx=4)
        s.text(f_of(f1) + 8, y + 4, f"{f1:.3f}", 11, INK, "start", 700 if ours else None, mono=True)
        # time bar
        s.rect(tx0, y - 9, t_of(p50) - tx0, 18, color, rx=4)
        s.text(t_of(p50) + 8, y + 4, f"{p50:g} ms", 11, INK, "start", 700 if ours else None, mono=True)
        s.text(tx1, y + 18, f"mean {mean:g} ms", 9.5, FAINT, "end", mono=True)
    # floor / ceiling markers on the F1 column
    yb0, yb1 = top + 40, top + 44 + 4 * rowh - 6
    for v, lab, c in ((FLOOR_TEST, "select-all floor 0.716", MUTED), (CEIL_TEST, "oracle ceiling 0.992", GREEN)):
        s.line(f_of(v), yb0, f_of(v), yb1, c, 1.2, dash="4 3")
        s.text(f_of(v) - 6, yb1 + 14, lab, 9.5, c, "end", mono=True)
    s.text(tx0, yb1 + 14, "heuristics deterministic: one run = mean", 9.5, FAINT)

    # size dependence strip
    sy = top + main_h + 16
    strip_h = 108                                  # was 96; extra bottom padding so the n = ... row clears the card edge
    s.card(40, sy, 880, strip_h)
    s.text(56, sy + 26, "Trafilatura time divided by ours, by page size (mean).", 12, INK, weight=700)
    s.text(56, sy + 44, "Above 1.0x we are faster. Short pages favour us; the longest pages favour trafilatura.", 10.5, MUTED)
    cw = 150
    for i, (bucket, n, r) in enumerate(SIZE_DEP):
        x = 56 + i * cw + 40
        col = AMBER if r > 1 else ("#9aa3ad" if r == 1 else "#6b7480")
        s.rect(x, sy + 58, cw - 12, 26, AMBER_T if r > 1 else GRAY_T, rx=6)
        s.text(x + 10, sy + 76, f"{bucket} tokens", 10, MUTED, mono=True)
        s.text(x + cw - 22, sy + 76, f"{r:.1f}x", 12, INK, "end", 700, mono=True)
        s.text(x + 10, sy + 94, f"n = {n}", 9, FAINT, mono=True)

    s.footer(sy + strip_h + 20, "14.9 F1 points over trafilatura, in a similar speed tier.",
             "",
             "source: results/speed-wmb.md (mini = 311m-table-bigru-fABC int8 ONNX, one-run timing); "
             "F1 = 3-seed test mean results/generalization.md; trafilatura 2.2.0, readability-lxml 0.9, resiliparse 1.0.9; docs/CLAIMS.md")
    s.save("cpu_speed_accuracy.svg")


# =====================================================================================
# 10. cross-board.svg -- the generalization matrix
# =====================================================================================
# colour = model (base/97m/mini); WMB- vs WCXB-trained is the "trained on X" label + in-domain opacity
XB = [
    ("311m-10 encoder", "WMB", BLUE, {"wmb": (0.9311, 0.0027), "wcxb": (0.8633, 0.0016), "daniel": (0.9175, 0.0027)}),
    ("311m-10 encoder", "WCXB", BLUE, {"wmb": (0.8479, 0.0107), "wcxb": (0.9209, 0.0007), "daniel": (0.8826, 0.0027)}),
    ("97m-6 int8 QAT", "WMB", PURPLE, {"wmb": (0.9256, 0.0048), "wcxb": (0.8507, 0.0030), "daniel": (0.8915, 0.0082)}),
    ("97m-6 int8 QAT", "WCXB", PURPLE, {"wmb": (0.8339, 0.0034), "wcxb": (0.9081, 0.0020), "daniel": (0.8700, 0.0082)}),
    ("311m table\n+ features, int8", "WMB", AMBER, {"wmb": (0.9010, 0.0031), "wcxb": (0.8474, 0.0089), "daniel": (0.8797, 0.0025)}),
    ("311m table\n+ features, int8", "WCXB", AMBER, {"wmb": (0.8387, 0.0108), "wcxb": (0.8876, 0.0046), "daniel": (0.7653, 0.0186)}),
]
XB_COLS = [("wmb", "WMB test545", "ROUGE-5", 0.7156, 0.9918), ("wcxb", "WCXB test511", "word-F1", 0.7201, 0.9933),
           ("daniel", "DAnIEL 1,689", "ROUGE-L macro", 0.4984, 0.9816)]


def cross_board():
    s = SVG(600)
    y0 = s.frame("Trained on one board, scored on all three",
                 "Each column is that board's own metric with its floor and ceiling: read down a column, never across a row. Test folds, 3 seeds, mean ± std.")
    top = y0 + 12
    s.card(40, top, 880, 392)
    cx = [(300, 190), (505, 190), (710, 190)]
    hy = top + 30
    for (x, w), (key, name, metric, floor, ceil) in zip(cx, XB_COLS):
        s.text(x + w / 2, hy, name, 12, INK, "middle", 700)
        s.text(x + w / 2, hy + 15, metric, 10, MUTED, "middle")
    s.line(56, hy + 26, 904, hy + 26, GRID)
    rowh = 50
    for i, (lab, trained, color, vals) in enumerate(XB):
        y = hy + 44 + i * rowh
        if i % 2 == 0 and i > 0:
            s.line(56, y - 14, 904, y - 14, GRID)
        if i % 2 == 0:
            for k, part in enumerate(lab.split("\n")):   # long labels wrap so they clear the "trained on" text
                s.text(56, y + 14 + k * 15, part, 12, INK, weight=700)
        s.circle(196, y + 10, 4.5, color)
        s.text(206, y + 14, f"trained on {trained}", 10.5, MUTED)
        for (x, w), (key, name, metric, floor, ceil) in zip(cx, XB_COLS):
            m, sd = vals[key]
            indom = key == trained.lower()
            bar_x0, bar_x1 = x + 10, x + w - 10
            v_of = lambda v: bar_x0 + (v - floor) / (ceil - floor) * (bar_x1 - bar_x0)
            s.rect(bar_x0, y + 4, bar_x1 - bar_x0, 8, GRAY_T, rx=4)
            s.rect(bar_x0, y + 4, max(v_of(m) - bar_x0, 0), 8, color, rx=4, op=1 if indom else 0.45)
            s.text(bar_x0, y + 28, f"{m:.4f} ± {sd:.4f}", 10.5, INK, "start", 700 if indom else None, mono=True)
            if indom:
                s.text(bar_x1, y + 28, "in-domain", 9.5, color, "end", 700)
    ny = hy + 44 + 6 * rowh + 2
    s.text(56, ny, "bar = position between that board's select-all floor (left end) and oracle ceiling (right end); faded = zero-shot", 10, FAINT)
    s.footer(top + 392 + 30, "Crossing boards costs 6 to 8 points either way.",
             "A WMB-trained encoder loses 5.8 pt onto WCXB; a WCXB-trained one loses 8.3 pt onto WMB, and the WMB-trained model leads on the neutral third board. The table arm without an encoder transfers worst to DAnIEL.",
             "source: results/generalization.md (test F1 per board, 3 seeds); floors and ceilings docs/CLAIMS.md")
    s.save("cross-board.svg")


# =====================================================================================
# 11. daniel-lolo.svg -- leave-one-language-out, as points under the ceiling
# =====================================================================================
LOLO = [  # language, iso, n, F1 mean, std, seeds, oracle ceiling  -- results/daniel-lolo.md test; ceilings docs/REPRODUCE.md
    ("Chinese", "zh", 401, 0.9595, 0.0027, 3, 0.9875),
    ("Greek", "el", 273, 0.9555, 0.0083, 3, 0.9922),
    ("English", "en", 475, 0.9507, 0.0005, 3, 0.9962),
    ("Polish", "pl", 274, 0.9174, 0.0030, 3, 0.9694),
    ("Russian", "ru", 266, 0.9021, 0.0042, 3, 0.9627),
]


def daniel_lolo():
    s = SVG(556)
    y0 = s.frame("DAnIEL: train on four languages, score the fifth",
                 "Leave-one-language-out on the appendix board. 311m-10, 4 epochs, 3 seeds; ROUGE-L F1 on the held-out language against that language's oracle ceiling.")
    top, rowh = y0 + 12, 52
    s.card(40, top, 880, 5 * rowh + 76)
    heads = [(56, "held-out language", "start"), (330, "held-out F1", "start"), (500, "oracle ceiling", "start"), (620, "points under the ceiling", "start")]
    for x, h, a in heads:
        s.text(x, top + 26, h.upper(), 9.5, FAINT, a, 700, ls=0.6)
    s.line(56, top + 36, 904, top + 36, GRID)
    bx0, bx1 = 620, 880
    b_of = lambda pts: bx0 + pts / 8 * (bx1 - bx0)
    for i, (lang, iso, n, m, sd, seeds, ceil) in enumerate(LOLO):
        y = top + 46 + i * rowh + rowh / 2
        s.text(56, y + 2, lang, 13, INK, weight=700)
        s.text(56, y + 18, f"{iso}, n = {n} pages", 10, FAINT, mono=True)
        s.text(330, y + 4, f"{m:.4f}", 15, INK, weight=700, mono=True)
        s.text(330 + tw("0.0000", 15, True, True) + 6, y + 4, f"± {sd:.4f}", 10.5, MUTED, mono=True)
        s.text(500, y + 4, f"{ceil:.4f}", 12, GREEN, mono=True)
        gap = (ceil - m) * 100
        s.rect(bx0, y - 8, b_of(gap) - bx0, 16, BLUE, rx=4)
        s.text(b_of(gap) + 8, y + 4, f"{gap:.1f} pt", 11, INK, "start", 700, mono=True)
        if i < 4:
            s.line(56, y + rowh / 2 - 2, 904, y + rowh / 2 - 2, GRID)
    for t in (0, 2, 4, 6, 8):
        s.text(b_of(t), top + 5 * rowh + 60, str(t), 10, FAINT, "middle", mono=True)
    s.text(bx0, top + 5 * rowh + 74, "0 = perfect selection of the gold paragraphs", 9.5, FAINT)

    s.footer(top + 5 * rowh + 76 + 30, "Out-of-language, not in-domain.",
             "A model that never saw the language lands 3 to 6 points under that language's oracle ceiling; Polish and Russian are hardest. The metric is reimplemented from the SIGIR 2025 paper (no scorer ships with the corpus) and is not yet established as like-for-like with the zero-shot column.",
             "source: results/daniel-lolo.md (held-out test, 3 seeds); per-language oracle ceilings docs/REPRODUCE.md; caveats docs/LIMITS.md")
    s.save("daniel-lolo.svg")


# =====================================================================================
# 12. gpu_speed_accuracy_full.svg / gpu_speed_accuracy_lite.svg -- sustained throughput vs MinerU-HTML v1.1
# =====================================================================================
GRAYC = "#6b7480"                                  # competitor gray (matches the SPEED baselines)
# (label, sustained pages/s full pipeline, ratio vs MinerU or "", F1, colour, band-outline).
# Throughput is one seed-0 run (results/throughput-wmb.md); F1 is the 3-seed predict ledger.
GPU_MINERU = ("MinerU-HTML v1.1", 1.16, "", 0.9306, GRAYC, False)
GPU_RESEARCH = [GPU_MINERU,
                ("htmlsift 311m-10", 18.4, "16×", 0.9311, BLUE, False),
                ("311m-10 + band", 24.6, "21×", 0.9311, BLUE, True),
                ("htmlsift 97m-6-qat", 30.2, "26×", 0.9256, PURPLE, False)]
GPU_PKG = [GPU_MINERU,
           ("htmlsift 311m-10 (band)", 24.6, "21×", 0.9311, BLUE, False)]
GPU_SOURCE = ("source: results/throughput-wmb.md, results/dripper-wmb.md (RTX 4090, seed-0 keeper); "
              "F1 = 3-seed predict ledger results/generalization.md; docs/CLAIMS.md")
# one (lead, rest) per green-box pointer; lead is bold, rest muted
GPU_FOOT_LITE = [("21x faster than MinerU-HTML v1.1 at the same accuracy.", "")]
GPU_FOOT_FULL = [("311m-10 + band: 21x faster than MinerU-HTML v1.1 at the same accuracy.", ""),
                 ("97m-6-qat: 26x faster, within 0.5 F1 pt.", "")]


def _gpu_cut(fname, subtitle, rows, foots):
    s = SVG(620)
    y0 = s.frame("htmlsift base vs MinerU-HTML v1.1", subtitle)
    top = y0 + 12
    rowh = 48
    card_h = 54 + len(rows) * rowh + 42
    s.card(40, top, 880, card_h)
    bx0, bx1, ax = 250, 740, 34
    t_of = lambda p: bx0 + p / ax * (bx1 - bx0)
    plot_top = top + 54
    axis_y = plot_top + len(rows) * rowh
    s.text(bx0, top + 32, "SUSTAINED PAGES / S", 10, FAINT, weight=700, ls=0.6)
    s.text(904, top + 32, "ROUGE-5 F1", 10, FAINT, "end", weight=700, ls=0.6)
    s.text(904, top + 46, "htmlsift 3-seed mean; MinerU one run", 9, FAINT, "end")
    for p in (0, 10, 20, 30):
        s.line(t_of(p), plot_top, t_of(p), axis_y, GRID, 1)
        s.text(t_of(p), axis_y + 18, str(p), 10, FAINT, "middle", mono=True)
    s.text((bx0 + bx1) / 2, axis_y + 36, "sustained pages / s (full pipeline, RTX 4090)", 10.5, MUTED, "middle")
    for i, (lab, pps, rat, f1, color, dashed) in enumerate(rows):
        y = plot_top + i * rowh + rowh / 2
        ours = color != GRAYC
        s.text(56, y + 4, lab, 12.5, INK, weight=700 if ours else None)
        w = max(t_of(pps) - bx0, 2)
        if dashed:                                 # band = same weights, drawn as an outline
            s.b.append(f'<rect x="{bx0}" y="{y - 9:g}" width="{w:g}" height="18" rx="4" '
                       f'fill="none" stroke="{color}" stroke-width="1.7" stroke-dasharray="5 3"/>')
        else:
            s.rect(bx0, y - 9, w, 18, color, rx=4)
        s.text(bx0 + w + 8, y + 4, f"{pps:g} pg/s" + (f"   {rat}" if rat else ""),
               11, INK, "start", 700 if rat else None, mono=True)
        s.text(904, y + 4, f"F1 {f1:.4f}", 11, MUTED, "end", mono=True)
    fy = top + card_h + 30
    for j, (lead, rest) in enumerate(foots):
        s.footer(fy + j * 24, lead, rest, GPU_SOURCE if j == len(foots) - 1 else "")
    s.save(fname)


def gpu_speed_accuracy_full():
    _gpu_cut("gpu_speed_accuracy_full.svg",
             "WMB test545, RTX 4090. All arms against MinerU-HTML v1.1; band is the bit-identical speed variant of 311m-10.",
             GPU_RESEARCH, GPU_FOOT_FULL)


def gpu_speed_accuracy_lite():
    _gpu_cut("gpu_speed_accuracy_lite.svg",
             "WMB test545, RTX 4090. Shipped model (311m-10, band attention) against MinerU-HTML v1.1.",
             GPU_PKG, GPU_FOOT_LITE)


if __name__ == "__main__":
    for fn in (concept, signals, architecture, tiers, annotation_example, divergence, layer_cut,
               data_efficiency, cpu_speed_accuracy, cross_board, daniel_lolo,
               gpu_speed_accuracy_full, gpu_speed_accuracy_lite):
        fn()
