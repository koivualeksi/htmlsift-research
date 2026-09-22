"""
Block labels -> prediction markdown (or pruned HTML): the scoring output path.

The 545's references are convert_main_content = render(extract_main_html(html)):
prune the DOM to the cc-selected elements, then render. So the honest way to score
against them is to manufacture our prediction the SAME way -- set cc-select on the
elements our selected blocks came from, prune, render -- not to paste block text
together (which loses every cross-block 5-gram).

The bridge from a rendered line back to its owning elements is core's per-line
provenance (render_tree's line_src), so this needs no marker injection and never
falls back -- the failure mode the marker approach had on 112 pages is gone.

prepare() does the label-independent render once per page; apply() does the
label-dependent mark+prune+render. The training val loop caches one Prepared per
page and re-applies it each epoch.
"""
from dataclasses import dataclass
from pathlib import Path

from core import render as cr
from vendors.shared.upstream import load
from vendors.wmb.adapter.sanitize import sanitize_tree

WMB = Path(__file__).resolve().parents[1]
SELECT_ATTR = "cc-select"                         # upstream main_html.SELECT_ATTR

_up = load(WMB / "upstream" / "main_html.py", "wmb_main_html")
extract_main_html = _up.extract_main_html
element_to_html = _up.element_to_html
HTML2TextWrapper = _up.HTML2TextWrapper


@dataclass
class Prepared:
    """One page's label-independent serialization state: the sanitized tree and,
    per non-empty block, the source elements that produced it."""
    root: object
    block_src: list
    url: str = ""


def prepare(html: str, url: str = "", n_blocks: int | None = None) -> Prepared:
    """Render the page with provenance and keep, per block, its source elements."""
    root = cr.parse(html)
    sanitize_tree(root)
    lines, line_src = cr.render_tree(root)
    block_src = [line_src[i] for i, ln in enumerate(lines) if ln.strip()]
    if n_blocks is not None:
        assert len(block_src) == n_blocks, (len(block_src), n_blocks)
    return Prepared(root, block_src, url)


def apply(p: Prepared, labels: list, output: str = "markdown") -> str:
    """Selected blocks -> cc-select on their source elements -> pruner -> render,
    manufacturing the output exactly as render(main_html) made the references.

    output="html" stops at the pruned main HTML instead of converting it; the two
    modes are the product's two output shapes, and their cost differs by the
    HTML2TextWrapper step. "markdown" is the scored representation (the WMB
    references are html2text), so scoring always runs through it."""
    assert len(labels) == len(p.block_src), (len(labels), len(p.block_src))
    if output not in ("markdown", "html"):
        raise ValueError(f"output must be 'markdown' or 'html', got {output!r}")
    selected = set()
    for src, lab in zip(p.block_src, labels):
        if lab == 1:
            selected |= src
    for el in p.root.iter():                       # clear any prior apply's marks
        if isinstance(el.tag, str) and el.get(SELECT_ATTR) is not None:
            del el.attrib[SELECT_ATTR]
    if not selected:
        return ""
    for el in selected:
        if isinstance(el.tag, str):
            el.set(SELECT_ATTR, "true")
    pred_main = extract_main_html(element_to_html(p.root))
    if not pred_main:
        return ""
    return pred_main if output == "html" else HTML2TextWrapper()(pred_main, p.url)
