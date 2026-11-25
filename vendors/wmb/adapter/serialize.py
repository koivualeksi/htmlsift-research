"""
Block labels -> prediction markdown (or pruned HTML): the scoring output path.

The scored object is the htmlsift package's own html output for the same labels: prune()
keeps exactly the text of the selected lines (render_tree's per-line pieces), then the
package's output sanitize. Research and package scores are therefore the same number;
the package's pruner is imported, never copied.

The 545's references are convert_main_content = html2text(extract_main_html(html)), so
the markdown is that same html2text over the pruned html -- not the package's own lxml
markdown, which would move the score through a renderer the board does not use, and
not pasted block text, which loses every cross-block 5-gram. WMB's extract_main_html
keeps selected elements whole; it builds the references and stays in the gates only.

prepare() does the label-independent render once per page; apply() prunes a copy, so the
training val loop caches one Prepared per page and re-applies it each epoch. The package
renders without this repo's WMB sanitize_tree; it is kept here, since it is model input.
"""
import copy
from dataclasses import dataclass
from pathlib import Path

from htmlsift._sanitize import sanitize as sanitize_output
from htmlsift.core import render as cr
from htmlsift.core.prune import prune
from lxml import html as lxml_html
from vendors.shared.upstream import load
from vendors.wmb.adapter.sanitize import sanitize_tree

WMB = Path(__file__).resolve().parents[1]
HTML2TextWrapper = load(WMB / "upstream" / "main_html.py", "wmb_main_html").HTML2TextWrapper


@dataclass
class Prepared:
    """One page's label-independent serialization state: the sanitized tree and, per
    non-empty block, the parts of elements its line holds (render_tree's pieces)."""
    root: object
    block_pieces: list
    url: str = ""


def blocks_of(lines: list, line_pieces: list) -> list:
    """The block view of a render: the pieces of the non-empty lines, the same rows the
    block builders keep."""
    return [line_pieces[i] for i, ln in enumerate(lines) if ln.strip()]


def prepare(html: str, url: str = "", n_blocks: int | None = None) -> Prepared:
    """Render the page with provenance and keep, per block, its line's pieces."""
    root = cr.parse(html)
    sanitize_tree(root)
    lines, _, line_pieces = cr.render_tree(root, pieces=True)
    block_pieces = blocks_of(lines, line_pieces)
    if n_blocks is not None:
        assert len(block_pieces) == n_blocks, (len(block_pieces), n_blocks)
    return Prepared(root, block_pieces, url)


def apply(p: Prepared, labels: list, output: str = "markdown") -> str:
    """Selected blocks -> the package's pruned html -> html2text, as the references were
    made. output="html" stops at the pruned html (the product's html output); scoring
    always runs through "markdown". No block selected -> ""."""
    assert len(labels) == len(p.block_pieces), (len(labels), len(p.block_pieces))
    if output not in ("markdown", "html"):
        raise ValueError(f"output must be 'markdown' or 'html', got {output!r}")
    selected = {i for i, lab in enumerate(labels) if lab == 1}
    if not selected:
        return ""
    root = copy.deepcopy(p.root)
    clone = dict(zip(p.root.iter(), root.iter(), strict=True))
    prune(root, [{(clone[el], *rest) for el, *rest in blk} for blk in p.block_pieces], selected)
    sanitize_output(root)
    pred_main = lxml_html.tostring(root, encoding="unicode")
    return pred_main if output == "html" else HTML2TextWrapper()(pred_main, p.url)
