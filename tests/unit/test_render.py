"""Unit: core render — the render_hidden toggle and line_src provenance.

render_hidden is the WMB/WCXB symmetry hinge: off (default) skips display:none
subtrees, matching WMB's pruner; on keeps them, for WCXB's hydrated-DOM reference.
line_src is the provenance that lets the block builder label each rendered line by
its DOM origin with no marker injection — every non-empty line traces to at least
one source element, structural separators to none.
"""
from core import render


def test_render_hidden_toggle():
    for decl in ("display:none", "display: none"):
        html = f'<html><body><p>Visible.</p><p style="{decl}">Hidden.</p></body></html>'
        assert "Hidden." not in render.render(html)
        assert "Visible." in render.render(html)
        assert "Hidden." in render.render(html, render_hidden=True)


def test_hidden_subtree_skipped_but_tail_renders():
    # the subtree of a display:none element is dropped, but el.tail is stored on
    # el and survives, so it still renders (core/render.py walk).
    html = ('<html><body><div>'
            '<p style="display:none">Hidden.</p>Tail text.'
            "</div></body></html>")
    out = render.render(html)
    assert "Hidden." not in out
    assert "Tail text." in out


def test_line_src_provenance():
    html = "<html><body><h1>Title</h1><p>Para <a href='/x'>link</a>.</p></body></html>"
    tree = render.parse(html)
    h1 = tree.xpath("//h1")[0]
    p = tree.xpath("//p")[0]
    a = tree.xpath("//a")[0]
    lines, src = render.render_tree(tree)
    assert len(lines) == len(src)
    assert src[lines.index("# Title")] == {h1}
    assert src[lines.index("Para link.")] == {p, a}
    # non-empty lines trace to >=1 element; structural separators carry none
    for line, s in zip(lines, src):
        assert bool(s) == bool(line.strip())


def test_render_empty_input():
    assert render.render("") == ""
    assert render.render("   ") == ""
