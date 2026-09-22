"""Unit: the WMB sanitizer and its ordering contract.

The headline invariant (CLAUDE.md §4, labels): cc-select is the label, and
sanitize strips it, so the block builder must harvest the main set from the raw
DOM *before* sanitizing. Inverting the order silently loses every label. The rest
lock the two sanitize layers: attribute hygiene is rendering-neutral, decomposing
injected elements is not, and legitimate markup (notranslate, real classes) and
the eval-visible text survive.
"""
import pytest

from core import render
from vendors.wmb.adapter import blocks as wmb_blocks
from vendors.wmb.adapter import sanitize

MAIN_HTML = (
    "<html><body>"
    "<nav>Site nav</nav>"
    '<div cc-select="true" data-anno-uid="7">'
    '<p class="mark-selected">Main content here.</p></div>'
    "<footer>Footer junk</footer>"
    "</body></html>"
)


def test_cc_select_stripped_by_sanitize():
    root = render.parse(MAIN_HTML)
    div = root.xpath("//div")[0]
    assert div.get("cc-select") == "true" and div.get("data-anno-uid") == "7"
    sanitize.sanitize_tree(root)
    assert div.get("cc-select") is None and div.get("data-anno-uid") is None


def test_harvest_before_sanitize_or_leak():
    before = wmb_blocks.compute_main_set(render.parse(MAIN_HTML))
    root = render.parse(MAIN_HTML)
    sanitize.sanitize_tree(root)
    after = wmb_blocks.compute_main_set(root)
    assert before                 # raw DOM: cc-select is readable, main set found
    assert not after              # sanitized first: the label is gone, main set empty


def test_label_page_labels_only_cc_select():
    blocks, labels = wmb_blocks.label_page(MAIN_HTML)
    labeled = [b for b, l in zip(blocks, labels) if l]
    assert labeled == ["Main content here."]


def test_residue_removed():
    residue = (
        "<html><body>"
        '<style id="cc-extraStyle">.x{color:red}</style>'
        '<span class="immersive-translate-target"'
        ' data-immersive-translate-translation-element-mark="1">t</span>'
        '<p cc-select="true" data-anno-uid="3" class="mark-selected"'
        ' style="user-select:none;color:blue">Body.</p>'
        "</body></html>"
    )
    root = render.parse(residue)
    sanitize.sanitize_tree(root)
    assert sanitize.residue_report(root) == []


def test_injected_content_removed_but_not_by_hygiene():
    injected = ('<html><body><p>Real.</p>'
                '<span class="immersive-translate-x">INJECTED</span></body></html>')
    kept = render.render(sanitize.sanitize_html(injected, decompose_injected=False))
    dropped = render.render(sanitize.sanitize_html(injected, decompose_injected=True))
    assert "INJECTED" in kept
    assert "INJECTED" not in dropped and "Real." in dropped


def test_hygiene_layer_is_render_neutral():
    hyg = ('<html><body><p class="mark-selected" data-anno-uid="9"'
           ' style="user-select:none;color:blue">Kept text.</p></body></html>')
    assert render.render(hyg) == render.render(
        sanitize.sanitize_html(hyg, decompose_injected=False))


def test_legit_class_and_notranslate_kept():
    cls = ('<html><body><p class="notranslate keepme mark-selected cc-unloaded">'
           "Text.</p></body></html>")
    root = render.parse(cls)
    sanitize.sanitize_tree(root)
    p = root.xpath("//p")[0]
    assert p.get("class") == "notranslate keepme"
    assert p.text == "Text."
