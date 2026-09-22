"""Gate: the vendored pruner extract_main_html, composed with the scorer,
reproduces the reference markdown byte-identically. Locked offline on an authored,
partially-selected page -- only the cc-select subtree and its ancestors survive,
and the end-to-end markdown is exact. The pruner is vendored verbatim (diff-empty
vs upstream, §9); this fixture locks the mechanism (there is no corpus pruner loop).
"""
from pathlib import Path

import pytest

from vendors.shared.upstream import load

ROOT = Path(__file__).resolve().parents[2]

main_html = load(ROOT / "vendors/wmb/upstream/main_html.py", "wmb_main_html")

EXPECTED_MD = "First real paragraph.\n"


@pytest.fixture(scope="module")
def labeled(fixtures_dir):
    return (fixtures_dir / "wmb" / "labeled.html").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def pruned(labeled):
    return main_html.extract_main_html(labeled)


def test_pruner_keeps_selected_and_ancestors(pruned):
    assert "First real paragraph." in pruned     # the cc-select block
    assert "<article" in pruned                   # its ancestor is retained


def test_pruner_drops_unselected(pruned):
    assert "Headline" not in pruned               # unselected sibling
    assert "Boilerplate promo paragraph." not in pruned
    assert "Home About Contact" not in pruned     # nav
    assert "Copyright 2026" not in pruned         # footer


def test_end_to_end_markdown_byte_identical(pruned):
    assert main_html.HTML2TextWrapper()(pruned, "") == EXPECTED_MD
