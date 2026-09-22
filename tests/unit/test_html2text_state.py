"""Unit: html2text.HTML2Text carries structural state across .handle() calls, so
the scorer must build a fresh HTML2TextWrapper per page. handle() resets only
self.start; an unclosed list leaves self.list non-empty, so the next page's list
renders as nested (indented, renumbered) instead of top-level -- silently breaking
the convert_main_content byte-identity gate.
"""
from pathlib import Path

from vendors.shared.upstream import load

ROOT = Path(__file__).resolve().parents[2]

main_html = load(ROOT / "vendors/wmb/upstream/main_html.py", "wmb_main_html")
Wrapper = main_html.HTML2TextWrapper

DIRTY = "<ol><li>a"                            # unclosed list: leaves self.list non-empty
PAGE = "<ol><li>x</li></ol>"                   # a top-level list, nested if state leaked


def test_fresh_instance_is_deterministic():
    assert Wrapper()(PAGE, "") == Wrapper()(PAGE, "")


def test_reused_instance_leaks_state():
    shared = Wrapper()
    shared(DIRTY, "")                          # dirty the converter's structural state
    reused = shared(PAGE, "")
    fresh = Wrapper()(PAGE, "")
    assert reused != fresh                     # same input, different output -> leak


def test_wrapper_ignores_links_and_images():
    out = Wrapper()('<p>see <a href="/x">text</a> and <img src="/y" alt="pic"></p>', "")
    assert "text" in out
    assert "/x" not in out and "/y" not in out and "pic" not in out
