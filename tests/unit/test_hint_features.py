"""Unit: group D (hidden_hint, button_like), the research screen WMB's blocks.py appends
after the package's ABC columns. The rules must match class tokens exactly, and adding D
must leave the ABC columns, and every column position an ABC run reads, unchanged.
"""
import numpy as np

from core.training import feature_dim, group_columns, zscore_stats
from vendors.wmb.adapter import blocks as wmb_blocks

ROWS = [   # (html, hidden_hint, button_like)
    ('<p>Plain text.</p>', 0, 0),
    ('<div class="overflow-hidden"><p>Overflow text.</p></div>', 0, 0),
    ('<div class="hiddenish"><p>Near-miss class.</p></div>', 0, 0),
    ('<h1 class="Visually-Hidden">Title</h1>', 1, 0),
    ('<div aria-hidden="true"><p>Aria hidden.</p></div>', 1, 0),
    ('<div aria-hidden="false"><p>Aria shown.</p></div>', 0, 0),
    ('<p hidden>Attr hidden.</p>', 1, 0),
    ('<p style="color: red; Visibility : Hidden">Style hidden.</p>', 1, 0),
    ('<p><a class="btn btn-primary" href="/b">Book now</a></p>', 0, 1),
    ('<p><span class="cta-large">Review</span></p>', 0, 1),
    ('<p><b class="btn">Bold btn class</b></p>', 0, 0),
    ('<p><a class="btnx" href="/x">Not a button</a></p>', 0, 0),
    ('<div role="button"><p>Role button.</p></div>', 0, 1),
    ('<form><input type="submit" value="Add to cart"><button>Buy</button></form>', 0, 1),
]
PAGE = "<html><body>" + "".join(h for h, _, _ in ROWS) + "</body></html>"


def test_rules():
    blocks, _labels, _box, _plain, fa = wmb_blocks.label_page(PAGE, "ABCD")
    assert len(blocks) == len(ROWS), blocks
    assert fa[:, -2:].tolist() == [[h, b] for _, h, b in ROWS]
    assert fa.shape[1] == feature_dim("ABCD") == feature_dim("ABC") + 2


def test_abc_unchanged_by_d():
    """An ABC run slices its columns from an ABCD feats.jsonl: same values, same z-score."""
    _b, _l, _x, _p, abc = wmb_blocks.label_page(PAGE, "ABC")
    *_, full = wmb_blocks.label_page(PAGE, "ABCD")
    assert np.array_equal(full[:, group_columns("ABC")], abc)
    for got, want in zip(zscore_stats([full], "ABC"), zscore_stats([abc], "ABC")):
        assert np.array_equal(got, want)
    assert group_columns("D") == [abc.shape[1], abc.shape[1] + 1]

