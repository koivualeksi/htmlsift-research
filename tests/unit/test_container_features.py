"""Unit: group E (container chars, link density, rows; page and in-container position), the
research screen WMB's blocks.py appends after D. Values hand-checked on a small page, and
adding E must leave every column an ABC or ABCD run reads, and its z-score, unchanged.
"""
import math

import numpy as np

from core.training import feature_dim, group_columns, zscore_stats
from vendors.wmb.adapter import blocks as wmb_blocks

# Rows: "# Title" 7, "First para." 11, "Second link." 12 (article); "- One" 5, "- Two" 5 (ul);
# "Solo footer line." 17, alone in its footer, so its container is body (all 6 rows, 57 chars).
PAGE = ('<html><body><article><h1>Title</h1><p>First para.</p>'
        '<p>Second <a href="/x">link</a>.</p></article>'
        '<ul><li><a href="/a">One</a></li><li><a href="/b">Two</a></li></ul>'
        '<footer><p>Solo footer line.</p></footer></body></html>')
ARTICLE = [math.log1p(30), 12 / 30, math.log1p(3)]
UL = [math.log1p(10), 1.0, math.log1p(2)]
BODY = [math.log1p(57), 22 / 57, math.log1p(6)]
WANT = [ARTICLE + [0.0, 0.0], ARTICLE + [0.2, 0.5], ARTICLE + [0.4, 1.0],
        UL + [0.6, 0.0], UL + [0.8, 1.0], BODY + [1.0, 1.0]]


def test_values():
    blocks, *_, fa = wmb_blocks.label_page(PAGE, "ABCDE")
    assert len(blocks) == 6, blocks
    assert fa.shape[1] == feature_dim("ABCDE") == feature_dim("ABCD") + 5
    np.testing.assert_allclose(fa[:, -5:], WANT, atol=1e-6)


def test_single_row_page():
    *_, fa = wmb_blocks.label_page("<p>Only.</p>", "E")
    np.testing.assert_allclose(fa, [[math.log1p(5), 0.0, math.log1p(1), 0.0, 0.0]], atol=1e-6)


def test_abc_abcd_unchanged_by_e():
    """ABC and ABCD runs slice their columns from an ABCDE feats.jsonl: same values, same
    z-score, and the stats a ckpt carries (full layout) are unchanged."""
    *_, full = wmb_blocks.label_page(PAGE, "ABCDE")
    for g in ("ABC", "ABCD"):
        *_, sub = wmb_blocks.label_page(PAGE, g)
        assert np.array_equal(full[:, group_columns(g)], sub)
        for got, want in zip(zscore_stats([full[:, group_columns(g)]], g), zscore_stats([sub], g)):
            assert np.array_equal(got, want)
        assert zscore_stats([full], g, full=True)[0] == [30, 32]


def test_e_zscore_columns():
    """cont_log_chars and cont_rows_log are z-scored, in the group's layout and the full one."""
    *_, full = wmb_blocks.label_page(PAGE, "ABCDE")
    assert zscore_stats([full[:, group_columns("ABCE")]], "ABCE")[0] == [30, 32, 33, 35]
    assert zscore_stats([full], "ABCE", full=True)[0] == [30, 32, 35, 37]
