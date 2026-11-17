"""Unit: core.training -- train_page (htmlsift's prep_page plus y and samples),
group_columns, and the train-fold z-score of the two unbounded scalars (computed on
train rows only, std floored so a constant column maps to 0, booleans untouched).
"""
import re

import numpy as np
import pytest
from htmlsift.core.features import apply_zscore

from core.training import group_columns, train_page, zscore_stats


def word_tok(text, add_special_tokens=True, **_):
    """Fast-tokenizer-shaped stub: one token per word, wrapped in two specials."""
    spans = [m.span() for m in re.finditer(r"\S+", text)]
    return {"input_ids": [0] + list(range(1, len(spans) + 1)) + [2],
            "offset_mapping": [(0, 0)] + spans + [(0, 0)],
            "special_tokens_mask": [1] + [0] * len(spans) + [1]}


def test_train_page_short_page():
    page = train_page(word_tok, "t1", ["a b", "c"], [1, 0], window=16, feats=[[1.0], [2.0]])
    assert page["tid"] == "t1"
    assert page["ids"].tolist() == [0, 1, 2, 3, 2]
    assert page["members"] == [[1, 2], [3]]
    assert page["y"].dtype == np.float32 and page["y"].tolist() == [1.0, 0.0]
    assert page["feats"].tolist() == [[1.0], [2.0]]
    assert page["samples"] == [(0, 5, 0, 2)]           # within the window: one sample


def test_train_page_without_labels():
    page = train_page(word_tok, None, ["a b", "c"], None, window=16)
    assert page["y"] is None
    assert page["samples"] == [(0, 5, 0, 2)]


def test_train_page_windows_long_page():
    blocks = ["a b", "c d", "e f", "g h"]                # 10 ids: 8 words + 2 specials
    page = train_page(word_tok, None, blocks, [0, 1, 1, 0], window=4, stride=2)
    assert page["window"] == 4 and page["stride"] == 2
    assert page["samples"] == [(0, 4, 0, 1), (2, 6, 1, 2), (4, 8, 2, 3), (6, 10, 3, 4)]


def test_group_columns():
    assert group_columns("ABC") == list(range(33))
    assert group_columns("BC") == [30, 31, 32]
    assert group_columns("AC") == list(range(30)) + [32]


def test_zscore_targets_only_the_two_scalars():
    # depth and log_chars, not has_link: in ABC layout they sit at 30 and 32
    idx, _, _ = zscore_stats([np.zeros((1, 33), dtype=np.float32)], "ABC")
    assert idx == [30, 32]


def test_zscore_stats_and_apply():
    # BC layout -> columns [depth, has_link, log_chars]; z-score idx [0, 2]
    rows = [np.array([[1, 1, 0]], dtype=np.float32),
            np.array([[5, 0, 4]], dtype=np.float32)]
    idx, mean, std = zscore_stats(rows, "BC")
    assert idx == [0, 2]
    assert mean.tolist() == pytest.approx([3.0, 2.0])   # depth (1,5), log_chars (0,4)
    assert std.tolist() == pytest.approx([2.0, 2.0])

    page = np.array([[1, 1, 0]], dtype=np.float32)
    out = apply_zscore(page, (idx, mean, std))
    assert out[0].tolist() == pytest.approx([-1.0, 1.0, -1.0])  # has_link untouched
    assert page.tolist() == [[1.0, 1.0, 0.0]]                   # input not mutated


def test_constant_scalar_column_maps_to_zero():
    rows = [np.array([[3, 1, 9]], dtype=np.float32),
            np.array([[3, 0, 9]], dtype=np.float32)]
    idx, mean, std = zscore_stats(rows, "BC")
    assert std.tolist() == [1.0, 1.0]           # floored, not 0
    out = apply_zscore(np.array([[3, 1, 9]], dtype=np.float32), (idx, mean, std))
    assert out[0].tolist() == pytest.approx([0.0, 1.0, 0.0])
