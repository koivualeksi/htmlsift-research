"""Unit: feature sizing and train-fold z-scoring in core.features. The raw
columns are gate-locked in tests/gate/test_render_features.py; here the concern is
feature_dim (head d_in without hardcoding K) and the z-score of the two unbounded
scalars -- computed on train rows only, applied to any fold, std floored so a
constant column maps to 0 rather than NaN, and booleans left untouched.
"""
import numpy as np
import pytest

from core.features import apply_zscore, feature_dim, zscore_stats


def test_feature_dim():
    assert feature_dim("ABC") == 33          # 30 (A) + 2 (B) + 1 (C)
    assert feature_dim("A") == 30
    assert feature_dim("B") == 2
    assert feature_dim("C") == 1
    assert feature_dim("AC") == 31           # B skipped, order still A..C
    assert feature_dim(None) == 0
    assert feature_dim("") == 0


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
