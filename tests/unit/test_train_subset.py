"""Unit: train_subset stratified sampling -- nesting, distribution tracking, determinism,
and exhaustion handling for --train-limit. Runs against a synthetic splits.json (no
corpus needed).
"""
import json

import pytest


# ---- synthetic splits.json fixture ----

LEVELS = {"easy": 300, "medium": 150, "hard": 50}   # 500 train pages
VAL_LEVELS = {"easy": 30, "medium": 15, "hard": 5}  # 50 val pages (same proportions)


def _make_splits(path):
    splits = {}
    idx = 0
    for level, count in LEVELS.items():
        for i in range(count):
            splits[f"train_{level}_{i:04d}"] = {"fold": "train", "level": level, "source": "full"}
            idx += 1
    for level, count in VAL_LEVELS.items():
        for i in range(count):
            splits[f"val_{level}_{i:04d}"] = {"fold": "val", "level": level, "source": "full"}
    # test records have no level -- should be ignored
    for i in range(20):
        splits[f"test_{i:04d}"] = {"fold": "test", "source": "both"}
    path.write_text(json.dumps(splits, indent=0), encoding="utf-8")
    return splits


@pytest.fixture
def fake_splits(tmp_path, monkeypatch):
    splits_path = tmp_path / "splits.json"
    splits = _make_splits(splits_path)
    monkeypatch.setattr("vendors.wmb.adapter.benchmark.SPLITS", splits_path)
    return splits


# ---- train_subset tests ----

def test_subset_returns_correct_size(fake_splits):
    from vendors.wmb.adapter.benchmark import WMBBenchmark
    for n in [10, 50, 100, 250, 500]:
        s = WMBBenchmark().train_subset(n)
        assert len(s) == n


def test_subset_capped_at_pool_size(fake_splits):
    from vendors.wmb.adapter.benchmark import WMBBenchmark
    pool_size = sum(LEVELS.values())
    s = WMBBenchmark().train_subset(pool_size + 100)
    assert len(s) == pool_size


def test_subsets_are_nested(fake_splits):
    from vendors.wmb.adapter.benchmark import WMBBenchmark
    sizes = [10, 25, 50, 100, 200, 400, 500]
    sets = {n: WMBBenchmark().train_subset(n) for n in sizes}
    for a, b in zip(sizes, sizes[1:]):
        assert sets[a] < sets[b], f"{a}-page set is not a strict subset of {b}-page set"


def test_subset_only_contains_train_ids(fake_splits):
    from vendors.wmb.adapter.benchmark import WMBBenchmark
    s = WMBBenchmark().train_subset(100)
    for tid in s:
        assert tid.startswith("train_"), f"non-train id in subset: {tid}"


def test_subset_is_deterministic(fake_splits):
    from vendors.wmb.adapter.benchmark import WMBBenchmark
    a = WMBBenchmark().train_subset(100)
    b = WMBBenchmark().train_subset(100)
    assert a == b


def test_different_seed_gives_different_subset(fake_splits):
    from vendors.wmb.adapter.benchmark import WMBBenchmark
    a = WMBBenchmark().train_subset(100, seed=1)
    b = WMBBenchmark().train_subset(100, seed=2)
    assert a != b


def test_distribution_tracks_val(fake_splits):
    from vendors.wmb.adapter.benchmark import WMBBenchmark
    val_total = sum(VAL_LEVELS.values())
    val_frac = {lv: c / val_total for lv, c in VAL_LEVELS.items()}
    for n in [50, 100, 250]:
        s = WMBBenchmark().train_subset(n)
        sub_counts = {}
        for tid in s:
            level = tid.split("_")[1]
            sub_counts[level] = sub_counts.get(level, 0) + 1
        for lv in val_frac:
            actual = sub_counts.get(lv, 0) / n
            # within 2 pages' worth of the target fraction
            assert abs(actual - val_frac[lv]) <= 2 / n, \
                f"n={n} level={lv}: target {val_frac[lv]:.3f} actual {actual:.3f}"
