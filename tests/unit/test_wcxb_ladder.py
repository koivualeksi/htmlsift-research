"""Unit: the WCXB page-type ladder -- nested rungs, a fixed val carve disjoint from every
rung, type filtering on every fold, and the unchanged train == val == dev board when no
page_type is set. Runs against a synthetic data dir (no corpus needed).
"""
import json

import pytest

from vendors.wcxb.adapter.benchmark import WCXBBenchmark

# dev: 40 article (2 with empty blocks), 10 forum; test: 6 article, 3 forum
DEV = [(f"dev-a{i:02d}", "article", i >= 38) for i in range(40)] + \
      [(f"dev-f{i:02d}", "forum", False) for i in range(10)]
TEST = [(f"test-a{i}", "article") for i in range(6)] + [(f"test-f{i}", "forum") for i in range(3)]


def _jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


@pytest.fixture
def data(tmp_path, monkeypatch):
    pages = [{"track_id": t, "split": "dev", "page_type": pt, "main_content": "ref"}
             for t, pt, _ in DEV]
    pages += [{"track_id": t, "split": "test", "page_type": pt, "main_content": "ref"}
              for t, pt in TEST]
    _jsonl(tmp_path / "wcxb.jsonl", pages)
    _jsonl(tmp_path / "labels.jsonl", [{"track_id": t, "labels": [1]} for t, _, _ in DEV])
    _jsonl(tmp_path / "blocks.jsonl",
           [{"track_id": t, "blocks": [] if empty else ["b"]} for t, _, empty in DEV]
           + [{"track_id": t, "blocks": ["b"]} for t, _ in TEST])
    monkeypatch.setattr("vendors.wcxb.adapter.benchmark.DATA", tmp_path)
    return tmp_path


def _val(b):
    return set(b._rows("val"))


def test_rungs_nested_and_deterministic(data):
    b = WCXBBenchmark("article", top=30)
    rungs = [b.train_subset(n) for n in (10, 20, 30)]
    assert [len(r) for r in rungs] == [10, 20, 30]
    assert rungs[0] < rungs[1] < rungs[2]
    assert WCXBBenchmark("article", top=30).train_subset(20) == rungs[1]


def test_val_is_fixed_carve_disjoint_from_rungs(data):
    b = WCXBBenchmark("article", top=30)
    val = _val(b)
    assert len(val) == 38 - 30
    assert not val & b.train_subset(30)
    assert val == _val(WCXBBenchmark("article", top=30))


def test_ladder_excludes_empty_and_other_types(data):
    order = WCXBBenchmark("article", top=30)._ladder()
    assert len(order) == 38
    assert all(t.startswith("dev-a") and int(t[5:]) < 38 for t in order)


def test_top_must_leave_val(data):
    with pytest.raises(AssertionError):
        WCXBBenchmark("forum", top=10)._ladder()


def test_rung_above_top_rejected(data):
    with pytest.raises(AssertionError):
        WCXBBenchmark("article", top=20).train_subset(30)


def test_train_without_ids_rejected(data):
    with pytest.raises(AssertionError):
        WCXBBenchmark("article", top=30).build_pages(None, "train", 512)


def test_folds_hold_only_the_type(data):
    b = WCXBBenchmark("forum", top=5)
    assert set(b._fold_blocks("test")) == {t for t, pt in TEST if pt == "forum"}
    assert set(b._fold_blocks("val")) == _val(b)
    assert all(t.startswith("dev-f") for t in _val(b))


def test_no_page_type_is_the_full_dev_board(data):
    b = WCXBBenchmark()
    assert set(b._rows("train")) == set(b._rows("val")) == {t for t, _, _ in DEV}
    assert set(b._fold_blocks("test")) == {t for t, _ in TEST}
