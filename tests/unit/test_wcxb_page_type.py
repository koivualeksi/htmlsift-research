"""Unit: the single-type WCXB board (wcxb-<type>) behind the data-efficiency ladder --
test twins never reach a rung, val shares no page or domain with any rung, rungs nest,
the domain-disjoint test subset, and an unknown type raises. Synthetic data files, no
corpus needed.
"""
import json

import pytest

from vendors.wcxb.adapter import benchmark as wb

N_DOMAINS = 240          # article dev domains, 1-4 pages each: 600 pages
TWINS = {3, 17, 40}      # dev page indices whose html is copied into a test page


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


@pytest.fixture
def data(tmp_path, monkeypatch):
    dev, test = [], []
    for d in range(N_DOMAINS):
        for k in range(1 + d % 4):
            i = len(dev)
            dev.append({"track_id": f"dev-{i:04d}", "split": "dev", "page_type": "article",
                        "url": f"https://www{k}.site{d}.com/p{k}", "html": f"<p>dev {i}</p>",
                        "main_content": f"dev {i}"})
    for i in range(30):
        dom = f"site{i}.com" if i < 10 else f"other{i}.org"      # 10 overlap dev domains
        html = dev[sorted(TWINS)[i]]["html"] if i < len(TWINS) else f"<p>test {i}</p>"
        test.append({"track_id": f"test-{i:04d}", "split": "test", "page_type": "article",
                     "url": f"https://{dom}/t", "html": html, "main_content": f"test {i}"})
    other = [{"track_id": "dev-9999", "split": "dev", "page_type": "product",
              "url": "https://shop.com/x", "html": "<p>x</p>", "main_content": "x"}]
    rows = dev + test + other
    _write(tmp_path / "wcxb.jsonl", rows)
    _write(tmp_path / "labels.jsonl", [{"track_id": r["track_id"], "labels": [1]}
                                        for r in rows if r["split"] == "dev"])
    _write(tmp_path / "blocks.jsonl", [{"track_id": r["track_id"], "blocks": ["b"]}
                                        for r in rows])
    monkeypatch.setattr(wb, "DATA", tmp_path)
    return {"dev": dev, "test": test}


def test_twins_never_train_or_val(data):
    b = wb.WCXBBenchmark("article")
    twins = {data["dev"][i]["track_id"] for i in TWINS}
    assert not twins & set(b._fold_rows("train"))
    assert not twins & set(b._fold_rows("val"))
    assert len(b._fold_rows("train")) + len(b._fold_rows("val")) == len(data["dev"]) - len(TWINS)


def test_val_disjoint_by_page_and_domain(data):
    b = wb.WCXBBenchmark("article")
    val = set(b._fold_rows("val"))
    assert len(val) >= wb.VAL_PAGES
    val_doms = {b._meta[t][1] for t in val}
    for n in (25, 50, 100, 10_000):
        rung = b.train_subset(n)
        assert not rung & val
        assert not {b._meta[t][1] for t in rung} & val_doms


def test_rungs_nest_and_cap(data):
    b = wb.WCXBBenchmark("article")
    sizes = [25, 50, 100, 200]
    sets = [b.train_subset(n) for n in sizes]
    assert [len(s) for s in sets] == sizes
    assert all(a < c for a, c in zip(sets, sets[1:]))
    assert b.train_subset(10_000) == set(b._fold_rows("train"))


def test_folds_hold_one_type(data):
    b = wb.WCXBBenchmark("article")
    assert "dev-9999" not in set(b._fold_rows("train")) | set(b._fold_rows("val"))
    assert set(b._fold_rows("test")) == {r["track_id"] for r in data["test"]}


def test_domain_disjoint_test(data):
    b = wb.WCXBBenchmark("article")
    assert b.test_domain_disjoint() == {f"test-{i:04d}" for i in range(10, 30)}


def test_all_types_board_unchanged(data):
    b = wb.WCXBBenchmark()
    assert b._fold_rows("train") == b._fold_rows("val")
    assert "dev-9999" in b._fold_rows("train")


def test_unknown_type_raises():
    with pytest.raises(ValueError):
        wb.WCXBBenchmark("recipe")


def test_subsets_score_beside_the_fold(data):
    from vendors.shared.bench_ops import score_subsets
    b = wb.WCXBBenchmark("article")
    assert b.subsets("val") == {} and wb.WCXBBenchmark().subsets("test") == {}
    probs = {t: [0.9] for t in b._fold_rows("test")}
    s = score_subsets(b, probs, "test", 0.5)
    assert set(s) == {"domain_disjoint"} and s["domain_disjoint"]["n"] == 20
    assert score_subsets(b, {"test-0000": [0.9]}, "test", 0.5) == {}
