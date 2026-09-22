"""Unit: LOLO CLI wiring -- holdout validation, ledger key separation, benchmark
name resolution, and the finetune _key integration. No corpus; the benchmark's
fold logic is already covered by its own ceiling gates."""
import json

import pytest

from trainers import ledger
from trainers.ledger import arm_key, record_key


# ---- holdout in record_key ----

def test_record_key_without_holdout_unchanged():
    r = {"model": "311m", "layers": 10, "head": "bigru", "feats": None, "seed": 0}
    assert record_key(r) == arm_key("311m", 10, "bigru", None, 0, 0)


def test_record_key_with_holdout_extends_key():
    r = {"model": "311m", "layers": 10, "head": "bigru", "feats": None, "seed": 0,
         "holdout": "Chinese"}
    k = record_key(r)
    base = arm_key("311m", 10, "bigru", None, 0, 0)
    assert k == base + ("Chinese",)


def test_record_key_different_holdouts_are_distinct():
    base = {"model": "311m", "layers": 10, "head": "bigru", "feats": None, "seed": 0}
    keys = {record_key({**base, "holdout": lang})
            for lang in ["Greek", "Polish", "Russian", "English", "Chinese"]}
    assert len(keys) == 5


def test_record_key_holdout_vs_no_holdout_distinct():
    base = {"model": "311m", "layers": 10, "head": "bigru", "feats": None, "seed": 0}
    assert record_key(base) != record_key({**base, "holdout": "Chinese"})


# ---- finetune _key with holdout ----

def test_finetune_key_with_holdout():
    from trainers.finetune import _key
    arm = {"model": "ibm-granite/granite-embedding-311m-multilingual-r2",
           "layers": 10, "head": "bigru", "feats": None, "cap": 0}
    k = _key(arm, seed=0, holdout="Chinese")
    assert k == arm_key("311m", 10, "bigru", None, 0, 0) + ("Chinese",)


def test_finetune_key_none_holdout_equals_no_holdout():
    from trainers.finetune import _key
    arm = {"model": "ibm-granite/granite-embedding-311m-multilingual-r2",
           "layers": 10, "head": "bigru", "feats": None, "cap": 0}
    assert _key(arm, seed=0, holdout=None) == _key(arm, seed=0)


# ---- ledger dedup with holdout ----

def test_ledger_dedup_separates_holdouts(tmp_path):
    local = tmp_path / "l.jsonl"
    r1 = {"model": "311m", "layers": 10, "head": "bigru", "feats": None,
          "seed": 0, "ts": "2026-01-01", "holdout": "Chinese"}
    r2 = {**r1, "holdout": "Greek"}
    local.write_text(
        json.dumps(r1) + "\n" + json.dumps(r2) + "\n", encoding="utf-8")
    led = ledger.Ledger(local, None, "daniel/r.jsonl")
    led.done = {led.key_fn(r) for r in [r1, r2]}
    assert led.is_done(record_key(r1))
    assert led.is_done(record_key(r2))
    assert record_key(r1) != record_key(r2)


# ---- benchmark name ----

def test_load_benchmark_daniel_eval_only():
    from vendors.shared.benchmark import load_benchmark
    b = load_benchmark("daniel")
    assert b.name == "daniel"
    assert b.holdout is None


def test_load_benchmark_daniel_holdout():
    from vendors.shared.benchmark import load_benchmark
    b = load_benchmark("daniel", holdout="Chinese")
    assert b.name == "daniel-zh"
    assert b.holdout == "Chinese"


def test_load_benchmark_daniel_holdout_all_five():
    from vendors.shared.benchmark import load_benchmark
    names = set()
    for lang in ["Greek", "Polish", "Russian", "English", "Chinese"]:
        b = load_benchmark("daniel", holdout=lang)
        names.add(b.name)
    assert names == {"daniel-el", "daniel-pl", "daniel-ru", "daniel-en", "daniel-zh"}


# ---- CLI validation (smoke) ----

def test_holdout_rejects_non_daniel():
    """--holdout on a non-daniel benchmark is an error."""
    from unittest.mock import patch
    with patch("sys.argv", ["finetune", "--benchmark", "wmb", "--holdout", "Chinese",
                            "--device", "cpu"]):
        with pytest.raises(SystemExit):
            from trainers.finetune import main
            main()


def test_daniel_without_holdout_rejects():
    """--benchmark daniel without --holdout is an error (no train fold)."""
    from unittest.mock import patch
    with patch("sys.argv", ["finetune", "--benchmark", "daniel",
                            "--device", "cpu"]):
        with pytest.raises(SystemExit):
            from trainers.finetune import main
            main()
