"""Unit: the result ledger's pure logic -- arm identity, --push resolution, shard
paths, and the latest-ts union merge_push applies before uploading. HF transport is
stubbed (hf_hub_download / upload_file monkeypatched on the module); the concern is
that keys normalize (missing cap reads as 0, None feats as "none"), resolve_repo's
three states hold, and the union drops no record and keeps the newest per arm. resume
and push against a live dataset are exercised later by the trainer smokes.
"""
import json
from pathlib import Path

import pytest
from huggingface_hub.errors import EntryNotFoundError

from trainers import ledger
from trainers.ledger import Ledger, arm_key, record_key, resolve_repo, sharded_path


def rec(model="97m", layers=6, head="bigru", feats=None, cap=0, seed=0,
        ts="2026-01-01", **extra):
    r = {"model": model, "layers": layers, "head": head, "feats": feats,
         "seed": seed, "ts": ts, **extra}
    if cap:                       # omit cap when 0 -- mirrors the pre-cap records
        r["cap"] = cap
    return r


def test_arm_key_normalizes_feats():
    assert arm_key("97m", 6, "bigru", None, 0, 0) == ("97m", 6, "bigru", "none", 0, 0)
    assert arm_key("97m", 6, "bigru", "ABC", 0, 0)[3] == "ABC"


def test_record_key_defaults_missing_cap_to_zero():
    # the pre-cap records carry no "cap"; they must read as cap 0, keyed like arm_key
    r = {"model": "97m", "layers": 6, "head": "bigru", "feats": None, "seed": 1}
    assert record_key(r) == arm_key("97m", 6, "bigru", None, 0, 1)
    assert record_key({**r, "cap": 64}) == arm_key("97m", 6, "bigru", None, 64, 1)


def test_resolve_repo_three_states(monkeypatch):
    monkeypatch.delenv("HF_RESULTS_DATASET", raising=False)
    assert resolve_repo(None) is None
    assert resolve_repo("me/runs") == "me/runs"
    monkeypatch.setenv("HF_RESULTS_DATASET", "me/env-runs")
    assert resolve_repo("") == "me/env-runs"


def test_resolve_repo_bare_push_without_env_exits(monkeypatch):
    monkeypatch.delenv("HF_RESULTS_DATASET", raising=False)
    with pytest.raises(SystemExit):
        resolve_repo("")


def test_sharded_path():
    base = "wmb/finetune_results.jsonl"
    assert sharded_path(base, "a") == "wmb/finetune_results.a.jsonl"
    assert sharded_path(base, "") == base            # unsharded passthrough


def test_resume_no_repo_is_empty(tmp_path):
    assert Ledger(tmp_path / "l.jsonl", None, "wmb/r.jsonl").resume() == set()


def test_resume_reads_local_without_download(tmp_path, monkeypatch):
    local = tmp_path / "l.jsonl"
    local.write_text(json.dumps(rec(seed=0)) + "\n" + json.dumps(rec(seed=1)) + "\n",
                     encoding="utf-8")
    monkeypatch.setattr(ledger, "hf_hub_download",
                        lambda *a, **k: pytest.fail("downloaded despite local present"))
    done = Ledger(local, "me/runs", "wmb/r.jsonl").resume()
    assert done == {record_key(rec(seed=0)), record_key(rec(seed=1))}


def test_resume_seeds_local_from_remote(tmp_path, monkeypatch):
    remote = tmp_path / "remote.jsonl"
    remote.write_text(json.dumps(rec(seed=2)) + "\n", encoding="utf-8")
    monkeypatch.setattr(ledger, "hf_hub_download", lambda *a, **k: str(remote))
    local = tmp_path / "l.jsonl"
    assert Ledger(local, "me/runs", "wmb/r.jsonl").resume() == {record_key(rec(seed=2))}
    assert local.exists()                            # remote copied down for resume


def test_resume_missing_remote_is_empty(tmp_path, monkeypatch):
    def missing(*a, **k):
        raise EntryNotFoundError("no file")
    monkeypatch.setattr(ledger, "hf_hub_download", missing)
    assert Ledger(tmp_path / "l.jsonl", "me/runs", "wmb/r.jsonl").resume() == set()


def test_append_stamps_ts_and_preserves_existing(tmp_path):
    local = tmp_path / "l.jsonl"
    led = Ledger(local, None, "wmb/r.jsonl")
    led.append({"model": "97m", "layers": 6, "head": "bigru", "feats": None, "seed": 0})
    led.append(rec(seed=1, ts="2020-01-01"))         # already stamped -> kept
    rows = [json.loads(l) for l in local.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["ts"]                             # stamped at write
    assert rows[1]["ts"] == "2020-01-01"             # not overwritten


def test_merge_push_unions_latest_ts(tmp_path, monkeypatch):
    # remote: X@Jan, Y@Jan.  local: X@Feb (newer), Z@Jan (new).  -> X(local), Y, Z
    remote = tmp_path / "remote.jsonl"
    remote.write_text(json.dumps(rec(seed=0, ts="2026-01-01", best_val=0.1)) + "\n" +
                      json.dumps(rec(seed=1, ts="2026-01-01", best_val=0.2)) + "\n",
                      encoding="utf-8")
    monkeypatch.setattr(ledger, "hf_hub_download", lambda *a, **k: str(remote))
    sent = {}
    monkeypatch.setattr(ledger, "upload_file",
                        lambda **k: sent.update(k, body=Path(k["path_or_fileobj"])
                                                .read_text(encoding="utf-8")))
    local = tmp_path / "l.jsonl"
    local.write_text(json.dumps(rec(seed=0, ts="2026-02-01", best_val=0.9)) + "\n" +
                     json.dumps(rec(seed=2, ts="2026-01-01", best_val=0.3)) + "\n",
                     encoding="utf-8")
    Ledger(local, "me/runs", "wmb/r.jsonl").merge_push("msg")

    merged = [json.loads(l) for l in local.read_text(encoding="utf-8").splitlines()]
    by_key = {record_key(r): r for r in merged}
    assert len(merged) == 3                                          # X, Y, Z, no dup
    assert by_key[record_key(rec(seed=0))]["best_val"] == 0.9        # newer local X won
    assert by_key[record_key(rec(seed=1))]["best_val"] == 0.2        # remote Y kept
    assert [r["ts"] for r in merged] == sorted(r["ts"] for r in merged)   # sorted by ts
    assert sent["path_in_repo"] == "wmb/r.jsonl" and sent["repo_id"] == "me/runs"
    assert sent["body"] == local.read_text(encoding="utf-8")         # uploaded == merged


def test_merge_push_noop_without_repo(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "upload_file",
                        lambda **k: pytest.fail("uploaded without a repo"))
    monkeypatch.setattr(ledger, "hf_hub_download",
                        lambda *a, **k: pytest.fail("downloaded without a repo"))
    local = tmp_path / "l.jsonl"
    local.write_text(json.dumps(rec()) + "\n", encoding="utf-8")
    led = Ledger(local, None, "wmb/r.jsonl")
    led.merge_push("msg")
