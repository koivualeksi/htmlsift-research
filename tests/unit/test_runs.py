"""Unit: the run store. The concern is identity and scoping: every run gets a fresh
folder that a collision cannot overwrite, only finished runs count, resume sees only its
own launch, a smoke can never be pushed, and a run without provenance or a reachable
private repo fails before training. HF is stubbed: snapshot_download returns a local
directory standing in for the dataset, HfApi records calls.
"""
import re
from datetime import datetime

import pytest
import yaml

from trainers import runs


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Staging at tmp/local, the 'remote' dataset at tmp/remote, provenance from env."""
    monkeypatch.setenv("GIT_SHA", "abc123")
    monkeypatch.setenv("GIT_DIRTY", "0")
    monkeypatch.setenv("HF_RESULTS_DATASET", "me/results")
    monkeypatch.setattr(runs, "LOCAL", tmp_path / "local")
    monkeypatch.setattr(runs, "SMOKE", tmp_path / "local" / "smoke")
    monkeypatch.setattr(runs, "host", lambda: {"gpu": None, "pod": None})
    remote = tmp_path / "remote"
    (remote / "runs").mkdir(parents=True)
    monkeypatch.setattr(runs, "snapshot_download", lambda *a, **k: str(remote))
    calls = []

    class Api:
        def __getattr__(self, name):
            return lambda *a, **k: calls.append((name, a, k))
    monkeypatch.setattr(runs, "HfApi", Api)
    return remote, calls


def finished(**kw):
    folder = runs.start("exp", "L1", {"model": "97m", "seed": 0}, **kw)
    runs.finish(folder, {"best_val": 0.9})
    return folder


def as_remote(remote, folder):
    """Simulate a completed push: copy run.yaml into the stand-in dataset."""
    dst = remote / "runs" / folder.name
    dst.mkdir()
    (dst / "run.yaml").write_bytes((folder / "run.yaml").read_bytes())


def test_id_format():
    rid = runs.new_id(datetime(2026, 11, 19, 12, 32, tzinfo=runs.TZ))
    assert re.fullmatch(r"2026-11-19T1232-[0-9a-f]{4}", rid)


def test_id_is_helsinki_time():
    from zoneinfo import ZoneInfo
    utc = datetime(2026, 11, 19, 10, 0, tzinfo=ZoneInfo("UTC"))
    assert runs.new_id(utc).startswith("2026-11-19T1200-")          # EET, UTC+2


def test_start_then_finish(env):
    folder = runs.start("exp", "L1", {"model": "97m"}, note="n")
    run = yaml.safe_load((folder / "run.yaml").read_text())
    assert folder.parent == runs.LOCAL and run["id"] == folder.name
    assert (run["ended"], run["result"]) == (None, None)
    assert run["provenance"]["git_sha"] == "abc123" and run["provenance"]["dirty"] is False
    runs.finish(folder, {"best_val": 0.9})
    run = yaml.safe_load((folder / "run.yaml").read_text())
    assert run["result"] == {"best_val": 0.9} and run["ended"] is not None
    with pytest.raises(RuntimeError, match="already finished"):
        runs.finish(folder, {})


def test_id_collision_does_not_overwrite(env, monkeypatch):
    monkeypatch.setattr(runs, "new_id", lambda: "2026-11-19T1200-aaaa")
    runs.start("exp", "L1", {})
    with pytest.raises(FileExistsError):
        runs.start("exp", "L1", {})


def test_smoke_lives_apart_and_cannot_push(env):
    remote, calls = env
    folder = finished(smoke=True)
    assert folder.parent == runs.SMOKE
    assert runs._local() == []                         # staging listing never sees smokes
    with pytest.raises(FileNotFoundError):
        runs.push(folder.name)
    assert runs.push_pending() == [] and calls == []


def test_push_one_commit_only_when_finished(env):
    remote, calls = env
    folder = runs.start("exp", "L1", {})
    with pytest.raises(RuntimeError, match="not finished"):
        runs.push(folder.name)
    runs.finish(folder, {})
    runs.push(folder.name)
    (name, _, kw), = calls
    assert name == "upload_folder" and kw["path_in_repo"] == f"runs/{folder.name}"
    assert kw["repo_id"] == "me/results" and kw["repo_type"] == "dataset"


def test_push_pending_skips_pushed_and_unfinished(env):
    remote, calls = env
    pushed, pending = finished(), finished()
    as_remote(remote, pushed)
    runs.start("exp", "L1", {})                        # still running
    assert runs.push_pending() == [pending.name]
    assert [k["path_in_repo"] for _, _, k in calls] == [f"runs/{pending.name}"]


def test_done_is_launch_scoped_and_finished_only(env):
    remote, _ = env
    a = finished()                                     # L1, staged
    as_remote(remote, finished())                      # L1, pushed (and staged)
    runs.start("exp", "L1", {"model": "97m", "seed": 1})        # L1, unfinished
    other = runs.start("exp", "L2", {"model": "97m", "seed": 2})
    runs.finish(other, {})
    done = runs.done("L1")
    assert {"model": "97m", "seed": 1} not in done and {"model": "97m", "seed": 2} not in done
    assert done and all(c == {"model": "97m", "seed": 0} for c in done)
    assert runs.done("L3") == []


def test_provenance_requires_dirty_flag(env, monkeypatch):
    monkeypatch.delenv("GIT_DIRTY")
    with pytest.raises(RuntimeError, match="GIT_DIRTY"):
        runs.git_provenance()


def test_provenance_without_git_raises(monkeypatch, tmp_path):
    monkeypatch.delenv("GIT_SHA", raising=False)
    monkeypatch.setattr(runs, "ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="no provenance"):
        runs.git_provenance()


def test_preflight(env, monkeypatch):
    _, calls = env

    class Info:
        private = True

    class Api:
        def auth_check(self, repo, **k):
            calls.append((repo, k))

        def repo_info(self, repo, **k):
            return Info()
    monkeypatch.setattr(runs, "HfApi", Api)
    runs.preflight()
    assert calls == [("me/results", {"repo_type": "dataset", "write": True})]
    with pytest.raises(RuntimeError, match="HF_MODELS_REPO"):
        runs.preflight(keep=True)
    Info.private = False
    with pytest.raises(RuntimeError, match="public"):
        runs.preflight()
    monkeypatch.delenv("HF_RESULTS_DATASET")
    with pytest.raises(RuntimeError, match="HF_RESULTS_DATASET"):
        runs.preflight()


def test_load_flattens(env):
    remote, _ = env
    as_remote(remote, finished())
    df = runs.load("exp")
    assert df.loc[0, "config.model"] == "97m" and df.loc[0, "result.best_val"] == 0.9
    assert runs.experiments().to_dict("records")[0]["runs"] == 1
