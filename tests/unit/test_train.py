"""Unit: the training entry point. The concern is which runs a launch trains and where they
land: a spec runs every config, a run.yaml one; a resumed launch skips only its own
finished runs; a smoke trains seed 0 at 1 epoch and is never pushed; a config the board
cannot train fails before any recipe runs. HF and the recipes are stubbed.
"""
import json
from types import SimpleNamespace

import pytest
import yaml

from trainers import runs, train

SPEC = """\
name: {name}
recipe: table
board: {board}
seeds: 2
sweep:
  head: [bigru, linear]
{extra}"""


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_SHA", "abc123")
    monkeypatch.setenv("GIT_DIRTY", "0")
    monkeypatch.setenv("HF_RESULTS_DATASET", "me/results")
    monkeypatch.delenv("LAUNCH", raising=False)
    monkeypatch.setattr(runs, "LOCAL", tmp_path / "local")
    monkeypatch.setattr(runs, "SMOKE", tmp_path / "local" / "smoke")
    monkeypatch.setattr(runs, "host", lambda: {"gpu": None, "pod": None})
    monkeypatch.setattr(runs, "preflight", lambda: None)
    monkeypatch.setattr(train, "PROBS", tmp_path / "probs")
    remote = tmp_path / "remote"
    (remote / "runs").mkdir(parents=True)
    monkeypatch.setattr(runs, "snapshot_download", lambda *a, **k: str(remote))
    pushed, trained = [], []
    monkeypatch.setattr(runs, "push", lambda run_id, root=None: pushed.append(run_id))

    def fake_run(cfg, ctx):
        trained.append((cfg, ctx.limit))
        print("training", cfg["head"], cfg["seed"])
        return {"best_val": 0.5}, [{"epoch": 1, "val": 0.5}]
    monkeypatch.setattr(train, "RECIPES", {"table": SimpleNamespace(run=fake_run)})
    boards = {"wmb": SimpleNamespace(name="wmb", holdout_groups=(), supports_train_limit=True),
              "wcxb": SimpleNamespace(name="wcxb", holdout_groups=(),
                                      supports_train_limit=False),
              "daniel": SimpleNamespace(name="daniel", holdout_groups=("Greek",),
                                        supports_train_limit=False)}
    monkeypatch.setattr(train, "load_benchmark", lambda b: boards[b])
    return SimpleNamespace(tmp=tmp_path, remote=remote, pushed=pushed, trained=trained)


def spec(env, name="exp", board="wmb", extra=""):
    p = env.tmp / f"{name}.yaml"
    p.write_text(SPEC.format(name=name, board=board, extra=extra), encoding="utf-8")
    return str(p)


def folders(root):
    return sorted(p.parent for p in root.glob("*/run.yaml"))


def test_spec_runs_every_config_and_pushes_each(env):
    train.main([spec(env)])
    assert [(c["head"], c["seed"]) for c, _ in env.trained] == \
        [("bigru", 0), ("bigru", 1), ("linear", 0), ("linear", 1)]
    fs = folders(env.tmp / "local")
    assert len(fs) == 4 and sorted(env.pushed) == [f.name for f in fs]
    run = yaml.safe_load((fs[0] / "run.yaml").read_text(encoding="utf-8"))
    assert run["experiment"] == "exp" and run["result"] == {"best_val": 0.5}
    assert json.loads((fs[0] / "curves.jsonl").read_text()) == {"epoch": 1, "val": 0.5}
    assert "training" in (fs[0] / "log.txt").read_text(encoding="utf-8")


def test_run_yaml_reruns_one_run_in_a_new_launch(env):
    train.main([spec(env)])
    first = folders(env.tmp / "local")[0]
    old = yaml.safe_load((first / "run.yaml").read_text(encoding="utf-8"))
    env.trained.clear()
    train.main([str(first / "run.yaml")])
    assert [c for c, _ in env.trained] == [old["config"]]
    new = [f for f in folders(env.tmp / "local") if f.name not in env.pushed[:4]]
    rerun = yaml.safe_load((new[0] / "run.yaml").read_text(encoding="utf-8"))
    assert rerun["experiment"] == "exp" and rerun["launch"] != old["launch"]


def test_resume_skips_only_this_launchs_finished_runs(env, monkeypatch):
    monkeypatch.setenv("LAUNCH", "L1")
    train.main([spec(env)])
    env.trained.clear()
    train.main([spec(env)])                       # same launch: everything done
    assert env.trained == []
    monkeypatch.setenv("LAUNCH", "L2")
    train.main([spec(env)])                       # new launch: everything again
    assert len(env.trained) == 4


def test_smoke_is_seed_zero_one_epoch_local_only(env):
    train.main([spec(env), "--smoke"])
    assert [(c["head"], c["seed"], c["epochs"], c["smoke"], lim)
            for c, lim in env.trained] == [("bigru", 0, 1, 20, 20), ("linear", 0, 1, 20, 20)]
    assert env.pushed == []
    assert len(folders(env.tmp / "local" / "smoke")) == 2
    assert folders(env.tmp / "local") == []


def test_smoke_takes_a_page_count(env):
    train.main([spec(env), "--smoke", "5"])
    assert {lim for _, lim in env.trained} == {5}


@pytest.mark.parametrize("board, extra, msg", [
    ("wcxb", "train_limit: 250\n", "train_limit not supported"),
    ("daniel", "", "set holdout"),
])
def test_board_checks_fail_before_any_run(env, board, extra, msg):
    with pytest.raises(ValueError, match=msg):
        train.main([spec(env, board=board, extra=extra)])
    assert env.trained == [] and not (env.tmp / "local").exists()


def test_unknown_holdout_fails_before_any_run(env):
    cfg = {"recipe": "finetune", "board": "daniel", "holdout": "Klingon",
           "train_limit": None, "seed": 0}
    with pytest.raises(ValueError, match="no holdout"):
        train.check_boards([cfg])


def test_a_failing_run_stays_unfinished_and_unpushed(env, monkeypatch):
    def boom(cfg, ctx):
        raise RuntimeError("diverged")
    monkeypatch.setattr(train, "RECIPES", {"table": SimpleNamespace(run=boom)})
    with pytest.raises(RuntimeError, match="diverged"):
        train.main([spec(env)])
    (f,) = folders(env.tmp / "local")
    assert yaml.safe_load((f / "run.yaml").read_text(encoding="utf-8"))["ended"] is None
    assert "diverged" in (f / "log.txt").read_text(encoding="utf-8")
    assert env.pushed == []
