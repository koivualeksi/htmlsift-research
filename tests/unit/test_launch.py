"""Unit: what launch.py tells a training pod. The concern is that the pod's command, the
boards it builds and the extra pages it fetches come from the spec itself, never from an
argument string, and that a spec the pod cannot see is refused on the laptop.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "deploy" / "pod"))
launch = pytest.importorskip("launch")

GIT = ("abc123", True)


@pytest.fixture
def tmp_spec(tmp_path, monkeypatch):
    """Write a spec into a stand-in repo's experiments/."""
    monkeypatch.setattr(launch, "ROOT", tmp_path)
    (tmp_path / "experiments").mkdir()

    def make(name, text):
        p = tmp_path / "experiments" / f"{name}.yaml"
        p.write_text(text.format(name=name), encoding="utf-8")
        return p
    return make


def test_train_env_from_spec():
    name, boards, env = launch.train_env(ROOT / "experiments" / "wmb-mini.yaml", False,
                                         "L1", GIT)
    assert (name, boards) == ("wmb-mini-L1", "wmb")
    assert env == {"ENTRY": "-m trainers.train experiments/wmb-mini.yaml",
                   "CHECK": "-m trainers.train experiments/wmb-mini.yaml --check",
                   "LAUNCH": "L1", "GIT_SHA": "abc123", "GIT_DIRTY": "1"}


def test_smoke_reaches_the_entry_not_the_check():
    name, _, env = launch.train_env(ROOT / "experiments" / "wmb-lite.yaml", True, "L1", GIT)
    assert name == "wmb-lite-L1-smoke"
    assert env["ENTRY"].endswith(" --smoke") and env["CHECK"].endswith(" --check")


def test_single_type_board_builds_its_base_board():
    _, boards, _ = launch.train_env(ROOT / "experiments" / "wcxb-article-ladder.yaml", False,
                                    "L1", GIT)
    assert boards == "wcxb"


def test_boards_and_extra_come_from_the_configs(tmp_spec):
    p = tmp_spec("x", "name: {name}\nrecipe: table\nboard: wmb\n"
                      "extra: custom\nsweep: {{extra_repeat: [0, 1]}}\n")
    _, boards, env = launch.train_env(p, False, "L1", GIT)
    assert boards == "wmb" and env["EXTRA"] == "custom"


def test_spec_outside_experiments_is_refused(tmp_path):
    p = tmp_path / "x.yaml"
    p.write_text("name: x\nrecipe: table\nboard: wmb\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="experiments/"):
        launch.train_env(p, False, "L1", GIT)


def test_bad_spec_fails_on_the_laptop(tmp_spec):
    p = tmp_spec("x", "name: {name}\nrecipe: table\nboard: wmb\nlayers: 6\n")
    with pytest.raises(ValueError, match="has no key"):
        launch.train_env(p, False, "L1", GIT)
