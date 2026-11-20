"""Unit: tools/promote.py names a kept run exactly as the old trainers named the keepers
already in the models repo, refuses what is not a model to ship, and never overwrites a
keeper unasked. No HF: HfApi and the run store are stubbed.
"""
import pytest

from tools import promote
from trainers.spec import DEFAULTS


def run(recipe, legacy=False, **over):
    cfg = {"recipe": recipe, "board": "wmb", "seed": 1, **DEFAULTS[recipe], **over}
    return {"id": "2026-11-20T2100-abcd", "config": cfg,
            **({"legacy": {"file": "wmb/table_results.jsonl"}} if legacy else {})}


@pytest.mark.parametrize("recipe, over, want", [
    ("finetune", {}, "wmb/ckpt/311m-10_s1.pt"),
    ("finetune", {"model": "97m", "layers": 6, "qat_w8a8": True}, "wmb/ckpt/97m-6-qat_s1.pt"),
    ("table", {}, "wmb/ckpt/311m-table-bigru-fABC_s1.pt"),
    ("table", {"feats": None}, "wmb/ckpt/311m-table-bigru-fnone_s1.pt"),
    ("table", {"model": "97m", "emb": "int8", "hidden": 128},
     "wmb/ckpt/97m-table-int8-h128-bigru-fABC_s1.pt"),
    ("table", {"extra": "custom", "extra_repeat": 5},
     "wmb/ckpt/311m-table+custom5-bigru-fABC_s1.pt"),
])
def test_target_matches_the_existing_keeper_names(recipe, over, want):
    assert promote.target(run(recipe, **over)) == want


def test_a_converted_table_run_keeps_its_old_tag():
    r = {"id": "x", "legacy": {}, "config": {"recipe": "table", "board": "wmb",
                                             "model": "97m-table-h128", "head": "bigru",
                                             "feats": "ABC", "seed": 0}}
    assert promote.target(r) == "wmb/ckpt/97m-table-h128-bigru-fABC_s0.pt"


@pytest.mark.parametrize("over", [{"train_limit": 500}, {"holdout": "Chinese"}])
def test_a_ladder_or_holdout_run_is_refused(over):
    with pytest.raises(SystemExit, match="not a model to ship"):
        promote.target(run("finetune", **over))


class Api:
    def __init__(self, existing):
        self.existing, self.uploads = existing, []

    def file_exists(self, repo, path, repo_type=None):
        return path in self.existing

    def upload_file(self, **kw):
        self.uploads.append(kw)


@pytest.fixture
def hub(monkeypatch):
    monkeypatch.setenv("HF_RESULTS_DATASET", "me/results")
    monkeypatch.setenv("HF_MODELS_REPO", "me/models")
    monkeypatch.setattr(promote.runs, "get", lambda run_id: run("table"))
    monkeypatch.setattr(promote, "hf_hub_download", lambda *a, **k: "/cache/ckpt.pt")

    def install(existing):
        api = Api(existing)
        monkeypatch.setattr(promote, "HfApi", lambda: api)
        return api
    return install


SRC = "runs/2026-11-20T2100-abcd/ckpt.pt"
DEST = "wmb/ckpt/311m-table-bigru-fABC_s1.pt"


def test_promote_copies_to_the_keeper_path_and_names_the_run(hub):
    api = hub({SRC})
    promote.main(["2026-11-20T2100-abcd"])
    (up,) = api.uploads
    assert (up["repo_id"], up["repo_type"], up["path_in_repo"]) == ("me/models", "model", DEST)
    assert "2026-11-20T2100-abcd" in up["commit_message"]


def test_a_run_without_keep_is_refused(hub):
    api = hub(set())
    with pytest.raises(SystemExit, match="not run with keep"):
        promote.main(["2026-11-20T2100-abcd"])
    assert api.uploads == []


def test_an_existing_keeper_needs_replace(hub):
    api = hub({SRC, DEST})
    with pytest.raises(SystemExit, match="--replace"):
        promote.main(["2026-11-20T2100-abcd"])
    promote.main(["2026-11-20T2100-abcd", "--replace"])
    assert len(api.uploads) == 1


def test_dry_run_copies_nothing(hub):
    api = hub({SRC})
    promote.main(["2026-11-20T2100-abcd", "--dry-run"])
    assert api.uploads == []
