"""Run store: one folder per run. runs/<id>/ on the HF results dataset is the record;
data/runs/<id>/ is where a run is staged before its push, and data/runs/smoke/<id>/ is
where a smoke run lives and stays. The id is fresh per run and nothing is deduplicated:
the same config run twice is two runs, told apart by what run.yaml records.

A run's run.yaml is written at start (ended and result null) and rewritten at finish;
only a finished run counts for resume or is pushed.
"""
import importlib.metadata
import os
import secrets
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml
from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "data" / "runs"
SMOKE = LOCAL / "smoke"
TZ = ZoneInfo("Europe/Helsinki")


def new_id(now=None):
    """Sortable id, Helsinki minute + 4 hex: 2026-11-19T1432-7f3a. Runs and launches."""
    now = (now or datetime.now(TZ)).astimezone(TZ)
    return f"{now:%Y-%m-%dT%H%M}-{secrets.token_hex(2)}"


def _now():
    return datetime.now(TZ).isoformat(timespec="seconds")


def _read(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _write(path, run):
    Path(path).write_text(yaml.safe_dump(run, sort_keys=False, allow_unicode=True),
                          encoding="utf-8")


def _repo(var):
    repo = os.environ.get(var)
    if not repo:
        raise RuntimeError(f"{var} is not set")
    return repo


def git_provenance():
    """(sha, dirty) from GIT_SHA/GIT_DIRTY when set -- a pod has a code tarball, no .git,
    so the launcher passes them -- else from the working tree. Neither is an error: a run
    that cannot name its code is not recorded."""
    if "GIT_SHA" in os.environ:
        if os.environ.get("GIT_DIRTY") not in ("0", "1"):
            raise RuntimeError("GIT_SHA is set without GIT_DIRTY=0|1")
        return os.environ["GIT_SHA"], os.environ["GIT_DIRTY"] == "1"
    try:
        def git(*a):
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True,
                                  check=True).stdout.strip()
        return git("rev-parse", "HEAD"), bool(git("status", "--porcelain"))
    except (OSError, subprocess.CalledProcessError) as e:
        raise RuntimeError("no provenance: no GIT_SHA in the env and no git work tree") from e


def provenance():
    sha, dirty = git_provenance()
    return {"git_sha": sha, "dirty": dirty,
            "htmlsift": importlib.metadata.version("htmlsift"),
            "code_snapshot": os.environ.get("CODE_SNAPSHOT")}


def host():
    import torch                          # lazy: the laptop-side readers never need it
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {"gpu": gpu, "pod": os.environ.get("RUNPOD_POD_ID")}


def preflight(keep=False):
    """Everything a non-smoke run needs to land, checked before any compute is spent:
    the results dataset (private, writable), the models repo when ckpts are kept, and
    provenance. launch.py runs it on the laptop before creating a pod."""
    api = HfApi()
    repos = {"HF_RESULTS_DATASET": "dataset"}
    if keep:
        repos["HF_MODELS_REPO"] = "model"
    for var, kind in repos.items():
        repo = _repo(var)
        api.auth_check(repo, repo_type=kind, write=True)
        if not api.repo_info(repo, repo_type=kind).private:
            raise RuntimeError(f"{var}={repo} is public; run records stay private")
    git_provenance()


def start(experiment, launch, config, *, kind="train", note="", smoke=False, root=None):
    """Create the run's folder and its run.yaml; return the folder. root overrides the
    staging dir (tests, the legacy converter)."""
    base = Path(root) if root else (SMOKE if smoke else LOCAL)
    run_id = new_id()
    folder = base / run_id
    folder.mkdir(parents=True, exist_ok=False)          # an id collision must not overwrite
    _write(folder / "run.yaml", {
        "id": run_id, "experiment": experiment, "launch": launch, "kind": kind,
        "note": note, "config": config, "provenance": provenance(), "host": host(),
        "started": _now(), "ended": None, "result": None})
    return folder


def finish(folder, result):
    run = _read(Path(folder) / "run.yaml")
    if run["ended"] is not None:
        raise RuntimeError(f"run {run['id']} already finished")
    run.update(ended=_now(), result=result)
    _write(Path(folder) / "run.yaml", run)


def push(run_id, root=None):
    """Upload a finished staged run's folder as one commit. Smoke runs live under
    smoke/, so a smoke id is not found here and cannot be pushed."""
    folder = Path(root or LOCAL) / run_id
    if _read(folder / "run.yaml")["ended"] is None:
        raise RuntimeError(f"run {run_id} is not finished")
    HfApi().upload_folder(repo_id=_repo("HF_RESULTS_DATASET"), repo_type="dataset",
                          folder_path=folder, path_in_repo=f"runs/{run_id}",
                          commit_message=f"run {run_id}")


def push_ckpt(run_id, path):
    """A kept ckpt goes to the models repo as runs/<id>.pt, not into the run folder."""
    HfApi().upload_file(repo_id=_repo("HF_MODELS_REPO"), repo_type="model",
                        path_or_fileobj=str(path), path_in_repo=f"runs/{run_id}.pt",
                        commit_message=f"ckpt {run_id}")


def _local(root=None):
    return [_read(p) for p in sorted(Path(root or LOCAL).glob("*/run.yaml"))]


def _remote():
    snap = snapshot_download(_repo("HF_RESULTS_DATASET"), repo_type="dataset",
                             allow_patterns="runs/*/run.yaml")
    return [_read(p) for p in sorted(Path(snap).glob("runs/*/run.yaml"))]


def push_pending(root=None):
    """Push every finished staged run that HF does not have yet (a push that failed at
    the end of a run). Returns the pushed ids."""
    remote = {r["id"] for r in _remote()}
    pending = [r["id"] for r in _local(root) if r["ended"] is not None and r["id"] not in remote]
    for run_id in pending:
        push(run_id, root)
    return pending


def done(launch, root=None):
    """Configs of this launch's finished runs, staged or pushed: a restarted pod skips
    these and nothing else. A new launch has none."""
    return [r["config"] for r in _remote() + _local(root)
            if r["launch"] == launch and r["ended"] is not None]


def get(run_id):
    for r in _remote():
        if r["id"] == run_id:
            return r
    raise KeyError(run_id)


def experiments():
    """Per experiment: run count and the latest start."""
    import pandas as pd
    df = pd.DataFrame([{"experiment": r["experiment"], "started": r["started"]}
                       for r in _remote()], columns=["experiment", "started"])
    return (df.groupby("experiment").agg(runs=("started", "size"), latest=("started", "max"))
            .reset_index())


def load(experiment):
    """One experiment's pushed runs as a DataFrame, nested keys flattened to dotted
    columns (config.model, result.best_val)."""
    import pandas as pd
    return pd.json_normalize([r for r in _remote() if r["experiment"] == experiment])
