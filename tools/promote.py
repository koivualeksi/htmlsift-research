"""Promote one kept run's checkpoint to the models repo ($HF_MODELS_REPO), where
bench/accuracy/predict.py, bench/speed/ and tools/release.py load keepers from:
runs/<id>/ckpt.pt on the results dataset -> <board>/ckpt/<name>_s<seed>.pt, named by its
recipe (trainers.recipes.<recipe>.name). The models-repo commit names the run.

    python tools/promote.py <run-id> --dry-run     # print the source and the target
    python tools/promote.py <run-id>               # copy it; --replace to overwrite a keeper

A train_limit or holdout run is not a model to ship, and is refused. Shipping a promoted
keeper is then a reviewed edit of the keeper constants in tools/release.py.
"""
import argparse
import importlib
import os

from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download

from trainers import runs


def target(run):
    """The models-repo path for a run's checkpoint. A converted table run predates the
    recipe config: its model field already holds the old tag (97m-table-h128)."""
    cfg = run["config"]
    for k in ("train_limit", "holdout"):
        if cfg.get(k) is not None:
            raise SystemExit(f"run {run['id']}: {k}={cfg[k]} is not a model to ship")
    if "legacy" in run and cfg["recipe"] == "table":
        name = f"{cfg['model']}-{cfg['head']}-f{cfg['feats'] or 'none'}"
    else:
        name = importlib.import_module(f"trainers.recipes.{cfg['recipe']}").name(cfg)
    return f"{cfg['board']}/ckpt/{name}_s{cfg['seed']}.pt"


def main(argv=None):
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--replace", action="store_true", help="overwrite an existing keeper")
    ap.add_argument("--dry-run", action="store_true", help="print, copy nothing")
    a = ap.parse_args(argv)
    results, models = os.environ.get("HF_RESULTS_DATASET"), os.environ.get("HF_MODELS_REPO")
    if not (results and models):
        raise SystemExit("set HF_RESULTS_DATASET and HF_MODELS_REPO")
    run, src = runs.get(a.run_id), f"runs/{a.run_id}/ckpt.pt"
    api = HfApi()
    if not api.file_exists(results, src, repo_type="dataset"):
        raise SystemExit(f"run {a.run_id} has no ckpt.pt: it was not run with keep")
    dest = target(run)
    exists = api.file_exists(models, dest)
    print(f"{results}/{src} -> {models}/{dest}" + (" (replaces the current keeper)" * exists))
    if exists and not a.replace:
        raise SystemExit(f"{dest} is already a keeper; --replace to overwrite it")
    if a.dry_run:
        return
    path = hf_hub_download(results, src, repo_type="dataset")
    api.upload_file(path_or_fileobj=path, path_in_repo=dest, repo_id=models, repo_type="model",
                    commit_message=f"promote run {a.run_id} -> {dest}")
    print("promoted")


if __name__ == "__main__":
    main()
