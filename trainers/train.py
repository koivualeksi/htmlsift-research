"""Training entry point: a sweep spec or one run's run.yaml in, one run folder per config
out, each pushed to the HF results dataset when it finishes.

    python -m trainers.train experiments/wmb-mini.yaml
    python -m trainers.train experiments/wmb-mini.yaml --smoke        # 20 pages, 1 epoch
    python -m trainers.train <dir>/run.yaml                           # rerun one run
    python -m trainers.train experiments/wmb-mini.yaml --check        # checks only

A smoke runs seed 0 of each config, lives under data/runs/smoke/ and is never pushed or
resumed. $LAUNCH resumes a launch (a restarted pod): its finished configs are skipped.
"""
import argparse
import json
import os
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import torch
import yaml

from trainers import runs, spec
from trainers.recipes import Ctx, finetune, table
from vendors.shared.benchmark import load_benchmark

RECIPES = {"finetune": finetune, "table": table}
PROBS = runs.ROOT / "data" / "probs"


def load(path):
    """(experiment, note, configs, from_spec) from a spec or a run.yaml (has config:)."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if "config" in data:
        spec.check(data["config"])
        return data["experiment"], data["note"], [data["config"]], False
    s = spec.load(path)
    return s.name, s.note, spec.expand(s), True


def check_boards(configs):
    """Board capabilities every config needs, checked before any compute."""
    boards = {}
    for cfg in configs:
        b = boards.setdefault(cfg["board"], load_benchmark(cfg["board"]))
        h = cfg.get("holdout")
        if h is not None and h not in b.holdout_groups:
            raise ValueError(f"{b.name} has no holdout {h!r}; one of {b.holdout_groups}")
        if b.holdout_groups and h is None:
            raise ValueError(f"{b.name} trains leave-one-out: set holdout, one of "
                             f"{b.holdout_groups}")
        if cfg["train_limit"] is not None and not b.supports_train_limit:
            raise ValueError(f"train_limit not supported for {b.name}")


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def run_one(experiment, launch, note, cfg, ctx, smoke):
    folder = runs.start(experiment, launch, cfg, note=note, smoke=smoke)
    ctx.folder, ctx.probs = folder, PROBS / folder.name
    print(f"== run {folder.name}: {json.dumps(cfg)}", flush=True)
    with open(folder / "log.txt", "w", encoding="utf-8") as log, \
            redirect_stdout(_Tee(sys.stdout, log)), redirect_stderr(_Tee(sys.stderr, log)):
        try:
            result, curves = RECIPES[cfg["recipe"]].run(cfg, ctx)
        except Exception:
            traceback.print_exc()
            raise
    (folder / "curves.jsonl").write_text(
        "".join(json.dumps(c) + "\n" for c in curves), encoding="utf-8")
    runs.finish(folder, result)
    if not smoke:
        runs.push(folder.name)
    print(f"== done {folder.name}: {json.dumps(result)}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("path", help="experiments/<name>.yaml, or a run's run.yaml")
    ap.add_argument("--smoke", nargs="?", type=int, const=20, default=None,
                    help="N pages per fold (default 20), 1 epoch, seed 0, never pushed")
    ap.add_argument("--check", action="store_true",
                    help="load, expand and run every check, then exit without training")
    args = ap.parse_args(argv)

    experiment, note, configs, from_spec = load(args.path)
    smoke = args.smoke is not None
    if smoke:
        if from_spec:
            configs = [c for c in configs if c["seed"] == 0]
        configs = [{**c, "epochs": 1, "smoke": args.smoke} for c in configs]
    check_boards(configs)
    if args.check:
        print(f"{experiment}: {len(configs)} runs, all checks pass", flush=True)
        return

    launch = os.environ.get("LAUNCH") or runs.new_id()
    done = []
    if not smoke:
        runs.preflight()
        done = runs.done(launch)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Capped: on a pod cpu_count is the host's cores, and each epoch's fork pool copies the
    # parent's pages as refcounts touch them; ~90 workers OOM-killed a 6-epoch run.
    workers = min(os.cpu_count(), 16) if sys.platform == "linux" else 1
    todo = [c for c in configs if c not in done]
    print(f"launch {launch}: {experiment}, {len(todo)} of {len(configs)} runs to do on {dev}"
          + (" (smoke)" if smoke else ""), flush=True)
    ctx = Ctx(dev=dev, folder=None, probs=None, limit=args.smoke, eval_workers=workers)
    for cfg in todo:
        run_one(experiment, launch, note, cfg, ctx, smoke)


if __name__ == "__main__":
    main()
