"""The wcxb-article data-efficiency ladder -> results/wcxb-article-ladder.md: word-F1 per
training-set size on val, test and test's domain-disjoint subset, for the mini and the base
encoder, each trained from scratch and from the shipped model (arm A, the specs with init).

Mini runs are scored here in their shipped form: each run's head is factored to rank 128
with the shipped WMB-fit C (vendors.wmb.adapter.lowrank) and run with its int8 low-rank
table through the keeper loader. Encoder runs are read from their run.yaml, which the pod
scored. Zero-shot is the shipped mini, scored the same way, and the shipped base, scored
from the probs bench/accuracy/predict.py exports for it (a GPU model; run that on a pod or
wait for it on CPU); from-scratch columns have no 0-page row. Third-party extractors are bench/accuracy/heuristics.py --board
wcxb-article. Scores are cached per checkpoint under data/ladder/, so a rerun resumes.

    python bench/accuracy/predict.py --ckpt wmb/ckpt/311m-10_s1.pt --benchmark wcxb-article         --fold {val,test} --out data/ladder/zero-shot-base-{val,test}.jsonl
    python bench/accuracy/ladder.py
"""
import json
import os
import statistics
from pathlib import Path

import torch
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

from core.loader import build_infer, resolve_ckpt
from core.quant import factor_keeper
from trainers import runs
from vendors.shared import bench_ops
from vendors.shared.arms import WINDOW
from vendors.shared.benchmark import load_benchmark
from vendors.wmb.adapter.lowrank import fit_or_load_C

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "ladder"
BOARD = "wcxb-article"
RANK = 128
POOL = 513                       # train_limit null: the whole training pool
COLUMNS = (("mini, scratch", "wcxb-article-ladder"),
           ("mini, arm A", "wcxb-article-ladder-init"),
           ("base, scratch", "wcxb-article-ladder-base"),
           ("base, arm A", "wcxb-article-ladder-base-init"))
MINI_ZERO_SHOT = "wmb/ckpt/311m-table-bigru-fABC-nb1-pf4-ema0.999-r128_s2.pt"
BASE_PROBS = "zero-shot-base-{}.jsonl"           # in CACHE: predict.py's export for wmb/ckpt/311m-10_s1.pt
METRICS = (("val", "val"), ("test", "test"), ("test_domain_disjoint", "disjoint"))


def _score(bench, ck, dev):
    """{val, test, test_domain_disjoint} word-F1 at 0.5 for one ckpt."""
    tok, infer, _kind, cap = build_infer(ck, dev)
    feats = bench_ops.carried_features(bench, ck)
    out = {}
    for fold in ("val", "test"):
        pr = bench.export_probs(fold, tok, infer, CACHE / "probs.jsonl", WINDOW, cap, feats)
        out[fold] = bench.score_fold(pr, fold, 0.5)["f1"]
        out.update({f"{fold}_{n}": s["f1"]
                    for n, s in bench_ops.score_subsets(bench, pr, fold, 0.5).items()})
    return out


def _base_zero_shot(bench):
    """The shipped base's zero-shot scores, from predict.py's exported probs."""
    out = {}
    for fold in ("val", "test"):
        path = CACHE / BASE_PROBS.format(fold)
        if not path.exists():
            raise SystemExit(f"{path} missing: run predict.py for the base first (module docstring)")
        pr = {r["track_id"]: r["probs"]
              for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())}
        out[fold] = bench.score_fold(pr, fold, 0.5)["f1"]
        out.update({f"{fold}_{n}": s["f1"]
                    for n, s in bench_ops.score_subsets(bench, pr, fold, 0.5).items()})
    return out


def _cached(key, compute):
    path = CACHE / "scores.jsonl"
    rows = {}
    if path.exists():
        rows = {r["key"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())}
    if key not in rows:
        rows[key] = {"key": key, **compute()}
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rows[key]) + "\n")
        print(f"{key}: " + " ".join(f"{k} {rows[key][k]:.4f}" for k, _ in METRICS), flush=True)
    return rows[key]


def _table_runs(bench, experiment, C, dev):
    """[(rung, scores)] for a mini experiment, each run factored to RANK and scored."""
    out = []
    for r in runs.load(experiment).to_dict("records"):
        def compute(r=r):
            p = hf_hub_download(os.environ["HF_RESULTS_DATASET"], f"runs/{r['id']}/ckpt.pt",
                                repo_type="dataset")
            ck = factor_keeper(torch.load(p, map_location="cpu", weights_only=False), C, RANK)
            return _score(bench, ck, dev)
        rung = r["config.train_limit"]
        out.append((POOL if rung != rung or rung is None else int(rung),
                    _cached(f"{r['id']}-r{RANK}", compute)))
    return out


def _encoder_runs(experiment):
    """[(rung, scores)] for an encoder experiment, from what its pod recorded."""
    out = []
    for r in runs.load(experiment).to_dict("records"):
        rung = r["config.train_limit"]
        out.append((POOL if rung != rung or rung is None else int(rung),
                    {"val": r["result.best_val"], "test": r["result.test_f1"],
                     "test_domain_disjoint": r["result.test_domain_disjoint_f1"]}))
    return out


def _cell(rows, rung, key):
    vals = [s[key] for g, s in rows if g == rung]
    if not vals:
        return "—"
    m = statistics.mean(vals)
    return f"{m:.4f}±{statistics.stdev(vals):.4f}" if len(vals) > 1 else f"{m:.4f}"


def main():
    load_dotenv()
    CACHE.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cpu")
    bench = load_benchmark(BOARD)
    C = fit_or_load_C(resolve_ckpt(MINI_ZERO_SHOT, dev)["model"], False)
    cols = {}
    for _label, exp in COLUMNS:
        kind = runs.load(exp)["config.recipe"].iloc[0]
        cols[exp] = _table_runs(bench, exp, C, dev) if kind == "table" else _encoder_runs(exp)
    cols["wcxb-article-ladder-init"].append(
        (0, _cached(MINI_ZERO_SHOT, lambda: _score(bench, resolve_ckpt(MINI_ZERO_SHOT, dev), dev))))
    cols["wcxb-article-ladder-base-init"].append((0, _base_zero_shot(bench)))
    rungs = sorted({g for rows in cols.values() for g, _ in rows})
    seeds = {exp: max(sum(1 for g, _ in rows if g == x) for x in rungs if x) for exp, rows in cols.items()}

    lines = ["WCXB articles -- data-efficiency ladder. Word-F1 (vendored) at threshold 0.5, mean±std "
             "over seeds. Train pages are nested subsets of the 513-page pool (dev articles minus "
             "79 byte-identical test twins and a 200-page val held out by domain); val is that "
             "held-out set; test is all 257 test articles; disjoint is the 155 of them whose "
             "domain no dev article has. Scratch starts from a random head (mini) or the "
             "pretrained backbone (base); arm A from the shipped model, whose zero-shot score is "
             "the 0-page row. Mini is scored at rank 128 with its int8 table (the shipped form); "
             "base as its pod scored it (GPU). Third-party extractors on the same pages: "
             "results/heuristics-wcxb-article.md.", "",
             "Regenerate: `python bench/accuracy/ladder.py`", "",
             "Seeds: " + ", ".join(f"{label} {seeds[exp]}" for label, exp in COLUMNS) + ".", ""]
    for key, name in METRICS:
        lines += [f"## {name}", "", "| train pages | " + " | ".join(l for l, _ in COLUMNS) + " |",
                  "|---" * (len(COLUMNS) + 1) + "|"]
        for g in rungs:
            lines.append(f"| {g} | " + " | ".join(_cell(cols[exp], g, key) for _, exp in COLUMNS) + " |")
        lines.append("")
    dest = ROOT / "results" / "wcxb-article-ladder.md"
    dest.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines) + f"\nwrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
