"""DAnIEL Benchmark: fits DAnIEL to a trainer / the cross-board infer tool -- pages
from blocks.jsonl + labels.jsonl, val + fold scored by ROUGE-L (per language) against
the <p>-derived gold. Implements vendors.shared.benchmark.Benchmark, modeled on
WCXBBenchmark (external gold, align-to-ref labels).

DAnIEL trains, if ever, leave-one-language-out: `holdout` names the held-out language
and folds resolve relative to it -- train/val = the other four, test = the held-out one
-- so a LOLO run looks to the trainer like any single-train-fold board and the 5-way
rotation is an outer loop of five benchmarks. holdout=None is the eval-only / zero-shot
mode: test = all 1,689, no train. The metric is never reimplemented -- score is eval.py's
ROUGE-L (per-language tokenized); the per-language macro table stays in eval.py, these
methods return the per-record scalar a trainer monitors on (as in WCXB/WMB).
"""
import json
from pathlib import Path

from core.prep_page import prep_page
from vendors.shared import bench_ops
from vendors.daniel.adapter.eval import ISO, LANGS, score
from vendors.shared.labeling import join_selected

DATA = Path(__file__).resolve().parents[1] / "data"
FEATS = DATA / "feats.jsonl"


class DanielBenchmark:
    supports_train_limit = False
    holdout_groups = tuple(LANGS)     # the five languages -- the LOLO holdout set (finetune --lolo)

    def __init__(self, holdout=None):
        assert holdout is None or holdout in LANGS, f"unknown holdout {holdout!r}"
        self.holdout = holdout
        self.name = f"daniel-{ISO[holdout]}" if holdout else "daniel"
        self._all = None          # {tid: (blocks, labels, ref, language)}, joined once
        self._folds = {}          # fold -> that subset, cached
        self._feats = None        # raw feats.jsonl {tid: [n, K]}, loaded once

    # ---- fold resolution (LOLO by language; holdout=None -> all 1,689, eval-only) ----

    def _fold_langs(self, fold):
        if self.holdout is None:
            assert fold == "test", "holdout=None is eval-only: fold must be 'test' (all 1,689)"
            return set(LANGS)
        if fold == "test":
            return {self.holdout}
        # train / val: the four non-holdout languages. val == train (no carve) --
        # the LOLO val-carve is a parked training decision.
        return set(LANGS) - {self.holdout}

    def _all_rows(self):
        if self._all is None:
            meta = {}
            with open(DATA / "daniel.jsonl", encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    meta[r["track_id"]] = (r["reference"], r["language"])
            labels = {}
            with open(DATA / "labels.jsonl", encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    labels[r["track_id"]] = r["labels"]
            rows = {}
            with open(DATA / "blocks.jsonl", encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    tid = r["track_id"]
                    ref, lang = meta[tid]
                    rows[tid] = (r["blocks"], labels[tid], ref, lang)
            self._all = rows
        return self._all

    def _fold_rows(self, fold):
        if fold not in self._folds:
            langs = self._fold_langs(fold)
            self._folds[fold] = {t: v for t, v in self._all_rows().items() if v[3] in langs}
        return self._folds[fold]

    # ---- Benchmark protocol ----

    def build_pages(self, tok, fold, window, cap=0, feats=None, limit=None, ids=None):
        rows = self._fold_rows(fold)
        items = [(tid, blk, lab) for tid, (blk, lab, _, _) in rows.items() if blk]
        return [prep_page(tok, tid, blk, lab, window, cap=cap,
                          feats=(feats[tid] if feats else None))
                for tid, blk, lab in items[:limit]]

    def _fold_blocks(self, fold):
        return {tid: r[0] for tid, r in self._fold_rows(fold).items()}

    def export_probs(self, fold, tok, infer_fn, out_path, window, cap=0, feats=None, limit=None):
        return bench_ops.export_probs(self._fold_blocks, fold, tok, infer_fn, out_path,
                                      window, cap, feats, limit)

    def build_val_scorer(self, val_pages, workers=1):
        """Mean per-record ROUGE-L F1 on the val pages (the four train languages).
        Caches each page's blocks + ref + language once; score_val assembles selected
        blocks (prob > 0.5) via labeling.join_selected and scores through eval.py's
        ROUGE-L -- the one prediction assembly, shared with eval.py / labels.py. workers
        ignored for now (a fork pool like WMB's is the parked perf call if LOLO training
        proves slow)."""
        rows = self._fold_rows("val")
        blocks = {p["tid"]: rows[p["tid"]][0] for p in val_pages}
        refs = {p["tid"]: rows[p["tid"]][2] for p in val_pages}
        langs = {p["tid"]: rows[p["tid"]][3] for p in val_pages}

        def score_val(pages, probs):
            f1 = [score(refs[p["tid"]],
                        join_selected(blocks[p["tid"]], [v > 0.5 for v in pr]),
                        langs[p["tid"]])["f1"]
                  for p, pr in zip(pages, probs)]
            return sum(f1) / len(f1)

        return score_val

    def score_fold(self, probs, fold, thr):
        """{n, prec, rec, f1} over the fold at threshold thr, via eval.py's ROUGE-L.
        Prediction = adapt(unescape(block)) for blocks with prob > thr, joined by newline
        -- exactly oracle_preds. Per-record mean (as WCXB/WMB); the per-language macro is
        eval.py's."""
        rows = self._fold_rows(fold)
        fs = []
        for tid, pr in probs.items():
            blk, _, ref, lang = rows[tid]
            assert len(pr) == len(blk), f"probs/blocks mismatch {tid}: {len(pr)} vs {len(blk)}"
            pred = join_selected(blk, [v > thr for v in pr])
            fs.append(score(ref, pred, lang))
        n = len(fs)
        return {"n": n, "prec": sum(s["prec"] for s in fs) / n,
                "rec": sum(s["rec"] for s in fs) / n, "f1": sum(s["f1"] for s in fs) / n}

    def score_by_language(self, probs, fold, thr):
        """Per-language ROUGE-L F1 (+ macro over present languages) -- the DAnIEL breakdown
        the Chinese claim needs; score_fold gives only the micro scalar. The ROUGE-L metric
        is eval.score (never reimplemented); this only groups by language, so a partial fold
        (a --limit run) scores the languages it has. Not in the Benchmark Protocol -- the
        driver taps it via hasattr for DAnIEL alone."""
        rows = self._fold_rows(fold)
        by = {}
        for tid, pr in probs.items():
            blk, _, ref, lang = rows[tid]
            pred = join_selected(blk, [v > thr for v in pr])
            by.setdefault(lang, []).append(score(ref, pred, lang)["f1"])
        per = {ISO[l]: sum(v) / len(v) for l, v in by.items()}
        return {"macro": sum(per.values()) / len(per), **per}

    def raw_features(self):
        if self._feats is None:
            self._feats = bench_ops.raw_features(FEATS)
        return self._feats

    def _train_ids(self):
        return self._fold_rows("train").keys()

    def load_features(self, group="ABC"):
        return bench_ops.load_features(self.raw_features(), self._train_ids(), group)

    def feature_stats(self):
        return bench_ops.feature_stats(self.raw_features(), self._train_ids())
