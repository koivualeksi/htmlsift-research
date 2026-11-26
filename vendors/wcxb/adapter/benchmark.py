"""WCXB Benchmark: fits WCXB to a trainer -- pages from blocks.jsonl + labels.jsonl
(dev only), val scored by word-F1 against main_content, fold F1 for a probs set.
Lifted out of the retired run.py; the metric is never reimplemented --
word_f1 is the vendored upstream scorer (via labels.wcxb_eval), and the per-record
mean mirrors WMBBenchmark.score_fold.

Implements vendors.shared.benchmark.Benchmark. WCXB has no carved val fold: dev is
both the train set and the val monitor (train == val), so build_pages resolves
"train" and "val" to the same dev pages; test is never trained on or labeled.

page_type narrows every fold to one type (board "wcxb-article"): the single-type
data-efficiency ladder. It does carve a val fold, since a ladder that selected epochs on
its own training pages would overstate the small rungs. Dev pages whose html is
byte-identical to a test page of the type are dropped first; the rest are split by
registrable domain into val (VAL_PAGES, no domain in train) and train.
"""
import hashlib
import json
import random
from pathlib import Path

from core.training import train_page
from vendors.shared import bench_ops
from vendors.shared.html_text import html_to_text
from vendors.wcxb.adapter.labels import join_selected, wcxb_eval
from vendors.wmb.adapter.split import domain_of

WCXB = Path(__file__).resolve().parents[1]
DATA = WCXB / "data"
FEATS = DATA / "feats.jsonl"
PAGE_TYPES = ("article", "collection", "documentation", "forum", "listing", "product",
              "service")
SEED = 20260815
VAL_PAGES = 200


class WCXBBenchmark:
    holdout_groups = ()

    def __init__(self, page_type=None):
        if page_type not in (None, *PAGE_TYPES):
            raise ValueError(f"unknown WCXB page type {page_type!r}; one of {PAGE_TYPES}")
        self.page_type = page_type
        # only a single-type board carves a val; on the full board val is the training
        # set, so a ladder would select each rung's epochs on its own training pages
        self.supports_train_limit = page_type is not None
        self.name = "wcxb" if page_type is None else f"wcxb-{page_type}"
        self._feats = None        # raw feats.jsonl {tid: [n, K]}, loaded once
        self._rows = {}           # split -> {tid: (blocks, labels|None, ref)}, cached
        self._page_types = {}     # tid -> page_type, populated by _fold_rows
        self._meta = {}           # tid -> (split, domain, html sha256), page_type only
        self._carve = None        # page_type only: {"train": ids, "val": ids}

    # ---- fold resolution (WCXB: train == val == dev; test is the 511) ----

    def _fold_rows(self, fold):
        """{tid: (blocks, labels|None, ref)} for a fold. Without page_type, train/val =
        dev (labels.jsonl selects it, + main_content ref); with it, train and val are the
        carve below. test = the split's ids from wcxb.jsonl (no labels, ref kept for
        scoring). Empty renders (blocks []) stay -- build_pages/export_probs handle them,
        score_fold scores pred="" as F1 0."""
        rows = self._split_rows("test" if fold == "test" else "dev")
        if self.page_type is None or fold == "test":
            return rows
        ids = self._carved()[fold]
        return {tid: r for tid, r in rows.items() if tid in ids}

    def _carved(self):
        """Dev of the type minus test twins, split by domain: whole domains go to val in
        a seeded order until it holds VAL_PAGES pages. Fixed per type, the same for
        every rung and seed."""
        if self._carve is None:
            dev = self._split_rows("dev")
            self._split_rows("test")
            test_sha = {s for sp, _, s in self._meta.values() if sp == "test"}
            pool = sorted(t for t in dev if self._meta[t][2] not in test_sha)
            by_dom = {}
            for t in pool:
                by_dom.setdefault(self._meta[t][1], []).append(t)
            doms = sorted(by_dom)
            random.Random(SEED).shuffle(doms)
            val = set()
            for d in doms:
                if len(val) >= VAL_PAGES:
                    break
                val.update(by_dom[d])
            self._carve = {"train": set(pool) - val, "val": val}
        return self._carve

    def test_domain_disjoint(self):
        """Test track_ids of the type whose domain no dev page of the type has: the
        second column of the ladder, free of the domain overlap WCXB's own split has."""
        if self.page_type is None:
            raise ValueError("test_domain_disjoint needs a page_type")
        self._split_rows("dev"), self._split_rows("test")
        dev_doms = {d for sp, d, _ in self._meta.values() if sp == "dev"}
        return {t for t, (sp, d, _) in self._meta.items() if sp == "test" and d not in dev_doms}

    def subsets(self, fold):
        """Named id subsets of a fold, scored beside the whole fold (bench_ops.score_subsets)."""
        if self.page_type is None or fold != "test":
            return {}
        return {"domain_disjoint": self.test_domain_disjoint()}

    def _split_rows(self, split):
        if split not in self._rows:
            labels = {}
            if split == "dev":
                with open(DATA / "labels.jsonl", encoding="utf-8") as f:
                    for line in f:
                        r = json.loads(line)
                        labels[r["track_id"]] = r["labels"]
            refs = {}
            with open(DATA / "wcxb.jsonl", encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    if r["split"] == split and self.page_type in (None, r["page_type"]):
                        tid = r["track_id"]
                        refs[tid] = r["main_content"] or ""
                        self._page_types[tid] = r["page_type"]
                        if self.page_type is not None:
                            self._meta[tid] = (split, domain_of(r["url"]),
                                               hashlib.sha256(r["html"].encode()).hexdigest())
            keep = set(labels) & set(refs) if split == "dev" else set(refs)
            rows = {}
            with open(DATA / "blocks.jsonl", encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    tid = r["track_id"]
                    if tid in keep:
                        rows[tid] = (r["blocks"], labels.get(tid), refs.get(tid, ""))
            self._rows[split] = rows
        return self._rows[split]

    def train_subset(self, n, seed=SEED):
        """n track_ids from the training pool (all of it if n >= the pool), as a seeded
        shuffle sliced as a prefix, so train_subset(25) is inside train_subset(50). Empty
        renders are left out, since build_pages drops them and they would shrink n."""
        pool = sorted(tid for tid, (blk, _, _) in self._fold_rows("train").items() if blk)
        random.Random(seed).shuffle(pool)
        return set(pool[:n])

    def raw_features(self):
        if self._feats is None:
            self._feats = bench_ops.raw_features(FEATS)
        return self._feats

    # ---- Benchmark protocol ----

    def build_pages(self, tok, fold, window, cap=0, feats=None, limit=None, ids=None):
        rows = self._fold_rows(fold)
        items = [(tid, blk, lab) for tid, (blk, lab, _) in rows.items()
                 if blk and (ids is None or tid in ids)]
        return [train_page(tok, tid, blk, lab, window, cap=cap,
                           feats=(feats[tid] if feats else None))
                for tid, blk, lab in items[:limit]]

    def build_val_scorer(self, val_pages, workers=1):
        """Mean word-F1 on the val pages themselves (dev = val, no carve). Cache each
        page's blocks + ref once; score_val assembles selected blocks (prob > 0.5) via
        labeling.join_selected and scores through the vendored word_f1 -- the one
        prediction assembly, shared with labels.py / evaluate.py. workers ignored
        (word_f1 is cheap; no fork pool, unlike WMB's ROUGE)."""
        rows = self._fold_rows("val")
        blocks = {p["tid"]: rows[p["tid"]][0] for p in val_pages}
        refs = {p["tid"]: rows[p["tid"]][2] for p in val_pages}

        def score_val(pages, probs):
            f1 = [wcxb_eval.word_f1(
                      join_selected(blocks[p["tid"]], [v > 0.5 for v in pr]),
                      refs[p["tid"]])[2]
                  for p, pr in zip(pages, probs)]
            return sum(f1) / len(f1)

        return score_val

    def _fold_blocks(self, fold):
        return {tid: r[0] for tid, r in self._fold_rows(fold).items()}

    def export_probs(self, fold, tok, infer_fn, out_path, window, cap=0, feats=None, limit=None):
        return bench_ops.export_probs(self._fold_blocks, fold, tok, infer_fn, out_path,
                                      window, cap, feats, limit)

    def score_fold(self, probs, fold, thr):
        """{n, prec, rec, f1} over the fold at threshold thr, via the vendored word_f1.
        Prediction = adapt(unescape(block)) for blocks with prob > thr, joined by newline
        -- exactly labels.py / evaluate.py. Metric vendored, mean ours (as in WMB)."""
        rows = self._fold_rows(fold)
        fs = []
        for tid, pr in probs.items():
            blk, _, ref = rows[tid]
            assert len(pr) == len(blk), f"probs/blocks mismatch {tid}: {len(pr)} vs {len(blk)}"
            pred = join_selected(blk, [v > thr for v in pr])
            fs.append(wcxb_eval.word_f1(pred, ref))
        n = len(fs)
        return {"n": n, "prec": sum(p for p, _, _ in fs) / n,
                "rec": sum(r for _, r, _ in fs) / n, "f1": sum(f for _, _, f in fs) / n}

    def score_text(self, outputs, fold):
        """{tid: (text, kind)} -> {n, prec, rec, f1}: an extractor's standard output judged
        by WCXB word-F1. html-kind is flattened to text (html_to_text), text-kind used as-is
        -- the same conversion for every extractor, ours included (bench/accuracy/heuristics.py).
        Metric is the vendored wcxb_eval.word_f1; per-record mean, as score_fold. The caller
        fixes the population (one shared set across methods); this scores exactly what it is
        given, so an empty prediction scores 0 against a non-empty ref."""
        rows = self._fold_rows(fold)
        fs = []
        for tid, (text, kind) in outputs.items():
            pred = html_to_text(text) if kind == "html" else text
            fs.append(wcxb_eval.word_f1(pred, rows[tid][2]))
        n = len(fs)
        return {"n": n, "prec": sum(p for p, _, _ in fs) / n,
                "rec": sum(r for _, r, _ in fs) / n, "f1": sum(f for _, _, f in fs) / n}

    def score_by_type(self, probs, fold, thr):
        """Per-page-type word-F1 means -- the WCXB breakdown parallel to DAnIEL's
        score_by_language. No macro key: the WCXB headline is the overall per-page
        mean (score_fold), not a macro over types."""
        rows = self._fold_rows(fold)
        by = {}
        for tid, pr in probs.items():
            blk, _, ref = rows[tid]
            pred = join_selected(blk, [v > thr for v in pr])
            by.setdefault(self._page_types[tid], []).append(wcxb_eval.word_f1(pred, ref)[2])
        return {t: sum(v) / len(v) for t, v in by.items()}

    def _train_ids(self):
        return self._fold_rows("train").keys()

    def load_features(self, group="ABC"):
        return bench_ops.load_features(self.raw_features(), self._train_ids(), group)

    def feature_stats(self, group="ABC"):
        return bench_ops.feature_stats(self.raw_features(), self._train_ids(), group)
