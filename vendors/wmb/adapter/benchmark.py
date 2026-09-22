"""WMB Benchmark: the board-specific glue that fits WMB to a trainer -- pages from
blocks.jsonl, val ROUGE-5 against convert_main_content, and fold F1 scoring. Lifted
out of the retired run.py; the science (fork-pool COW val scorer, train-fold z-score,
cmc-null handling) moves verbatim, gated by the finetune smoke.

Implements vendors.shared.benchmark.Benchmark.
"""
import json
import multiprocessing as mp
import random
from collections import Counter, defaultdict
from pathlib import Path

from core.constants import THRESHOLD
from core.prep_page import prep_page
from vendors.shared import bench_ops
from vendors.wmb.adapter import eval as wmb_eval, serialize

WMB = Path(__file__).resolve().parents[1]
SPLITS = WMB / "data" / "splits.json"
FEATS = WMB / "data" / "feats.jsonl"


# Fork-pool val scoring holds the per-page serializer state + cmc refs in a module
# global so forked workers inherit it (COW) rather than pickling lxml state per job.
_SCORE_SNAP = None


def _score_one(job):
    tid, labels = job
    prepared, refs = _SCORE_SNAP
    return wmb_eval.calc_rouge_n_score(refs[tid], serialize.apply(prepared[tid], labels))["f1"]


class WMBBenchmark:
    name = "wmb"
    holdout_groups = ()               # no leave-one-out rotation
    supports_train_limit = True       # the data-efficiency ladder (pool prefixes)

    def __init__(self):
        self._feats = None               # raw feats.jsonl {tid: [n, K]}, loaded once
        self._folds = {}                 # fold -> (pages, blocks), cached for score_fold

    def train_subset(self, n, seed=20260815):
        """Stratified subset of the training pool, matched to val's level distribution.

        Returns a set of n track_ids (or all training ids if n >= pool size). Subsets
        are nested: train_subset(250) is a strict subset of train_subset(500), etc.,
        because the sequence is built by Bresenham interleave and sliced as a prefix."""
        with open(SPLITS, encoding="utf-8") as f:
            splits = json.load(f)

        train_by_level = defaultdict(list)
        val_level_counts = Counter()
        for tid, info in splits.items():
            fold = info.get("fold")
            level = info.get("level")
            if fold == "train" and level is not None:
                train_by_level[level].append(tid)
            elif fold == "val" and level is not None:
                val_level_counts[level] += 1

        # deterministic shuffle: sort then shuffle each bucket with a chained RNG
        rng = random.Random(seed)
        for level in sorted(train_by_level):
            train_by_level[level].sort()
            rng.shuffle(train_by_level[level])

        pool_size = sum(len(v) for v in train_by_level.values())
        n = min(n, pool_size)

        # val fractions per level; levels in train but not val get target 0
        val_total = sum(val_level_counts.values())
        target_frac = {lv: val_level_counts.get(lv, 0) / val_total for lv in train_by_level}

        # Bresenham interleave: greedily pick from the most underrepresented level
        cursor = {lv: 0 for lv in train_by_level}
        picked = {lv: 0 for lv in train_by_level}
        sequence = []
        for _ in range(n):
            best_level, best_deficit = None, -float("inf")
            for lv in sorted(train_by_level):
                if cursor[lv] >= len(train_by_level[lv]):
                    continue
                # deficit = target fraction - actual fraction so far
                actual = picked[lv] / (len(sequence) + 1) if sequence else 0
                deficit = target_frac[lv] - actual
                if deficit > best_deficit:
                    best_deficit = deficit
                    best_level = lv
            sequence.append(train_by_level[best_level][cursor[best_level]])
            cursor[best_level] += 1
            picked[best_level] += 1

        result = set(sequence)
        assert len(result) == n, f"train_subset: expected {n}, got {len(result)}"
        return result

    def raw_features(self):
        if self._feats is None:
            self._feats = bench_ops.raw_features(FEATS)
        return self._feats

    def _train_ids(self):
        return wmb_eval.fold_ids("train")

    def load_features(self, group="ABC"):
        return bench_ops.load_features(self.raw_features(), self._train_ids(), group)

    def feature_stats(self):
        return bench_ops.feature_stats(self.raw_features(), self._train_ids())

    def build_pages(self, tok, fold, window, cap=0, feats=None, limit=None, ids=None):
        fold_set = ids if ids is not None else wmb_eval.fold_ids(fold)
        blk = wmb_eval.load_blocks(fold_set)
        items = [(tid, b) for tid, b in blk.items() if b["blocks"]]
        return [prep_page(tok, tid, b["blocks"], b["labels"], window, cap=cap,
                          feats=(feats[tid] if feats else None))
                for tid, b in items[:limit]]

    def _fold_blocks(self, fold):
        return {tid: b["blocks"]
                for tid, b in wmb_eval.load_blocks(wmb_eval.fold_ids(fold)).items()}

    def export_probs(self, fold, tok, infer_fn, out_path, window, cap=0, feats=None, limit=None):
        return bench_ops.export_probs(self._fold_blocks, fold, tok, infer_fn, out_path,
                                      window, cap, feats, limit)

    def build_val_scorer(self, val_pages, workers=1):
        global _SCORE_SNAP
        src = wmb_eval.load_pages({p["tid"] for p in val_pages})
        prepared = {p["tid"]: serialize.prepare(src[p["tid"]]["html"], src[p["tid"]]["url"],
                                                len(p["members"])) for p in val_pages}
        refs = {p["tid"]: src[p["tid"]]["cmc"] for p in val_pages}
        _SCORE_SNAP = (prepared, refs)
        if workers > 1:
            import jieba
            jieba.initialize()           # warm in parent so forked workers inherit the dict

        def score_val(pages, probs):
            jobs = [(p["tid"], [int(v > THRESHOLD) for v in pr]) for p, pr in zip(pages, probs)]
            if workers > 1:
                with mp.get_context("fork").Pool(workers) as pool:
                    f1 = pool.map(_score_one, jobs, chunksize=max(1, len(jobs) // (workers * 4)))
            else:
                f1 = [_score_one(j) for j in jobs]
            return sum(f1) / len(f1)

        return score_val

    def score_fold(self, probs, fold, thr):
        pages, blocks = self._fold_data(fold)
        sub = {tid: blocks[tid] for tid in probs}          # score exactly what was exported
        preds = wmb_eval.predict(pages, sub, lambda tid, b: [int(v > thr) for v in probs[tid]])
        fs = [wmb_eval.calc_rouge_n_score(pages[tid]["cmc"], p)
              for tid, p in preds.items() if pages[tid]["cmc"] is not None]
        n = len(fs)
        return {"n": n, "prec": sum(f["prec"] for f in fs) / n,
                "rec": sum(f["rec"] for f in fs) / n, "f1": sum(f["f1"] for f in fs) / n}

    def score_text(self, outputs, fold):
        """{tid: (text, kind)} -> {n, prec, rec, f1}: the board's F1 for any extractor's
        STANDARD output. html-kind is html2text'd (the WMB scorer) then ROUGE-5 vs cmc;
        text-kind is scored as-is. This is the ONLY place html2text runs -- ours and the
        third-party baselines are judged the same way. cmc-null pages drop (as score_fold)."""
        pages, _ = self._fold_data(fold)
        fs = []
        for tid, (text, kind) in outputs.items():
            cmc = pages[tid]["cmc"]
            if cmc is None:
                continue
            pred = serialize.HTML2TextWrapper()(text, pages[tid]["url"]) if kind == "html" else text
            fs.append(wmb_eval.calc_rouge_n_score(cmc, pred))
        n = len(fs)
        return {"n": n, "prec": sum(f["prec"] for f in fs) / n,
                "rec": sum(f["rec"] for f in fs) / n, "f1": sum(f["f1"] for f in fs) / n}

    def _fold_data(self, fold):
        if fold not in self._folds:
            ids = wmb_eval.fold_ids(fold)
            self._folds[fold] = (wmb_eval.load_pages(ids), wmb_eval.load_blocks(ids))
        return self._folds[fold]
