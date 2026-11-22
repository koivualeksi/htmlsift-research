"""Common Crawl yield: how many pages a table ckpt's text gets past the FineWeb-Edu and DCLM
filters, against trafilatura's stored scores on the same pages.

The ckpt is plugged into htmlsift's Extractor in place of its bundle, so text comes out
of the package's own render + selection path with no export or release. The table runs
int8, as mini ships. `--ckpt mini` scores the installed package's own bundle instead.
Private data (koivualeksi/htmlsift-bench), pinned; pages, texts and per-page scores stay
under data/ccbench/, and a full run appends its summary row to data/ccbench/results.jsonl.
Linux only: the render is libxml2-version-specific, and extraction forks one CPU worker
per core over the pages (the table model is CPU-sized, as mini ships).

    python bench/ccbench.py --ckpt wmb/ckpt/311m-table-bigru-fABC_s1.pt --limit 300   # reproduces scores/mini.jsonl
    python bench/ccbench.py --ckpt data/runs/<run>/ckpt.pt --drop-title
"""
import argparse
import gzip
import json
import os
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

from htmlsift.extractor import Extractor
from htmlsift.loader import Model
from core.loader import build_infer, resolve_ckpt
from vendors.shared.arms import WINDOW

REPO, REVISION = "koivualeksi/htmlsift-bench", "0ac5a811ca972c83b861d4358e5f1b74a13ce904"
FOLDER = "2026-11-18-bench-en"
N_PAGES, POOL = 7177, 32399
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "ccbench"

LID = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
EN_MIN = 0.65
DCLM = ("mlfoundations/fasttext-oh-eli5", "openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin")
DCLM_PASS = 0.018112
EDU = ("HuggingFaceFW/fineweb-edu-classifier", "284663cbb2dabf9bda30d8f8cc49601251ee1631")
EDU_PASS = 3.0
CHUNK, BATCH = 96, 16          # the reference's Edu batching: per 96 pages, length-sorted 16s
N_BOOT = 2000


def fetch(name):
    return Path(hf_hub_download(REPO, f"{FOLDER}/{name}", repo_type="dataset",
                                revision=REVISION, local_dir=str(DATA)))


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def extractor(ck_ref):
    """An Extractor whose model is the ckpt: table int8, its feature group and carried
    z-score, the window it trained with. "mini" is the installed package's bundle."""
    if ck_ref == "mini":
        return Extractor("mini", threads=1)
    ck = resolve_ckpt(ck_ref, torch.device("cpu"))
    if ck.get("kind") != "table":
        raise SystemExit("ccbench scores table ckpts only")
    if ck.get("feats") not in (None, "ABC"):
        # the carried stats index the full research layout; Model.features z-scores the
        # group's own columns, and the two agree only when the group is a prefix of it
        raise SystemExit(f"feats {ck['feats']!r} unsupported: carried z-score is full-layout")
    tok, infer_fn, _kind, cap = build_infer({**ck, "int8": True}, torch.device("cpu"))
    ex = Extractor("mini")
    ex._model = Model(mode="mini", tok=tok, infer_fn=infer_fn, feats=ck.get("feats"),
                      zscore=ck.get("zscore"), cap=cap, window=WINDOW)
    return ex


def n_workers():
    """CPUs this container may use: the cgroup v2 quota when set (a pod's nproc reports
    the host), else the affinity mask."""
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            return max(1, int(quota) // int(period))
    except OSError:
        pass
    return len(os.sched_getaffinity(0))


_EX = None          # the extractor, set before the fork so workers share it copy-on-write
_DROP_TITLE = False


def _init_worker():
    torch.set_num_threads(1)


def _extract(html):
    try:
        return _EX.extract(html, drop_title=_DROP_TITLE) or "", None
    except Exception as e:  # noqa: BLE001 -- a failing page scores as empty text, as the reference
        return "", type(e).__name__


def extract_all(ex, pages, drop_title):
    """(text, error) per page, in page order, over a forked worker pool."""
    global _EX, _DROP_TITLE
    _EX, _DROP_TITLE = ex, drop_title
    n = n_workers()
    print(f"  extracting on {n} CPU workers", flush=True)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    with get_context("fork").Pool(n, initializer=_init_worker) as pool:
        return pool.map(_extract, [p["html"] for p in pages], chunksize=8)


class Scorer:
    """en / tokens / dclm / edu for one text, as htmlsift-data's mine/score.py."""

    def __init__(self, device):
        import fasttext
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        lid = DATA / "lid.176.bin"
        if not lid.exists():
            import urllib.request
            urllib.request.urlretrieve(LID, lid)
        self.lid = fasttext.load_model(str(lid))
        self.dclm = fasttext.load_model(hf_hub_download(*DCLM))
        self.tok = AutoTokenizer.from_pretrained(EDU[0], revision=EDU[1])
        self.edu = AutoModelForSequenceClassification.from_pretrained(EDU[0], revision=EDU[1])
        self.edu.to(device).eval()
        self.device = device

    def en(self, text):
        if not text.strip():
            return False
        labels, probs = self.lid.predict(text.replace("\n", " "), k=1)
        return labels[0] == "__label__en" and float(probs[0]) >= EN_MIN

    def dclm_prob(self, text):
        if not text.strip():
            return 0.0
        labels, probs = self.dclm.predict(" ".join(text.strip().splitlines()))
        return float(probs[0]) if labels[0] == "__label__hq" else 1.0 - float(probs[0])

    def n_tokens(self, text):
        return len(self.tok(text, add_special_tokens=False)["input_ids"]) if text else 0

    @torch.no_grad()
    def edu_scores(self, texts):
        out = [0.0] * len(texts)
        todo = sorted((i for i, t in enumerate(texts) if t.strip()), key=lambda i: len(texts[i]))
        for k in range(0, len(todo), BATCH):
            part = todo[k:k + BATCH]
            enc = self.tok([texts[i] for i in part], return_tensors="pt", padding="longest",
                           truncation=True).to(self.device)
            logits = self.edu(**enc).logits.squeeze(-1).float().cpu().tolist()
            for i, v in zip(part, logits, strict=True):
                out[i] = float(v)
        return out


def score_pages(ex, scorer, pages, out_path, drop_title):
    extracted = extract_all(ex, pages, drop_title)
    rows = []
    with open(out_path, "w", encoding="utf-8") as f:
        for k in range(0, len(pages), CHUNK):
            texts, errs = map(list, zip(*extracted[k:k + CHUNK]))
            edus = scorer.edu_scores(texts)
            for p, text, err, e in zip(pages[k:k + CHUNK], texts, errs, edus, strict=True):
                row = {"id": p["id"], "en": scorer.en(text), "tokens": scorer.n_tokens(text),
                       "dclm": scorer.dclm_prob(text), "edu": e, "text": text}
                if err:
                    row["error"] = err
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows.append(row)
            print(f"  {k + len(texts)}/{len(pages)} pages", flush=True)
    return rows


def passes(r, metric):
    return r["en"] and r[metric] >= (DCLM_PASS if metric == "dclm" else EDU_PASS)


def change(a, b, w, strata, seed=0):
    """sum(w a) / sum(w b) - 1 with a 95 % CI, paired bootstrap of pages within strata."""
    rng = np.random.default_rng(seed)
    sa, sb = np.zeros(N_BOOT), np.zeros(N_BOOT)
    for s in sorted(set(strata.tolist())):
        m = np.flatnonzero(strata == s)
        idx = m[rng.integers(0, len(m), (N_BOOT, len(m)))]
        sa += (w * a)[idx].sum(1)
        sb += (w * b)[idx].sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):    # a small --limit can draw sb == 0
        lo, hi = np.percentile(sa / sb - 1, [2.5, 97.5])
    return {"rel": float((w * a).sum() / (w * b).sum() - 1), "lo": float(lo), "hi": float(hi)}


def compare(new, base, meta):
    """Edu / DCLM change in passing pages and passing tokens, new vs base."""
    ids = list(meta)
    w = np.array([meta[i]["weight"] for i in ids], dtype=float)
    strata = np.array([meta[i]["stratum"] for i in ids])
    out = {}
    for metric in ("edu", "dclm"):
        pn = np.array([passes(new[i], metric) for i in ids], dtype=float)
        pb = np.array([passes(base[i], metric) for i in ids], dtype=float)
        tn = pn * np.array([new[i]["tokens"] for i in ids], dtype=float)
        tb = pb * np.array([base[i]["tokens"] for i in ids], dtype=float)
        out[f"{metric}_pages"] = change(pn, pb, w, strata)
        out[f"{metric}_tokens"] = change(tn, tb, w, strata)
    return out


def fmt(c):
    def p(x):
        return f"{x * 100:+.1f}".replace(".", ",")
    return f"{p(c['rel'])} % [{p(c['lo'])}, {p(c['hi'])}]"


def load_bench():
    """(meta, pages, trafilatura rows), fetched at the pin and checked."""
    meta = {r["id"]: r for r in read_jsonl(fetch("pages.jsonl"))}
    with gzip.open(fetch("html.jsonl.gz"), "rt", encoding="utf-8") as f:
        pages = [json.loads(line) for line in f if line.strip()]
    if len(pages) != N_PAGES or round(sum(r["weight"] for r in meta.values())) != POOL:
        raise SystemExit(f"bench data off: {len(pages)} pages, pool "
                         f"{sum(r['weight'] for r in meta.values()):.0f}")
    base = {r["id"]: r for r in read_jsonl(fetch("scores/trafilatura.jsonl"))}
    return meta, pages, base


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help="table ckpt, local path or models-repo-relative; 'mini' for the package bundle")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                    help="the Edu classifier's device; extraction is always CPU")
    ap.add_argument("--limit", type=int, default=None, help="smoke: first N pages, no summary row")
    ap.add_argument("--drop-title", action="store_true",
                    help="extract with drop_title=True, as trafilatura keeps the title out of its "
                         "text; off reproduces the stored scores/mini.jsonl")
    args = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    meta, pages, base = load_bench()
    pages = pages[:args.limit]
    stem = Path(args.ckpt).stem
    dev = torch.device(args.device)
    print(f"ccbench: {stem}, {len(pages)} pages, Edu on {dev}", flush=True)
    out = DATA / f"{stem}{'-droptitle' if args.drop_title else ''}.jsonl"
    rows = score_pages(extractor(args.ckpt), Scorer(dev), pages, out, args.drop_title)
    new = {r["id"]: r for r in rows}
    res = compare(new, base, {i: meta[i] for i in new})
    res.update(n=len(rows), errors=sum("error" in r for r in rows), revision=REVISION,
               drop_title=args.drop_title)
    print(f"{stem}  vs trafilatura, {len(rows)} pages, weighted to {POOL} CC pages\n"
          f"  Edu passing pages     {fmt(res['edu_pages'])}\n"
          f"  DCLM passing pages    {fmt(res['dclm_pages'])}\n"
          f"  DCLM passing tokens   {fmt(res['dclm_tokens'])}\n"
          f"  Edu passing tokens    {fmt(res['edu_tokens'])}  (a few long pages; unstable)\n"
          f"  extraction errors     {res['errors']}\n"
          f"  per-page rows         {out}", flush=True)
    if args.limit is None:
        rec = {"ckpt": args.ckpt, **res, "ts": datetime.now(timezone.utc).isoformat()}
        with open(DATA / "results.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
