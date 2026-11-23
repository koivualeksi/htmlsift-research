"""Parity gate for the mini artifact: the built ONNX graphs, run through the shipped
export/infer.load_mini forward, must reproduce the torch mini bit-for-bit or the build
does not ship. Value-independent -- random ids / block membership / features are a
complete check (the tests/gate/test_pool rationale), so no corpus is needed. §4.

parity() returns (flips, n, max_dp); the caller decides -- release.py refuses to push on
any flip, tests/gate/test_onnx_parity.py asserts zero. Runs the SHIPPED forward (load_mini),
not a re-implementation, so it gates exactly what production runs. parity_pages() is the
same comparison over a board's real prepared pages; the CLI builds a keeper's bundle and
runs both, the record behind the published parity claim:

    python -m export.gate wmb/ckpt/<arm>_s<seed>.pt --benchmark wmb --fold test
"""
import argparse
import tempfile

import numpy as np
import torch
from transformers import AutoTokenizer

from core.model import infer_table_page
from export.infer import load_mini


def _compare(infer_fn, emb, head, pages):
    """(flips at 0.5, blocks, max |dp|) of the shipped forward against the torch reference."""
    dev = torch.device("cpu")
    flips = n = 0
    max_dp = 0.0
    for page in pages:
        pt = infer_table_page(emb, head, page, dev)      # torch reference
        pf = infer_fn(page)                              # shipped onnx forward
        flips += int(((pt >= 0.5) != (pf >= 0.5)).sum())
        n += len(pt)
        max_dp = max(max_dp, float(np.abs(pt - pf).max(initial=0.0)))
    return flips, n, max_dp


def _random_pages(V, K, trials, seed):
    rng = np.random.default_rng(seed)
    for _ in range(trials):
        T = int(rng.integers(20, 500))
        ids = rng.integers(0, V, T).astype(np.int64)
        nb = int(rng.integers(1, 40))
        members = [[] for _ in range(nb)]
        for t in range(T):                          # ascending within block; some empty
            members[int(rng.integers(nb))].append(t)
        feats = rng.standard_normal((nb, K)).astype(np.float32)
        yield {"ids": ids, "members": members, "feats": feats}


def parity(table_p, head_p, emb, head, d_in, trials=20, seed=0):
    """Compare the built graphs (via load_mini) against the torch reference (emb, head)
    over `trials` random pages; return (flips, n, max_dp) at the 0.5 threshold. emb/head
    are the references build.py returned -- the int8 (or low-rank) embedder and the head."""
    V, H = emb.tok.weight.shape
    return _compare(load_mini(table_p, head_p), emb, head,
                    _random_pages(V, d_in - H, trials, seed))


def parity_pages(table_p, head_p, emb, head, pages):
    """parity() over real prepared pages (ids, members, feats); returns (flips, n, max_dp)."""
    return _compare(load_mini(table_p, head_p), emb, head, pages)


def main(argv=None):
    from core.loader import resolve_ckpt
    from export.build import build_mini
    from vendors.shared import bench_ops
    from vendors.shared.arms import WINDOW
    from vendors.shared.benchmark import load_benchmark
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", help="table keeper, local path or models-repo relative")
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--fold", default="test")
    a = ap.parse_args(argv)
    ck = resolve_ckpt(a.ckpt, torch.device("cpu"))
    with tempfile.TemporaryDirectory() as tmp:
        table_p, head_p, emb, head, d_in = build_mini(ck, tmp)
        print(f"built {table_p.name} {table_p.stat().st_size / 1e6:.1f} MB | "
              f"{head_p.name} {head_p.stat().st_size / 1e6:.1f} MB (d_in {d_in})", flush=True)
        flips, n, dp = parity(table_p, head_p, emb, head, d_in)
        print(f"random: {flips}/{n} flips, max |dp| {dp:.1e}", flush=True)
        bench = load_benchmark(a.benchmark)
        pages = bench.build_pages(AutoTokenizer.from_pretrained(ck["model"]), a.fold, WINDOW,
                                  feats=bench_ops.carried_features(bench, ck))
        rflips, rn, rdp = parity_pages(table_p, head_p, emb, head, pages)
        print(f"{a.benchmark} {a.fold}: {rflips}/{rn} flips over {len(pages)} pages, "
              f"max |dp| {rdp:.1e}", flush=True)
    if flips or rflips:
        raise SystemExit("ONNX parity FAILED")


if __name__ == "__main__":
    main()
