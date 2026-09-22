"""Parity gate for the mini artifact: the built ONNX graphs, run through the shipped
export/infer.load_mini forward, must reproduce the torch mini bit-for-bit or the build
does not ship. Value-independent -- random ids / block membership / features are a
complete check (the tests/gate/test_pool rationale), so no corpus is needed. §4.

parity() returns (flips, n); the caller decides -- release.py refuses to push on any
flip, tests/gate/test_onnx_parity.py asserts zero. Runs the SHIPPED forward (load_mini),
not a re-implementation, so it gates exactly what production runs.
"""
import numpy as np
import torch

from core.model import infer_table_page
from export.infer import load_mini


def parity(table_p, head_p, emb, head, d_in, trials=20, seed=0):
    """Compare the built graphs (via load_mini) against the torch reference (emb, head)
    over `trials` random pages; return (flips, n) at the 0.5 threshold. emb/head are the
    references build.py returned -- the int8-dequantized embedder and the trained head."""
    infer_fn = load_mini(table_p, head_p)
    rng = np.random.default_rng(seed)
    V, H = emb.tok.weight.shape
    K = d_in - H
    dev = torch.device("cpu")
    flips = n = 0
    for _ in range(trials):
        T = int(rng.integers(20, 500))
        ids = rng.integers(0, V, T).astype(np.int64)
        nb = int(rng.integers(1, 40))
        members = [[] for _ in range(nb)]
        for t in range(T):                          # ascending within block; some empty
            members[int(rng.integers(nb))].append(t)
        feats = rng.standard_normal((nb, K)).astype(np.float32)
        page = {"ids": ids, "members": members, "feats": feats}
        pt = infer_table_page(emb, head, page, dev)      # torch reference
        pf = infer_fn(page)                              # shipped onnx forward
        for a, b in zip(pt, pf):
            flips += (a >= 0.5) != (b >= 0.5)
            n += 1
    return flips, n
