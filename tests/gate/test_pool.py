"""
Gate: core.model.pool is byte-identical to the naive per-token reference loop it
replaced. The vectorized build (numpy index tensors, O(blocks)) must feed the same
index_add on the same hidden in the same order, so the pooled output is bit-for-bit
identical -- which is why the swap needed no retraining and cannot move any §4
number. Value-independent, so random hidden + varied membership is a complete check
(empty blocks, multi-token blocks, sub-range windows, tok_off > 0).
"""

import numpy as np
import torch

from core.model import pool


def _naive(hidden, members, b_lo, b_hi, tok_off):
    dev = hidden.device
    mem_idx, blk_idx, counts = [], [], []
    for k, bi in enumerate(range(b_lo, b_hi)):
        mem = members[bi]
        counts.append(max(len(mem), 1))
        mem_idx.extend(m - tok_off for m in mem)
        blk_idx.extend([k] * len(mem))
    pooled = hidden.new_zeros((b_hi - b_lo, hidden.shape[1]))
    if mem_idx:
        pooled = pooled.index_add(0, torch.tensor(blk_idx, device=dev),
                                  hidden[torch.tensor(mem_idx, device=dev)])
    return pooled / torch.tensor(counts, device=dev, dtype=pooled.dtype).unsqueeze(1)


def test_pool_byte_identical_random():
    rng = np.random.default_rng(0)
    for _ in range(300):
        off = int(rng.integers(0, 20))
        T = int(rng.integers(1, 400))
        hidden = torch.randn(T, 384)
        nb = int(rng.integers(1, 40))
        members = [[] for _ in range(nb)]
        for t in range(T):                       # ascending within block; some empty
            members[int(rng.integers(nb))].append(t + off)
        b_lo = int(rng.integers(0, nb))
        b_hi = int(rng.integers(b_lo + 1, nb + 1))
        assert torch.equal(pool(hidden, members, b_lo, b_hi, off),
                           _naive(hidden, members, b_lo, b_hi, off))


def test_pool_byte_identical_edges():
    for members, T in ([[], [], []], 5), ([[0]], 1), ([[0, 1, 2]], 3), ([[2], [], [0, 1]], 3):
        h = torch.randn(T, 384)
        assert torch.equal(pool(h, members, 0, len(members), 0),
                           _naive(h, members, 0, len(members), 0))
