"""
Board-agnostic model: encoder, per-block heads, pooling, windowing, inference.

A page's blocks are joined with '\n' and encoded in one forward pass; each
block's token states are mean-pooled to a vector; a per-block head emits one
logit per block; sigmoid @ 0.5 selects. The encoder is a granite-embedding
backbone truncated to N layers with a fresh trainable final LayerNorm. Arm
matrix and discipline: core/CLAUDE.md.
"""

import math

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel

from core.prep_page import window_starts


def build_encoder(model_name, n_layers, dtype=torch.float32, device="cpu",
                  attn="sdpa"):
    """Load the backbone, keep its first n_layers, and replace the pretrained
    final_norm with a fresh trainable LayerNorm (re-stabilizes the truncated
    layer output). Dims come from the config, so the 97m (hidden 384) and 311m
    (hidden 768) share this path.

    dtype is the encoder PARAMETER dtype; fp32 is the reference — bf16 params at
    enc-lr 2e-5 round most updates to nothing (root CLAUDE.md §7). The forward
    uses bf16 autocast on CUDA regardless. The resolved dtype is printed to
    satisfy the model gate (§4)."""
    model = AutoModel.from_pretrained(model_name, attn_implementation=attn,
                                      dtype=dtype)
    n_full = len(model.layers)
    assert 1 <= n_layers <= n_full, f"n_layers {n_layers} outside 1..{n_full}"
    model.layers = model.layers[:n_layers]
    model.final_norm = nn.LayerNorm(model.config.hidden_size, eps=1e-5)
    model.to(device)
    dtypes = "+".join(sorted({str(p.dtype).replace("torch.", "")
                              for p in model.parameters()}))
    print(f"encoder: {model_name} — {n_layers} of {n_full} layers, hidden "
          f"{model.config.hidden_size} "
          f"({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params), "
          f"fresh final LayerNorm, param dtype {dtypes}", flush=True)
    return model


# Heads map pooled block vectors [1, n_blocks, d_in] to per-block logits
# [1, n_blocks]. LinearHead is the LR ablation (no sequence model); BiGRUHead is
# the reference. d_in = encoder hidden + any concatenated feature columns.

class LinearHead(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.out = nn.Linear(d_in, 1)

    def forward(self, x, mask=None):
        return self.out(x).squeeze(-1)      # per-block; padding is handled in the loss


class BiGRUHead(nn.Module):
    def __init__(self, d_in, hidden=256, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(d_in, hidden, num_layers=1, bidirectional=True,
                          batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(2 * hidden, 1)

    def forward(self, x, mask=None):
        if mask is None:
            h, _ = self.gru(x)
        else:
            # pack so the bidirectional GRU never reads padding -- outputs on real
            # blocks are then identical to the unbatched per-page forward.
            lengths = (~mask).sum(1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False)
            h, _ = nn.utils.rnn.pad_packed_sequence(
                self.gru(packed)[0], batch_first=True, total_length=x.shape[1])
        return self.out(self.drop(h)).squeeze(-1)


class TransformerHead(nn.Module):
    """Sequence head that COMPARES blocks instead of accumulating a recurrent
    state: near-identical block vectors attend to each other directly (the shape
    of evidence for repeated boilerplate). A published-negative arm (root
    CLAUDE.md §7), ported faithfully. Long pages are chunked with 50%%-stride
    block windows and the same most-interior ownership as the encoder stitch;
    positions are within-window sinusoids, so train and inference see one range
    regardless of page length."""

    def __init__(self, d_in, d_model=512, nhead=8, layers=4, dropout=0.2,
                 window=2048):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model) if d_in != d_model else nn.Identity()
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=4 * d_model, dropout=dropout,
            batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(enc_layer, layers)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, 1)
        self.d_model = d_model
        self.window = window

    def _pos(self, n, device, dtype):
        pos = torch.arange(n, device=device, dtype=torch.float32)[:, None]
        idx = torch.arange(0, self.d_model, 2, device=device, dtype=torch.float32)
        div = torch.exp(idx * (-math.log(10000.0) / self.d_model))
        pe = torch.zeros(n, self.d_model, device=device, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.to(dtype)[None]

    def forward(self, x, mask=None):  # x: [B, n_blocks, d_in]; mask [B, n] True=pad
        h = self.proj(x)
        n, w = h.shape[1], self.window
        if n <= w:
            h = self.enc(h + self._pos(n, h.device, h.dtype), src_key_padding_mask=mask)
        else:
            # windowed path stays per-page: long pages are singleton-batched, so no pad.
            assert h.shape[0] == 1 and mask is None, "windowed transformer path is per-page only"
            starts = window_starts(n, w)
            pos = self._pos(w, h.device, h.dtype)
            parts = []
            for k, s in enumerate(starts):
                z = self.enc(h[:, s:s + w] + pos)
                lo, hi = stitch_bounds(starts, w, k)
                parts.append(z[:, lo:hi])
            h = torch.cat(parts, dim=1)
            assert h.shape[1] == n, (h.shape[1], n)
        return self.out(self.drop(h)).squeeze(-1)  # [1, n_blocks]


def build_head(kind, d_in, hidden=256, dropout=0.2, **kw):
    if kind == "bigru":
        return BiGRUHead(d_in, hidden=hidden, dropout=dropout)
    if kind == "linear":
        return LinearHead(d_in)
    if kind == "transformer":
        return TransformerHead(d_in, dropout=dropout, **kw)
    raise ValueError(f"unknown head {kind!r}")


def pool(hidden, members, b_lo, b_hi, tok_off):
    """Differentiable mean-pool of encoder states hidden [w, H] into per-block
    vectors [b_hi-b_lo, H]. members hold page-level token indices; tok_off is the
    page index of hidden[0], so m - tok_off is the window-local row. A block with
    no tokens pools to a zero vector (sum of nothing / 1). The index tensors are
    built with numpy (O(blocks), not O(tokens)); byte-identical to the naive loop,
    gated in tests/gate/test_pool.py."""
    dev = hidden.device
    blocks = members[b_lo:b_hi]
    lengths = np.fromiter((len(m) for m in blocks), np.int64, len(blocks))
    counts = torch.tensor(np.maximum(lengths, 1), device=dev,
                          dtype=hidden.dtype).unsqueeze(1)
    pooled = hidden.new_zeros((b_hi - b_lo, hidden.shape[1]))
    if lengths.sum():
        mem_idx = np.concatenate([np.asarray(m, np.int64) for m in blocks if m]) - tok_off
        blk_idx = np.repeat(np.arange(len(blocks), dtype=np.int64), lengths)
        pooled = pooled.index_add(0, torch.from_numpy(blk_idx).to(dev),
                                  hidden[torch.from_numpy(mem_idx).to(dev)])
    return pooled / counts


def cat_feats(x, feats, b_lo, b_hi):
    """Concatenate per-block feature columns onto pooled vectors x [n, H] ->
    [n, H+K]. feats is the page's [n_blocks, K] array (or None); the b_lo:b_hi
    slice selects the blocks x covers -- a training window, or the whole page at
    inference. No-op when feats is None (the text-only arm)."""
    if feats is None:
        return x
    f = torch.as_tensor(feats[b_lo:b_hi], dtype=x.dtype, device=x.device)
    return torch.cat([x, f], dim=1)


def stitch_bounds(starts, window, k):
    """Window-local [lo, hi) slice that window k owns under most-interior
    ownership: each unit goes to the window whose centre is nearest, so a window
    keeps from the midpoint of its overlap with the previous window to the
    midpoint of its overlap with the next (0 / window at the two ends). Shared by
    the encoder stitch (infer_page) and TransformerHead, so the rule lives once
    (root CLAUDE.md §4)."""
    s = starts[k]
    lo = 0 if k == 0 else (starts[k - 1] + window + s) // 2 - s
    hi = window if k == len(starts) - 1 else (s + window + starts[k + 1]) // 2 - s
    return lo, hi


def encode_window(encoder, ids_t, autocast):
    """Encode a 1-D token window -> hidden states [w, H] (fp32). bf16 autocast on
    the forward when enabled (CUDA); a no-op on CPU."""
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        h = encoder(input_ids=ids_t[None],
                    attention_mask=torch.ones_like(ids_t)[None]
                    ).last_hidden_state[0]
    return h.float()


@torch.no_grad()
def pool_page(encoder, page, dev, autocast):
    """Full-page pooled block vectors [n_blocks, H] -- the encoder half of
    inference, no features and no head. A page within its window encodes in one
    pass; a longer page encodes in overlapping windows stitched by
    most-interior-token ownership -- each page token is taken from the window
    whose centre is nearest -- then the whole page is pooled. The page carries
    its own window/stride, so this cannot diverge from how the page was sampled
    for training. Shared by infer_page and the frozen screen so the stitch lives
    once."""
    window, stride = page["window"], page.get("stride")
    ids = torch.from_numpy(page["ids"]).to(dev)
    n = len(ids)
    if n <= window:
        hidden = encode_window(encoder, ids, autocast)
    else:
        hidden = torch.full((n, encoder.config.hidden_size), float("nan"),
                            device=dev)
        starts = window_starts(n, window, stride)
        for k, s in enumerate(starts):
            h = encode_window(encoder, ids[s:s + window], autocast)
            lo, hi = stitch_bounds(starts, window, k)
            hidden[s + lo:s + hi] = h[lo:hi]
        assert not torch.isnan(hidden).any(), f"stitch gap {page['tid']}"
    return pool(hidden, page["members"], 0, len(page["members"]), 0)


@torch.no_grad()
def infer_page(encoder, head, page, dev, autocast):
    """Full-page inference -> per-block probs (np.float32): pool the page, concat
    any structural features, run the head."""
    x = pool_page(encoder, page, dev, autocast)
    x = cat_feats(x, page.get("feats"), 0, len(page["members"]))
    probs = torch.sigmoid(head(x[None])[0])
    return probs.float().cpu().numpy()


class TableEmbedder(nn.Module):
    """Granite's token-embedding stage with the transformer stack removed: the
    frozen tok_embeddings + its embedding LayerNorm, nothing else. No attention,
    so a token's vector is position-independent -- the whole page embeds in one
    gather, with no window, stitch or mask. This is the no-encoder CPU arm; the
    only trainable component is the downstream head."""

    def __init__(self, tok_embeddings, norm, hidden_size):
        super().__init__()
        self.tok = tok_embeddings
        self.norm = norm
        self.hidden_size = hidden_size

    def forward(self, ids):                       # ids [T] -> [T, H]
        return self.norm(self.tok(ids))


def build_table_embedder(model_name, int8=False, device="cpu"):
    """Load the backbone, keep ONLY its embedding stage (drop every layer and the
    final norm), freeze it. int8 fake-quants the table per-row (core.quant) -- the
    packaging lever; here the table is ~all of the params. fp32 params, printed to
    satisfy the model gate (§4)."""
    m = AutoModel.from_pretrained(model_name, dtype=torch.float32)
    if int8:
        from core.quant import emb_int8_fakequant
        emb_int8_fakequant(m)
    emb = TableEmbedder(m.embeddings.tok_embeddings, m.embeddings.norm,
                        m.config.hidden_size).to(device)
    emb.requires_grad_(False).eval()
    print(f"table-embedder: {model_name} embeddings only, hidden "
          f"{emb.hidden_size}, {'int8 table' if int8 else 'fp32 table'}, "
          f"param dtype {str(next(emb.parameters()).dtype).replace('torch.','')}",
          flush=True)
    return emb


@torch.no_grad()
def pool_table_page(embedder, page, dev):
    """Pooled block embeddings [n_blocks, H] for one page: embed the whole token
    sequence in one pass (no window -- table vectors are position-independent) and
    mean-pool by block membership. Returns a tensor on dev."""
    ids = torch.from_numpy(page["ids"]).to(dev)
    return pool(embedder(ids), page["members"], 0, len(page["members"]), 0)


@torch.no_grad()
def pool_table_pages(embedder, pages, dev):
    """Frozen pooled block embeddings per page for head training, as CPU float32
    [n_blocks, H] -- the table-embedder analogue of frozen.pool_pages."""
    return [pool_table_page(embedder, p, dev).float().cpu().numpy() for p in pages]


@torch.no_grad()
def infer_table_page(embedder, head, page, dev):
    """Table-model inference -> per-block probs (np.float32): pool the page's frozen
    table embeddings, concat any structural features, run the head. The no-encoder
    analogue of infer_page (no transformer forward, no window)."""
    x = pool_table_page(embedder, page, dev)
    x = cat_feats(x, page.get("feats"), 0, len(page["members"]))
    return torch.sigmoid(head(x[None])[0]).float().cpu().numpy()
