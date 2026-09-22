"""
Frozen screen: run the encoder once per layer, then train light heads on the
cached block embeddings. This is the cheap front of the funnel -- the archive's
"screen every depth frozen, then fine-tune the promising ones". The expensive
encoder forward is shared across every head and feature set at a layer; only the
head trains, so a whole layers x heads array costs one forward per layer plus a
pile of near-free head fits. No encoder weights move, so nothing is cached to
disk or across environments: embeddings live in RAM for one layer and are
discarded (root CLAUDE.md -- no persisted cache, no provenance to guard).
"""
import numpy as np
import torch
import torch.nn.functional as F

from core.model import build_head, pool_page
from trainers.xgb import fit_xgb_head


def pool_pages(encoder, pages, dev, autocast):
    """Frozen pooled block embeddings per page, as CPU float32 arrays [n_blocks,
    H] for head training. One encoder forward per page (model.pool_page); the
    encoder must already be frozen and in eval()."""
    return [pool_page(encoder, p, dev, autocast).float().cpu().numpy()
            for p in pages]


def _batches(xs, order, batch_size, max_len, max_tokens=1 << 30):
    """Walk `order` (indices pre-sorted by length) into padded batches. Yields
    (idx, (x[B,T,d], mask[B,T] True=pad)); a page longer than max_len is its own
    batch as (idx, None) so the transformer's windowed path stays per-page. A batch
    caps at batch_size pages AND at max_tokens padded blocks (B*T) -- the token cap
    shrinks long-page batches so the transformer's O(B*T^2) attention can't OOM."""
    i = 0
    while i < len(order):
        if len(xs[order[i]]) > max_len:
            yield [order[i]], None
            i += 1
            continue
        idx = [order[i]]                              # always take at least one
        i += 1
        while (i < len(order) and len(idx) < batch_size
               and len(xs[order[i]]) <= max_len
               and (len(idx) + 1) * len(xs[order[i]]) <= max_tokens):
            idx.append(order[i])
            i += 1
        T = max(len(xs[k]) for k in idx)
        x = np.zeros((len(idx), T, xs[idx[0]].shape[1]), np.float32)
        mask = np.ones((len(idx), T), bool)
        for r, k in enumerate(idx):
            x[r, :len(xs[k])] = xs[k]
            mask[r, :len(xs[k])] = False
        yield idx, (x, mask)


def _forward(head, xs, idx, packed, dev):
    """Head logits for one batch: [B,T] with (x,mask), or [1,n] for a singleton."""
    if packed is None:
        return head(torch.from_numpy(xs[idx[0]]).to(dev)[None]), None
    x, mask = packed
    mt = torch.from_numpy(mask).to(dev)
    return head(torch.from_numpy(x).to(dev), mt), mt


def train_head_frozen(head, train_x, train_y, val_x, val_pages, score_val, dev,
                      *, epochs=4, head_lr=1e-3, seed=0, batch_size=64,
                      max_len=2048, max_tokens=32768):
    """Train a per-block head on FROZEN, pre-pooled inputs, BATCHED over pages
    (length-bucketed, padded, masked). Batching feeds the GPU many pages at once
    -- the sequence heads (BiGRU/Transformer) were GPU-starved at batch 1. The
    masked forward is per-block identical to the unbatched path (packed GRU /
    masked attention), but the optimizer now steps once per batch (mini-batch
    gradient), so vals shift slightly vs the old per-page SGD. train_x/val_x are
    lists of [n_blocks, d_in] aligned with train_y / val_pages. Selects the epoch
    best on score_val and returns {val, epoch}."""
    torch.manual_seed(seed)
    pos = sum(float(y.sum()) for y in train_y)
    tot = sum(len(y) for y in train_y)
    pos_weight = torch.tensor((tot - pos) / max(pos, 1.0), device=dev)
    opt = torch.optim.AdamW(head.parameters(), lr=head_lr)
    by_len = sorted(range(len(train_x)), key=lambda i: len(train_x[i]))
    val_order = sorted(range(len(val_x)), key=lambda i: len(val_x[i]))
    # Linear LR warmup over the first 10% of steps, then hold: a 4-layer
    # transformer from a cold high-lr start diverges (val -> 0); warmup plus the
    # grad-clip below stabilise it, matching train.py. BiGRU/linear are unaffected.
    n_steps = epochs * sum(1 for _ in _batches(train_x, by_len, batch_size, max_len, max_tokens))
    warmup = max(1, n_steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup))
    rng = np.random.default_rng(seed)
    best, best_state = {"val": -1.0, "epoch": 0}, None
    for epoch in range(1, epochs + 1):
        head.train()
        batches = list(_batches(train_x, by_len, batch_size, max_len, max_tokens))
        rng.shuffle(batches)
        for idx, packed in batches:
            opt.zero_grad()
            logits, mt = _forward(head, train_x, idx, packed, dev)
            B, T = logits.shape
            y = np.zeros((B, T), np.float32)
            for r, k in enumerate(idx):
                y[r, :len(train_y[k])] = train_y[k]
            yt = torch.from_numpy(y).to(dev)
            per = F.binary_cross_entropy_with_logits(
                logits, yt, pos_weight=pos_weight, reduction="none")
            valid = torch.ones_like(logits) if mt is None else (~mt).float()
            ((per * valid).sum() / valid.sum()).backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            sched.step()

        head.eval()
        probs = [None] * len(val_x)
        with torch.no_grad():
            for idx, packed in _batches(val_x, val_order, batch_size, max_len, max_tokens):
                p = torch.sigmoid(_forward(head, val_x, idx, packed, dev)[0]).cpu().numpy()
                for r, k in enumerate(idx):
                    probs[k] = p[r][:len(val_x[k])]
        v = score_val(val_pages, probs)
        if v > best["val"]:
            best = {"val": v, "epoch": epoch}
            best_state = {k: t.cpu().clone() for k, t in head.state_dict().items()}
        print(f"  epoch {epoch}: val {v:.4f}"
              f"{'  *' if epoch == best['epoch'] else ''}", flush=True)

    head.load_state_dict(best_state)
    return best


def fit_torch_head(head_kind, hidden, z, train_pages, val_pages, etr, eva, train_y,
                   score_val, dev, *, epochs=4, seed=0, batch_size=64, head_lr=1e-3):
    """Build a per-block head and train it on the cached embeddings, concatenating the
    z-scored features z ({track_id: [n, F]}, or None for text-only) onto each page's
    embedding. The shared torch path for the frozen family; returns (best {val, epoch},
    head) -- head carries the best-epoch weights, for the table trainer's --keep."""
    def cat(pages, emb):
        return [np.concatenate([e, z[p["tid"]]], 1) if z is not None else e
                for p, e in zip(pages, emb)]
    tx, vx = cat(train_pages, etr), cat(val_pages, eva)
    head = build_head(head_kind, d_in=tx[0].shape[1], hidden=hidden).to(dev)
    best = train_head_frozen(head, tx, train_y, vx, val_pages, score_val, dev,
                             epochs=epochs, seed=seed, batch_size=batch_size, head_lr=head_lr)
    return best, head


def fit_head(head_kind, raw, z, train_pages, val_pages, etr, eva, train_y,
             score_val, dev, *, hidden, epochs, seed, batch_size, head_lr, use_emb=True):
    """Dispatch to the xgb or torch fitter on the cached embeddings. Returns
    (best {val, epoch}, head): head is None for xgboost (no torch module), else the
    trained head carrying best-epoch weights (for --keep). use_emb is the xgb-only
    'append the block's own embedding' flag -- the torch path takes features-only
    through zero-width etr/eva, so it needs no such flag."""
    if head_kind == "xgboost":
        return fit_xgb_head(raw, train_pages, val_pages, etr, eva, train_y,
                            score_val, seed, use_emb=use_emb), None
    return fit_torch_head(head_kind, hidden, z, train_pages, val_pages, etr, eva,
                          train_y, score_val, dev, epochs=epochs, seed=seed,
                          batch_size=batch_size, head_lr=head_lr)
