"""The two epoch loops, one run each: train for N epochs, score val after every epoch,
leave the model holding the best epoch's weights. train_encoder fine-tunes an
encoder+head on token windows (recipe finetune); train_head trains a head alone on
frozen, pre-pooled block inputs, batched over pages (recipe table).
"""
import numpy as np
import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from htmlsift.extractor import THRESHOLD
from core.model import cat_feats, encode_window, infer_page, pool


def _eval(encoder, head, val_pages, score_val, dev, autocast):
    """Inference over val -> board-agnostic block P/R/F1 (at 0.5) plus the
    board's selection scalar from score_val(val_pages, probs)."""
    encoder.eval(), head.eval()
    probs = [infer_page(encoder, head, p, dev, autocast) for p in val_pages]
    tp = fp = fn = 0
    for p, pr in zip(val_pages, probs):
        pred, y = pr > THRESHOLD, p["y"] > 0.5
        tp += int((pred & y).sum())
        fp += int((pred & ~y).sum())
        fn += int((~pred & y).sum())
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"block_p": prec, "block_r": rec,
            "block_f1": 2 * prec * rec / max(prec + rec, 1e-9),
            "val": score_val(val_pages, probs)}


def train_encoder(encoder, head, train_pages, val_pages, score_val, dev, *,
                  epochs=4, enc_lr=2e-5, head_lr=1e-3, wd=0.01, warmup=200,
                  grad_clip=1.0, seed=0, out_path=None, meta=None, steps=None):
    """Fine-tune encoder+head on train_pages, selecting on
    score_val(val_pages, probs) (higher is better). Leaves encoder+head holding
    the best-val weights and returns {best_val, best_epoch, history}. A checkpoint
    is written only when out_path is given (keep), since these are large. steps, as in
    train_head, replaces epochs with a budget of at least that many steps, scored 4 times:
    4 epochs of a few pages end before the 200-step warmup does."""
    torch.manual_seed(seed)
    autocast = dev.type == "cuda"

    samples = [(pi, si) for pi, p in enumerate(train_pages)
               for si in range(len(p["samples"]))]
    pos = sum(float(p["y"][lo:hi].sum())
              for p in train_pages for _, _, lo, hi in p["samples"])
    tot = sum(hi - lo for p in train_pages for _, _, lo, hi in p["samples"])
    pos_weight = torch.tensor((tot - pos) / max(pos, 1.0), device=dev)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    no_decay = ("bias", "norm")
    named_enc = list(encoder.named_parameters())
    opt = torch.optim.AdamW([
        {"params": [p for n, p in named_enc
                    if not any(k in n.lower() for k in no_decay)],
         "lr": enc_lr, "weight_decay": wd},
        {"params": [p for n, p in named_enc
                    if any(k in n.lower() for k in no_decay)],
         "lr": enc_lr, "weight_decay": 0.0},
        {"params": list(head.parameters()), "lr": head_lr, "weight_decay": 0.0},
    ])
    every = 1
    if steps is not None:
        epochs = -(-steps // len(samples))
        every = -(-epochs // 4)
    total_steps = len(samples) * epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda step: min((step + 1) / max(warmup, 1),
                              max(0.0, 1 - step / total_steps)))
    print(f"train: {len(samples)} samples over {len(train_pages)} pages, "
          f"pos_weight {float(pos_weight):.2f}, {total_steps} steps", flush=True)

    rng = np.random.default_rng(seed)
    best, best_state, history = {"val": -1.0, "epoch": 0}, None, []
    for epoch in range(1, epochs + 1):
        encoder.train(), head.train()
        rng.shuffle(samples)
        total_loss = 0.0
        for pi, si in samples:
            p = train_pages[pi]
            s, e, b_lo, b_hi = p["samples"][si]
            opt.zero_grad()
            hidden = encode_window(
                encoder, torch.from_numpy(p["ids"][s:e]).to(dev), autocast)
            x = pool(hidden, p["members"], b_lo, b_hi, s)
            x = cat_feats(x, p.get("feats"), b_lo, b_hi)
            y = torch.from_numpy(p["y"][b_lo:b_hi]).to(dev)
            loss = loss_fn(head(x[None])[0], y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(head.parameters()), grad_clip)
            opt.step()
            sched.step()
            total_loss += float(loss.detach())

        if epoch % every and epoch != epochs:
            continue
        m = _eval(encoder, head, val_pages, score_val, dev, autocast)
        m["loss"] = total_loss / len(samples)
        m["epoch"] = epoch
        history.append(m)
        star = ""
        if m["val"] > best["val"]:
            best = {"val": m["val"], "epoch": epoch}
            best_state = (
                {k: v.cpu().clone() for k, v in encoder.state_dict().items()},
                {k: v.cpu().clone() for k, v in head.state_dict().items()})
            star = "  *"
        print(f"epoch {epoch}: loss {m['loss']:.4f}  block P {m['block_p']:.4f} "
              f"R {m['block_r']:.4f} F1 {m['block_f1']:.4f}  val {m['val']:.4f}"
              f"{star}", flush=True)

    encoder.load_state_dict(best_state[0])
    head.load_state_dict(best_state[1])
    if out_path is not None:
        torch.save({**(meta or {}), "epoch": best["epoch"], "val": best["val"],
                    "encoder": best_state[0], "head_state": best_state[1]}, out_path)
        print(f"saved best (epoch {best['epoch']}, val {best['val']:.4f}) -> "
              f"{out_path}", flush=True)
    return {"best_val": best["val"], "best_epoch": best["epoch"],
            "history": history}


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


def train_head(head, train_x, train_y, val_x, val_pages, score_val, dev, objective,
               *, epochs=4, head_lr=1e-3, seed=0, batch_size=64,
               max_len=2048, max_tokens=32768, ema=0.0, steps=None):
    """Train a per-block head on FROZEN, pre-pooled inputs, batched over pages
    (length-bucketed, padded, masked; the masked forward is per-block identical to the
    unbatched path). train_x/val_x are lists of [n_blocks, d_in] aligned with train_y /
    val_pages; objective (trainers.objective) gives the batch loss. ema > 0 scores, selects
    and keeps an exponential moving average of the weights (that decay per step) while the
    raw weights train on. steps, when given, replaces epochs with a budget of at least that
    many optimiser steps, scored 4 times as the 4-epoch run is: a per-step EMA over the
    handful of steps a small train set gets in 4 epochs is still near its first step's
    weights. Leaves head holding the best scored weights and returns
    {best_val, best_epoch, history}, one history entry per scoring."""
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(head.parameters(), lr=head_lr)
    by_len = sorted(range(len(train_x)), key=lambda i: len(train_x[i]))
    val_order = sorted(range(len(val_x)), key=lambda i: len(val_x[i]))
    per_epoch = sum(1 for _ in _batches(train_x, by_len, batch_size, max_len, max_tokens))
    if steps is not None:
        epochs = -(-steps // per_epoch)
    every = -(-epochs // 4) if steps is not None else 1
    # Linear LR warmup over the first 10% of steps, then hold: a 4-layer
    # transformer from a cold high-lr start diverges (val -> 0); warmup plus the
    # grad-clip below stabilise it. BiGRU/linear are unaffected.
    n_steps = epochs * per_epoch
    warmup = max(1, n_steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup))
    rng = np.random.default_rng(seed)
    scored = head
    if ema:
        avg = AveragedModel(head, multi_avg_fn=get_ema_multi_avg_fn(ema), use_buffers=True)
        scored = avg.module
    best, best_state, history = {"val": -1.0, "epoch": 0}, None, []
    for epoch in range(1, epochs + 1):
        head.train()
        batches = list(_batches(objective.augment(train_x), by_len, batch_size, max_len,
                                max_tokens))
        rng.shuffle(batches)
        total_loss = 0.0
        for idx, packed in batches:
            opt.zero_grad()
            logits, mt = _forward(head, train_x, idx, packed, dev)
            B, T = logits.shape
            y = np.zeros((B, T), np.float32)
            for r, k in enumerate(idx):
                y[r, :len(train_y[k])] = train_y[k]
            yt = torch.from_numpy(y).to(dev)
            valid = torch.ones_like(logits) if mt is None else (~mt).float()
            loss = objective.loss(logits, yt, valid, idx)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            sched.step()
            if ema:
                avg.update_parameters(head)
            total_loss += float(loss.detach())

        if epoch % every and epoch != epochs:
            continue
        scored.eval()
        probs = [None] * len(val_x)
        with torch.no_grad():
            for idx, packed in _batches(val_x, val_order, batch_size, max_len, max_tokens):
                p = torch.sigmoid(_forward(scored, val_x, idx, packed, dev)[0]).cpu().numpy()
                for r, k in enumerate(idx):
                    probs[k] = p[r][:len(val_x[k])]
        v = score_val(val_pages, probs)
        history.append({"epoch": epoch, "val": v, "loss": total_loss / len(batches)})
        if v > best["val"]:
            best = {"val": v, "epoch": epoch}
            best_state = {k: t.cpu().clone() for k, t in scored.state_dict().items()}
        print(f"  epoch {epoch}: val {v:.4f}"
              f"{'  *' if epoch == best['epoch'] else ''}", flush=True)

    head.load_state_dict(best_state)
    return {"best_val": best["val"], "best_epoch": best["epoch"], "history": history}
