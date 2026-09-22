"""
Board-agnostic training loop: fine-tune an encoder+head on prepared pages,
selecting the checkpoint by a caller-supplied val score. One arm, one seed --
the sweep that chains arms x seeds and aggregates their numbers lives in the
board adapter, not here.
"""

import numpy as np
import torch
import torch.nn as nn

from core.constants import THRESHOLD
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


def train(encoder, head, train_pages, val_pages, score_val, dev, *,
          epochs=4, enc_lr=2e-5, head_lr=1e-3, wd=0.01, warmup=200,
          grad_clip=1.0, seed=0, out_path=None, meta=None):
    """Fine-tune encoder+head on train_pages, selecting on
    score_val(val_pages, probs) (higher is better). Leaves encoder+head holding
    the best-val weights and returns a summary dict. A checkpoint is written only
    when out_path is given -- the sweep sets it for keeper arms alone, since these
    are large. One seed; the caller runs the 3-seed system."""
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

        m = _eval(encoder, head, val_pages, score_val, dev, autocast)
        m["loss"] = total_loss / len(samples)
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
