"""The table recipe's training loss, built from a run's config: per-block BCE, weighted
per block. With no method active every weight is 1 and the loss is plain masked BCE
with pos_weight = neg/pos."""
import numpy as np
import torch
import torch.nn.functional as F
from htmlsift.extractor import THRESHOLD


def len_weight(chars, cap, floor):
    """chars/cap clipped to [floor, 1]: full weight from cap chars up."""
    return np.clip(np.asarray(chars, np.float32) / cap, floor, 1.0)


class Objective:
    def __init__(self, cfg, train_y, train_chars, dev, train_box=None):
        """train_y, train_chars and train_box are per train page, in train_head's train_x
        order. train_box is only read with fp_box set."""
        cap, side = cfg.get("len_cap"), cfg.get("len_side", "both")
        self.w = [np.ones(len(c), np.float32) for c in train_chars]
        if cap:
            # the ramp on the rows of len_side (pos: gold content, neg: the rest), 1 elsewhere
            self.w = [np.where({"both": True, "pos": y == 1, "neg": y == 0}[side],
                               len_weight(c, cap, cfg["len_floor"]), 1.0).astype(np.float32)
                      for c, y in zip(train_chars, train_y)]
        if cfg.get("page_norm"):
            # WMB F1 is a mean over pages: each page's weights sum to 1, so a page counts
            # once in the loss however many rows it has.
            self.w = [w / w.sum() for w in self.w]
        # By weight, not row count: content rows run longer, so a count-based pos_weight
        # on top of a length weight would tilt the loss toward content.
        pos = sum(float((w * y).sum()) for w, y in zip(self.w, train_y))
        neg = sum(float((w * (1 - y)).sum()) for w, y in zip(self.w, train_y))
        self.pos_weight = torch.tensor(neg / pos, device=dev)
        # The weight a pick of each gold-free block costs: fp_weight, or with fp_box only
        # inside a gold-free box of at least fp_box lines (blocks.box_sizes), 1 elsewhere.
        fp, box_min = cfg.get("fp_weight", 1.0), cfg.get("fp_box")
        self.fp = [np.where(b >= box_min, fp, 1.0).astype(np.float32) if box_min
                   else np.full(len(y), fp, np.float32)
                   for y, b in zip(train_y, train_box or [None] * len(train_y))]

    def loss(self, logits, y, valid, idx):
        """Weighted mean of per-block BCE over the valid (non-pad) blocks; idx names the
        train page of each batch row. Dividing by the weight sum, not the block count,
        keeps a weight from scaling the effective learning rate."""
        per = F.binary_cross_entropy_with_logits(
            logits, y, pos_weight=self.pos_weight, reduction="none")
        w = np.zeros(tuple(logits.shape), np.float32)
        fp = np.ones(tuple(logits.shape), np.float32)
        for r, k in enumerate(idx):
            w[r, :len(self.w[k])] = self.w[k]
            fp[r, :len(self.fp[k])] = self.fp[k]
        w = valid * torch.from_numpy(w).to(logits.device)
        # Online hard negatives: a gold-free block the model currently selects costs its
        # fp weight. Weight 1 multiplies by 1, leaving the loss bit-identical.
        picked = torch.sigmoid(logits.detach()) > THRESHOLD
        w = w * torch.where((y == 0) & picked, torch.from_numpy(fp).to(logits.device), 1.0)
        return (per * w).sum() / w.sum()
