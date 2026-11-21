"""The table recipe's training loss, built from a run's config. A training method is a
per-block weight factor; factors multiply, so combining methods is a spec, not code.
With no factor active the loss is plain masked BCE with pos_weight = neg/pos."""
import torch
import torch.nn.functional as F


class Objective:
    def __init__(self, cfg, train_y, dev):
        pos = sum(float(y.sum()) for y in train_y)
        tot = sum(len(y) for y in train_y)
        self.pos_weight = torch.tensor((tot - pos) / max(pos, 1.0), device=dev)
        self.factors = []                       # one per active method, none yet

    def loss(self, logits, y, valid):
        """Weighted mean of per-block BCE over the valid (non-pad) blocks. Dividing by
        the weight sum, not the block count, keeps a factor from scaling the
        effective learning rate."""
        per = F.binary_cross_entropy_with_logits(
            logits, y, pos_weight=self.pos_weight, reduction="none")
        w = valid
        for f in self.factors:
            w = w * f(logits.detach(), y)
        return (per * w).sum() / w.sum()
