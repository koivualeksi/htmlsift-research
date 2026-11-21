"""The table recipe's training loss, built from a run's config: per-block BCE, weighted
per block, plus optional penalty terms on the probabilities. With no method active every
weight is 1 and the loss is plain masked BCE with pos_weight = neg/pos."""
import numpy as np
import torch
import torch.nn.functional as F
from htmlsift.core.features import GROUPS
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
        self.gamma, self.smooth = cfg.get("focal_gamma", 0), cfg.get("label_smooth", 0.0)
        self.nb, self.box_mass = cfg.get("smooth_weight", 0.0), cfg.get("box_mass", 0.0)
        if self.box_mass:
            # blocks.jsonl stores each line's box size, not its box: a DOM subtree renders to
            # consecutive lines, so a box is taken as a maximal run of lines sharing one size
            # >= fp_box. Two adjacent boxes of equal size merge into one. -1 is off-box.
            self.box_id = [np.where(b >= box_min,
                                    np.cumsum(np.r_[True, b[1:] != b[:-1]]) - 1, -1)
                           for b in train_box]
        self.text_drop, self.feat_drop = cfg.get("text_drop", 0.0), cfg.get("feat_drop", 0.0)
        self.n_feat = sum(len(GROUPS[g]) for g in cfg.get("feats") or "")
        # Its own stream, so the drop draws leave train_head's batch shuffle untouched.
        self.rng = np.random.default_rng((cfg.get("seed", 0), 1))

    def augment(self, xs):
        """Modality dropout over the train pages for one epoch: one draw per page picks
        text-dropped, feat-dropped or intact, so a page never loses both. Rows are
        [text | feats] (table.run's cat); zeroed feats sit at the train mean of the
        z-scored columns. An intact page is passed through uncopied."""
        out = []
        for x, u in zip(xs, self.rng.random(len(xs))):
            if u < self.text_drop + self.feat_drop:
                d = x.shape[1] - self.n_feat
                x = x.copy()
                x[:, slice(None, d) if u < self.text_drop else slice(d, None)] = 0
            out.append(x)
        return out

    def loss(self, logits, y, valid, idx):
        """Weighted mean of per-block BCE over the valid (non-pad) blocks; idx names the
        train page of each batch row. Dividing by the weight sum, not the block count,
        keeps a weight from scaling the effective learning rate."""
        # Label smoothing softens the BCE targets only: pos_weight and the hard-negative
        # picks below stay on the hard labels. eps 0 leaves y exact.
        per = F.binary_cross_entropy_with_logits(
            logits, y * (1 - self.smooth) + self.smooth / 2, pos_weight=self.pos_weight,
            reduction="none")
        # Focal: |y - p| is 1 - p_correct. Not detached, as in standard focal loss, so the
        # factor also carries gradient; gamma 0 multiplies by exactly 1.
        per = per * (y - torch.sigmoid(logits)).abs() ** self.gamma
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
        loss = (per * w).sum() / w.sum()
        if self.nb:
            # Neighbour smoothness over pairs of equal gold label only, so a true content
            # edge stays free to be sharp while an isolated flip inside a run is pulled back.
            p = torch.sigmoid(logits)
            pair = valid[:, 1:] * valid[:, :-1] * (y[:, 1:] == y[:, :-1])
            loss = loss + self.nb * (pair * (p[:, 1:] - p[:, :-1]) ** 2).sum() / pair.sum()
        if self.box_mass:
            # A box's mean prob against 0, so a whole gold-free container is pushed down
            # together. Ids are offset by r * L to be unique across the batch.
            B, L = logits.shape
            bid = np.full((B, L), -1, np.int64)
            for r, k in enumerate(idx):
                bid[r, :len(self.box_id[k])] = np.where(self.box_id[k] >= 0,
                                                        self.box_id[k] + r * L, -1)
            bid = torch.from_numpy(bid).to(logits.device)
            on = bid >= 0
            n = torch.bincount(bid[on], minlength=B * L)
            s = torch.zeros(B * L, device=logits.device).index_add_(0, bid[on],
                                                                    torch.sigmoid(logits)[on])
            mean = s[n > 0] / n[n > 0]
            # A batch can hold no box (one long page alone, say): its term is then 0.
            loss = loss + self.box_mass * -torch.log(1 - mean + 1e-6).sum() / max(len(mean), 1)
        return loss
