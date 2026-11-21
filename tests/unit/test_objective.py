"""With no method active, Objective.loss must be bit-identical to the masked BCE the
table recipe trained every shipped mini with, so moving the loss out of the loop moves
no number."""
import numpy as np
import torch
import torch.nn.functional as F

from trainers.objective import Objective

DEV = torch.device("cpu")


def test_default_loss_matches_masked_bce():
    rng = np.random.default_rng(0)
    train_y = [(rng.random(n) > 0.6).astype(np.float32) for n in (7, 13, 4)]
    obj = Objective({}, train_y, DEV)

    pos = sum(float(y.sum()) for y in train_y)
    tot = sum(len(y) for y in train_y)
    assert float(obj.pos_weight) == np.float32((tot - pos) / pos)

    logits = torch.from_numpy(rng.standard_normal((3, 13)).astype(np.float32))
    y = torch.from_numpy((rng.random((3, 13)) > 0.6).astype(np.float32))
    mask = torch.zeros(3, 13, dtype=torch.bool)
    mask[0, 7:] = mask[2, 4:] = True                  # padding
    for valid in ((~mask).float(), torch.ones_like(logits)):
        per = F.binary_cross_entropy_with_logits(
            logits, y, pos_weight=obj.pos_weight, reduction="none")
        want = (per * valid).sum() / valid.sum()
        assert torch.equal(obj.loss(logits, y, valid), want)
