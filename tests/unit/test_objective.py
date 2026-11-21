"""With no method active, Objective.loss must be bit-identical to the masked BCE the
table recipe trained every shipped mini with, so moving the loss out of the loop moves
no number. With len_cap set, the loss is the length-weighted mean it claims to be."""
import numpy as np
import torch
import torch.nn.functional as F

from trainers.objective import Objective, len_weight

DEV = torch.device("cpu")


def test_default_loss_matches_masked_bce():
    rng = np.random.default_rng(0)
    train_y = [(rng.random(n) > 0.6).astype(np.float32) for n in (7, 13, 4)]
    obj = Objective({}, train_y, [np.ones(len(y)) for y in train_y], DEV)

    pos = sum(float(y.sum()) for y in train_y)
    tot = sum(len(y) for y in train_y)
    assert float(obj.pos_weight) == np.float32((tot - pos) / pos)

    logits = torch.from_numpy(rng.standard_normal((3, 13)).astype(np.float32))
    y = torch.from_numpy((rng.random((3, 13)) > 0.6).astype(np.float32))
    mask = torch.zeros(3, 13, dtype=torch.bool)
    mask[0, 7:] = mask[2, 4:] = True                  # padding
    valid = (~mask).float()
    per = F.binary_cross_entropy_with_logits(
        logits, y, pos_weight=obj.pos_weight, reduction="none")
    want = (per * valid).sum() / valid.sum()
    assert torch.equal(obj.loss(logits, y, valid, [0, 1, 2]), want)


def test_len_weight_ramp():
    w = len_weight([0, 20, 40, 100, 200, 5000], cap=200, floor=0.2)
    assert np.allclose(w, [0.2, 0.2, 0.2, 0.5, 1.0, 1.0])


def test_len_weighted_loss():
    # pos_weight balances the classes by weight; each batch row takes the weights of
    # the train page idx names.
    train_y = [np.array([1, 0, 0], np.float32), np.array([0, 1], np.float32)]
    chars = [np.array([400, 10, 100]), np.array([60, 200])]
    obj = Objective({"len_cap": 200, "len_floor": 0.2}, train_y, chars, DEV)
    # weights: page 0 [1.0, 0.2, 0.5], page 1 [0.3, 1.0]; pos 2.0, neg 1.0
    assert np.isclose(float(obj.pos_weight), 0.5)

    logits = torch.tensor([[0.3, -1.2, 0.0], [2.0, 0.5, 0.0]])
    y = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    valid = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    w = torch.tensor([[0.3, 1.0, 0.0], [1.0, 0.2, 0.5]])
    per = F.binary_cross_entropy_with_logits(
        logits, y, pos_weight=obj.pos_weight, reduction="none")
    assert torch.allclose(obj.loss(logits, y, valid, [1, 0]), (per * w).sum() / w.sum())
