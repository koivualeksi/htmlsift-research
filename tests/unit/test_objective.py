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


def test_page_norm_is_mean_of_page_means():
    # Each page's weights sum to 1, so a full batch's loss is the mean over pages of each
    # page's mean BCE, whatever the page lengths.
    train_y = [np.array([1, 0, 0, 0], np.float32), np.array([0, 1], np.float32)]
    obj = Objective({"page_norm": True}, train_y, [np.ones(4), np.ones(2)], DEV)
    # weights: page 0 [.25] * 4, page 1 [.5] * 2; pos .75, neg 1.25
    assert np.isclose(float(obj.pos_weight), 1.25 / 0.75)

    logits = torch.tensor([[0.3, -1.2, 0.7, 0.1], [2.0, 0.5, 0.0, 0.0]])
    y = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    valid = torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0]])
    per = F.binary_cross_entropy_with_logits(
        logits, y, pos_weight=obj.pos_weight, reduction="none")
    want = (per[0].mean() + per[1, :2].mean()) / 2
    assert torch.allclose(obj.loss(logits, y, valid, [0, 1]), want)


def test_len_side_weights_one_class():
    train_y = [np.array([1, 0, 1, 0], np.float32)]
    chars = [np.array([20, 20, 400, 400])]
    neg = Objective({"len_cap": 200, "len_floor": 0.2, "len_side": "neg"}, train_y, chars, DEV)
    pos = Objective({"len_cap": 200, "len_floor": 0.2, "len_side": "pos"}, train_y, chars, DEV)
    assert np.allclose(neg.w[0], [1.0, 0.2, 1.0, 1.0])
    assert np.allclose(pos.w[0], [0.2, 1.0, 1.0, 1.0])


def test_fp_weight_upweights_selected_negatives():
    # Only a gold-free block scored above the threshold is multiplied; a selected
    # content block and an unselected negative keep weight 1.
    train_y = [np.array([1, 0, 0], np.float32)]
    obj = Objective({"fp_weight": 4.0}, train_y, [np.ones(3)], DEV)
    logits = torch.tensor([[2.0, 1.5, -1.0]])         # picked, picked, not picked
    y = torch.tensor([[1.0, 0.0, 0.0]])
    valid = torch.ones(1, 3)
    w = torch.tensor([[1.0, 4.0, 1.0]])
    per = F.binary_cross_entropy_with_logits(
        logits, y, pos_weight=obj.pos_weight, reduction="none")
    assert torch.allclose(obj.loss(logits, y, valid, [0]), (per * w).sum() / w.sum())


def test_fp_box_upweights_only_picks_in_big_boxes():
    # With fp_box 5, a picked gold-free block costs fp_weight only inside a gold-free box
    # of at least 5 lines; a picked one in a smaller box (or none) keeps weight 1.
    train_y = [np.array([1, 0, 0, 0], np.float32)]
    box = [np.array([0, 6, 3, 5])]
    obj = Objective({"fp_weight": 4.0, "fp_box": 5}, train_y, [np.ones(4)], DEV, box)
    logits = torch.tensor([[2.0, 1.5, 1.5, -1.0]])    # picked, picked, picked, not picked
    y = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    valid = torch.ones(1, 4)
    w = torch.tensor([[1.0, 4.0, 1.0, 1.0]])
    per = F.binary_cross_entropy_with_logits(
        logits, y, pos_weight=obj.pos_weight, reduction="none")
    assert torch.allclose(obj.loss(logits, y, valid, [0]), (per * w).sum() / w.sum())


def _two_blocks(cfg):
    # One content and one gold-free block, so pos_weight is 1 and drops out of the
    # hand-computed values.
    obj = Objective(cfg, [np.array([1, 0], np.float32)], [np.ones(2)], DEV)
    logits, y = torch.tensor([[2.0, -1.0]]), torch.tensor([[1.0, 0.0]])
    return float(obj.loss(logits, y, torch.ones(1, 2), [0]))


def test_focal_matches_hand_computed():
    # mean of -log(p_c) * (1 - p_c)^2 with p_c = sigmoid(2), 1 - sigmoid(-1)
    assert np.isclose(_two_blocks({"focal_gamma": 2}), 0.0122308102, rtol=1e-6)


def test_focal_downweights_confident_correct_block():
    # A block scored right at p 0.95 keeps under 1 % of its BCE; one scored wrong keeps
    # most of it.
    train_y = [np.array([1, 0], np.float32)]
    plain, focal = (Objective(c, train_y, [np.ones(2)], DEV) for c in ({}, {"focal_gamma": 2}))
    logits, y = torch.tensor([[3.0, 3.0]]), torch.tensor([[1.0, 0.0]])
    ratio = [float(focal.loss(logits, y, v, [0]) / plain.loss(logits, y, v, [0]))
             for v in (torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0]]))]
    assert ratio[0] < 0.01 and ratio[1] > 0.9


def test_label_smooth_matches_hand_computed():
    # eps 0.1: targets 0.95 and 0.05 in the BCE
    assert np.isclose(_two_blocks({"label_smooth": 0.1}), 0.2950948493, rtol=1e-6)


def _extra_term(cfg, train_y, box, logits, valid):
    # The term on top of the default loss: both share pos_weight, as neither cfg moves it.
    y = torch.zeros_like(logits)
    for r, t in enumerate(train_y):
        y[r, :len(t)] = torch.from_numpy(t)
    chars = [np.ones(len(t)) for t in train_y]
    arm, base = (Objective(c, train_y, chars, DEV, box) for c in (cfg, {}))
    idx = list(range(len(train_y)))
    return float(arm.loss(logits, y, valid, idx) - base.loss(logits, y, valid, idx))


def test_smooth_weight_matches_hand_computed():
    # Pairs counted: page 0 (0, 1) and (2, 3), page 1 (0, 1). Page 0's (1, 2) straddles a
    # gold boundary and page 1's later pairs touch padding, so neither counts.
    train_y = [np.array([1, 1, 0, 0], np.float32), np.zeros(2, np.float32)]
    logits = torch.tensor([[2.0, 0.0, -1.0, 1.0], [0.5, -0.5, 3.0, -3.0]])
    valid = torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0]])
    term = _extra_term({"smooth_weight": 2.0}, train_y, None, logits, valid)
    assert np.isclose(term, 2 * 0.1395146109, rtol=1e-5)


def test_box_mass_matches_hand_computed():
    # fp_box 3: boxes are page 0 rows 1-3 (size 3) and 4-7 (size 4, a different box next
    # door), and page 1 rows 0-2. The content row, the size-2 box and padding do not count.
    train_y = [np.r_[1, np.zeros(9)].astype(np.float32), np.zeros(3, np.float32)]
    box = [np.array([0, 3, 3, 3, 4, 4, 4, 4, 2, 2]), np.array([3, 3, 3])]
    logits = torch.tensor([[2.0, -1.0, 0.0, 1.0, -2.0, -1.0, 0.0, 1.0, 3.0, 3.0],
                           [0.5, -1.5, -0.5, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]])
    valid = torch.ones(2, 10)
    valid[1, 3:] = 0
    term = _extra_term({"box_mass": 1.0, "fp_box": 3}, train_y, box, logits, valid)
    assert np.isclose(term, 0.5710367515, rtol=1e-5)


def _drop_pages(cfg):
    # Rows are [text | ABC feats], as table.run concatenates them.
    rng = np.random.default_rng(0)
    xs = [rng.standard_normal((n, 8 + 33)).astype(np.float32) + 1 for n in (5, 3)]
    obj = Objective({"feats": "ABC", **cfg}, [np.eye(len(x), dtype=np.float32)[0] for x in xs],
                    [np.ones(len(x)) for x in xs], DEV)
    return xs, obj.augment(xs)


def test_default_augment_leaves_pages_untouched():
    xs, out = _drop_pages({})
    assert all(o is x for o, x in zip(out, xs))


def test_text_drop_zeroes_text_keeps_feats():
    xs, out = _drop_pages({"text_drop": 1.0})
    for o, x in zip(out, xs):
        assert not o[:, :8].any() and np.array_equal(o[:, 8:], x[:, 8:])
        assert x[:, :8].all()                         # the input page is not mutated


def test_feat_drop_zeroes_feats_keeps_text():
    xs, out = _drop_pages({"feat_drop": 1.0})
    for o, x in zip(out, xs):
        assert not o[:, 8:].any() and np.array_equal(o[:, :8], x[:, :8])
