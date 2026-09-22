"""XGBoost per-block head -- the "why not just XGBoost" baseline.

A gradient-boosted tree over per-block rows: the off-the-shelf tabular default a
practitioner reaches for first. It is here to be measured, not tuned -- stock
params, identical for every arm -- so the comparison answers "why not JUST
XGBoost" and not "why not a tuned XGBoost". A tree has no recurrence, so
page_matrix hands each block its own features plus its neighbours', the page
mean and its position: the cross-block context a sequence head gets for free,
without which the tree would lose for lack of context rather than for being a
tree. Features are passed raw (trees are scale-invariant, so the z-scoring the
torch heads need is skipped); the pooled embedding, when present, is the block's
own. Output is the same {track_id: probs} the torch heads export, so the board's
--probs eval scores it unchanged.
"""
import numpy as np
import xgboost as xgb

# A practitioner's five-minute default, deliberately untuned and identical for
# every arm (see module docstring). scale_pos_weight is per-fit, below.
PARAMS = dict(n_estimators=400, max_depth=6, learning_rate=0.1, subsample=0.8,
              colsample_bytree=0.8, tree_method="hist", eval_metric="logloss")


def page_matrix(feats, emb=None):
    """[n, K] raw block features (+ optional [n, H] pooled embedding) -> [n, D].
    Each row carries its own features, the previous and next block's (clamped at
    the page edges), the page mean, three position columns (normalised index,
    page length, raw index) and, when given, the block's own embedding."""
    n = feats.shape[0]
    prev = np.vstack([feats[:1], feats[:-1]])
    nxt = np.vstack([feats[1:], feats[-1:]])
    pmean = np.tile(feats.mean(0), (n, 1))
    idx = np.arange(n, dtype=np.float32)
    pos = np.stack([idx / max(n - 1, 1), np.full(n, n, np.float32), idx], axis=1)
    parts = [feats, prev, nxt, pmean, pos]
    if emb is not None:
        parts.append(emb)
    return np.concatenate(parts, axis=1).astype(np.float32)


def _stack(pages, use_emb):
    """Pages -> (X [total_blocks, D], y [total_blocks] or None, page spans). A
    page is (track_id, feats [n, K], emb [n, H] or None, labels [n] or None); y
    is None when any page carries no labels (eval-only)."""
    mats, labels, spans, off = [], [], [], 0
    have_y = all(y is not None for _, _, _, y in pages)
    for tid, feats, emb, y in pages:
        X = page_matrix(feats, emb if use_emb else None)
        mats.append(X)
        if have_y:
            labels.append(y)
        spans.append((tid, off, off + len(X)))
        off += len(X)
    Y = np.concatenate(labels) if have_y else None
    return np.concatenate(mats, axis=0), Y, spans


def fit_predict(train_pages, eval_pages, use_emb, seed=0):
    """Fit a stock XGBClassifier on the train blocks, return {track_id: probs}
    (per-block keep probability) for the eval pages. scale_pos_weight balances
    the minority positive label from the train fold's own rate."""
    Xtr, ytr, _ = _stack(train_pages, use_emb)
    spw = (len(ytr) - ytr.sum()) / max(ytr.sum(), 1.0)
    clf = xgb.XGBClassifier(scale_pos_weight=spw, random_state=seed, **PARAMS)
    clf.fit(Xtr, ytr.astype(int))
    Xva, _, spans = _stack(eval_pages, use_emb)
    p = clf.predict_proba(Xva)[:, 1]
    return {tid: p[lo:hi] for tid, lo, hi in spans}


def fit_xgb_head(raw, train_pages, val_pages, etr, eva, train_y, score_val, seed, use_emb=True):
    """Fit the xgb head on the cached embeddings + raw features (raw: {track_id: [n, F]},
    or None for text-only), score val. The shared xgb path for the frozen family; one
    fit, no epochs, so it returns {val, epoch: 0} beside the torch heads' result."""
    def feats(tid, n):
        return raw[tid] if raw is not None else np.zeros((n, 0), np.float32)
    tr = [(p["tid"], feats(p["tid"], len(y)), e, y) for p, e, y in zip(train_pages, etr, train_y)]
    va = [(p["tid"], feats(p["tid"], len(p["y"])), e, None) for p, e in zip(val_pages, eva)]
    pr = fit_predict(tr, va, use_emb=use_emb, seed=seed)
    v = score_val(val_pages, [pr[p["tid"]] for p in val_pages])
    print(f"  val {v:.4f}", flush=True)
    return {"val": v, "epoch": 0}
