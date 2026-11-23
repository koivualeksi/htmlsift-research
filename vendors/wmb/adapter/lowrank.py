"""Derive rank-r keepers from a trained table arm: fit C (the second moment of pooled int8
table block vectors) on WMB train only, factor each seed's head through it
(core.quant.factor_keeper), write <arm>-r<rank>_s<seed>.pt under data/lowrank/. No training.

C depends on the int8 table and the train renders, not on the head, so one C serves every
seed and every arm; it is cached and refused if the htmlsift version changed. Run on the
WSL/pod render (libxml2-version-specific), never a Windows one.

    python -m vendors.wmb.adapter.lowrank 311m-table-bigru-fABC-nb1-pf4-ema0.999 --rank 128
"""
import argparse
from pathlib import Path

import htmlsift
import numpy as np
import torch
from transformers import AutoTokenizer

from core.loader import resolve_ckpt
from core.model import build_table_embedder
from core.quant import GRU_IH, factor_keeper, second_moment
from vendors.shared.arms import WINDOW
from vendors.shared.benchmark import load_benchmark

OUT = Path(__file__).resolve().parents[1] / "data" / "lowrank"


def fit_or_load_C(model, refit):
    path = OUT / f"C_{model.replace('/', '_')}_int8_train.npz"
    if path.exists() and not refit:
        z = np.load(path)
        if str(z["htmlsift"]) != htmlsift.__version__:
            raise SystemExit(f"{path} was fit under htmlsift {z['htmlsift']}, "
                             f"this is {htmlsift.__version__}; --refit")
        print(f"C: cached {path.name}, {int(z['n'])} blocks", flush=True)
        return z["C"]
    tok = AutoTokenizer.from_pretrained(model)
    pages = load_benchmark("wmb").build_pages(tok, "train", WINDOW)
    C, n = second_moment(build_table_embedder(model, int8=True), pages)
    np.savez(path, C=C, n=n, htmlsift=htmlsift.__version__)
    print(f"C: fit on {n} train blocks ({len(pages)} pages) -> {path.name}", flush=True)
    return C


def weighted_error(ck, lk, C):
    """||(W_e - A B) C^1/2||_F / ||W_e C^1/2||_F: the share of the head's input the rank
    drops, measured where block vectors occur."""
    H, r = C.shape[0], lk["rank"]
    W = np.concatenate([ck["head_state"][k][:, :H].double().numpy() for k in GRU_IH])
    A = np.concatenate([lk["head_state"][k][:, :r].double().numpy() for k in GRU_IH])
    D = W - A @ lk["proj"].double().numpy()
    return float(np.sqrt(np.trace(D @ C @ D.T) / np.trace(W @ C @ W.T)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", help="table keeper name in the models repo, wmb/ckpt/<arm>_s<seed>.pt")
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--refit", action="store_true", help="fit C even if cached")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    cks = {s: resolve_ckpt(f"wmb/ckpt/{a.arm}_s{s}.pt", torch.device("cpu"))
           for s in map(int, a.seeds.split(","))}
    models = {ck["model"] for ck in cks.values()}
    if len(models) != 1 or any(ck.get("kind") != "table" for ck in cks.values()):
        raise SystemExit(f"{a.arm}: seeds must be table keepers on one model, got {models}")
    C = fit_or_load_C(models.pop(), a.refit)
    for s, ck in cks.items():
        lk = factor_keeper(ck, C, a.rank)
        path = OUT / f"{a.arm}-r{a.rank}_s{s}.pt"
        torch.save(lk, path)
        print(f"s{s}: weighted error {weighted_error(ck, lk, C):.4f} -> {path.name}", flush=True)


if __name__ == "__main__":
    main()
