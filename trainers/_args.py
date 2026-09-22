"""Shared trainer CLI. base_parser() holds the run-grid and ledger args every trainer
accepts; head_parser() adds the head-training hyperparameters the frozen-head trainers
(frozen, features_only, embedding_table) share. Each trainer builds its parser as
argparse.ArgumentParser(parents=[base_parser(), ...]) and adds only its mode-unique flags
(--layers, --caps, --emb, ...). parse_feats normalizes the shared --feats string.
"""
import argparse


def base_parser():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--benchmark", default="wmb",
                    help="board name; validated against load_benchmark's registry")
    ap.add_argument("--models", default="311m,97m")
    ap.add_argument("--heads", default="bigru")
    ap.add_argument("--feats", default="none",
                    help="feature groups per arm, comma list (e.g. none,ABC)")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--eval-workers", type=int, default=1,
                    help="processes for per-epoch val scoring (Linux fork pool); 1 = serial")
    ap.add_argument("--out", default=None, help="output dir (default data/runs/<benchmark>)")
    ap.add_argument("--limit", type=int, default=None, help="cap pages per fold for a CPU smoke")
    ap.add_argument("--holdout", default=None,
                    help="held-out language for DAnIEL LOLO (Greek/Polish/Russian/English/Chinese)")
    ap.add_argument("--shard", default="",
                    help="push shard tag; distinct per parallel pod (default: unsharded)")
    ap.add_argument("--push", nargs="?", const="", default=None,
                    help="push the results shard to an HF dataset and resume from it; "
                         "bare --push uses $HF_RESULTS_DATASET, --push <repo> overrides")
    return ap


def head_parser():
    """Head-training hyperparameters shared by the frozen-head trainers (frozen,
    features_only, embedding_table); added as a second parent alongside base_parser."""
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--batch-size", type=int, default=8, help="head-training batch size (pages per step)")
    ap.add_argument("--head-lr", type=float, default=2e-3, help="head learning rate")
    return ap


def parse_feats(csv):
    """--feats string -> group list, "none"/"" mapped to None (a text-only arm)."""
    return [None if f in ("", "none") else f for f in csv.split(",")]
