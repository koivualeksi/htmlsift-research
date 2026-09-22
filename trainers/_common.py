"""Shared trainer scaffolding: the run-constant setup every mode trainer repeats, plus
the base Ctx it hangs on. Each trainer subclasses Ctx for its mode-specific state and
builds it with e.g. FtCtx(**setup(args, tag), feats_z=..., keep=...) -- so adding a
common field later touches only this file, never a trainer.
"""
import argparse
from dataclasses import dataclass
from pathlib import Path

import torch

from trainers import ledger
from vendors.shared.benchmark import load_benchmark

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Ctx:
    """The run-constant values every trainer shares -- a plain data bag (no logic). Mode
    trainers subclass this to add their own state (feats_z, keep, raw_f, ...)."""
    bench: object
    dev: torch.device
    out_dir: Path
    args: argparse.Namespace
    led: ledger.Ledger


def setup(args, tag):
    """The preamble every trainer runs: pick the benchmark, resolve the device, make the
    run dir, open the ledger (resolve_repo + shard) and resume its done-set. tag is the
    result-file stem (finetune_results / frozen_results / ...). Returns the base-Ctx
    fields as kwargs for the trainer's own Ctx subclass."""
    bench = load_benchmark(args.benchmark, getattr(args, "holdout", None))
    dev = torch.device(args.device)
    out_dir = Path(args.out) if args.out else ROOT / "data" / "runs" / bench.name
    out_dir.mkdir(parents=True, exist_ok=True)
    led = ledger.Ledger(out_dir / f"{tag}.jsonl", ledger.resolve_repo(args.push),
                        ledger.sharded_path(f"{bench.name}/{tag}.jsonl", args.shard),
                        ledger.resolve_models_repo(args.push))
    led.resume()
    return dict(bench=bench, dev=dev, out_dir=out_dir, args=args, led=led)
