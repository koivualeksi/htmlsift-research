"""Recipes: what one run does. Each module exposes run(cfg, ctx) -> (result, curves),
where cfg is one config from spec.expand and ctx the runtime values that are not config.
"""
from dataclasses import dataclass, field
from pathlib import Path

import torch

from vendors.shared.benchmark import load_benchmark


@dataclass
class Ctx:
    dev: torch.device
    folder: Path            # the run folder; ckpt.pt goes here when keep
    probs: Path             # local prob exports, never pushed
    limit: int | None = None   # pages per fold, smokes only
    eval_workers: int = 1
    cache: dict = field(default_factory=dict)


def cached(ctx, name, key, build):
    """One-slot cache: the value build() gave for key, rebuilt when key changes. Runs of
    one spec arrive with shared preparation adjacent (seeds innermost), so one slot per
    name hits without grouping code, and the old value is dropped before the new build."""
    if name in ctx.cache and ctx.cache[name][0] == key:
        return ctx.cache[name][1]
    ctx.cache.pop(name, None)
    ctx.cache[name] = (key, build())
    return ctx.cache[name][1]


def bench(ctx, cfg):
    return cached(ctx, "bench", (cfg["board"], cfg.get("holdout")),
                  lambda: load_benchmark(cfg["board"], cfg.get("holdout")))
