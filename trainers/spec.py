"""Sweep spec: a committed experiments/<name>.yaml that expands into runs. A launcher
only -- each run records its own expanded config in run.yaml, and that is what tables
read. Validation happens at load, so a typo fails on the laptop rather than on a pod.
"""
import itertools
from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

REQUIRED = object()

# Per recipe: the axes a sweep may cross, with the value a run takes when the spec does
# not sweep it (REQUIRED: the spec must). Defaults are written into every run's config,
# so a run.yaml stays complete if a default here later changes.
AXES = {
    "finetune": {"model": REQUIRED, "head": REQUIRED, "feats": None, "layers": "full",
                 "cap": 0, "train_limit": None, "holdout": None},
    "table": {"model": REQUIRED, "head": REQUIRED, "feats": None, "emb": REQUIRED,
              "train_limit": None, "extra_repeat": None},
}
# Per recipe: fixed training keys and their defaults; never crossed.
TRAIN = {
    "finetune": {"epochs": 4, "hidden": 256, "qat_w8a8": False, "quant_eval": False},
    "table": {"epochs": 4, "hidden": 256, "batch_size": 8, "head_lr": 2.0e-3},
}
EXTRAS = ("custom",)


@dataclass(frozen=True)
class Spec:
    name: str
    recipe: str
    board: str
    sweep: dict
    seeds: int
    train: dict = field(default_factory=dict)
    extra: str | None = None
    export_test: bool = False
    keep: bool = False
    note: str = ""

    def __post_init__(self):
        if self.recipe not in AXES:
            raise ValueError(f"unknown recipe {self.recipe!r}; one of {sorted(AXES)}")
        axes = AXES[self.recipe]
        if unknown := set(self.sweep) - set(axes):
            raise ValueError(f"{self.recipe} has no axis {sorted(unknown)}; one of {list(axes)}")
        if missing := [a for a, d in axes.items() if d is REQUIRED and a not in self.sweep]:
            raise ValueError(f"{self.recipe} must sweep {missing}")
        for axis, values in self.sweep.items():
            if not isinstance(values, list) or not values:
                raise ValueError(f"sweep.{axis} must be a non-empty list, got {values!r}")
            if len(set(map(repr, values))) != len(values):
                raise ValueError(f"sweep.{axis} repeats a value: {values}")
        if unknown := set(self.train) - set(TRAIN[self.recipe]):
            raise ValueError(f"{self.recipe} has no train key {sorted(unknown)}; "
                             f"one of {list(TRAIN[self.recipe])}")
        if not isinstance(self.seeds, int) or isinstance(self.seeds, bool) or self.seeds < 1:
            raise ValueError(f"seeds must be a positive int, got {self.seeds!r}")
        if self.extra is not None and self.extra not in EXTRAS:
            raise ValueError(f"unknown extra {self.extra!r}; one of {EXTRAS}")
        if self.extra is None and "extra_repeat" in self.sweep:
            raise ValueError("sweep.extra_repeat needs an extra")
        if self.recipe == "table" and self.export_test and not self.keep:
            raise ValueError("table export_test scores the kept ckpt: set keep")


def load(path):
    """experiments/<name>.yaml -> Spec. name must match the file stem, so a copied spec
    cannot silently write its runs under another experiment."""
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if unknown := set(data) - {f.name for f in fields(Spec)}:
        raise ValueError(f"{path.name}: unknown key {sorted(unknown)}")
    if data.get("name") != path.stem:
        raise ValueError(f"{path.name}: name {data.get('name')!r} must equal the file stem")
    return Spec(**data)


def expand(spec):
    """Spec -> one flat config dict per run: swept axes crossed in YAML order, seeds
    innermost, unswept axes and train keys at their defaults."""
    axes = {a: d for a, d in AXES[spec.recipe].items() if d is not REQUIRED}
    if spec.extra is not None:
        axes["extra_repeat"] = 1
    train = {**TRAIN[spec.recipe], **spec.train}
    configs = []
    for combo in itertools.product(*spec.sweep.values()):
        swept = dict(zip(spec.sweep, combo))
        for seed in range(spec.seeds):
            configs.append({
                "recipe": spec.recipe, "board": spec.board, "extra": spec.extra,
                **{a: swept.get(a, axes.get(a)) for a in AXES[spec.recipe]},
                "seed": seed, **train,
                "export_test": spec.export_test, "keep": spec.keep,
            })
    return configs
