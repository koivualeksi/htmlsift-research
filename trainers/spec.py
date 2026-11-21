"""Sweep spec: a committed experiments/<name>.yaml that expands into runs. A launcher
only -- each run records its own expanded config in run.yaml, and that is what tables
read. Validation happens at load, so a typo fails on the laptop rather than on a pod.

A spec names only what differs from its recipe's DEFAULTS: a key at the top level is
fixed for every run, a key under sweep: is a list crossed into runs.
"""
import itertools
from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

# Per recipe: every config key a run has, at the shipped config (tools/release.py:
# finetune = base 311m-10, table = mini 311m-table-bigru-fABC). Written in full into every
# run's config, so a run.yaml stays complete if a default here later changes.
DEFAULTS = {
    "finetune": {"model": "311m", "layers": 10, "head": "bigru", "feats": None, "cap": 0,
                 "hidden": 256, "epochs": 4, "train_limit": None, "holdout": None,
                 "qat_w8a8": False, "quant_eval": False,
                 "export_test": True, "keep": False},
    "table": {"model": "311m", "emb": "fp32", "head": "bigru", "feats": "ABC",
              "hidden": 256, "epochs": 4, "batch_size": 8, "head_lr": 2.0e-3,
              "train_limit": None, "extra": None, "extra_repeat": None,
              "len_cap": None, "len_floor": None, "page_norm": False,
              "export_test": True, "keep": True},
}
EXTRAS = ("custom",)
EMBS = ("fp32", "int8")


@dataclass(frozen=True)
class Spec:
    name: str
    recipe: str
    board: str
    seeds: int = 3
    sweep: dict = field(default_factory=dict)
    fixed: dict = field(default_factory=dict)   # top-level config keys of the yaml
    note: str = ""

    def __post_init__(self):
        if self.recipe not in DEFAULTS:
            raise ValueError(f"unknown recipe {self.recipe!r}; one of {sorted(DEFAULTS)}")
        keys = DEFAULTS[self.recipe]
        for where, d in (("key", self.fixed), ("sweep key", self.sweep)):
            if unknown := set(d) - set(keys):
                raise ValueError(f"{self.recipe} has no {where} {sorted(unknown)}; "
                                 f"one of {list(keys)}")
        if both := set(self.fixed) & set(self.sweep):
            raise ValueError(f"{sorted(both)} both fixed and swept")
        for k, values in self.sweep.items():
            if not isinstance(values, list) or not values:
                raise ValueError(f"sweep.{k} must be a non-empty list, got {values!r}")
            if len(set(map(repr, values))) != len(values):
                raise ValueError(f"sweep.{k} repeats a value: {values}")
        if not isinstance(self.seeds, int) or isinstance(self.seeds, bool) or self.seeds < 1:
            raise ValueError(f"seeds must be a positive int, got {self.seeds!r}")
        expand(self)                              # runs check() on every config


def check(cfg):
    """Rules on one expanded config (or a run.yaml's), so a bad combination of swept
    values fails at load, before any compute."""
    if cfg.get("extra") is not None and cfg["extra"] not in EXTRAS:
        raise ValueError(f"unknown extra {cfg['extra']!r}; one of {EXTRAS}")
    if cfg.get("extra") is None and cfg.get("extra_repeat") is not None:
        raise ValueError("extra_repeat needs an extra")
    if cfg.get("extra") is not None and cfg["train_limit"] is not None:
        raise ValueError("extra with train_limit is unsupported (the z-score refit "
                         "would have to decide whether extra pages count)")
    if cfg.get("quant_eval") and cfg.get("qat_w8a8"):
        raise ValueError("quant_eval scores int8 variants against the fp32 model; "
                         "incompatible with qat_w8a8")
    if cfg["recipe"] == "table":
        if cfg["emb"] not in EMBS:
            raise ValueError(f"unknown emb {cfg['emb']!r}; one of {EMBS}")
        if cfg["export_test"] and not cfg["keep"]:
            raise ValueError("table export_test scores the kept ckpt: set keep")
        cap, floor = cfg["len_cap"], cfg["len_floor"]
        if cap is not None and not (isinstance(cap, int) and cap >= 1
                                    and isinstance(floor, (int, float)) and 0 < floor <= 1):
            raise ValueError(f"len_cap must be a positive int and len_floor in (0, 1], "
                             f"got {cap!r}, {floor!r}")
        if not isinstance(cfg["page_norm"], bool):
            raise ValueError(f"page_norm must be true or false, got {cfg['page_norm']!r}")


def load(path):
    """experiments/<name>.yaml -> Spec. name must match the file stem, so a copied spec
    cannot silently write its runs under another experiment."""
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data.get("name") != path.stem:
        raise ValueError(f"{path.name}: name {data.get('name')!r} must equal the file stem")
    meta = {f.name for f in fields(Spec)} - {"fixed"}
    return Spec(**{k: v for k, v in data.items() if k in meta},
                fixed={k: v for k, v in data.items() if k not in meta})


def expand(spec):
    """Spec -> one flat config dict per run: swept keys crossed in YAML order, seeds
    innermost, everything else at its fixed value or the recipe default."""
    base = {**DEFAULTS[spec.recipe], **spec.fixed}
    configs = []
    for combo in itertools.product(*spec.sweep.values()):
        cfg = {**base, **dict(zip(spec.sweep, combo))}
        if cfg.get("extra") is not None and cfg["extra_repeat"] is None:
            cfg["extra_repeat"] = 1
        cfg = {"recipe": spec.recipe, "board": spec.board, **cfg}
        check(cfg)
        configs += [{**cfg, "seed": seed} for seed in range(spec.seeds)]
    return configs
