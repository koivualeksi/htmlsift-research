"""
Board-agnostic arm matrix: the model registry and the axis-crossing that turns
the sweep CLI (--models/--layers/--heads/--caps) into arm dicts. The trainers
(trainers/*.py) import this and walk the arms; each board's data source and val
metric reach the loop through its Benchmark (vendors/<board>/adapter/benchmark.py).
"""

WINDOW = 8192
M311 = "ibm-granite/granite-embedding-311m-multilingual-r2"
M97 = "ibm-granite/granite-embedding-97m-multilingual-r2"
MODELS = {"311m": (M311, 22), "97m": (M97, 12)}
SHORT = {hf: k for k, (hf, _) in MODELS.items()}   # HF path -> short arm key


def parse_layers(spec, full):
    """'full' -> [full]; 'all' -> 1..full; else a comma list of ints/ranges
    (7,11,22 or 1-22), clamped to this model's depth."""
    if spec == "full":
        return [full]
    if spec == "all":
        return list(range(1, full + 1))
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return [L for L in out if 1 <= L <= full]


def build_arms(models, layers, heads, caps, hidden, feats=(None,)):
    """Cross the axes into arm dicts; each names only what differs from the
    reference (bigru, cap 0, no feats). Transformer carries its head-lr exception.
    feats is the structural-feature axis: a group string like "ABC" (arm carries
    feats and gains a -f suffix) or None (text-only)."""
    arms = []
    for m in models:
        hf, full = MODELS[m]
        for n_layers in parse_layers(layers, full):
            for head in heads:
                for cap in caps:
                    for ft in feats:
                        name = m + f"-{n_layers}"
                        if head != "bigru":
                            name += f"-{head}"
                        if cap:
                            name += f"-cap{cap}"
                        if ft:
                            name += f"-f{ft}"
                        arm = {"name": name, "model": hf, "layers": n_layers,
                               "head": head, "cap": cap, "hidden": hidden,
                               "feats": ft}
                        if head == "transformer":
                            arm["head_lr"] = 3e-4
                        arms.append(arm)
    return arms
