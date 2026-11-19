"""Unit: sweep-spec loading and expansion. The concern is that a spec either expands into
exactly the runs it names, each with a complete config, or fails at load -- a typo must
never fall back to a default and launch the wrong sweep.
"""
import pytest

from trainers import spec as S

TABLE = """\
name: {name}
recipe: table
board: wmb
extra: custom
sweep:
  model: [97m]
  emb: [lowrank128]
  head: [bigru]
  feats: [AB]
  extra_repeat: [0, 1, 5]
seeds: 3
train: {{epochs: 4, batch_size: 8, head_lr: 2.0e-3}}
export_test: true
keep: true
note: ""
"""


def write(tmp_path, text, name="wmb-mini"):
    p = tmp_path / f"{name}.yaml"
    p.write_text(text.format(name=name), encoding="utf-8")
    return p


def table(**kw):
    base = dict(name="x", recipe="table", board="wmb",
                sweep={"model": ["97m"], "emb": ["fp32"], "head": ["bigru"]}, seeds=1)
    return S.Spec(**{**base, **kw})


def test_load_and_expand(tmp_path):
    runs = S.expand(S.load(write(tmp_path, TABLE)))
    assert len(runs) == 9
    assert [(r["extra_repeat"], r["seed"]) for r in runs] == \
        [(x, s) for x in (0, 1, 5) for s in range(3)]          # seeds innermost
    assert runs[0] == {"recipe": "table", "board": "wmb", "extra": "custom",
                       "model": "97m", "head": "bigru", "feats": "AB", "emb": "lowrank128",
                       "train_limit": None, "extra_repeat": 0, "seed": 0,
                       "epochs": 4, "hidden": 256, "batch_size": 8, "head_lr": 2.0e-3,
                       "export_test": True, "keep": True}


def test_unswept_axes_and_train_take_defaults():
    (run,) = S.expand(S.Spec(name="x", recipe="finetune", board="wmb",
                             sweep={"model": ["97m"], "head": ["bigru"]}, seeds=1))
    assert (run["layers"], run["cap"], run["feats"], run["holdout"]) == ("full", 0, None, None)
    assert (run["epochs"], run["qat_w8a8"]) == (4, False)


def test_extra_repeat_defaults_to_one_only_with_extra():
    assert S.expand(table())[0]["extra_repeat"] is None
    assert S.expand(table(extra="custom"))[0]["extra_repeat"] == 1


def test_name_must_match_stem(tmp_path):
    p = write(tmp_path, TABLE)
    p.rename(tmp_path / "other.yaml")
    with pytest.raises(ValueError, match="file stem"):
        S.load(tmp_path / "other.yaml")


def test_unknown_top_level_key(tmp_path):
    with pytest.raises(ValueError, match="unknown key"):
        S.load(write(tmp_path, TABLE + "sedes: 3\n"))


@pytest.mark.parametrize("kw, msg", [
    (dict(recipe="frozen"), "unknown recipe"),
    (dict(sweep={"model": ["97m"], "emb": ["fp32"], "head": ["bigru"], "models": ["x"]}),
     "no axis"),
    (dict(sweep={"model": ["97m"], "head": ["bigru"]}), "must sweep"),
    (dict(sweep={"model": "97m", "emb": ["fp32"], "head": ["bigru"]}), "non-empty list"),
    (dict(sweep={"model": [], "emb": ["fp32"], "head": ["bigru"]}), "non-empty list"),
    (dict(sweep={"model": ["97m", "97m"], "emb": ["fp32"], "head": ["bigru"]}), "repeats"),
    (dict(train={"lr": 1e-3}), "no train key"),
    (dict(seeds=0), "positive int"),
    (dict(seeds=True), "positive int"),
    (dict(extra="other"), "unknown extra"),
    (dict(sweep={"model": ["97m"], "emb": ["fp32"], "head": ["bigru"],
                 "extra_repeat": [1]}), "needs an extra"),
    (dict(export_test=True), "set keep"),
])
def test_invalid_spec_raises(kw, msg):
    with pytest.raises(ValueError, match=msg):
        table(**kw)
