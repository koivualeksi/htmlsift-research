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
model: 97m
sweep:
  extra_repeat: [0, 1, 5]
seeds: 3
note: ""
"""


def write(tmp_path, text, name="wmb-mini"):
    p = tmp_path / f"{name}.yaml"
    p.write_text(text.format(name=name), encoding="utf-8")
    return p


def table(**kw):
    return S.Spec(**{**dict(name="x", recipe="table", board="wmb", seeds=1), **kw})


def test_load_and_expand(tmp_path):
    runs = S.expand(S.load(write(tmp_path, TABLE)))
    assert len(runs) == 9
    assert [(r["extra_repeat"], r["seed"]) for r in runs] == \
        [(x, s) for x in (0, 1, 5) for s in range(3)]          # seeds innermost
    assert runs[0] == {"recipe": "table", "board": "wmb",
                       "model": "97m", "emb": "fp32", "head": "bigru", "feats": "ABC",
                       "hidden": 256, "epochs": 4, "batch_size": 8, "head_lr": 2.0e-3,
                       "train_limit": None, "extra": "custom", "extra_repeat": 0,
                       "len_cap": None, "len_floor": None, "len_side": "both",
                       "page_norm": False, "fp_weight": 1.0, "fp_box": None,
                       "focal_gamma": 0, "label_smooth": 0.0, "text_drop": 0.0, "feat_drop": 0.0, "plain": False,
                       "export_test": True, "keep": True, "seed": 0}


@pytest.mark.parametrize("recipe", ["finetune", "table"])
def test_empty_spec_is_the_shipped_config(recipe):
    runs = S.expand(S.Spec(name="x", recipe=recipe, board="wmb"))
    assert [r["seed"] for r in runs] == [0, 1, 2]
    assert {k: v for k, v in runs[0].items() if k not in ("recipe", "board", "seed")} \
        == S.DEFAULTS[recipe]


def test_shipped_defaults():
    ft, tb = S.DEFAULTS["finetune"], S.DEFAULTS["table"]
    assert (ft["model"], ft["layers"], ft["head"], ft["feats"]) == ("311m", 10, "bigru", None)
    assert (tb["model"], tb["emb"], tb["head"], tb["feats"]) == ("311m", "fp32", "bigru", "ABC")


def test_sweep_crosses_in_yaml_order():
    runs = S.expand(table(sweep={"model": ["97m", "311m"], "feats": [None, "AB"]}))
    assert [(r["model"], r["feats"]) for r in runs] == \
        [("97m", None), ("97m", "AB"), ("311m", None), ("311m", "AB")]


def test_extra_repeat_defaults_to_one_only_with_extra():
    assert S.expand(table())[0]["extra_repeat"] is None
    assert S.expand(table(fixed={"extra": "custom"}))[0]["extra_repeat"] == 1


def test_name_must_match_stem(tmp_path):
    p = write(tmp_path, TABLE)
    p.rename(tmp_path / "other.yaml")
    with pytest.raises(ValueError, match="file stem"):
        S.load(tmp_path / "other.yaml")


def test_unknown_top_level_key(tmp_path):
    with pytest.raises(ValueError, match="has no key"):
        S.load(write(tmp_path, TABLE + "sedes: 3\n"))


@pytest.mark.parametrize("kw, msg", [
    (dict(recipe="frozen"), "unknown recipe"),
    (dict(fixed={"layers": 10}), "has no key"),                 # finetune-only key
    (dict(sweep={"models": ["x"]}), "has no sweep key"),
    (dict(fixed={"model": "97m"}, sweep={"model": ["311m"]}), "both fixed and swept"),
    (dict(sweep={"model": "97m"}), "non-empty list"),
    (dict(sweep={"model": []}), "non-empty list"),
    (dict(sweep={"model": ["97m", "97m"]}), "repeats"),
    (dict(seeds=0), "positive int"),
    (dict(seeds=True), "positive int"),
    (dict(fixed={"extra": "other"}), "unknown extra"),
    (dict(sweep={"extra_repeat": [1]}), "needs an extra"),
    (dict(fixed={"keep": False}), "set keep"),
    (dict(sweep={"keep": [True, False]}), "set keep"),        # one swept value is enough
    (dict(fixed={"emb": "lowrank128"}), "unknown emb"),
    (dict(fixed={"extra": "custom", "train_limit": 250}), "extra with train_limit"),
    (dict(fixed={"len_cap": 200}), "len_floor in"),
    (dict(fixed={"len_cap": 0, "len_floor": 0.2}), "len_floor in"),
    (dict(fixed={"len_cap": 200, "len_floor": 1.5}), "len_floor in"),
    (dict(fixed={"page_norm": "yes"}), "page_norm must be"),
    (dict(fixed={"len_side": "neg"}), "needs len_cap"),
    (dict(fixed={"len_cap": 200, "len_floor": 0.2, "len_side": "gold"}), "len_side must"),
    (dict(fixed={"fp_weight": 0}), "fp_weight must"),
    (dict(fixed={"fp_box": 5}), "needs fp_weight"),
    (dict(fixed={"fp_weight": 2.0, "fp_box": 0}), "fp_box must"),
    (dict(fixed={"fp_weight": 2.0, "fp_box": 5, "extra": "custom"}), "only WMB"),
    (dict(fixed={"focal_gamma": -1}), "focal_gamma must"),
    (dict(fixed={"label_smooth": 0.5}), "label_smooth must"),
    (dict(fixed={"label_smooth": -0.1}), "label_smooth must"),
    (dict(fixed={"text_drop": 0.6, "feat_drop": 0.6}), "sum <= 1"),
    (dict(fixed={"feat_drop": -0.1}), "sum <= 1"),
    (dict(fixed={"text_drop": 0.3, "feats": None}), "need feats"),
    (dict(fixed={"plain": 1}), "plain must be"),
    (dict(fixed={"plain": True, "extra": "custom"}), "only WMB"),
    (dict(board="wcxb", fixed={"plain": True}), "only WMB"),
])
def test_invalid_spec_raises(kw, msg):
    with pytest.raises(ValueError, match=msg):
        table(**kw)


def test_quant_eval_with_qat_raises():
    with pytest.raises(ValueError, match="incompatible with qat_w8a8"):
        S.Spec(name="x", recipe="finetune", board="wmb",
               fixed={"quant_eval": True}, sweep={"qat_w8a8": [False, True]})


def test_committed_specs_load():
    from pathlib import Path
    paths = sorted(Path(__file__).resolve().parents[2].glob("experiments/*.yaml"))
    assert paths
    for p in paths:
        S.expand(S.load(p))


def test_finetune_rejects_table_key():
    with pytest.raises(ValueError, match="has no key"):
        S.Spec(name="x", recipe="finetune", board="wmb", fixed={"emb": "fp32"})


def test_run_yaml_from_before_a_key_passes_check():
    # A run.yaml written before len_cap etc. existed ran with them off; re-running it
    # from that yaml must not fail on the missing keys.
    old = {k: v for k, v in S.expand(table())[0].items()
           if k not in ("len_cap", "len_floor", "len_side", "page_norm", "fp_weight",
                        "fp_box", "focal_gamma", "label_smooth", "text_drop", "feat_drop",
                        "plain")}
    S.check(old)
