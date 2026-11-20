"""Unit: the results reader's cell identity. A cell separates the arm, train_limit,
holdout and the epoch budget. A cell that merges any of those averages two populations
and reports the wrong one -- the failure that put a train_limit=4000 run in the
full-pool table.
No HF transport: render() is pure over authored records, and the legacy loader is read
from a run folder written here.
"""
import json

import pytest
import yaml

from tools.render_tables import (_epochs, collapse, from_experiment, from_ledgers, group_key,
                                 record, render, render_matrix, render_runs)

FROZEN = {"model": "311m", "layers": 12, "head": "bigru", "feats": "ABC",
          "seed": 0, "ts": "2026-01-01", "best_val": 0.89, "frozen": True}


def rec(seed=0, ts="2026-01-01", epochs=8, **extra):
    return {"model": "311m", "layers": 10, "head": "bigru", "feats": None, "cap": 0,
            "seed": seed, "ts": ts, "best_val": 0.9,
            "val_by_epoch": [0.9] * epochs, **extra}


def exported(seed=0, f1=0.93, test_n=544, **extra):
    return rec(seed=seed, test_f1=f1, test_prec=0.93, test_rec=0.96, test_n=test_n, **extra)


# ---- group_key separates the arm, train_limit, holdout and the epoch budget ----

def test_arm_fields_and_seed_are_the_key_head():
    assert group_key(rec(seed=2))[:6] == ("311m", 10, "bigru", "none", 0, 2)


def test_different_holdouts_are_distinct_cells():
    assert len({group_key(rec(holdout=h)) for h in ("Greek", "Polish", "Chinese")}) == 3

def test_train_limit_is_a_distinct_cell():
    assert group_key(rec()) != group_key(rec(train_limit=2000))


def test_different_train_limits_are_distinct_cells():
    assert len({group_key(rec(train_limit=n)) for n in (125, 250, 500)}) == 3


def test_holdout_is_a_distinct_cell():
    assert group_key(rec()) != group_key(rec(holdout="Chinese"))


def test_epoch_budget_is_a_distinct_cell():
    assert group_key(rec(epochs=4)) != group_key(rec(epochs=8))


def test_record_without_curves_keys_without_raising():
    # frozen / table / feature records carry best_val only -- no curves, no cap
    assert _epochs(FROZEN) is None
    assert group_key(FROZEN)[-1] is None


# ---- collapse ----

def test_latest_ts_wins_within_a_cell():
    groups = collapse([rec(ts="2026-01-01", best_val=0.1),
                       rec(ts="2026-02-01", best_val=0.9)])
    assert [r["best_val"] for g in groups for r in g] == [0.9]


def test_seeds_pool_into_one_cell():
    groups = collapse([rec(seed=s) for s in (0, 1, 2)])
    assert len(groups) == 1 and len(groups[0]) == 3


def test_ladder_does_not_overwrite_the_full_pool():
    # the live failure: a newer train-limit run collapsing onto the full-pool arm
    groups = collapse([rec(ts="2026-01-01", best_val=0.90),
                       rec(ts="2026-09-13", best_val=0.88, train_limit=2000)])
    assert sorted(r["best_val"] for g in groups for r in g) == [0.88, 0.90]


# ---- render ----

def test_ladder_and_holdout_render_as_separate_tables():
    out = render([rec(best_val=0.90),
                  rec(best_val=0.88, train_limit=2000),
                  rec(best_val=0.85, holdout="Chinese")], "cap")
    assert "## Training-set size" in out and "## Held-out language" in out
    assert out.index("## Training-set size") < out.index("## Held-out language")


def test_epoch_column_appears_only_when_curves_exist():
    assert "| layers | ep |" in render([rec(epochs=4), rec(epochs=8)], "cap")
    out = render([FROZEN], "cap")
    assert "| layers |" in out and " ep " not in out


def test_seeds_aggregate_but_populations_do_not():
    out = render([rec(seed=s, best_val=v) for s, v in ((0, 0.90), (1, 0.92))]
                 + [rec(seed=0, best_val=0.10, train_limit=125)], "cap")
    assert "0.9100±0.0141" in out          # the two seeds pooled
    assert "0.1000" in out                 # the ladder rung, alone, in its own table


# ---- the two bases ----

def test_both_bases_render_with_their_own_headings():
    out = render([rec(best_val=0.95, val_by_epoch=[0.80, 0.90])], "cap")
    assert "## val (epoch-mean)" in out and "## val (best-epoch)" in out
    assert out.index("epoch-mean") < out.index("best-epoch")   # §6 basis first
    assert "0.8500" in out and "0.9500" in out                 # mean of curve, best epoch


def test_curveless_ledger_renders_one_basis_and_no_heading():
    out = render([FROZEN], "cap")
    assert "val (" not in out and "0.8900" in out


def test_basis_heading_is_combined_with_the_section_title():
    out = render([rec(train_limit=125)], "cap")
    assert "## Training-set size -- val (epoch-mean)" in out


# ---- the test fold ----

def test_test_table_lists_only_exported_arms():
    out = render([exported(seed=0, f1=0.93), exported(seed=1, f1=0.95),
                  rec(seed=2, layers=22)], "cap")           # 22 never exported
    assert ("| 311m | 10 | 8 | bigru/none | 2 | 544 | 0.9400±0.0141 | 0.9300±0.0000 "
            "| 0.9600±0.0000 |") in out
    assert "| 311m | 22 |" not in out.split("## test")[1]


def test_test_seed_count_may_be_below_the_val_seed_count():
    # three val seeds, one exported -- the test cell must not imply three
    out = render([rec(seed=0), rec(seed=1), exported(seed=2, f1=0.9356)], "cap")
    assert "| 311m | 10 | 8 | bigru/none | 1 | 544 | 0.9356 | 0.9300 | 0.9600 |" in out


def test_no_test_table_without_exported_records():
    assert "## test" not in render([rec()], "cap")


def test_test_population_varies_by_row_not_by_ledger():
    # DAnIEL LOLO: the test fold is the held-out language, so n differs row to row
    out = render([exported(seed=0, holdout="Greek", test_n=273),
                  exported(seed=0, holdout="English", test_n=475)], "cap")
    assert "| Greek | 311m | 10 | 8 | bigru/none | 1 | 273 |" in out
    assert "| English | 311m | 10 | 8 | bigru/none | 1 | 475 |" in out


def test_holdout_section_renders_its_own_test_table():
    # LOLO's whole result is its test fold -- a full-pool-only test table would drop it
    out = render([exported(seed=0, holdout="Greek", test_n=273)], "cap")
    assert "## Held-out language -- test" in out
    assert "| holdout | model | layers |" in out


# ---- the generalization matrix ----

def gen(model="wmb-311m-10", board="wmb", seed=0, n=544, f1=0.93, ts="2026-01-01"):
    return {"model": model, "trained_on": model.split("-")[0], "eval_board": board,
            "seed": seed, "n": n, "prec": 0.93, "rec": 0.96, "f1": f1, "ts": ts}


def test_matrix_bolds_in_domain_from_the_recorded_trained_on():
    out = render_matrix([gen(board="wmb", f1=0.93), gen(board="wcxb", n=511, f1=0.86)], "cap")
    assert "**0.9300**" in out and "| 0.8600 |" in out


def test_matrix_raises_on_a_smoke_run():
    # a 20-page laptop smoke would otherwise overwrite the 544-page pod run on ts alone
    with pytest.raises(ValueError, match="smoke records"):
        render_matrix([gen(ts="2026-01-01"), gen(ts="2026-09-13", n=20, f1=0.5)], "cap")


def test_matrix_population_is_per_board():
    # wcxb's 511 is full for wcxb, not a short wmb run
    out = render_matrix([gen(board="wmb", n=544), gen(board="wcxb", n=511)], "cap")
    assert "0.9300" in out


# ---- the run loader ----

LEGACY = {"repo": "htmlsift-runs", "file": "wmb/finetune_results.jsonl", "line": 4,
          "ts": "2026-09-06T14:18:46+00:00"}


def folder(tmp_path, curves=None, legacy=LEGACY, experiment="legacy-wmb-finetune",
           ended="2026-09-06T17:18:46+03:00"):
    run = {"id": "2026-09-06T1718-f3ff", "experiment": experiment, "ended": ended,
           "config": {"recipe": "finetune", "board": "wmb", "model": "311m", "layers": 10,
                      "head": "bigru", "feats": None, "cap": 0, "seed": 0},
           "result": {"best_val": 0.9, "best_epoch": 2, "ckpt": "ckpt.pt"}}
    if legacy:
        run["legacy"] = legacy
    (tmp_path / "run.yaml").write_text(yaml.safe_dump(run), encoding="utf-8")
    if curves:
        (tmp_path / "curves.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in curves), encoding="utf-8")
    return tmp_path


def test_a_converted_run_rebuilds_its_ledger_line(tmp_path):
    order, _, rec = record(folder(tmp_path, curves=[{"epoch": 1, "val": 0.8, "loss": 0.5},
                                                    {"epoch": 2, "val": 0.9, "loss": 0.4}]))
    assert order == (0, "wmb/finetune_results.jsonl", 4)
    assert rec == {"model": "311m", "layers": 10, "head": "bigru", "feats": None, "cap": 0,
                   "seed": 0, "best_val": 0.9, "best_epoch": 2,
                   "val_by_epoch": [0.8, 0.9], "loss_by_epoch": [0.5, 0.4],
                   "ts": "2026-09-06T14:18:46+00:00"}


def test_a_run_without_curves_has_no_epoch_lists(tmp_path):
    _, _, rec = record(folder(tmp_path))
    assert "val_by_epoch" not in rec and _epochs(rec) is None


def test_a_new_run_takes_its_end_in_utc_and_orders_after_legacy(tmp_path):
    order, _, rec = record(folder(tmp_path, legacy=None, experiment="generalization"))
    assert rec["ts"] == "2026-09-06T14:18:46+00:00"
    assert order > (0, "z", 10**9)


def run(file=None, experiment="x"):
    return {"experiment": experiment, **({"legacy": {"file": file}} if file else {})}


def test_ledger_selection_takes_shards_but_not_a_sibling_ledger():
    sel = from_ledgers(["wmb/finetune_results.jsonl"])
    assert sel(run("wmb/finetune_results.jsonl")) and sel(run("wmb/finetune_results.tl311.jsonl"))
    assert not sel(run("wmb/finetune_qat_results.jsonl"))


def test_ledger_selection_excludes_by_prefix():
    sel = from_ledgers(["wcxb/finetune_results.jsonl"], exclude=["wcxb/finetune_results.wcxbtl-"])
    assert sel(run("wcxb/finetune_results.wcxb311.jsonl"))
    assert not sel(run("wcxb/finetune_results.wcxbtl-forum-fabc.jsonl"))


def test_ledger_selection_never_takes_a_new_run():
    assert not from_ledgers(["wmb/finetune_results.jsonl"])(run())


def test_experiment_selection_adds_the_converted_ledger():
    sel = from_experiment("generalization", also=from_ledgers(["generalization_results.jsonl"]))
    assert sel(run(experiment="generalization"))
    assert sel(run("generalization_results.311m.jsonl"))
    assert not sel(run(experiment="wmb-base"))


# ---- runs from the run store ----

def new(seed=0, val=0.86, test=0.90, sha="115b03b5c8", dirty=False, run_id=None, **cfg):
    return {"id": run_id or f"2026-11-20T2138-000{seed}",
            "config": {"recipe": "table", "board": "wmb", "hidden": 256, "seed": seed, **cfg},
            "provenance": {"git_sha": sha, "dirty": dirty},
            "result": {"best_val": val, "test_f1": test, "test_n": 544}}


def test_seeds_pool_into_one_row_and_fixed_keys_are_stated_once():
    out = render_runs([new(0, val=0.86), new(1, val=0.88)], "cap")
    assert "| 2 | 0.8700±0.0141 | 0.9000±0.0000 |" in out
    assert "Config: `recipe=table`, `board=wmb`, `hidden=256`" in out
    assert "Test n: 544." in out


def test_a_varying_key_splits_rows():
    out = render_runs([new(0, hidden=128, val=0.80), new(0, hidden=256, val=0.86)], "cap")
    assert "| hidden | seeds | val | test |" in out
    assert "| 128 | 1 | 0.8000 |" in out and "| 256 | 1 | 0.8600 |" in out
    assert "hidden=" not in out


def test_a_rerun_adds_seeds_and_is_not_deduplicated():
    out = render_runs([new(0, val=0.86), new(0, val=0.84, run_id="2026-11-21T0900-aaaa")], "cap")
    assert "| 2 | 0.8500±0.0141 |" in out
    assert "2026-11-21T0900-aaaa" in out


def test_each_run_is_listed_with_its_commit():
    out = render_runs([new(0), new(1, dirty=True)], "cap")
    assert "| 2026-11-20T2138-0000 | 0 | 115b03b | 0.8600 | 0.9000 |" in out
    assert "| 2026-11-20T2138-0001 | 1 | 115b03b+dirty |" in out


def test_a_column_shows_only_when_a_run_carries_it():
    assert "test int8" not in render_runs([new(0)], "cap")
