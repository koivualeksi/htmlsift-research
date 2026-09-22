"""Unit: the results reader's cell identity. The writer (trainers.ledger.record_key)
separates train_limit and holdout; the reader adds the epoch budget, which no key
carries. A cell that merges any of those averages two populations and reports the
wrong one -- the failure that put a train_limit=4000 run in the full-pool table.
No HF transport: render() is pure over authored records.
"""
import pytest

from tools.download_results import _epochs, collapse, group_key, render, render_matrix

FROZEN = {"model": "311m", "layers": 12, "head": "bigru", "feats": "ABC",
          "seed": 0, "ts": "2026-01-01", "best_val": 0.89, "frozen": True}


def rec(seed=0, ts="2026-01-01", epochs=8, **extra):
    return {"model": "311m", "layers": 10, "head": "bigru", "feats": None, "cap": 0,
            "seed": seed, "ts": ts, "best_val": 0.9,
            "val_by_epoch": [0.9] * epochs, **extra}


def exported(seed=0, f1=0.93, test_n=544, **extra):
    return rec(seed=seed, test_f1=f1, test_prec=0.93, test_rec=0.96, test_n=test_n, **extra)


# ---- group_key separates what record_key separates, plus the epoch budget ----

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
