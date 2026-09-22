"""§7/§4 bench gate: the CPU speed harness refuses a cross-population ratio. A
competitor that produced output but shares no timed page with ours must raise, not
silently ratio over an empty intersection (which used to surface as a StatisticsError
deep inside report()).
"""
import pytest

from bench.speed.cpu_compare import report


def test_report_refuses_disjoint_competitor():
    # htmlsift timed {a, b}; the competitor timed only {c}. Non-empty but disjoint --
    # the refusal fires before any scoring, so board/tokens/env are never touched.
    methods = {"htmlsift": {"a": None, "b": None}, "trafilatura": {"c": None}}
    scorable = {"a", "b", "c"}
    notes = {n: {"empty": 0, "raised": 0} for n in methods}
    with pytest.raises(SystemExit, match="cross-population"):
        report(methods, board=None, fold="test", tokens={}, pop="test545",
               env=[], scorable=scorable, notes=notes)


def test_report_refuses_when_ours_scores_nothing():
    # the existing guard: if htmlsift itself covers no scorable page, there is no
    # ratio baseline at all.
    methods = {"htmlsift": {"z": None}, "trafilatura": {"a": None}}
    scorable = {"a"}
    notes = {n: {"empty": 0, "raised": 0} for n in methods}
    with pytest.raises(SystemExit, match="no scorable page"):
        report(methods, board=None, fold="test", tokens={}, pop="test545",
               env=[], scorable=scorable, notes=notes)
