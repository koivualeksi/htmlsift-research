"""Unit: the vendored board scorers, never reimplemented (CLAUDE.md §8).

WMB: calc_rouge_n_score — ROUGE-N F1 over jieba tokens, n=5 by default (the board
metric). WCXB: word_f1 — bag-of-words F1 over \\w+ lowercase tokens, and tokenize,
whose one surviving marker is the underscore (PROTOCOL.md). These bound the oracle
ceilings, so a drift here decouples every score from the artifact.
"""
from pathlib import Path

import pytest

from vendors.shared.upstream import load

ROOT = Path(__file__).resolve().parents[2]

rouge = load(ROOT / "vendors/wmb/upstream/rouge_utils.py", "wmb_rouge_utils")
wcxb = load(ROOT / "vendors/wcxb/upstream/evaluate.py", "wcxb_evaluate")

# ROUGE-5 needs >=5 tokens to form an n-gram; keep the perfect-match samples long.
LONG = "the quick brown fox jumps over the lazy dog again"


def test_rouge_identical_is_perfect():
    s = rouge.calc_rouge_n_score(LONG, LONG)
    assert s == {"prec": pytest.approx(1.0), "rec": pytest.approx(1.0),
                 "f1": pytest.approx(1.0)}


def test_rouge_both_empty_is_perfect():
    assert rouge.calc_rouge_n_score("", "")["f1"] == 1.0


def test_rouge_disjoint_is_zero():
    assert rouge.calc_rouge_n_score(LONG, "")["f1"] == 0.0
    assert rouge.calc_rouge_n_score("aaa bbb ccc ddd eee",
                                    "vvv www xxx yyy zzz")["f1"] == 0.0


def test_rouge_default_n_is_5_and_order_sensitive():
    a = "one two three four five six seven eight"
    b = "eight seven six five four three two one"
    assert rouge.calc_rouge_n_score(a, b) == rouge.calc_rouge_n_score(a, b, n=5)
    # reversed: every unigram still matches, but the 5-grams do not
    assert rouge.calc_rouge_n_score(a, b, n=1)["f1"] == pytest.approx(1.0)
    assert rouge.calc_rouge_n_score(a, b)["f1"] < 1.0


def test_word_f1_identical_case_and_order_invariant():
    assert wcxb.word_f1("Hello, WORLD!", "hello world") == (1.0, 1.0, 1.0)
    assert wcxb.word_f1("a b c", "c b a") == (1.0, 1.0, 1.0)


def test_word_f1_empty_cases():
    assert wcxb.word_f1("", "") == (1.0, 1.0, 1.0)
    assert wcxb.word_f1("x", "") == (0.0, 0.0, 0.0)      # ref empty, pred not
    assert wcxb.word_f1("", "x") == (0.0, 0.0, 0.0)      # pred empty, ref not


def test_word_f1_multiset_partial():
    for pred, ref in (("a a b", "a b b"), ("a b c", "a b d")):
        p, r, f1 = wcxb.word_f1(pred, ref)
        assert (p, r, f1) == pytest.approx((2 / 3, 2 / 3, 2 / 3))


def test_tokenize_lowercase_keeps_underscore():
    assert wcxb.tokenize("Hello, world_1! 3C") == ["hello", "world_1", "3c"]
    assert wcxb.tokenize("") == []
    assert wcxb.tokenize(None) == []
