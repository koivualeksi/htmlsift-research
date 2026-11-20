"""Unit: LOLO wiring -- benchmark name resolution. No corpus; the benchmark's fold logic
is already covered by its own ceiling gates."""


# ---- benchmark name ----

def test_load_benchmark_daniel_eval_only():
    from vendors.shared.benchmark import load_benchmark
    b = load_benchmark("daniel")
    assert b.name == "daniel"
    assert b.holdout is None


def test_load_benchmark_daniel_holdout():
    from vendors.shared.benchmark import load_benchmark
    b = load_benchmark("daniel", holdout="Chinese")
    assert b.name == "daniel-zh"
    assert b.holdout == "Chinese"


def test_load_benchmark_daniel_holdout_all_five():
    from vendors.shared.benchmark import load_benchmark
    names = set()
    for lang in ["Greek", "Polish", "Russian", "English", "Chinese"]:
        b = load_benchmark("daniel", holdout=lang)
        names.add(b.name)
    assert names == {"daniel-el", "daniel-pl", "daniel-ru", "daniel-en", "daniel-zh"}

