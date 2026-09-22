"""
Domain-grouped train/val fold over the 7,280 training pool.

Grouping unit = registrable domain (eTLD+1 via tldextract's bundled public-suffix
snapshot, offline); no domain straddles the folds. Domains are bucketed by
(majority language, majority level) of their pages; within each bucket they are
shuffled under a fixed seed and assigned to val until the bucket's val share
reaches VAL_FRAC, skipping any domain that would overshoot target * 1.4 so one
large domain cannot swamp a fold.

Writes the fold into splits.json: every source=="full" record becomes train or
val; the board (both/test) keeps fold "test". Idempotent -- the pool is keyed off
source, which split never changes.

    python vendors/wmb/adapter/split.py
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import tldextract

DATA = Path(__file__).resolve().parents[1] / "data"
COMBINED = DATA / "wmb.jsonl"
SPLITS = DATA / "splits.json"
VAL_FRAC = 0.10
SEED = 20260815

# offline: bundled public-suffix snapshot only, never the network
_extract = tldextract.TLDExtract(suffix_list_urls=(), fallback_to_snapshot=True)


def domain_of(url: str) -> str:
    ext = _extract(url or "")
    if ext.domain and ext.suffix:
        return f"{ext.domain}.{ext.suffix}".lower()
    return (ext.domain or ext.subdomain or url or "unknown").lower()


def main():
    with open(SPLITS, encoding="utf-8") as f:
        splits = json.load(f)

    pool = []
    with open(COMBINED, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            tid = r["track_id"]
            if splits[tid]["source"] != "full":
                continue
            m = r.get("meta") or {}
            pool.append({
                "track_id": tid,
                "domain": domain_of(r.get("url", "")),
                "language": m.get("language") or "unknown",
                "level": m.get("level") or "unknown",
            })
    assert len(pool) == 7280

    by_domain = defaultdict(list)
    for p in pool:
        by_domain[p["domain"]].append(p)

    buckets = defaultdict(list)
    for dom, ps in by_domain.items():
        lang = Counter(p["language"] for p in ps).most_common(1)[0][0]
        level = Counter(p["level"] for p in ps).most_common(1)[0][0]
        buckets[(lang, level)].append(dom)

    rng = random.Random(SEED)
    val_domains = set()
    for key in sorted(buckets):
        doms = sorted(buckets[key])
        rng.shuffle(doms)
        target = sum(len(by_domain[d]) for d in doms) * VAL_FRAC
        got = 0
        for d in doms:
            if got >= target:
                break
            size = len(by_domain[d])
            if got + size > target * 1.4:
                continue
            val_domains.add(d)
            got += size

    for p in pool:
        fold = "val" if p["domain"] in val_domains else "train"
        splits[p["track_id"]]["fold"] = fold
        splits[p["track_id"]]["level"] = p["level"]

    fold_of_domain = defaultdict(set)
    for p in pool:
        fold_of_domain[p["domain"]].add(splits[p["track_id"]]["fold"])
    assert all(len(fs) == 1 for fs in fold_of_domain.values()), "domain straddles folds"

    n_val = sum(1 for p in pool if splits[p["track_id"]]["fold"] == "val")
    assert len(val_domains) == 545, f"val domains {len(val_domains)} != 545"
    assert n_val == 732, f"val pages {n_val} != 732"

    with open(SPLITS, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=0)

    print(f"{len(by_domain)} domains, {len(pool)} pool pages")
    print(f"val: {len(val_domains)} domains, {n_val} pages ({n_val / len(pool):.1%})")
    for key in sorted(buckets):
        ps = [p for d in buckets[key] for p in by_domain[d]]
        nv = sum(1 for p in ps if splits[p["track_id"]]["fold"] == "val")
        print(f"  {key}: {nv}/{len(ps)} val ({nv / len(ps):.1%})")


if __name__ == "__main__":
    main()
