"""
Collapse the two acquired WMB files into one deduplicated corpus.
    - webmainbench.jsonl (7809 entries)
    - WebMainBench_545 (545 entries)

These overlap on 529 entries, 545 brings 16 new entries to main table

Merged in following way:
    - Big table is source truth for html, url, convert_main_content
    - Small table is source truth for groundtruth_convert

Outputs collapsed file of 7825 entries and a splits.json containing track_id and source
"""

import argparse
import json
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
FULL = DATA / "webmainbench.jsonl"
SUB = DATA / "WebMainBench_545.jsonl"
COMBINED = DATA / "wmb.jsonl"
SPLITS = DATA / "splits.json"


def main(delete_original=True):
    sub = {}
    with open(SUB, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            tid = r["track_id"]
            assert tid not in sub, f"duplicate track_id in 545: {tid}"
            sub[tid] = r
    assert len(sub) == 545

    splits, seen, n_full = {}, set(), 0
    with open(FULL, encoding="utf-8") as f, open(COMBINED, "w", encoding="utf-8") as out:
        for line in f:
            r = json.loads(line)
            tid = r["track_id"]
            assert tid not in seen, f"duplicate track_id in 7809: {tid}"
            seen.add(tid)
            n_full += 1
            if tid in sub:
                r["groundtruth_content"] = sub[tid]["groundtruth_content"]
                splits[tid] = {"source": "both", "fold": "test"}
            else:
                splits[tid] = {"source": "full", "fold": None}
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
        for tid, r in sub.items():
            if tid not in seen:
                splits[tid] = {"source": "test", "fold": "test"}
                out.write(json.dumps(r, ensure_ascii=False) + "\n")

    assert n_full == 7809
    counts = {"full": 0, "test": 0, "both": 0}
    for v in splits.values():
        counts[v["source"]] += 1
    assert len(splits) == 7825
    assert counts["both"] == 529
    assert counts["full"] == 7280
    assert counts["test"] == 16

    with open(SPLITS, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=0)

    print(f"{COMBINED.name}: {len(splits)} records "
          f"(full-only {counts['full']}, both {counts['both']}, test-only {counts['test']})")
    print(f"{SPLITS.name}: {len(splits)} track_id -> source")

    if delete_original:
        FULL.unlink()
        SUB.unlink()
        print(f"deleted originals: {FULL.name}, {SUB.name} (re-run acquire to restore)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep-original", action="store_true",
                        help="keep the two acquired source files "
                             "(default: delete them after collapse; re-run acquire to restore)")
    main(delete_original=not parser.parse_args().keep_original)
