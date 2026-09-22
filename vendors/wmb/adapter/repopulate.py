"""
Re-render convert_main_content from main_html and check it against the stored value.

The 7,809 full-set records (source full/both) carry the benchmark's official,
calibrated convert_main_content: render main_html and verify it reproduces that
value (the render gate), but keep the stored bytes. The 16 test-only records
carry the 545's own stale/null value: discard it, regenerate main_html from
cc-select where present, render, and fill. 2fe1c202 has no cc-select and is
excluded (main_html and convert_main_content left null).

Rewrites data/wmb.jsonl in place. Report is bucketed by source from splits.json.

    python vendors/wmb/adapter/repopulate.py
"""

import json
from pathlib import Path

from vendors.shared.upstream import load

WMB = Path(__file__).resolve().parents[1]
DATA = WMB / "data"
COMBINED = DATA / "wmb.jsonl"
SPLITS = DATA / "splits.json"
CC = 'cc-select="true"'


def load_upstream():
    mod = load(WMB / "upstream" / "main_html.py", "wmb_main_html")
    return mod.extract_main_html, mod.HTML2TextWrapper


def main():
    extract_main_html, HTML2TextWrapper = load_upstream()
    with open(SPLITS, encoding="utf-8") as f:
        splits = json.load(f)

    stats = {
        "full": {"matched": 0, "mismatched": 0},
        "both": {"matched": 0, "mismatched": 0},
        "test": {"filled": 0, "excluded": 0},
    }
    mismatched = []
    tmp = COMBINED.with_name(COMBINED.name + ".tmp")

    with open(COMBINED, encoding="utf-8") as f, open(tmp, "w", encoding="utf-8") as out:
        for line in f:
            r = json.loads(line)
            src = splits[r["track_id"]]["source"]
            if src in ("full", "both"):
                rendered = HTML2TextWrapper()(r["main_html"], r.get("url", ""))
                if rendered == r["convert_main_content"]:
                    stats[src]["matched"] += 1
                else:
                    stats[src]["mismatched"] += 1
                    mismatched.append(r["track_id"])
            elif CC in r["html"]:
                r["main_html"] = extract_main_html(r["html"])
                r["convert_main_content"] = HTML2TextWrapper()(r["main_html"], r.get("url", ""))
                stats["test"]["filled"] += 1
            else:
                r["main_html"] = None
                r["convert_main_content"] = None
                stats["test"]["excluded"] += 1
            out.write(json.dumps(r, ensure_ascii=False) + "\n")

    assert stats["full"]["matched"] + stats["full"]["mismatched"] == 7280
    assert stats["both"]["matched"] + stats["both"]["mismatched"] == 529
    assert stats["test"]["filled"] + stats["test"]["excluded"] == 16
    # the render gate (§4): render(main_html) reproduces the stored value
    # byte-identically across the full-set, 7809/7809.
    assert not mismatched, (
        f"render gate: {len(mismatched)} of 7809 render(main_html) != stored "
        f"convert_main_content (first 10: {[i[:8] for i in mismatched[:10]]})")
    tmp.replace(COMBINED)

    for src in ("full", "both"):
        s = stats[src]
        print(f"{src:5} ({s['matched'] + s['mismatched']}): "
              f"matched {s['matched']}, mismatched {s['mismatched']}")
    t = stats["test"]
    print(f"test  ({t['filled'] + t['excluded']}): filled {t['filled']}, excluded {t['excluded']}")


if __name__ == "__main__":
    main()
