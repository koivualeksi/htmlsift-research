"""
Collapse the acquired WCXB tree into one corpus, WMB-shaped.

WCXB ships as dev/ and test/ trees of per-page ground-truth JSON + wild HTML,
not WMB's two overlapping JSONL files. This folds them into a single
data/wcxb.jsonl (one record per page) and data/splits.json (track_id -> source,
fold) -- the layout the WMB pipeline uses.

file_id stems are reused across splits (139 collide), so the key namespaces the
split: track_id = f"{split}-{file_id}". We use WCXB's own split verbatim, no custom
fold: dev is the training set (fold "train"), test is the evaluation board (fold
"test", never trained on). The source tree is NOT deleted: their evaluate.py reads
it directly and re-downloading is rate-limited.

Wild HTML is not guaranteed UTF-8: decode strict, fall back to charset_normalizer,
count fallbacks.

    python vendors/wcxb/adapter/collapse.py
"""

import json
from pathlib import Path

from vendors.shared.acquire import read_html

DATA = Path(__file__).resolve().parents[1] / "data"
COMBINED = DATA / "wcxb.jsonl"
SPLITS = DATA / "splits.json"

N = {"dev": 1497, "test": 511}


def page_type(gt_json: dict) -> str:
    # mirrors upstream/evaluate.py get_page_type: primary type, category -> collection
    pt = (gt_json.get("_internal") or {}).get("page_type")
    prim = pt.get("primary", "article") if isinstance(pt, dict) else (pt if isinstance(pt, str) else "article")
    return "collection" if prim == "category" else prim


def main():
    splits = {}
    n_fallback = 0
    with open(COMBINED, "w", encoding="utf-8") as out:
        for split in ("dev", "test"):
            gt_dir = DATA / split / "ground-truth"
            html_dir = DATA / split / "html"
            ids = sorted(p.stem for p in gt_dir.glob("*.json"))
            assert len(ids) == N[split], f"{split}: {len(ids)} != {N[split]}"
            for fid in ids:
                d = json.loads((gt_dir / f"{fid}.json").read_text(encoding="utf-8"))
                gt = d.get("ground_truth") or {}
                html, fb = read_html(html_dir / f"{fid}.html")
                n_fallback += fb
                tid = f"{split}-{fid}"
                out.write(json.dumps({
                    "track_id": tid,
                    "split": split,
                    "file_id": fid,
                    "url": d.get("url") or "",
                    "page_type": page_type(d),
                    "html": html,
                    "main_content": gt.get("main_content") or "",
                    "with": gt.get("with") or [],
                    "without": gt.get("without") or [],
                    "title": gt.get("title") or "",
                }, ensure_ascii=False) + "\n")
                splits[tid] = {"source": split, "fold": "test" if split == "test" else "train"}

    assert len(splits) == 2008
    src = {"dev": 0, "test": 0}
    for v in splits.values():
        src[v["source"]] += 1
    assert src == {"dev": 1497, "test": 511}
    with open(SPLITS, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=0)

    print(f"{COMBINED.name}: {len(splits)} records (dev {src['dev']}, test {src['test']})")
    print(f"{SPLITS.name}: {len(splits)} track_id -> source, fold")
    print(f"charset fallback on {n_fallback} html files")


if __name__ == "__main__":
    main()
