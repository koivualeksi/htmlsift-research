# vendors/wcxb — Web Content Extraction Benchmark

Second first-class board: word-level F1 via their `evaluate.py`, used verbatim.
2,008 pages, 1,613 domains, 7 page types. Dev 1,497 and test 511.
**Never train on the 511.**

## Source

| | |
|---|---|
| dataset | https://huggingface.co/datasets/murrough-foley/web-content-extraction-benchmark |
| licence | CC-BY-4.0 |
| repo | https://github.com/Murrough-Foley/web-content-extraction-benchmark |
| paper | https://arxiv.org/abs/2605.21097 |
| leaderboard | https://webcontentextraction.org/ |
| Zenodo DOI | 10.5281/zenodo.19316874 |

## What acquire downloads

`acquire.py` pulls the whole dataset repo at pinned revision
`be72432cfa012ac918af47010bf106a2801afeef` with one `snapshot_download` into
`data/` (gitignored) — a directory tree, not JSONL: `dev/` and `test/`, each with
`ground-truth/*.json` + `html/*.html`, plus top-level `evaluate.py`, `LICENSE`,
`README.md`, `metadata.json`.

| split | pages | files |
|---|---|---|
| dev | 1,497 | 2,994 (gt + html) |
| test | 511 | 1,022 (gt + html) |

At 4,016 page files, per-file sha256 is not listed; instead acquire computes the
sha256 of the sorted `sha256  relpath` manifest of every downloaded file (the
`.cache` bookkeeping excluded) and asserts it equals

    e803b06b223bf932f7c682e13903f947d18835ad9d84bbd86e13a1061de18ad2

recorded as `MANIFEST_SHA256` in `adapter/acquire.py`; one byte anywhere trips it.
acquire also asserts gt/html stems pair exactly and the split counts (1,497 /
511). The tree is kept, not deleted — their `evaluate.py` reads it.

    python vendors/wcxb/adapter/acquire.py

## Vendored in upstream/

`evaluate.py` — verbatim from the dataset repo, which ships it at the tree root,
CC-BY-4.0, pinned revision `be72432cfa012ac918af47010bf106a2801afeef` (git blob
`46dae2c204d1e3cd173a405e5ec531f697cbcacb`). No edits: check out the dataset at
that revision and the diff is empty. sha256
`f33188453d26791b75ee644ab666960636a8a4e9e2f34c4b313e0ca6e7d6a4aa`. Its `LICENSE`
(CC-BY-4.0) sits beside it, sha256
`dcd40aa527474a0d1d20270038591be5f4149963140561e9adfff28ef7818e9c`.

It is the WCXB metric: `word_f1` (word-level precision/recall/F1 over `\w+`
lowercased multiset overlap) on plain-text `main_content`, with `tokenize`. The
adapter's `labels.py` imports these to build and score dev labels, and the test
board is scored by this file verbatim — never reimplemented (§8, §9). Its
`WCEB_DIR = Path(__file__).parent` is left as-is: `word_f1`/`tokenize` are
path-independent, and the CLI's `load_ground_truth` runs from the `data/` tree
where the same file ships, not from `upstream/`.

Ground truth here is LLM-assisted then human-reviewed, not purely human like
WMB's. That is a disclosure line in the writeup; see `docs/LIMITS.md`.
