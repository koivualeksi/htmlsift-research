# vendors/custom — annotated hard pages

Extra training pages, never a board: no val, no test, no metric of their own. They
enter WMB's train fold only; WMB val selects and WMB test scores, unchanged.

## Source

| | |
|---|---|
| dataset | `koivualeksi/htmlsift-annotations` (Hugging Face, private) |
| revision | `88ba5c55e402e5bc6c9dde66ac96945e7ea78e94` |
| pages | 112, Common Crawl pages where mini kept boilerplate that trafilatura dropped |
| labels | per block, 1 = main, guideline v1 (WMB policy) |
| licence | labels ours; page HTML is third-party and not redistributable |

The dataset is private because the page HTML cannot be redistributed, so this adapter
reproduces only with read access to it. Nothing acquired or derived is committed.

## What acquire downloads

`pages/<id>.json` and `html/<id>.html.gz` for all 112 pages, into `data/raw/`. The tree
is pinned by manifest sha256 (sorted `sha256  relpath` lines over both directories):
`849833c06e816922ee6f7bb1878927e9b6cb7103500280858d7eb6135ccedf95`.

## Blocks

`adapter/blocks.py` re-renders each page on the WMB block path and keeps a page only if
the render equals the blocks it was labelled on; a mismatch drops the page rather than
realigning its labels. Writes `data/blocks.jsonl` and `data/feats.jsonl` in WMB's
shapes. Must run on Linux with the libxml2 recorded in each page (2.14.6).

    python vendors/custom/adapter/acquire.py
    python vendors/custom/adapter/blocks.py

At the pinned revision: 112/112 kept, no URL shared with WMB; custom domains also
present in WMB: 3 in train, 1 in test.
