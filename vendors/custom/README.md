# vendors/custom — annotated hard pages

Extra training pages, never a board: no val, no test, no metric of their own. They
enter WMB's train fold only; WMB val selects and WMB test scores, unchanged.

## Source

| | |
|---|---|
| dataset | `koivualeksi/htmlsift-annotations` (Hugging Face, private) |
| revision | `88a2564ef63192ba08f795b5af59045b1fa8ca26` |
| pages | 355, Common Crawl pages mined where mini kept boilerplate (footers, side navigation) or failed DCLM where trafilatura passed |
| labels | per block, 1 = main, guideline v1 (WMB policy) |
| licence | labels ours; page HTML is third-party and not redistributable |

The dataset is private because the page HTML cannot be redistributed, so this adapter
reproduces only with read access to it. Nothing acquired or derived is committed.

## What acquire downloads

`pages/<id>.json` and `html/<id>.html.gz` for all 355 pages, into `data/raw/`. The tree
is pinned by manifest sha256 (sorted `sha256  relpath` lines over both directories):
`7129fc7ce18a739382f57859a229debdc118c09471c14c266d86de6a74202271`.

## Blocks

`adapter/blocks.py` re-renders each page on the WMB block path and keeps a page only if
the render equals the blocks it was labelled on; a mismatch drops the page rather than
realigning its labels. Writes `data/blocks.jsonl` and `data/feats.jsonl` in WMB's
shapes. Must run on Linux with the libxml2 recorded in each page (2.14.6).

    python vendors/custom/adapter/acquire.py
    python vendors/custom/adapter/blocks.py

At the pinned revision: 355/355 kept, no URL shared with WMB; custom domains also
present in WMB: 5 in train, 1 in val, 1 in test.
