Common Crawl yield -- how many pages an extractor's text gets past the FineWeb-Edu and DCLM quality filters, against trafilatura on the same pages. **Private data.** The page set is `koivualeksi/htmlsift-bench` at revision `0ac5a811`. Its HTML is not redistributable, so this table cannot be reproduced from the published sources; it is context only, not a board.

Regenerate (Linux, pinned `htmlsift` 0.2.1, `pip install -e .[ccbench]`):

    python bench/ccbench.py --ckpt wmb/ckpt/311m-table-bigru-fABC_s1.pt [--drop-title]
    python bench/ccbench.py --ckpt wmb/ckpt/311m-table-bigru-fABC-nb1-pf4-ema0.999_s1.pt [--drop-title]

**Page set:**
- 7,177 pages from CC-MAIN-2026-43, drawn in three strata: edu 2,677, dclm 3,000 and rest
  1,500.
- Weights scale them to a pool of 32,399 pages.

**Filters:**
- English means fastText lid.176 English with p ≥ 0.65.
- A page passes Edu at a FineWeb-Edu classifier score ≥ 3 on its English text. It passes DCLM
  at a DCLM fastText score ≥ 0.018112.
- Changes are weighted relative changes against the reference extractor. The 95 % intervals
  come from a paired bootstrap of pages within strata, 2,000 draws.

**Extractors:**
- trafilatura is its stored output on the same pages.
- `mini` is the shipped mini, s1. `candidate` is `ema 0.999 + smooth_weight 1 + page_f1 4`, s1
  (`docs/objective.md`). Both run with the int8 table through the package's own `Extractor`.
- `drop_title` removes the extracted title before scoring. trafilatura's text carries no title.
  The setting changes scoring only, not the model or the labels.

## Totals

Edu pages are unweighted, because every Edu-passing page has weight 1. The other counts are raw,
with the weighted count in brackets.

| extractor | Edu pages | DCLM pages | English pages | mean tokens | empty texts |
|---|---|---|---|---|---|
| trafilatura | 436 | 2,926 (6,753) | 6,817 (30,517) | 885 | 59 (302) |
| mini | 383 | 2,990 (6,885) | 6,365 (27,993) | 1,145 | 433 (2,380) |
| candidate | 402 | 2,994 (7,060) | 6,233 (26,985) | 1,090 | 514 (2,912) |
| mini, drop_title | 428 | 3,009 (6,943) | 6,316 (27,663) | 1,140 | 557 (3,088) |
| candidate, drop_title | 433 | 3,027 (7,153) | 6,189 (26,682) | 1,084 | 640 (3,689) |

## Against trafilatura

Relative change in passing pages and passing tokens, weighted:

| extractor | Edu pages | Edu tokens | DCLM pages | DCLM tokens |
|---|---|---|---|---|
| mini | -12.2 % [-18.0, -6.0] | +26.6 % [-20.2, +120.4] | +2.0 % [-1.8, +5.9] | +41.6 % [+33.0, +51.5] |
| candidate | -7.8 % [-13.8, -1.9] | +36.4 % [-10.7, +128.8] | +4.5 % [+0.4, +9.1] | +40.5 % [+31.7, +50.6] |
| mini, drop_title | -1.8 % [-7.5, +4.3] | +39.8 % [-7.8, +134.3] | +2.8 % [-0.9, +6.7] | +42.4 % [+33.6, +52.5] |
| candidate, drop_title | -0.7 % [-6.3, +5.5] | +42.9 % [-3.8, +135.5] | +5.9 % [+1.8, +10.4] | +41.0 % [+32.0, +51.2] |

## Candidate against mini

| setting | Edu pages | Edu tokens | DCLM pages | DCLM tokens |
|---|---|---|---|---|
| title kept | +5.0 % [+0.6, +9.5] | +7.7 % [+1.4, +19.2] | +2.5 % [+0.0, +5.1] | -0.8 % [-2.6, +0.9] |
| drop_title | +1.2 % [-2.6, +5.4] | +2.2 % [-2.6, +9.3] | +3.0 % [+0.4, +5.8] | -1.0 % [-2.6, +0.6] |

## Reading

- **The title is most of the Edu gap.** The Edu classifier reads only the first 512 tokens.
  Once the title is dropped, both models are level with trafilatura on Edu pages, and both
  intervals span zero.
- **DCLM pages.** The candidate leads trafilatura by +5.9 % [+1.8, +10.4]. It also leads the
  shipped mini by +3.0 %.
- **Tokens.** The models pass about 41 % more DCLM tokens, because the text they keep on
  passing pages is longer.
- **Empty texts are mostly bot-challenge pages.** 322 of the 7,177 pages (1,256 weighted) are
  bot-challenge or JavaScript-wall interstitials. On these, trafilatura returns a line such as
  "Please wait while your request is being verified..." and the models return nothing.
  - That line passes DCLM on 305 of them, 1,045 weighted. This is 15 % of trafilatura's weighted
    DCLM passes, earned on pages with no content.
  - The models leave 318 or 319 of these pages empty, out of their 557 and 640 empty texts.
- **With the challenge pages removed** (trafilatura text of 60 tokens or fewer that matches a
  challenge phrase), DCLM pages against trafilatura become +21.6 % [+17.2, +26.2] for the mini
  and +25.3 % [+20.7, +30.3] for the candidate, both with drop_title. Edu is unchanged.
- **Remaining empties.** The candidate leaves 321 non-challenge pages empty. On those pages,
  trafilatura's median text is 70 tokens, and it passes DCLM on 52 pages (166 weighted). These
  are the model's real misses, analysed in `docs/mini-error-groups.md`.
