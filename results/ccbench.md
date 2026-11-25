Common Crawl yield -- how many pages an extractor's text gets past the FineWeb-Edu and DCLM quality filters, against trafilatura on the same pages. **Private data.** The page set is `koivualeksi/htmlsift-bench` at revision `0ac5a811`. Its HTML is not redistributable, so this table cannot be reproduced from the published sources; it is context only, not a board.

Regenerate (Linux, `htmlsift` 0.3.0, `pip install -e .[ccbench]`):

    python bench/ccbench.py --ckpt mini --drop-title --dedup

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
- `mini 0.3.0` is the bundle `htmlsift` 0.3.0 ships (`311m-table-bigru-fABC-nb1-pf4-ema0.999-r128`,
  seed 2, `docs/CLAIMS.md`), through the package's own `Extractor` with `drop_title` and `dedup`.
- `drop_title` removes the extracted title before scoring. trafilatura's text carries no title.
- `dedup` drops repeated lines from the extracted text.
- Both settings change only the extracted text, not the model or the labels.

## Totals

Edu pages are unweighted, because every Edu-passing page has weight 1. The other counts are raw,
with the weighted count in brackets. Mean tokens is over all 7,177 pages.

| extractor | Edu pages | DCLM pages | English pages | mean tokens | empty texts |
|---|---|---|---|---|---|
| trafilatura | 436 | 2,926 (6,753) | 6,817 (30,517) | 885 | 59 (302) |
| mini 0.3.0, drop_title, dedup | 436 | 3,003 (6,966) | 6,194 (26,819) | 1,078 | 637 (3,605) |

## Against trafilatura

Relative change in passing pages and passing tokens, weighted:

| extractor | Edu pages | Edu tokens | DCLM pages | DCLM tokens |
|---|---|---|---|---|
| mini 0.3.0, drop_title, dedup | +0.0 % [-5.6, +6.3] | +41.9 % [-4.7, +134.6] | +3.1 % [-0.8, +7.3] | +40.4 % [+31.9, +50.7] |

## Reading

- **Edu pages.** The Edu classifier reads only the first 512 tokens, and trafilatura's text
  carries no title, so the mini is scored with its title dropped. It is then level with
  trafilatura on Edu pages, and the interval spans zero.
- **DCLM pages.** The 0.3.0 mini leads trafilatura by +3.1 % [-0.8, +7.3]; the interval spans zero.
- **Tokens.** The mini passes about 40 % more DCLM tokens, because the text it keeps on passing
  pages is longer.
- **Empty texts are mostly bot-challenge pages.** 322 of the 7,177 pages (1,256 weighted) are
  bot-challenge or JavaScript-wall interstitials. On these, trafilatura returns a line such as
  "Please wait while your request is being verified..." and the right output is nothing.
  - That line passes DCLM on 305 of them, 1,045 weighted. This is 15 % of trafilatura's weighted
    DCLM passes, earned on pages with no content.
  - The 0.3.0 mini returns nothing on 319 of the 322; they are half of its 637 empty texts.
- **With the challenge pages removed** (trafilatura text of 60 tokens or fewer that contains a
  challenge phrase, case-insensitive; `docs/mini-error-groups.md`), on the other 6,855 pages the
  0.3.0 mini against trafilatura: Edu pages +0.0 % [-5.8, +6.4], DCLM pages +22.0 % [+17.7, +26.7],
  DCLM tokens +40.7 % [+32.1, +51.0].
- **Remaining empties.** The 0.3.0 mini's other 318 empty pages (2,388 weighted) are its real
  misses, analysed in `docs/mini-error-groups.md`.
