# Annotation divergence: "main content" is a policy, not a fact

WMB and WCXB do not just contain different pages — they encode different *labeling
policies*. On some page types they draw the boundary of "main content" in opposite
places. This is why every cross-board number in `CLAIMS.md` is reported as its own cell
and never blended, and why a model never trains on a board it is scored against.

The clearest case is the **collection** (e-commerce category) page.

## The measurement

The same model (311m-10 encoder), trained on each benchmark's labels and scored on
WCXB test511, by page type (word-F1, 3 seeds each; `results/generalization.md`). Where
the two training policies disagree, the WMB-trained model underperforms the WCXB-trained
one:

| WCXB page type | n | trained on WMB | trained on WCXB | gap |
|---|---|---|---|---|
| **collection** | 34 | **0.5306** | **0.8795** | **+34.9** |
| product | 28 | 0.8275 | 0.9348 | +10.7 |
| listing | 40 | 0.6667 | 0.7587 | +9.2 |
| article | 257 | 0.9342 | 0.9723 | +3.8 |
| service | 59 | 0.8274 | 0.8561 | +2.9 |
| forum | 51 | 0.8668 | 0.8601 | &#8722;0.7 |
| documentation | 42 | 0.9561 | 0.9496 | &#8722;0.7 |

The gap is concentrated on **collections**: +34.9 pt, three times the next type. On
forums and documentation the two policies agree within 1 pt. Picture:
`assets/benchmark-divergence.svg`.

## The mechanism

Same page kind, opposite call. A collection page is a heading plus a product grid; the
two benchmarks label it in opposite directions. Picture: `assets/annotation-example.svg`;
full source text for three matched pairs in `divergence-examples.md`.

- **WCXB keeps blurb and grid.** `upliftdesk.com/standing-desks` (dev-0643) — main
  content is the heading, the intro blurb and all 30 desks with prices: *"Standing desks ·
  Improve your health, comfort, and productivity … · UPLIFT V3 Standing Desk $599 · UPLIFT
  L-Shaped Standing Desk, 3-Leg $1149 · …"*, 65 of 178 rendered blocks. Across the 117
  WCXB dev collection pages, 27 carry an intro blurb and every one of them is kept; no
  collection page labels a blurb as boilerplate.
- **WMB keeps the heading and blurb and drops the grid.**
  `apeainthepod.com/maternity/tees-and-tanks` — main content is **2 of 809 blocks** (the
  heading and the intro paragraph); all 241 priced products, each with its own name
  block, are labeled boilerplate.

The pattern holds across pure-grid category pages: `lilyboutique.com/party-dresses`
(2/374 kept), `autopia-carcare.com/vinyl` (2/444 kept) — heading kept, grid dropped —
against WCXB collections that keep the grid (`analuisa.com/collections/…`,
`giro.com/…/lights`).

## The control

Where the policies point at the same thing, they agree — so this is a real policy
difference, not a WMB annotation that is simply sparse. On **article** pages from
domains present in *both* corpora, the two keep the same thing:

- `arxiv.org` — both keep the abstract.
- `nytimes.com` — both keep the article body.

Articles are also the type where the cross-board gap is small (+3.8 pt, most of which is
format, not policy).

## Why it matters

- **Cross-board numbers are policy comparisons, not model rankings.** A WMB-trained
  model "failing" on WCXB collections (0.53) has not failed — it is applying WMB's
  policy faithfully to a page WCXB annotates differently.
- **No board is both trained on and reported.** The two policies are incompatible on
  collections, so mixing them would train a model to satisfy neither.
- **The Chinese/DAnIEL caveats (`LIMITS.md`) are the same phenomenon** at the language
  level: what counts as content is an annotation choice, and it does not always transfer.

## The heuristics are policies too

The same split shows up in off-the-shelf extractors, which is the practical form of the
argument. Scored zero-shot on WCXB and DAnIEL (`CLAIMS.md`, `results/heuristics-wcxb.md`,
`results/heuristics-daniel.md`), each heuristic is a specialist. trafilatura, built for
articles, is the top heuristic on WCXB (0.8584 word-F1) and drops to next-to-last on DAnIEL's
multilingual news (0.8265 ROUGE-L). readability does the reverse, first on DAnIEL (0.8925) and
last on WCXB (0.7653). resiliparse trails on both. None is wrong; each fits one policy. A
learned policy is what generalizes: base clears every heuristic on both boards, and mini sits
just behind the leader on each while staying consistent.

The transfer is directional, and our own models show it too. A WMB-trained encoder carried to
DAnIEL scores 0.9175 macro; a WCXB-trained one scores 0.8826 (`CLAIMS.md`). Training on one
policy and moving to a third board is not free, the same effect measured on collections above,
now across languages.

## Caveats

- The corpora are disjoint, so no byte-identical page exists in both; pairs are matched
  by page **kind** (and, for the article control, by domain).
- WMB has no page-type labels, and its policy on product grids is split, not uniform. Of
  the 224 WMB pages with 20 or more priced blocks, 96 drop the grid (under 10% of priced
  blocks kept), 82 keep it (over 90%), 46 are mixed. WCXB's policy on collections is
  uniform. A model trained on WMB learns the split and pays for it on every WCXB
  collection page; the heading-kept / grid-dropped pattern shown here is the WMB
  majority, not a rule.
- The table arm shows the same shape with a smaller collection gap (+25.6 pt,
  `results/generalization.md`); the encoder is the published model, so it is the one drawn.
