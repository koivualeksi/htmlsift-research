# Objective changes on the mini

The shipped mini (`experiments/wmb-mini.yaml`) trains with one loss: masked BCE over rows, with
pos_weight balancing the classes by character weight. The WMB metric is something else: a per-page
ROUGE-5 F1 over characters, averaged over pages. A row-level loss treats a stray 2,000-character
block the same on a page with 300 gold characters as on a page with 20,000, while the metric
punishes the first far harder. This section records which changes to the training objective close
that gap and which do not. The model, the features (ABC), the inputs and the data stay fixed
throughout.

Every arm and seed is in `results/wmb-mini-objective.md`, rendered from the run store:

    python tools/render_tables.py

Each arm is one spec under `experiments/`:

    python -m trainers.train experiments/wmb-mini-pf-confirm.yaml

## Protocol

- **Selection uses val732 only.** test545 is reported, never used to choose.
- **Noise floor.** Two runs of the same commit and seed on the same card differ by about 0.006
  val and 0.009 test, because the GPU is not deterministic. So a single seed cannot resolve a
  difference of 0.005.
- **Seeds.** A screen runs 3 seeds. An arm that leads at 3 seeds is confirmed at 6 or more
  before it is called a result.
- **Greedy order.** Changes are stacked one at a time on the best base so far. Each arm is
  compared against the base it was added to, not against the shipped mini.

## What was tried

| change | mechanism | outcome |
|---|---|---|
| `ema` 0.995 / 0.999 | Exponential moving average of the head's weights. Val scoring, epoch selection and the kept checkpoint use the average. | Gain. The largest single step: val +0.008. |
| `smooth_weight` 1 / 3 / 10 | (p_i - p_i+1)^2 over adjacent rows with the same gold label. It pulls isolated flips back to their neighbours and leaves gold edges free. | Weight 1 gains on top of EMA. Weight 3 ties it; weight 10 loses on val. |
| `page_f1` 0.5 / 1 / 2 / 4 / 8 | 1 minus a char-weighted soft F1 per page, added to the BCE: the metric's own unit. A page with no gold scores 100 / (selected chars + 100). | Gain at every weight. Weight 4 is best on val. Weight 8 has a similar test score but a lower val. |
| `fp_weight`, `fp_box`, `box_mass` | Hard negatives: a gold-free row scored above 0.5 costs more, everywhere or only inside gold-free containers. `box_mass` pushes whole gold-free boxes down. | Ties or losses. |
| `focal_gamma` 2 | Focal loss. | Loses. |
| `len_cap` / `len_floor` | Weights each row's loss by its length, clipped. | Loses. |
| `page_norm` | Makes the BCE a mean over pages. | Ties. |
| `label_smooth` 0.05 | BCE targets 0.975 / 0.025. | Ties alone and with EMA. |
| `feat_drop` 0.3 | Zeroes the feature columns on 30 % of pages per epoch. | Ties: no measurable effect with or without `page_f1` at 6 seeds. |
| `text_drop` 0.3 | Zeroes the text columns on 30 % of pages per epoch. | Loses. |
| `head_lr` 5e-4 / 1e-3 / 4e-3 | Learning rate, against the 2e-3 default. | All lose. |
| `epochs` 6, `ema` 0.9995 | A longer horizon for the average. | Ties: no gain. |

## Result

The chosen arm is `ema 0.999 + smooth_weight 1 + page_f1 4`. ROUGE-5 F1 on val732, the selection
fold, mean ± std over seeds:

| arm | seeds | val |
|---|---|---|
| 0.2.x recipe | 3 | 0.8613±0.0026 |
| ema 0.999 + smooth_weight 1 | 6 | 0.8706±0.0026 |
| + page_f1 4 | 9 | 0.8739±0.0023 |

The gain over the 0.2.x recipe is +1.3 pt val. On test545 the change is read off the delivered
models, both through today's package (`CLAIMS.md`): the 0.2.x mini 0.8975 against the 0.3.0 mini
0.9114, +1.4 pt. That difference also carries the rank-128 table and a different seed.

Why `page_f1` helps:
- It is the only change that sees the page. BCE charges each row on its own, so it cannot tell
  that a stray block on a page with little gold hurts more than the same block on a long page.
- `page_f1` charges that block what the metric charges it.
- The hard-negative variants aim at the same false positives row by row and do not reach them.

Why the gain is real:
- At 9 seeds, every seed of the chosen arm is above the 6-seed base's mean on val.
- On test545 (`results/wmb-mini-objective.md`), 8 of the 9 seeds are above that base's test mean
  and the lowest equals it.
- The int8 table loses nothing, so the gain survives the 0.2.x numerics. The 0.3.0 numerics, the
  table factored to rank 128, cost 0.0008 on WMB test at the shipped seed (`CLAIMS.md`).

The arm was selected on WMB alone. Off WMB the delivered 0.3.0 mini gains on DAnIEL and is level
on WCXB, as the next section shows.

## Off-board checks

The checkpoints are `wmb/ckpt/311m-table-bigru-fABC-nb1-pf4-ema0.999_s{0,1,2}.pt` in the models
repo. Off WMB the change is read off the delivered models, zero-shot, both through today's package
(`output="text"`, `CLAIMS.md`):

| board | 0.2.x mini (s1) | 0.3.0 mini (s2) |
|---|---|---|
| WCXB test511, word-F1 | 0.8528 | 0.8529 |
| DAnIEL 1,689, ROUGE-L macro | 0.8818 | 0.8925 |

- **WCXB.** Level (+0.0001).
- **DAnIEL.** +1.1 pt.
- **Common Crawl.** Yield is in `results/ccbench.md`, 0.3.0 mini only.

## Limits

- **One recipe.** Every arm is the table + BiGRU recipe. Whether `page_f1` helps the encoder
  arms is untested.
- **Selection on WMB.** The objective was chosen on WMB val. The off-board checks are
  single-seed evaluations of the delivered models, not a second selection.
- **Shipped at rank 128.** `htmlsift` 0.3.0 ships this arm, seed 2, with its table factored to
  rank 128 (`CLAIMS.md`). The val numbers in this document are the full-width table (`CLAIMS.md`).
