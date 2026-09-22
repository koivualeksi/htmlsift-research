# PROTOCOL

Metric, splits, rendering and labeling per board. Numbers carry their population
and basis; the commands that regenerate them are in `REPRODUCE.md`.

## Model

Board-agnostic (`core/`). A page's blocks are `\n`-joined and encoded in one
forward pass by a granite-embedding backbone truncated to N of its layers with a
fresh trainable final LayerNorm (parameters fp32, bf16 autocast on the forward).
Each block's tokens are mean-pooled (fast-tokenizer offset_mapping) to one vector;
a per-block head (BiGRU / linear / transformer) emits a logit; sigmoid at 0.5
selects. Pages over 8,192 tokens are encoded in 50%-overlap windows and stitched by
most-interior-token ownership -- a trained-in property, so training samples and
inference cannot diverge. Training is per-block BCE with `pos_weight`, AdamW
(encoder 2e-5 / head 1e-3), warmup then linear decay, checkpoint on best val. Every
arm runs at 3 seeds, compared on epoch means; sub-1-pt val differences are not
decision-grade. `trainers/finetune.py --benchmark <board>` generates the ablation
arms from the CLI axes and drives this loop; each board's data and val metric reach
it through that board's `Benchmark` (`vendors/<board>/adapter/benchmark.py`)
(`REPRODUCE.md`).

## WMB

**Metric.** ROUGE-5 F1 against `convert_main_content`, their `calc_rouge_n_score`
vendored and never reimplemented (`vendors/wmb/upstream/`). Both sides are tokenized
with `jieba.lcut`, n=5, arithmetic mean over records. The reference is rendered
markdown: `html2text==2025.4.15` (the scorer) over the stored `main_html`, reproduced
byte-for-byte by the `repopulate` render gate (7,809/7,809; `REPRODUCE.md`). Test is
n=544, not 545, because one board page (`2fe1c202`) carries no `cmc` reference and the
scorer reads `r.get('cmc')` and skips it.

**Split.** Domain-grouped so no domain spans folds. `tldextract` (offline snapshot)
groups the pool by registered domain, seed 20260815, 10% to val (545 domains / 732
pages), and zero train/val domain overlap is asserted, not assumed. The 545-page test
board and the 529 pages it shares with the corpus stay out of training: the pool is
7,809 minus 529, i.e. 7,280 (6,548 train + 732 val), and "no 545 `track_id` in the
pool" is an assertion in the split, not a comment.

**Rendering.** `core/render.py` (lxml), `render_hidden=False` (the default). WMB's
reference is built by a pruner that also skips `display:none`, so the model input is
rendered under the same visibility rule as the reference. The renderer, the flag and
the serializer must be identical across blocks, inference and eval-prediction
assembly, or labels do not transfer.

**Labeling.** DOM provenance by construction, with no matcher and no marker injection.
`core.render_tree`'s `line_src` maps each rendered line to the DOM elements that
produced it, and a line is positive iff its elements fall in the main set. The
`cc-select` annotation is harvested **before** sanitization strips it; inverting that
order is silent label leakage. Every page labels exactly: 7,825/7,825, 0 fallbacks.

**Sanitize.** Applied symmetrically to the model input and to label generation, so the
two cannot diverge. Eval references stay verbatim, since they define the score.

**Ceiling.** The oracle (gold labels through the DOM serializer to ROUGE-5) bounds
every model result and is the eval gate, frozen and asserted: test 0.991769, val
0.978383 (`REPRODUCE.md`). Drift in the render config, the `offset_mapping` mean-pool,
the window boundaries, the 0.5 threshold or the serializer trips it.

## WCXB

**Metric.** Word-level F1 against the plain-text `main_content` reference: tokens
`\w+` lowercased, multiset (bag-of-words) overlap, macro-mean over pages. Scored
by their `evaluate.py` verbatim (`vendors/wcxb/upstream/evaluate.py`), never
reimplemented. The reference is plain text, not rendered — unlike WMB's
`convert_main_content`. Because the metric is bag-of-words, block order does not
affect the score; labeling is pure subset selection.

**Split.** WCXB's dev/test split is used verbatim: dev = training pool (1,497),
test = eval board (511), never labeled or trained on. No custom val fold — the
reporting basis is epoch-mean at a fixed 0.5 threshold, which needs no carved val.

**Rendering.** `core/render.py` (lxml), `render_hidden=True` for WCXB. WCXB's
reference is captured from a hydrated DOM, so content shipped
`display:none`-until-hydration is main content here; WMB keeps the default
`render_hidden=False` because its reference is built by a pruner that also skips
`display:none`. The flag must be identical across blocks, inference and
eval-prediction assembly, or labels do not transfer. `adapt` maps a rendered
markdown block into the `\w+` token space (the only marker `\w+` does not already
wash out is the underscore: emphasis `_..._` dropped, escaped `\_` kept).

**Labeling** (dev only). No DOM annotations, so per-block 0/1 targets are derived:
align reference lines to blocks (length-weighted LCS + order-constrained
containment), recover split paragraphs (revcont), cover `with[]` / drop
`without[]` snippets, then hill-climb the exact word-F1 to a fixpoint. The
per-page fixpoint is the training ceiling. Every page is emitted with its ceiling;
nothing is dropped. The incremental score is cross-checked against the vendored
`word_f1` on every page and raises on drift.

**Leakage.** 139 dev/test stems are byte-identical duplicates (27% of test); the
benchmark is used as-is (upstream issues filed), `track_id` namespaces the split,
and the board number is not de-contaminated. See `LIMITS.md`.

## DAnIEL (appendix, eval-only)

Metric ROUGE-L against the `<p>`-derived gold, per language (el / en / pl / ru / zh),
macro-mean. Eval-only, never trained on except the leave-one-language-out probe
(`LIMITS.md`). Rendering is `core/render.py` with `render_hidden=False`. Labels are the
word-F1-optimal block selection (align + hill-climb) built on the shared labeling
primitives (`vendors/shared/`), scored to a frozen selection ceiling 0.9870. It carries
the Chinese claim WMB cannot support (`LIMITS.md`).

## Features (table-embedding arm, board-agnostic)

Structural columns concatenated at the BiGRU head input for the table-embedding
arm. The manifest -- column order and group map -- is `core/features.py`;
computed post-hoc from `core.render_tree`'s `line_src` (the DOM elements behind
each block), so no second parse and no renderer change. Booleans for presence;
the two scalars z-scored on train-fold stats. Gathered on the same render by
`vendors/wmb/adapter/blocks.py --feats <groups>` (e.g. `--feats ABC`) into
`data/feats.jsonl`, parallel to `blocks.jsonl`; the pass runs only when `--feats`
is given, so a run that does not want features pays nothing.

A column earns its place only if it is **absent from the tokenized block text and
not recoverable by the BiGRU**. The block vector is a mean-pool of token
embeddings -- a bag of words -- so it cannot see structure markdown flattened,
links the renderer prints as plain text, or the block's own length.

| group | columns | kind | why it is not in the text |
|---|---|---|---|
| A | `tag_*` x 30 (`main` .. `div`) | boolean | ancestor-or-self tag presence; the structure markdown flattens, and the marker the mean-pool dilutes for tags that keep one |
| B | `depth` | scalar (z-scored) | DOM nesting depth never appears in the rendered text |
| B | `has_link` | boolean | an `<a>` ancestor; the renderer prints `<a>` as bare text, so link-ness is invisible in the output |
| C | `log_chars` | scalar (z-scored) | block length -- destroyed the moment the tokens are mean-pooled |

**Dropped, with reason.** The class/id name heuristics (`cls_negative` /
`cls_positive` / `cls_byline`): the only expensive tier (per-ancestor string scan),
and the head learns boilerplate indirectly from A across pages; the archive
ablation ladder showed it wash-to-negative. The marker flags (`heading_level`,
`is_bullet`, `is_numbered`, `is_blockquote`, `is_table_row`, `is_separator_row`):
redundant with A's tag booleans and already present in the text as markdown
markers. `position`: the BiGRU carries block order.
