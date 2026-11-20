# core — scope

Board-agnostic model — encoder, heads, pooling, quant, the keeper loader, and the
training-only page helpers. Render, prep, features and band are imported from the
`htmlsift` package (htmlsift.core), never copied here. Nothing here names a benchmark; each
board wires its data and its val metric in through `vendors/<board>/adapter`. The
training spine (the epoch loops, the recipes) lives in `trainers/`.

## The model

One encoder forward per page, no decode steps. Blocks joined by `\n` -> encoder
-> mean-pool each block's tokens (fast-tokenizer offset_mapping) -> per-block
head -> sigmoid @ 0.5 -> merge consecutive positives into ranges. Whole-page
attention contextualizes each block; each vector stays anchored to its own
token span.

Encoder = a granite-embedding backbone truncated to N of its layers, pretrained
final_norm replaced by a fresh trainable LayerNorm. Params fp32 with bf16
autocast on the forward (mixed precision); resolved param dtype printed every
run. All dims read from encoder.config — nothing hardcoded to 768.

Windowing (window 8192) is trained-in: pages over the window train on
50%-stride window samples; inference stitches windows by most-interior-token
ownership, then pools and runs the head over the full page.

## Arm matrix

| encoder      | hidden | depths     | notes                                            |
|--------------|--------|------------|--------------------------------------------------|
| granite-311m | 768    | 22 / 11 / 7| fp32                                             |
| granite-97m  | 384    | 12 / 6     | 6-layer also QAT int8 + int8 embeddings, CPU-run |

Provisional published picks (configs; numbers and the selection live in
docs/CLAIMS.md): 311m-11-fp32 and 97m-6-int8. Subject to results.

## Ablation axes

- head: LR (per-block linear) · Transformer (sequence) · BiGRU (sequence; 2 hidden sizes)
- cap: per-block token cap (page width), trained-in
- features: structural columns off the same lxml render (htmlsift.core.features:
  30 tag booleans + depth + has_link + log_chars; class heuristics and marker
  flags dropped), gated by
  `blocks.py --feats` into feats.jsonl, concatenated at the head input. Spec and
  the dropped columns: docs/PROTOCOL.md

## Discipline

Every arm at 3 seeds; nothing adopted on one seed. Epoch-mean basis; sub-1-pt
val is not decision-grade. Test looks are deliberate and ledgered — root
CLAUDE.md §4 / §6.

## Files

- model.py     — encoder build, heads, pooling, inference
- loader.py    — keeper ckpt -> runnable (tok, infer_fn, kind, cap); resolve_ckpt is the only IO
- training.py  — train_page (htmlsift's prep_page + labels + training samples), train-fold
                 z-score stats, group-subset slicing
- quant.py     — int8 for the 97m-6 CPU arm: QAT (keeper) + PTQ foil + emb-table int8

From htmlsift.core: render (provenance renderer), prep_page (tokenize, block membership,
windowing, cap), features (structural columns), band (chunked band attention; bit-identical,
speed-only).

The training spine is `trainers/` (`loop.py` the two epoch loops, `recipes/` what one run does),
not here — it consumes these primitives, never inference.
