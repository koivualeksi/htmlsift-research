# REPRODUCE

Regenerate the published artifacts from a clean clone. Read only this file to
run; the per-board `README.md` files carry provenance (source, commit, sha256)
and are reference, not run steps.

## Environment

- Python >= 3.11
- Install the reproduction dependencies (pinned):

      pip install -e ".[repro]"

  `-e` installs the repo as an editable namespace package: the repo root goes on
  `sys.path`, so `core/`, `vendors/` and `trainers/` import absolutely and
  `python trainers/<mode>.py` runs against the working tree, not an installed
  snapshot. It also resolves the pinned reproduction dependencies:
  `huggingface-hub==1.12.2`,
  `lxml==6.1.1`, `html2text==2025.4.15`, `tldextract==5.3.2`, `jieba==0.42.1`
  and `rouge_score==0.1.2`. The `html2text` and `lxml` pins are load-bearing for
  byte-identity (render and DOM serialization); `jieba` and `rouge_score` are
  load-bearing for the scores. Another version does not reproduce
  `convert_main_content` or the ROUGE numbers. (`lxml` 6.1.1 was verified
  byte-identical to the 5.4.0 baseline across the whole corpus: the lxml render
  of model input and labels, 7,825/7,825 (`wmb/blocks.py`).)

- GPU training or model eval additionally needs the torch stack:

      pip install -e ".[repro,train]"

  `torch==2.11.0`, `transformers==5.15.0`, `numpy==2.2.6`, `xgboost==3.4.1`. A
  local GPU is sufficient -- nothing here requires RunPod. On a pod or a newer
  card, install the matching `torch` from the pytorch cu128 index first.

## WMB training data

    python vendors/wmb/adapter/prepare.py

Downloads the source (~1.3 GB), builds the corpus, and writes the training data
under `vendors/wmb/data/`:

- `wmb.jsonl` — the merged, `track_id`-keyed corpus (7,825 records)
- `splits.json` — per-record source, train/val fold, and stratification `level`
- `blocks.jsonl` — per-line blocks and labels (the training data)

`prepare.py` runs `acquire -> collapse -> repopulate -> split -> blocks`, and every
stage prints its own gate result and stops the chain on failure. The `repopulate`
stage is the render gate: the `html2text==2025.4.15` scorer reproduces
`convert_main_content` from each stored `main_html` byte-for-byte, 7,809/7,809 (the
7,280-page train pool plus the 529 shared with the board). Labels are DOM provenance
by construction — `core.render`'s per-line source maps each rendered line to its
elements — so every page is labeled exactly, with no matcher and no fallback. The
final self-check asserts all 7,825 records are present:

    blocks.jsonl: 7825 pages, every page labeled (no fallback)

`acquire` downloads ~1.3 GB and `collapse` deletes the two source files after
merging, so a full re-run re-downloads. To rebuild only the blocks off an
existing `wmb.jsonl`, run `python vendors/wmb/adapter/blocks.py`; add
`--feats ABC` to also gather the structural features on the same render into
`feats.jsonl` (table-embedding arm; column spec in `PROTOCOL.md`).

## WMB model training

Selection is val-only, so a training sweep records each arm's val ROUGE-5 and
exports no test probs by default:

    python trainers/finetune.py --models 311m,97m --layers all --seeds 3

Arms are generated from the CLI axes (`--models`, `--layers`, `--heads`, `--caps`,
`--feats`); each arm x seed trains through `core` (encoder truncated to N layers +
fresh final LayerNorm + head, per-block BCE, checkpoint on best val) and appends a
val-scored row to `data/runs/wmb/finetune_results.jsonl` (`{model, layers, head,
feats, cap, seed, best_val, best_epoch, val_by_epoch, f1_by_epoch, loss_by_epoch,
frozen:False, ts}` -- the per-epoch arrays let the basis be best-epoch or epoch-mean,
chosen at analysis time from one run, never re-run per basis). `--layers` takes `full` (max
depth, the default), `all` (1..max), or a list/range like `7,11,22`. `--feats ABC`
trains the table-embedding arm — structural columns concatenated at the head,
z-scored on the train fold — and requires `feats.jsonl` (from `blocks.py --feats ABC`
above). A local CPU smoke run before any GPU spend:

    python trainers/finetune.py --models 97m --layers 6 --seeds 1 --device cpu --limit 20 --epochs 1

`--push` (with `$HF_RESULTS_DATASET`, or `--push <repo>`) uploads the results shard
and resumes from it, skipping done arms; parallel pods each own a shard (auto-derived
from the arm selection, or set with `--shard`) and `tools/download_results.py` merges
them into `results/wmb-finetune.md`. `--seeds N` runs seeds 0..N-1; `--seed-ids 1,2` runs
exactly those seeds, so a later batch extends an existing set without redoing seed 0
(records pool by seed in `download_results.py`), or one arm's seeds fan across pods (pair
with `--shard` to
keep their push files distinct). `--qat-w8a8` trains the int8 arm into a separate
`finetune_qat_results.*` ledger. GPU training (RunPod) is metered and launched by
hand.

Test is spent once on the chosen keeper, not on every arm: re-run the keeper with
`--export-test` to write `data/runs/wmb/probs/<arm>_<fold>_s<seed>.jsonl`, scored through
the eval gate below and folded into that arm's ledger row as `test_f1`.

`--train-limit` takes a comma list and runs the rungs sequentially in one process,
stratified nested prefixes of the training pool matched to val's level distribution.
It is the data-efficiency ladder in `CLAIMS.md`; val-only, so no `--export-test` and
no `--keep` (keeping a checkpoint across multiple limits raises):

    python trainers/finetune.py --models 311m --layers 10 --seeds 3 \
        --train-limit 125,250,500,1000,2000,4000 --push

`train_limit` is part of the ledger key, so a rung never collides with the full-pool
run of the same arm, and `download_results.py` renders the ladder as its own table.

## WMB evaluation

The board metric is ROUGE-5 F1 against `convert_main_content` (their
`calc_rouge_n_score`, vendored). Predictions are manufactured through the DOM
serializer, so the oracle (gold labels) bounds every model result and is the
eval gate:

    python vendors/wmb/adapter/eval.py            # test fold
    python vendors/wmb/adapter/eval.py --fold val # val fold

Expected, frozen from this repo's run and asserted on every run:

    ORACLE/test545 vs convert_main_content: n=544  P 0.9901  R 0.9945  F1 0.9918
    ceiling gate OK: 0.991769 == 0.991769
    ORACLE/val732  vs convert_main_content: n=732  P 0.9753  R 0.9858  F1 0.9784
    ceiling gate OK: 0.978383 == 0.978383

Drift in the render config, the serializer or the scorer trips the ceiling
assertion. A model checkpoint's exported per-block probabilities are scored the
same way:

    python vendors/wmb/adapter/eval.py --probs <probs.jsonl> [--fold val] [--threshold 0.5]

The select-all floor (every block selected) is scored the same way — context for a
model number, not a gate (`CLAIMS.md`):

    python vendors/wmb/adapter/eval.py --selectall [--fold val]

## WCXB training data

    python vendors/wcxb/adapter/prepare.py

Runs acquire -> collapse -> blocks -> labels and writes, under
`vendors/wcxb/data/`:

- `wcxb.jsonl` — dev+test corpus, `track_id`-keyed (2,008 records)
- `splits.json` — `track_id` -> source, fold (dev = train 1,497 / test 511)
- `blocks.jsonl` — per-page rendered blocks (2,008 pages, `render_hidden=True`)
- `labels.jsonl` — per-block 0/1 targets, dev only (1,497 pages)

Each stage asserts its invariant (stem pairing + manifest sha256 in acquire;
1,497 / 511 / 2,008 counts in collapse; 2,008 pages in blocks). The labeler's
self-check is the ceiling, frozen and asserted:

    label ceiling (mean F1): 0.9890   (dev, n=1471 scoreable)

WCXB's dev/test split is used verbatim (no custom fold); test is never labeled.
The source tree is kept — their `evaluate.py` reads it — so a re-run re-verifies
against the cache. To rebuild only blocks+labels off an existing `wcxb.jsonl`:
`python vendors/wcxb/adapter/blocks.py` then `labels.py`; add `--feats ABC` to the
`blocks.py` call to also write `feats.jsonl` (table-embedding arm).

## WCXB model training

Train arms on WCXB-dev and export their test per-block probabilities:

    python trainers/finetune.py --benchmark wcxb --models 311m,97m --layers all --seeds 3 --export-test

Same `core` loop, ledger and CLI axes as WMB (`trainers/finetune.py`), fit to WCXB
through its `Benchmark` (`vendors/wcxb/adapter/benchmark.py`): pages are built from
`blocks.jsonl` + `labels.jsonl` (dev only), and the val monitor is word-F1 against
`main_content` (text assembly, not a DOM re-render). There is no carved val fold —
dev is both the train set and the selection monitor — so the honest number is the
test board below; `--export-test` writes each arm's val+test probs to
`data/runs/wcxb/probs/<arm>_<fold>_s<seed>.jsonl` and folds test F1 into the
`finetune_results.jsonl` row. A local CPU smoke run before any GPU spend:

    python trainers/finetune.py --benchmark wcxb --models 97m --layers 6 --seeds 1 --device cpu --limit 20 --epochs 1 --export-test

## WCXB evaluation

The board metric is word-level F1 against plain-text `main_content`, scored by
their `evaluate.py` verbatim. Predictions are the selected blocks assembled into
text; the oracle (best selection) bounds every model result and is the gate:

    python vendors/wcxb/adapter/evaluate.py --split test --oracle

The oracle ceiling is frozen and asserted at **F1 0.9933** over the 511 test pages
(`ORACLE_CEILING`); the shipped `evaluate.py` prints its own per-type table.

A model's output is scored the same way — either a run's exported per-block probs
(`--probs`, thresholded and assembled into text exactly as the oracle) or a
`{file_id: text}` json (`--preds`):

    python -X utf8 vendors/wcxb/adapter/evaluate.py --split test --probs <probs.jsonl> [--threshold 0.5]
    python vendors/wcxb/adapter/evaluate.py --split test --preds <model_out.json>

`--probs` scores the whole split and asserts its count (511 test / 1,497 dev), so
a partial smoke export does not pass — it is the full-board scorer. The select-all
floor is scored the same way (`CLAIMS.md`):

    python -X utf8 vendors/wcxb/adapter/evaluate.py --split test --selectall

Both write `data/eval/wcxb_test_<tag>_preds.json`, re-scorable with the shipped
scorer beside the tree (add `-X utf8` on Windows — their loader omits `encoding=`):

    python vendors/wcxb/data/evaluate.py --split test --results <preds.json> --per-type

## DAnIEL eval board

Appendix board, eval-only, ROUGE-L against the `<p>`-derived gold (metric and
caveats in `docs/LIMITS.md`). Build the corpus, blocks and oracle labels:

    python vendors/daniel/adapter/prepare.py   # acquire -> collapse -> blocks -> labels

Writes, under `vendors/daniel/data/`:

- `daniel.jsonl` — 1,689 paired records (html + `<p>`-derived gold + language)
- `blocks.jsonl` — per-page rendered blocks (`render_hidden=False`)
- `labels.jsonl` — per-block oracle labels (align + word-F1-optimal selection)

Each stage asserts its invariant (manifest sha256 + 1,689 / 2,120 / 2,089 counts
in acquire; 1,689 and the per-language split el 273 / pl 274 / ru 266 / zh 401 /
en 475 in collapse; 1,689 in blocks and labels). The labeler's self-check is the
word-F1 selection ceiling, frozen and asserted:

    word-F1 selection ceiling: 0.9870

The board metric is ROUGE-L per language; the oracle (gold-labeled block-text
join) bounds every model result and is the gate:

    python vendors/daniel/adapter/eval.py

Expected, frozen and asserted (ROUGE-L macro F1), with the per-language table:

    ORACLE/daniel1689 macro F1 0.9816 (micro 0.9838)
      el 0.9922  pl 0.9694  ru 0.9627  en 0.9962  zh 0.9875

A model or baseline's text output is scored the same way (`--preds` is a
`{track_id: text}` json), and the select-all floor with `--selectall` (`CLAIMS.md`):

    python vendors/daniel/adapter/eval.py --preds <out.json>
    python vendors/daniel/adapter/eval.py --selectall

## DAnIEL leave-one-language-out

The one DAnIEL run that trains. `--lolo` walks all five languages in one process:
each holdout trains on the other four, scores the held-out language as its test fold,
and writes to its own `daniel-<iso>/` ledger namespace. `holdout` is part of the
ledger key. Guarded to DAnIEL, and incompatible with `--holdout` or `--keep`:

    python trainers/finetune.py --benchmark daniel --lolo \
        --models 311m --layers 10 --seeds 3 --export-test --push

A single language instead of all five:

    python trainers/finetune.py --benchmark daniel --holdout Chinese \
        --models 311m --layers 10 --seeds 3 --export-test

## Cross-board generalization

Run a keeper checkpoint over every board, each scored by that board's own metric.
`--model` is `<board>-<arm>`, split on the first hyphen to locate
`<board>/ckpt/<arm>_s<seed>.pt` in the model repo:

    python bench/accuracy/predict.py --model wmb-311m-10 --boards wmb,wcxb,daniel --push

Rows land in `generalization_results.jsonl` at the dataset root. Records carry `n`;
a run that did not cover a whole board is a smoke and `download_results.py` refuses
to render the ledger until it is removed.

## CPU speed comparison (§7)

`bench/speed/cpu_compare.py` times our model against the third-party extractors (trafilatura,
resiliparse, readability) on WMB CPU-side and scores each through the board metric.
Speed is extraction only, html2text off the clock (§7). Every method is scored over one
population — the cmc-scored pages — on both axes: an empty return is timed and scores as
an empty prediction (F1 0 vs non-empty gold), only a crash is excluded (asterisk). The
published figure is the **ratio** with its size dependence, never an absolute: a cgroup
CPU cap fixes the thread count, not the host microarchitecture, so absolute ms does not
carry across machines. On CPU we are not faster than the heuristics (`LIMITS.md`); the
ratio is reported honestly and size-bucketed. The published figure is the shipped mini
(`--onnx`, `CLAIMS.md`); an encoder keeper via `--ckpt`/`--model` is slower and not the
CPU competitor.

Run it in the pinned container so the render stack (Python 3.13 + lxml 6.1.1, the §4
render-gate versions) and the extractor versions are fixed, and the report records the
CPU model, the visible core count and the pinned versions it ran on:

    docker build -f deploy/docker/Dockerfile -t htmlsift-bench .
    docker run --rm --cpus=2 --cpuset-cpus=0,1 -v "$PWD/vendors/wmb/data:/repo/vendors/wmb/data" -v "$PWD/311m-10_s0.pt:/repo/ckpt.pt" htmlsift-bench --ckpt /repo/ckpt.pt --fold test --threads 2

Data and the keeper `.pt` mount at run time, never baked (§2); on Git Bash prefix
`MSYS_NO_PATHCONV=1` and use forward-slash host paths. `--cpuset-cpus` pins which cores
and `--cpus` how many. The vCPU count is part of the claim, not just hygiene: our ONNX
forward gains on more cores (ORT intra-op) while a single trafilatura call cannot (GIL-bound),
so handing ORT more threads than trafilatura would manufacture an unfair win (§7) — pin the
same low count for both. A modest 2 is the conservative choice; the single-threaded lxml
render+prep floor caps the end-to-end gain regardless. The run writes `results/speed-wmb.md`
for the `--onnx` mini, or `results/speed-wmb-<model>.md` for an encoder keeper.

## GPU throughput vs MinerU-HTML v1.1 (§7)

Two harnesses, two pods, one card type. `bench/speed/gpu_throughput.py` batches an encoder
keeper over WMB test545 and reports sustained pages/s (full pipeline and model-only) with
batch-1 p50/p90; `bench/speed/gpu_dripper.py` runs MinerU-HTML v1.1 through vLLM over the
same pages and scores it html-mode through `WMB.score_text`. The competitor's stack (vLLM
0.11.1, torch 2.9) collides with `[train]`, so it always runs on its own pod from the
`vllm/vllm-openai` image (`pyproject.toml` `[bench-gpu]`, `deploy/pod/pod_dripper.sh`). The
comparison holds only on the same card type — the published rows are all RTX 4090:

    python deploy/pod/launch.py --bench gpu_throughput --gpu "<card>" --name thr311  --args "--model 311m-10 --device cuda"
    python deploy/pod/launch.py --bench gpu_throughput --gpu "<card>" --name thr311b --args "--model 311m-10 --device cuda --band"
    python deploy/pod/launch.py --bench gpu_throughput --gpu "<card>" --name thr97q  --args "--model 97m-6-qat --device cuda"
    python deploy/pod/launch.py --dripper --gpu "<card>" --name dripper-v11 --args "--device cuda"

Both harnesses print their report and write it to `results/` on the pod; nothing is pushed.
Pull the pod logs off the volume and copy the tables into `results/throughput-wmb.md`
(one row per arm; the writer merges rows) and `results/dripper-wmb.md`:

    python tools/upload_data.py --stage logs --run thr311      # likewise thr311b, thr97q, dripper-v11

Speed rows are one run of the seed-0 keeper; the F1 beside them is the 3-seed predict ledger
(`results/generalization.md`). Pods are metered and human-launched (§2).

## Results tables

Every ledger on the HF dataset renders to `results/`:

    python tools/download_results.py

Eight files: the WMB frozen screen, WMB fine-tune and its QAT ledger, the three WCXB
ledgers, the DAnIEL LOLO holdouts, and the generalization matrix. Val tables render
once per basis — epoch-mean (the §6 selection basis) and best-epoch (what the kept
checkpoint realizes) — with the epoch budget as a row dimension, since neither basis
is comparable across budgets. Full-pool runs, the train-limit ladder and the LOLO
holdouts are separate tables: one cell never averages two populations.
