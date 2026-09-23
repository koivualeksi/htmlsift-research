# CLAIMS

Every published number and the command that regenerates it. Populations:
`test545` / `val732` (WMB), `test511` / `dev1497` (WCXB), `daniel1689`. Ceilings and
floors are label-selection numbers; model numbers carry their own basis, n and seed
count on every row.

## WMB oracle ceilings

The 545 board metric is ROUGE-5 F1 against `convert_main_content` (their
`calc_rouge_n_score`, vendored). The oracle is gold DOM labels through the
serializer — not a hill-climb — so it bounds every model result and is the eval
gate. Frozen from this repo's run of the lxml provenance pipeline:

**Test DOM oracle: F1 0.991769** (test545, n=544 — `2fe1c202` carries no `cmc`).
P 0.9901 / R 0.9945.

    python vendors/wmb/adapter/eval.py            # test fold

**Val DOM oracle: F1 0.978383** (val732, n=732). P 0.9753 / R 0.9858.

    python vendors/wmb/adapter/eval.py --fold val

The html2text-pipeline baseline was 0.994356 / 0.982753; the lxml renderer costs
-0.26% test / -0.44% val.

## WCXB ceilings

**Test oracle ceiling: F1 0.9933** (test511, their `evaluate_results`,
`render_hidden=True`). Per type: article .998, product .994, collection .993,
documentation .992, service .992, listing .986, forum .979.

    python vendors/wcxb/adapter/evaluate.py --split test --oracle

**Dev label ceiling: mean F1 0.9890** (dev1471 scoreable, `render_hidden=True`).
Per type: article .996, service .995, documentation .992, product .976, forum
.980, listing .969, collection .967.

    python vendors/wcxb/adapter/prepare.py     # or labels.py off an existing wcxb.jsonl

Both are label-selection ceilings — the best word-F1 block selection reaches, a
lower bound on the exact (knapsack) optimum.

## Select-all floors

Every rendered block selected, scored through each board's own metric — the floor a
model must clear, since a model that just selects everything lands here. Recall is
~0.98 across every board; the floor is set by precision, i.e. how much boilerplate
the page carries. A baseline, not a gate — no assertion.

| board | population | n | P | R | F1 |
|---|---|---|---|---|---|
| WMB — ROUGE-5 vs `convert_main_content` | test545 | 544 | 0.6218 | 0.9873 | **0.7156** |
| WMB | val732 | 732 | 0.5265 | 0.9776 | **0.6359** |
| WCXB — word-F1 | test511 | 511 | 0.6094 | 0.9886 | **0.7201** |
| WCXB | dev1497 | 1497 | 0.5871 | 0.9782 | **0.6971** |
| DAnIEL — ROUGE-L macro | daniel1689 | 1689 | 0.3626 | 0.9825 | **0.4984** |
| DAnIEL — ROUGE-L micro | daniel1689 | 1689 | 0.3680 | 0.9849 | **0.5049** |

    python vendors/wmb/adapter/eval.py --selectall [--fold val]
    python -X utf8 vendors/wcxb/adapter/evaluate.py --split test --selectall   # or --split dev
    python vendors/daniel/adapter/eval.py --selectall

WMB and WCXB sit near 0.72 on test; DAnIEL's floor is ~0.50 because ROUGE-L (LCS) is
broken by interleaved boilerplate where bag-of-words word-F1 and short-n-gram ROUGE-5
are not — so DAnIEL discriminates a select-everything model far harder. Per type /
language the floor swings wide (WCXB test article 0.807 vs collection 0.490; DAnIEL
en 0.543 vs ru 0.372), so a select-all model's mean hides that it is useless on
list- and boilerplate-heavy pages.

## Model boards

Selection is val-only. The test fold is spent on the arms re-run with `--export-test`
(`REPRODUCE.md`), which is why its seed count is its own and is stated per row. Every
number below is this repo's own run. The full sweeps regenerate into `results/`:

    python tools/download_results.py

### WMB test545 — ROUGE-5 F1 vs `convert_main_content`, n=544

| arm | epochs | seeds | F1 | P | R |
|---|---|---|---|---|---|
| 311m-10 | 4 | 3 | **0.9311±0.0027** | 0.9306 | 0.9573 |
| 311m-10 | 8 | 3 | 0.9308±0.0022 | 0.9305 | 0.9577 |
| 311m-22 | 8 | 3 | 0.9348±0.0043 | 0.9363 | 0.9590 |
| 97m-6 | 8 | 3 | 0.9215±0.0046 | 0.9233 | 0.9507 |
| 97m-6-qat (W8A8) | 4 | 3 | 0.9256±0.0048 | — | — |
| 97m-12 | 8 | 3 | 0.9224±0.0038 | 0.9222 | 0.9526 |
| mini — 311m-table-bigru-fABC (int8 table) | — | 3 | 0.9010±0.0031 | — | — |

Read against the select-all floor 0.7156 and the DOM oracle ceiling 0.9918 above.
Every row is the arm's own `--export-test` run except the 97m-6-qat and mini rows,
whose numbers come from the `bench/accuracy/predict.py` ledger over the saved keeper
(pod render, 3 seeds; `results/generalization.md`). That ledger reports F1 only, so
P and R are shown only on the `--export-test` rows. The published models are `base`
(311m-10) and the mini; 97m-6-qat is the unpublished lite arm. The QAT training export read 0.9261±0.0046 on the same W8A8
weights; the 0.05 pt gap is run-to-run GPU noise, and only the predict figure is cited.
97m-12 against 97m-6 is +0.09 pt at 3 seeds each — flat. 311m-22 at 3 seeds is
0.9348±0.0043, +0.37 over 311m-10 but with overlapping spreads — not decision-grade;
see `LIMITS.md` on depth.

### WMB val732 — the selection surface, ROUGE-5 F1, full pool, 8 epochs, 3 seeds

epoch-mean is the §6 basis; best-epoch is what the kept checkpoint realizes.

| arm | epoch-mean | best-epoch |
|---|---|---|
| 311m-10 | 0.8940±0.0011 | 0.8986±0.0028 |
| 311m-11 | 0.8915±0.0019 | 0.8971±0.0024 |
| 311m-12 | 0.8941±0.0009 | 0.9016±0.0042 |
| 311m-13 | 0.8965±0.0009 (n=2) | 0.9029±0.0006 (n=2) |
| 311m-22 | 0.8943±0.0024 | 0.9009±0.0014 |
| 97m-6 | 0.8819±0.0012 | 0.8903±0.0008 |
| 97m-6-qat (W8A8) | 0.8820±0.0009 | 0.8880±0.0026 |
| 97m-12 | 0.8908±0.0012 | 0.8953±0.0024 |

**int8 QAT is accuracy-neutral** on the 97m-6 arm: +0.01 pt on epoch-mean, −0.23 on
best-epoch, same population, same 8-epoch budget, 3 seeds each.

**Post-hoc int8 on the 97m-6 fp32 master** (best-epoch val732 / test545 @0.5, n=544, 3 seeds):

| variant | val732 | test545 |
|---|---|---|
| fp32 | 0.8903±0.0008 | 0.9215±0.0046 |
| emb-int8 | 0.8901±0.0004 | 0.9216±0.0047 |
| PTQ-dynamic | 0.8865±0.0005 | 0.9177±0.0025 |

emb-int8 (the embedding table only) is flat; PTQ-dynamic loses 0.38 pt on both folds and is
CPU-only dynamic int8 with no real speedup, so it is the dead foil — QAT above, retrained and
accuracy-neutral, is the int8 that ships. Regenerate with `--quant-eval` (reads `ptq_`/`emb_int8_`
ledger fields).

**Structural features are redundant once the encoder fine-tunes**, on both backbones
(epoch-mean, 8 epochs, 3 seeds): 311m-10 none 0.8940 vs `fABC` 0.8942; 97m-12 0.8908 vs
0.8911. They are not redundant frozen — see the frozen screen in `results/`.

### WCXB test511 — word-F1 by their `evaluate.py`, n=511

| arm | epochs | seeds | F1 | P | R |
|---|---|---|---|---|---|
| 311m-10 | 4 | 3 | **0.9209±0.0007** | 0.9214 | 0.9404 |
| 97m-6-qat (W8A8) | 4 | 3 | 0.9096±0.0009 | 0.9142 | 0.9331 |

Floor 0.7201, ceiling 0.9933. WCXB carves no val fold, so the selection monitor is
dev1497 (0.9357±0.0014 epoch-mean / 0.9628±0.0007 best-epoch for 311m-10).

### Data efficiency — WMB val732, 4 epochs, 3 seeds, 311m-10

Stratified nested prefixes of the 6,548-page training split (`--train-limit`). The ceiling
row is the same arm on the full split at the same epoch budget: 0.8966±0.0035 epoch-mean /
0.9024±0.0035 best-epoch.

| pages | epoch-mean | best-epoch |
|---|---|---|
| 125 | 0.8079±0.0035 | 0.8349±0.0060 |
| 250 | 0.8421±0.0077 | 0.8614±0.0037 |
| 500 | 0.8624±0.0012 | 0.8690±0.0034 |
| 1000 | 0.8640±0.0241 | 0.8829±0.0032 |
| 2000 | 0.8824±0.0016 | 0.8887±0.0029 |
| 4000 | 0.8868±0.0044 | 0.8972±0.0017 |

### DAnIEL leave-one-language-out — ROUGE-L F1, 311m-10, 4 epochs

Train on four languages, score the fifth, so n is that language's page count and the
number is out-of-language, not in-domain. Metric is ours, not vendored (`LIMITS.md`).

| held out | n | seeds | F1 |
|---|---|---|---|
| Greek | 273 | 3 | 0.9555±0.0083 |
| Polish | 274 | 3 | 0.9174±0.0030 |
| Russian | 266 | 3 | 0.9021±0.0042 |
| English | 475 | 3 | 0.9507±0.0005 |
| Chinese | 401 | 3 | 0.9595±0.0027 |

### Cross-board generalization

Each keeper run over every board through `bench/accuracy/predict.py`, scored by that board's own
metric. Comparable down a column, never across a row. The matrix is large; regenerate it
rather than reading it here:

    python bench/accuracy/predict.py --model wmb-311m-10 --boards wmb,wcxb,daniel --push
    python tools/download_results.py            # -> results/generalization.md

Headline, test folds, 3 seeds: `wmb-311m-10` scores WMB 0.9311±0.0027, WCXB
0.8633±0.0016, DAnIEL macro 0.9175±0.0027. Trained the other way round, `wcxb-311m-10`
scores WCXB 0.9209±0.0007, WMB 0.8479±0.0107, DAnIEL 0.8826±0.0027 — a WMB-trained
model loses 5.76 pt moving onto WCXB, a WCXB-trained one loses 8.32 pt moving onto WMB,
and the WMB-trained model is ahead on the neutral third board.

The WCXB per-type rows for the WMB-trained encoders (`results/generalization.md`) locate
the gap: `wmb-311m-10` against `wcxb-311m-10` is −34.9 pt on collection, −10.7 on product,
−9.2 on listing, −3.8 on article, and level on forum and documentation (+0.7 each) — the
two boards' annotation policies disagree on list-shaped pages, not on prose.

### Heuristics on WCXB and DAnIEL (zero-shot)

The same third-party extractors scored on the two boards outside the WMB training policy, so
the accuracy claim can be read off-policy. Our models are 3-seed keepers from the
generalization ledger (scored on their selected-block text); the heuristics are single
deterministic runs of their native text output, except readability, which has no text mode and
is flattened (`vendors/shared/html_text.py`). One population per board, the block-scored pages;
an empty return scores 0. Comparable down a column, never across.

    python bench/accuracy/heuristics.py --board wcxb
    python bench/accuracy/heuristics.py --board daniel

WCXB test511 (word-F1, n=511) and DAnIEL 1,689 (ROUGE-L, macro over five languages):

| extractor | WCXB | DAnIEL |
|---|---|---|
| base (311m-10, 3 seeds) | 0.8633±0.0016 | 0.9175±0.0027 |
| mini (int8 table, 3 seeds) | 0.8474±0.0089 | 0.8797±0.0025 |
| trafilatura | 0.8584 | 0.8265 |
| readability | 0.7653 | 0.8925 |
| resiliparse | 0.7909 | 0.7094 |

resiliparse empty on 10/511 (WCXB) and 1/1,689 (DAnIEL), scored 0; trafilatura empty on 3/511
(WCXB). base clears every heuristic on both boards. Each heuristic tops one board and drops on
the other: trafilatura leads WCXB and is next-to-last on DAnIEL, readability the reverse. mini
sits just behind the board leader on each and ahead of the other two. Dev cross-check
(`results/heuristics-wcxb-dev.md`): our trafilatura 0.8132 and resiliparse 0.7711 land within
~2-3 pt of the public WCXB leaderboard (0.791 / 0.797), a version and metric-detail gap. Full
tables and pins in `results/heuristics-wcxb.md` and `results/heuristics-daniel.md`.

### CPU speed + F1 vs third-party extractors

The shipped mini (`311m-table-bigru-fABC` s1, int8 table + BiGRU head, ONNX) against
trafilatura, readability and resiliparse — CPU single-thread (`onnxruntime intra=1`), WMB
test545. One population, the 544 cmc-scored pages, on both speed and F1: an empty return is
timed and scores as an empty prediction (F1 0 vs non-empty gold); only a crash is excluded.
Full table, pins and size-dependence in `results/speed-wmb.md`:

    python bench/speed/cpu_compare.py --onnx --model 311m-table-bigru-fABC --seed 1 --fold test --threads 1

**Accuracy is the win.** ROUGE-5 F1 (the board metric, html2text of each extractor's HTML):
ours **0.8994**, readability 0.8016, trafilatura 0.7525, resiliparse 0.7135 (empty on 4 of
544, scored 0) — +14.7 pt over trafilatura, the strongest heuristic on speed.

**Speed is the heuristic tier, not faster than it.** On the p50 ours sits level with
trafilatura and a shade slower (trafilatura ≈0.9× our time); readability is ~1.7-2× and
resiliparse ~15-20× faster, both far behind on F1. The ratio is size-dependent: ours beats
trafilatura on pages under ~4k tokens and trafilatura pulls ahead on the >32k whales. No
absolute "N× faster" claim — see `LIMITS.md` / §7. The encoder arms are slower than the
table and are not the CPU competitor.

The figures above are single-thread (`--threads 1`). ORT intra-op threads speed the forward
but not the single-threaded lxml render+prep floor, so end-to-end gains are sub-linear and
capped; a threaded-ORT-vs-single-thread-trafilatura comparison is the §7 landmine and is not
run. Threading is a latency lever, not a tier change (`LIMITS.md`).

### GPU throughput vs MinerU-HTML v1.1 — WMB test545, RTX 4090

The encoder keepers against MinerU-HTML v1.1 (HunYuan 0.5B, the Dripper lineage, vLLM
0.11.1, max context 131,072), both on an RTX 4090 over the same 545 pages, both scored
html-mode through `WMB.score_text`. Speed rows are one run of the seed-0 keeper (fp32
weights, bf16 autocast, batch ≤64, B×T cap 8,192); the F1 beside each is the 3-seed
predict ledger above. Full pipeline is CPU render + prep + GPU model, serial; MinerU's
call is HTML in, HTML out and cannot be split, so only the full-pipeline rows compare.
Tables and regenerate lines in `results/throughput-wmb.md` and `results/dripper-wmb.md`:

    python bench/speed/gpu_throughput.py --model 311m-10 --device cuda [--band]
    python bench/speed/gpu_throughput.py --model 97m-6-qat --device cuda
    python bench/speed/gpu_dripper.py --device cuda        # its own vLLM pod, deploy/pod/pod_dripper.sh

| system | sustained pg/s, full pipeline | batch-1 p50 | batch-1 p90 | F1 |
|---|---|---|---|---|
| MinerU-HTML v1.1 | 1.16 | 844 ms | 3,343 ms | 0.9306 (1 run) |
| 311m-10 | 18.4 | 6.7 ms, model only | 73.6 ms | 0.9311±0.0027 |
| 311m-10, band attention | 24.6 | 7.4 ms, model only | 45.2 ms | same weights, bit-identical |
| 97m-6-qat (fp32 forward) | 30.2 | 4.2 ms, model only | 37.8 ms | 0.9256±0.0048 |

**Throughput at matched accuracy is the claim.** 311m-10 ties MinerU-HTML on F1 (0.9311
against 0.9306, inside one seed spread) at 16× its sustained rate, 21× with band. The
median-latency ratio (~126×) is not quoted as a throughput figure: whale pages set the
sustained rate — pages over the 8,192-token window are encoded window by window, our p90
is 11× our p50 where theirs is 4×. Our batch-1 latency is model-only and theirs is the whole call,
so the latency columns are shown side by side and not ratioed. The 97m-6-qat runs its fp32
masters here (int8 is a CPU lever; GPU int8 is unmeasured). Same card type is the
condition: no GPU number is compared across cards or against a CPU extractor.
