# LIMITS

Scope limits and honest caveats. Retractions and cross-board claims sit with each
board.

## Model

Board numbers are in `CLAIMS.md`. GPU is measured on one card type against one
competitor and carries one regime-matched claim (below); the mini targets the CPU
regime, and CPU/GPU numbers are never interchanged.

**Read against the select-all floor.** Word-F1 (WCXB) and ROUGE-5 (WMB) forgive
boilerplate: selecting every rendered block already scores test 0.7156 / 0.7201
(WMB / WCXB), so a board number is read as its margin over that floor, never in the
absolute (floors and per-type/-language spread in `CLAIMS.md`). DAnIEL's ROUGE-L
floor is far lower (macro 0.4984) because its LCS metric is broken by interleaved
boilerplate, making it the board that discriminates a select-everything model. The
no-encoder table-embedding arm is the one to watch (the shipped `mini` is its int8
`fABC` keeper): it can collapse to select-all and still post ~0.72 on the first-class
boards.

**Selection is val-only, and 311m depth is not decision-grade.** Architecture is chosen on val
(WMB val732); the 545 test is spent on the arms re-run with `--export-test` — a ledgered
look, not a tuning surface. Val cannot separate the top 311m depths: 10 through 22 sit
inside 0.25 pt on epoch-mean, 0.58 pt on best-epoch, well under the sub-1-pt bar. That
flatness is exactly the pattern whose archive version was refuted on test, so it is not
a result. On test the 97m half is settled: 97m-6 0.9215±0.0046 against 97m-12
0.9224±0.0038 (n=544, 3 seeds each) — flat, so the small encoder is depth-saturated. The
311m half is now 3 seeds too: 311m-22 0.9348±0.0043 against 311m-10's 0.9311±0.0027
(+0.37, n=544), but the seed spreads overlap and a gap that size is inside the noise, so
**no 311m depth claim is made in either direction** — neither "flat" nor "deeper wins".
The archive's +2.09 deep-advantage is not reproduced.

**Single-seed screens are filters, not results.** Every adopted number is 3-seed,
epoch-mean. A one-seed screen misleads at this resolution: the 97m 6-vs-7 gap read
+1.36 at one seed and +0.26 at three seeds (best-epoch), a fifth the size.
best-epoch cherry-picks a post-plateau high (val plateaus by ~epoch 3-4 while train
loss falls to zero), so epoch-mean is the reported basis.

**The global-readout observation is mid-to-deep only, not a law.** Granite's
full-attention readout depths (depth ≡ 1 mod 3) tend to be tighter and slightly
higher under fine-tuning at mid-to-deep positions, but the effect does not
generalize: at shallow depth it reverses (97m-4, a global-readout depth, is the
worst and noisiest arm), and the deepest control (311m-21) is single-seed and
unresolved. It is reported as an observation, not a mechanism to lean on.

**Retracted / not claimed.** "Doubling encoder depth loses ~1 pt" stays retracted — it
was measured under a bf16 optimiser bug that froze most of the encoder's weight mass.
This repo's own fp32 311m-22 is now scored at 3 seeds (val and test545, `CLAIMS.md`):
+0.37 over 311m-10, inside the seed noise, so it neither reproduces the archive's loss
nor establishes a gain. No GPU claim beyond the regime-matched one below. No full-set WMB number
(training uses 7,280 of 7,809; published full-set boards are context only, every
baseline re-run under this harness).

**Optimizations are separated by objective, never multiplied.** int8 (QAT) is
accuracy-neutral on the 97m-6 lite arm — +0.01 pt epoch-mean, −0.23 best-epoch, 3 seeds
each on val732 (`CLAIMS.md`); the dynamic-PTQ arm is its dead foil; int8 speed is
host-scoped, not an accuracy gate. int8 on the 311m encoder is unmeasured in every form
(no PTQ, embedding-int8 or QAT run exists for it) and the 97m evidence is not carried
across. Caps are a page-width/speed lever, accuracy-flat within a one-seed val screen —
not an accuracy win. Features are redundant once the encoder fine-tunes, now shown on
both backbones (`CLAIMS.md`), while remaining worth 1.5-3 pt frozen.

**The CPU speed number is a ratio at the heuristic tier — not a speed win.**
`bench/speed/cpu_compare.py` times the shipped mini (int8 table, ONNX) against trafilatura,
readability and resiliparse over WMB test545 (544 cmc-scored pages, one population on both
axes, CPU single-thread); `results/speed-wmb.md` and `CLAIMS.md` carry the figure. Ours
wins on F1 (0.8994 vs trafilatura's 0.7525) and sits at trafilatura's speed tier —
level-to-slightly-slower on the p50, ahead on pages under ~4k tokens, behind on the >32k
whales. Never "faster than trafilatura / resiliparse" in the absolute, and never a single
blended factor: the published figure is the measured ratio with its population and its
size-dependence. The encoder arms are slower than the table; GPU is a separate regime
with its own paragraph below.

**Threads do not change the tier.** The mini's ONNX forward parallelizes with ORT intra-op
threads (thread-safe), but end-to-end gains are sub-linear and capped: the single-threaded
lxml render+prep floor threads cannot reduce. A single extraction is threadable for us and
not for trafilatura (GIL-bound), so at an equal core budget our per-page latency can edge
ahead — but for throughput both saturate all cores (us via a thread pool on one session,
trafilatura via worker processes) and the tier is set by single-thread efficiency, where
trafilatura is level or ahead. Giving ORT more threads while pinning trafilatura to one is
the §7 landmine; the comparison is fair only at an equal core budget, which is why the
harness pins a low thread count. Threading is a latency lever, not a speed win.

**The GPU number is sustained throughput at matched accuracy, not the median-latency
ratio.** One comparison exists: the encoder keepers against MinerU-HTML v1.1 on the same
card type (RTX 4090), same 545 pages, both scored html-mode (`CLAIMS.md`,
`results/throughput-wmb.md`, `results/dripper-wmb.md`). 311m-10 ties it on F1 (0.9311±0.0027
against 0.9306) at 16× its sustained rate, 21× with band attention. Three caveats travel
with that figure. (1) The median-latency ratio is ~126× and is not the throughput ratio:
pages over the 8,192-token window are encoded window by window at batch one, so a few
whales set the sustained rate — our p90 is 11× our p50 where MinerU's is 4×. An earlier "~50×" scoping was the median
figure read as throughput; it is withdrawn. (2) Our batch-1 latency is model-only (the
CPU render and prep are timed apart) while MinerU's is the whole HTML-to-HTML call, so
the latency columns are shown, never ratioed. (3) Speed rows are one run of the seed-0
keeper with fp32 weights under bf16 autocast; band attention is bit-identical and so
shares the fp32 F1, and GPU int8 is unmeasured. No GPU figure is compared across card
types or against a CPU extractor.

## WMB

**The board metric scores a perfect extraction as zero on short pages.**
`calc_rouge_n_score` fixes `n=5`, so a reference tokenizing to fewer than 5 tokens has
no 5-grams and precision/recall/F1 collapse to 0 whatever the prediction — a correct
extraction of a one-heading page is scored identically to a wrong one. We score the
board their way regardless, since every leaderboard entry eats the same artifact, but it puts
a floor under how close any extractor can get to the ceiling.

**We train on 7,280 of 7,809.** The 529 pages sharing a `track_id` with the 545-page
test board are held out of the training pool (§4), so no full-set WMB number is
produced here and published full-set boards are context only.

**The raw html carries the annotators' selection marks.** WMB's `html` is the page as
it sat in the annotation tool, and the gold elements carry `class="mark-selected"` (with
a `cc-select` attribute and an injected stylesheet that styles it); the marker is present
in the raw html of all 545 test pages. `vendors/wmb/adapter/sanitize.py` strips it from
model input and label generation alike, so no htmlsift model ever sees it, and the eval
references are untouched. A competitor fed the raw html does see it: MinerU-HTML keeps
class attributes, so its 0.9306 (`CLAIMS.md`) is measured on input that carries the
label. The archive measured the marker's effect on MinerU at +0.22 pt with a confidence
interval spanning zero; it is not re-measured here, and the F1 tie is read with that in
mind.

## WCXB

**Annotation is LLM-assisted then human-reviewed**, not purely human like WMB's —
a disclosure line for the writeup.

**Reference captured from a hydrated DOM.** WCXB's `main_content` reflects what is
visible after JS hydration, while the shipped `html/*.html` is the pre-hydration
static page. We render WCXB with `render_hidden=True` so content shipped
`display:none`-until-hydration is included: 34% of dev pages carry hidden text and
the flag lifts the dev ceiling 0.9832 -> 0.9890. Where content is instead fetched
client-side (not merely hidden), the static HTML lacks it entirely and no render
setting recovers it.

**Residual tail: 17 dev pages below F1 0.80**, all characterized, none an
extractor bug — Discourse/SPA forums whose thread is client-fetched and absent
from the static HTML; product pages whose prose ships only in `<script>` JSON
(pre-hydration snapshot); one page with malformed HTML (a premature `</body>`
strands the article outside lxml's child chain); one whose reference is the page's
`<meta description>` SEO tagline, not body text. A browser-rendered input would
close most of this gap, but the frozen benchmark cannot be re-captured. The same
modes cap the test board (forum .979).

**139 dev/test duplicates.** 139 stems are byte-identical across dev and test (27%
of test); a model trained on dev has seen them. We use the benchmark as-is
(upstream issues filed) and do not de-contaminate the board — every leaderboard
entry eats the same split. `track_id` namespaces the collision.

**Ceilings are lower bounds.** The 0.9890 / 0.9933 ceilings are hill-climb
fixpoints — the best word-F1 block selection found, a lower bound on the exact
knapsack optimum, not a proven maximum.

## DAnIEL

**A third annotation policy.** DAnIEL's gold is `<p>`-segmented news paragraphs,
distinct from WMB's markdown `convert_main_content` (which keeps tables, lists and
formulas) and WCXB's word-level plaintext. Absolute scores are not comparable
across boards; DAnIEL measures whether a WMB-trained extractor survives an
annotation convention it never saw, not the quantity WMB or WCXB report.

**`table-abc` zero-shot normalizes DAnIEL's features by the training board's yardstick.**
The `table-abc` arm z-scores its two scalar features (`depth`, `log_chars`), and in the
zero-shot condition a WMB-trained model carries WMB train-fold stats and applies them to
DAnIEL's raw features -- foreign and multilingual (Chinese `log_chars` counts characters,
not words), normalized by a yardstick fit on WMB. This is the honest zero-shot form
(everything about the model is frozen at training and applied blind), not a per-board
refit. A leakage-free way to remove the mismatch later is an external structural-stats
estimate (e.g. random CommonCrawl pages) -- never stats fit on the eval boards themselves.
Applies only to `table-abc`; the encoder keeper is text-only.

**The scored gold is ours, not shipped.** The corpus stores gold as `<p>`-wrapped
HTML and the metric lives only in the SIGIR 2025 paper — there is no `evaluate.py`
to vendor. `collapse.py` derives the gold string: parse the fragment, drop
`<script>`/`<style>`, join each `<p>`'s text by newline, collapse intra-line
whitespace. Unlike WMB (ROUGE vendored) and WCXB (`evaluate.py` verbatim), the
DAnIEL metric is reimplemented from the paper and is labeled as such.

**Eval set is 1,689 of 2,120 golds.** 431 `reference/` files have no paired HTML
input and are dropped; only documents with both html and gold are scoreable.
Per-language: el 273 / pl 274 / ru 266 / zh 401 / en 475.

**Two small gold warts, disclosed not fixed.** One reference leaks a `<script>`
(removed by the `drop_tree` above); 13 carry the corpus's own nonstandard `&X;`
entities (`&E;`, `&Partner;` — likely name anonymization), left literal because
inventing a decoding is worse and they sit identically in the gold. 0 of the 1,689
golds cleaned to empty and 0 html files needed a charset fallback.

**Narrow and old.** 1,689 news pages, ~2012-era, one genre. It is the appendix board —
eval-only for the WMB and WCXB keepers, and the one board with enough Chinese pages
(401) to resolve what WMB cannot: the WMB zh margin does not replicate there (545 n=27
+16.46 vs val n=14 +2.24, p=0.43). Nothing else rests on it.

**The LOLO runs are the one exception to "eval-only".** `--lolo` trains on four
languages and scores the fifth (`CLAIMS.md`); those checkpoints are not keepers and no
DAnIEL-trained model is carried to another board. One caveat before anything is read
into the per-language spread: the LOLO scorer has not been checked against the
cross-board matrix's `by_lang` for metric identity, so the LOLO column and the zero-shot
column are not yet established as like-for-like.
