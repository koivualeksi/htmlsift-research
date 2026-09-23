# AGENTS.md

Orientation for anyone working in this repo, human or agent.

## What this is

One job: someone reads the writeup, clones this repo, and reproduces a number.
Nothing else earns a place here. This is the reproduction harness behind
[`htmlsift`](https://github.com/koivualeksi/htmlsift), a contextual web
main-content extractor; the models are trained, benchmarked, and checked here, and
published through the `htmlsift` package.

It is a public artifact, and its credibility is partly typographic: no emoji, no
decorative headers, no marketing prose, no file that exists only to look thorough.
A comment explains *why*, never *what*. A broken invariant should raise, not be
swallowed by a fallback that hides the bug.

## Layout

One directory per benchmark under `vendors/`, split by provenance rather than by
function.

| path | contents |
|---|---|
| `vendors/<board>/upstream/` | their code, vendored verbatim. Diffs empty against the commit named in that board's `README.md`. Their licence and headers, no edits. |
| `vendors/<board>/adapter/` | our code for that board: acquire, resolve, clean, label, benchmark. Fits their data and format to this repo. |
| `vendors/<board>/data/` | corpus landing site. Gitignored, created by the adapter, never committed. |
| `vendors/<board>/README.md` | provenance: repo, path, commit, licence, and the sha256 of every acquired file. |
| `vendors/shared/` | board-agnostic data-prep and sweep config used by more than one board and never run at inference (labeling primitives, the arm matrix). |
| `core/` | board-agnostic inference primitives only: encoder, heads, pooling, render, prep, features, quant, band, the keeper loader. Nothing names a benchmark; nothing trains. |
| `trainers/` | board-agnostic entry points and their machinery: the training spine, the result ledger transport, and one file per mode (`finetune`, `frozen`, `features_only`, `embedding_table`). Runs training, never inference. |
| `bench/` | offline harnesses that consume the library layer and produce writeup numbers. `accuracy/` is cross-board F1 through the `Benchmark` seam; `speed/` is the WMB speed spine (CPU compare, GPU throughput, per-stage latency). |
| `export/` | the shipped `mini` artifact's lifecycle: graph build, the torch-free onnxruntime forward, and the parity gate that reproduces the torch mini bit-for-bit. |
| `tools/` | laptop-side data and results transport: raw board inputs up to storage, result ledgers down into `results/*.md`. Run by hand, never at inference. |
| `deploy/` | how the pinned stack is materialised to run: `deploy/pod/` (metered provisioning) and `deploy/docker/` (local images). Never imported by the reproduction pipeline. |
| `results/` | rendered result tables the writeup and README cite. Committed; raw artifacts are not. |
| `tests/` | the gate tests below, plus unit tests. |

## The reproduction gate

Gate module by module as work proceeds, not once at the end. A module is not done
until its invariant passes on this repo's own run.

| module | invariant |
|---|---|
| render | the html2text scorer reproduces `convert_main_content` from stored `main_html` byte-identically, 7,809/7,809. Model input and labels render through `core/render.py` (lxml). |
| pruner | vendored `extract_main_html` is byte-identical to upstream; composed with the scorer it reproduces the reference markdown byte-identically on the fixture. |
| corpus | no board `track_id` in the training pool; pool = 7,809 − 529 = 7,280, asserted. |
| labels | DOM-provenance labels for every page, 0 fallbacks, no matcher; harvested before sanitization strips the selection marker. |
| sanitize | applied symmetrically to model input and label generation; eval references stay verbatim. |
| split | domain-grouped, fixed seed, zero domain overlap asserted. |
| eval | ROUGE vendored, never reimplemented; oracle DOM ceiling asserted (test 0.9918 / val 0.9784). |
| model | encoder dtype resolves to fp32; every run prints its resolved parameter dtype. |
| bench | speed comparisons run only over pages both sides measured; the harness refuses otherwise. |
| export (mini) | ONNX parity reproduces the torch mini bit-for-bit through the shipped forward: 0 flips at the 0.5 threshold. |

Drift anywhere in the html2text config, the mean-pooling, the window boundaries, the
0.5 threshold, or the DOM serializer silently decouples the writeup from the artifact.

## Conventions

- **Adapt around upstream code, never inside it.** If `upstream/` is not verbatim,
  the claim that ROUGE and `extract_main_html` are vendored rather than reimplemented
  cannot be checked by anyone.
- **Acquisition always starts from the published source.** A new user has no local
  copy, and that new user is the only audience this repo has.
- **Read results down a column, never across a row.** The benchmarks encode different
  labelling policies, so cross-board numbers are policy comparisons, not model rankings.
  No model is trained on a board it is scored against.
- **No full-set WMB number.** Training uses 6,548 of 7,809; full-set boards are cited
  as context only, and every baseline is re-run under this harness.
- **Never commit data, weights, caches, or run outputs.** Corpora are read from
  configured paths. `data/` is ignored in full.

## Documents

| need | file |
|---|---|
| metric, splits, rendering, labelling per board | `docs/PROTOCOL.md` |
| every published number and the command that regenerates it | `docs/CLAIMS.md` |
| corpus acquisition, environment, run order | `docs/REPRODUCE.md` |
| scope limits, caveats, and retractions | `docs/LIMITS.md` |

## Licence

GPL-3.0-or-later. This repo imports `html2text` as the scorer (WMB's metric is
defined against its output), which is acceptable here and only here: it is a
reproduction harness, not a distributed library. That is the reason the published
package renders with lxml instead and can be Apache-2.0.
