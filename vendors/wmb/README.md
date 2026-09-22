# vendors/wmb — WebMainBench

Training pool and first-class board: ROUGE-5 F1 against `convert_main_content`,
their function verbatim, on the 545.

## Source

| | |
|---|---|
| dataset | https://huggingface.co/datasets/opendatalab/WebMainBench |
| licence | Apache-2.0 |
| benchmark repo | https://github.com/opendatalab/WebMainBench |
| eval code and html2text config | https://github.com/opendatalab/MinerU-HTML |
| paper | https://arxiv.org/abs/2511.16397 |

## What acquire downloads

| file | records | role |
|---|---|---|
| `webmainbench.jsonl` | 7,809 | training pool source |
| `WebMainBench_545.jsonl` | 545 | test board, never trained on |

Pinned revision `5da0972e9b58d0c7891ae75053ced97c268f52e3` (last modified
2026-04-08). Recorded in `adapter/acquire.py` so a mismatch raises, and repeated
here for the reader:

| file | bytes | sha256 |
|---|---|---|
| `webmainbench.jsonl` | 1,354,734,941 | `85765fe798f07c14eb1c92945046eaa56e0da59663f70b9c498647d7dfd78884` |
| `WebMainBench_545.jsonl` | 109,097,918 | `0efaa4b49a45e320a27fe6e5a0b6aad5b57259fc3321ac3448519cacc74c537e` |

Run `python vendors/wmb/adapter/acquire.py` to land both in `data/` (gitignored).

## Vendored in upstream/

`main_html.py` — verbatim from the WebMainBench toolkit
(`webmainbench/utils/main_html.py`), Apache-2.0, commit
`9d991bdc00c57b57521499494d96be85c31317ba`. No edits: clone WebMainBench at that
commit and the diff is empty. sha256
`d78988e7c283c09a75a3d4e80d4c0bf096817af3ade13c18fdc8b3d80d398a9e`.

It provides both halves of their ground-truth pipeline: `extract_main_html` (the
cc-select DOM pruner) and `HTML2TextWrapper` (the html2text render config —
`bodywidth=0`, `ignore_links`, `ignore_images`, `baseurl`). Its
`LICENSE.webmainbench` sits beside it.

`rouge_utils.py` — verbatim from a *different* upstream, MinerU-HTML
(`eval_baselines/utils.py`), Apache-2.0, commit
`73cf266690befd209cae7e6fdff9716d5b31a976`. No edits: clone MinerU-HTML at that
commit and the diff is empty. sha256
`38009f6ea1e0f942dddfb6a0d6d36c677097d43193b698768deaa67f4260200d`. Its own
`LICENSE.mineru-html` sits beside it.

It is the WMB scoring function, `calc_rouge_n_score`: ROUGE-N (n=5) over
`jieba.lcut` tokens on both sides via `rouge_score` internals. Vendored, never
reimplemented — this is the metric the 545 board is scored on.

Leakage, the null-reference trap, the annotation residue and the reference
invariant live in `docs/PROTOCOL.md` and are not restated here.

## Adapter pipeline

`adapter/` turns the downloaded source into training data. One command runs the
whole chain from the published source:

    python vendors/wmb/adapter/prepare.py

It writes `data/blocks.jsonl` (the per-line training data) alongside
`data/wmb.jsonl` (the merged corpus) and `data/splits.json` (source + fold). The
stages, and why each exists:

| stage | what it does | why |
|---|---|---|
| `acquire` | downloads the two source files at the pinned revision, sha256-checked | reproduction starts from the published source, byte-verified |
| `collapse` | merges them into one `wmb.jsonl` keyed by `track_id`, recording each record's source in `splits.json` | the files overlap (529 shared) and neither is a superset (16 test-only); one keyed corpus fixes the training pool at 7,809 − 529 = 7,280 and removes the leakage ambiguity |
| `repopulate` | re-renders `main_html` and checks it reproduces the stored `convert_main_content` byte-for-byte; fills the 16 test-only records | proves our html2text version and config match theirs exactly before anything is built on it |
| `split` | domain-grouped train/val fold (seed 20260815), zero domain overlap asserted | pages from one site must not straddle train and val, or the validation set leaks |
| `blocks` | renders each sanitized page to markdown lines and labels each line main/junk | the per-line training data; a line is main iff one of its DOM source elements is cc-selected, exact provenance with no marker injection and no matcher fallback (structural lines with no source take a deterministic neighbour rule) |

`sanitize` is not a separate stage: it is the residue stripper `blocks` calls to
clean each page before rendering (annotation marks are the label and must not
reach the model; the translation-extension elements are boilerplate in no real
page). Run on its own it is a self-test, not part of the chain.

Each stage asserts its own invariant and stops the chain on failure. Individual
stages run standalone for partial rebuilds; `docs/REPRODUCE.md` has the run order
and the expected output.
