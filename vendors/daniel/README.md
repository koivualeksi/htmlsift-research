# vendors/daniel — DAnIEL

Appendix board, **eval-only, never training**. 1,689 news pages with paired HTML:
el 273, pl 274, ru 266, zh 401, en 475. It carries the Chinese claim that WMB
cannot support, where the zh margin does not replicate.

The metric is not shipped with the corpus. The SIGIR 2025 paper that benchmarks
extractors on DAnIEL scores ROUGE-L plus precision/recall/F1 per language; there
is no `evaluate.py` to vendor, so the adapter implements the metric and
`docs/LIMITS.md` discloses it.

## Source

| | |
|---|---|
| corpus | https://github.com/rundimeco/waddle — path `corpora/Corpus_daniel_v2.1` |
| licence | GPL-3.0 (whole `waddle` repo); compatible with this repo's GPL-3.0-or-later |
| paper | https://dl.acm.org/doi/10.1145/3726302.3730234 (SIGIR '25) |
| paper PDF | https://maurelf.users.greyc.fr/docs/conferences/SIGIR_2025_paper_1968.pdf |

## What acquire downloads

DAnIEL lives in a GitHub repo, not on HuggingFace, so `acquire.py` pulls the repo
tarball at pinned commit `be875ce12ad9e23ef68ef7f3b0605db6d06bcdb1` (last pushed
2023-02-02) and extracts only the `corpora/Corpus_daniel_v2.1` subtree into
`data/` (gitignored).

| item | count | role |
|---|---|---|
| `doc_lg.json` | 2,089 entries | `{doc_id: language}` map |
| `html/` | 1,689 files | raw HTML inputs, named `YYYYMMDD_domain_hash`, no extension |
| `reference/` | 2,120 files | main-content gold, `<p>`-wrapped paragraphs, same naming |

The eval set is the **1,689** documents with both an HTML input and a gold
reference (every `html/` file is paired; 431 `reference/` files have no HTML and
are unusable). The full 2,120/2,089 counts are the original corpus, built for
epidemic-event detection; only the 1,689 HTML-paired subset is a content
extraction benchmark.

GitHub's generated tarball wrapper is not byte-stable, so integrity is anchored
on file *contents*, the way WCXB anchors its tree: `acquire` computes the sha256
of the sorted `sha256  relpath` manifest of every extracted file and asserts it
equals

    7f5723fb20a7d533bf6d2334a09e3917746ac5d0a58bbe50dc7f01a1a9203502

recorded as `MANIFEST_SHA256` in `adapter/acquire.py`; one byte anywhere trips it.
acquire also asserts the three counts above and that every `html/` file has a
paired `reference/`. The tree is kept, not deleted — it is the eval input and gold.

    python vendors/daniel/adapter/acquire.py

## Vendored in upstream/

Nothing. Unlike WMB (`main_html.py`, `rouge_utils.py`) and WCXB (`evaluate.py`),
the DAnIEL corpus ships no scoring code — the metric exists only in the SIGIR
paper. There is no upstream file to diff, so nothing is vendored; the metric is
implemented in `adapter/` and disclosed.

Gold here is `<p>`-segmented main text — a third annotation policy, distinct from
WMB's markdown `convert_main_content` and WCXB's word-level plaintext. That
divergence is a disclosure line in the writeup; see `docs/LIMITS.md`.

## Adapter pipeline

`adapter/` turns the downloaded corpus into the eval board. Eval-only, so there is
no split and no training data — the chain builds the reference corpus, the
rendered blocks, and the oracle labels that set the ceiling.

| stage | what it does | why |
|---|---|---|
| `acquire` | downloads the tarball at the pinned commit, manifest-sha256-checked | reproduction starts from the published source, byte-verified |
| `collapse` | pairs html to reference, attaches language, cleans the `<p>` gold to text → `daniel.jsonl` (1,689) | one keyed corpus; drops the 431 reference-only orphans; the gold transform is defined and disclosed (`docs/LIMITS.md`) |
| `blocks` | renders each page to markdown lines → `blocks.jsonl` | the block units a classifier sees; `render_hidden=False` (static ~2012 pages) |
| `labels` | aligns the gold to the blocks and hill-climbs a word-F1-optimal selection → `labels.jsonl` | the oracle selection; its word-F1 ceiling **0.9870** gates labeler/render drift; reuses `vendors/shared/labeling.py`, jieba-tokenized for zh |
| `eval` | ROUGE-L per language; oracle = gold-labeled block-text join | the board metric; oracle macro F1 **0.9816** bounds every model result and is the gate |

`prepare.py` runs acquire → collapse; `blocks`, `labels` and `eval` run
standalone. `docs/REPRODUCE.md` has the run order and expected output.
