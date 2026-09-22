# Tests

The equivalence gate as a fast, offline test suite: the §4 mechanism invariants
that couple the writeup to the artifact, checked in seconds on tiny authored
fixtures. Corpus-scale reproduction is not re-checked here -- each pipeline stage
asserts its own gate and raises on failure (see the last section).

## Running

    pip install -e ".[repro,test]"   # render/score libs (lxml, numpy, ...) + pytest
    pytest                          # unit + gate, no corpus, seconds

## Tiers

- **unit** -- pure logic on authored minimal HTML / synthetic arrays.
- **gate** -- byte-identity against the committed fixtures in `fixtures/`.

Board scope is the filename prefix; fixtures live in `fixtures/<source>/`.

### unit -- `tests/unit/`

| file | locks | §4 |
|---|---|---|
| `test_render.py` | `render_hidden` toggle, hidden-subtree-but-tail, `line_src` provenance, empty input | render, labels |
| `test_sanitize.py` | cc-select harvested before sanitize (leakage guard), residue removal, layer-2 render-neutral, layer-1 drop, notranslate kept | sanitize, labels |
| `test_html2text_state.py` | fresh `HTML2TextWrapper` per page; reuse leaks list state | render |
| `test_rouge.py` | vendored `calc_rouge_n_score` (n=5) and `word_f1` / `tokenize` | eval |
| `test_features.py` | `feature_dim`, train-fold z-scoring, constant-column std floor | features |
| `test_model.py` | `build_encoder` fp32 + N-layer truncation, head `d_in`, `stitch_bounds` ownership | model |

### gate -- `tests/gate/` (fixtures in `tests/fixtures/`)

| file | locks | §4 |
|---|---|---|
| `test_render_features.py` | render blocks **and** feature vectors byte-identical | render |
| `test_pruner.py` | `extract_main_html` + scorer end-to-end markdown byte-identical | pruner |
| `test_split.py` | `domain_of` eTLD+1 grouping key | split |

## Corpus-scale gates live in the pipeline

The full-corpus invariants are asserted where they are produced -- running the
pipeline **is** the gate (§4: "Reproduction is not a test look") -- so they are
not duplicated as tests:

| invariant | asserted by |
|---|---|
| source sha256 / manifest (all three boards) | `acquire.py` (raises on mismatch) |
| corpus counts 7825 / 7280 / 529 / 16 | `wmb/collapse.py` |
| scorer reproduces `convert_main_content`, full set 7809/7809 | `wmb/repopulate.py` (`assert not mismatched`) |
| domain split 545 domains / 732 pages, zero straddle | `wmb/split.py` |
| labels 7825, no fallback | `wmb/blocks.py` |
| oracle ceilings (WMB test/val, WCXB test/dev) | `wmb/eval.py`, `wcxb/evaluate.py`, `wcxb/labels.py` |
