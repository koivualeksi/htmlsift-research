# Test fixtures

Committed inputs and golden outputs for the `unit` and `gate` tiers, the one
carve-out from the corpus ignore (`.gitignore`: `!tests/**/fixtures/**`). Keep
every file small enough to read in the diff — authored minimal HTML and the
blocks / feature vectors / labels it is expected to produce, never a copied
corpus page. Whole-fold reproduction reads the real corpus and lives in
`tests/full`, not here.

Layout: one subdirectory per source — `core/`, `wmb/`, `wcxb/`, `daniel/`.
