"""Board-agnostic inference constants. Dependency-free on purpose: the torch-free
`[repro]` scorers (vendors/*/adapter/eval*.py) import from here, so they share the
exact values the torch training and inference paths use. core/model.py imports torch,
so it cannot be that shared home."""

# Sigmoid decision threshold (§4). One source for the whole pipeline: keeper selection
# (trainers/train.py and score_val), the board eval CLIs (--threshold default), and the
# shipped extractor. Selecting and reporting on the same value is the invariant; drift
# here silently decouples the writeup from the artifact.
THRESHOLD = 0.5
