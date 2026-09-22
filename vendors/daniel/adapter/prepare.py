"""
End-to-end DAnIEL data preparation: acquire -> collapse -> blocks -> labels.
Turns the published source into the eval corpus and its label ceiling
(data/daniel.jsonl, data/blocks.jsonl, data/labels.jsonl). Each stage asserts
its own invariant and raises on failure, stopping the chain.

    python vendors/daniel/adapter/prepare.py

Eval-only, so there is no split stage -- like WCXB, DAnIEL's gold is external
(<p>-derived) with no fold to carve. blocks renders each page and labels aligns
the gold to the blocks to the word-F1 fixpoint; both are needed for eval.py's
oracle ceiling gate. Individual stages stay runnable for partial rebuilds.
"""

from vendors.daniel.adapter import (
    acquire,
    collapse,
    blocks,
    labels
)

STAGES = [
    ("acquire", acquire.acquire),
    ("collapse", collapse.main),
    ("blocks", blocks.main),
    ("labels", labels.main),
]


def main():
    for name, run in STAGES:
        print(f"\n=== {name} ===", flush=True)
        run()
    print("\ndaniel data prep complete: data/daniel.jsonl, "
          "data/blocks.jsonl, data/labels.jsonl")


if __name__ == "__main__":
    main()
