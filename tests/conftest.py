"""Shared pytest configuration.

Both tiers run on a clean checkout with no corpus: unit (pure logic on authored
inputs) and gate (byte-identity against the committed fixtures under
tests/fixtures). Corpus-scale invariants are asserted in the pipeline stages
themselves -- each stage raises on a broken gate -- not re-checked here.
"""
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="session")
def fixtures_dir():
    return FIXTURES
