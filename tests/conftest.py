"""Shared test setup.

Why: the Stage 1-6 tests were written when there was one search source and
they patch `agents.search_client.search` only. Since Stage 7 the default is
two sources, so without this fixture those tests would call Semantic Scholar
for real. Every test therefore starts with a single source; the Stage 7 tests
(test_retrieval.py) set config.SOURCES explicitly to exercise several.
"""
import pytest

import config


@pytest.fixture(autouse=True)
def single_source_by_default(monkeypatch):
    monkeypatch.setattr(config, "SOURCES", ["openalex"])
