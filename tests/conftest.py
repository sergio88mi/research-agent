"""Shared test setup.

Why a single source by default: the Stage 1-6 tests were written when there
was one search source and they patch `agents.search_client.search` only.
Since Stage 7 the default is two sources, so without this fixture those tests
would call Semantic Scholar for real. Every test therefore starts with a
single source; the Stage 7 tests (test_retrieval.py) set config.SOURCES
explicitly to exercise several.

Why a throw-away cache directory: the clients read and write the disk cache.
A test must never be answered from, or leave a file in, the cache/ folder the
real runs use.
"""
import pytest

import config


@pytest.fixture(autouse=True)
def single_source_by_default(monkeypatch):
    monkeypatch.setattr(config, "SOURCES", ["openalex"])


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CACHE_DIR", str(tmp_path / "cache"))
