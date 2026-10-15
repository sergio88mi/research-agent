"""Disk cache tests: what was stored comes back, and nothing else does."""
import os
from unittest.mock import patch

import pytest

import config
from agents import cache


def test_a_request_never_seen_before_is_a_miss():
    assert cache.get("openalex", "some query|20") is None


def test_put_then_get_round_trips_and_keeps_non_ascii_text():
    value = {"results": [{"title": "Étude des agents – 智能体"}], "count": 1}
    cache.put("openalex", "q|20", value)
    assert cache.get("openalex", "q|20") == value


def test_the_same_request_to_two_services_is_stored_separately():
    # Why: OpenAlex and Semantic Scholar are asked the same query string; one must not answer for the other.
    cache.put("openalex", "q|20", {"results": ["a"]})
    cache.put("semanticscholar", "q|20", {"data": ["b"]})
    assert cache.get("openalex", "q|20") == {"results": ["a"]}
    assert cache.get("semanticscholar", "q|20") == {"data": ["b"]}


def test_the_cache_directory_is_created_on_first_write():
    assert not os.path.exists(config.CACHE_DIR)
    cache.put("crossref", "10.1/x", {})
    assert os.path.isdir(config.CACHE_DIR) and cache.get("crossref", "10.1/x") == {}


def test_a_damaged_cache_file_counts_as_not_asked_before():
    # Why: a run stopped half-way through a write must not make every later run of that request fail (remediation #20).
    cache.put("openalex", "q|20", {"results": ["a"]})
    path = cache._key("openalex", "q|20")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write('{"results": ["a"')                      # cut off mid-file
    assert cache.get("openalex", "q|20") is None
    cache.put("openalex", "q|20", {"results": ["b"]})    # and it can be written again
    assert cache.get("openalex", "q|20") == {"results": ["b"]}


def test_an_interrupted_write_keeps_the_previous_answer_and_leaves_nothing_behind():
    # Why: Ctrl-C can arrive in the middle of a write. The answer already stored must survive it.
    cache.put("openalex", "q|20", {"results": ["a"]})

    def cut_off(value, fh, **kwargs):
        fh.write('{"results": ["b"')
        raise KeyboardInterrupt

    with patch("agents.cache.json.dump", side_effect=cut_off):
        with pytest.raises(KeyboardInterrupt):
            cache.put("openalex", "q|20", {"results": ["b"]})
    assert cache.get("openalex", "q|20") == {"results": ["a"]}
    assert [name for name in os.listdir(config.CACHE_DIR) if not name.endswith(".json")] == []
