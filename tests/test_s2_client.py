"""Stage 7 tests: Semantic Scholar parsing and request shape, using a recorded response - no network."""
import json
import os
from unittest.mock import patch

import config
from agents import s2_client
from agents.s2_client import _clean_abstract, _normalise

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "semanticscholar_sample.json")


def _load():
    with open(FIXTURE, encoding="utf-8") as fh:
        return json.load(fh)


def test_normalise_takes_doi_from_external_ids_and_tags_the_source():
    paper = _normalise(_load()["data"][0])
    assert paper.doi == "10.1109/WiSEE57913.2025.11229863"
    assert paper.authors == ["Zainab Rafique", "Muhammad Wasim"]
    assert paper.year == 2025 and paper.source == "semanticscholar"


def test_abstract_whitespace_is_collapsed_and_empty_stays_none():
    # Why: Semantic Scholar pads some abstracts with newlines; the threshold must not count an empty one.
    paper = _normalise(_load()["data"][0])
    assert paper.abstract == "Large Language Models (LLMs) have quickly pushed the frontiers of autonomous agents."
    assert _clean_abstract("\n \n") is None and _clean_abstract(None) is None


def test_normalise_never_invents_a_doi():
    paper = _normalise(_load()["data"][1])
    assert paper.doi is None and paper.abstract is None and paper.authors == []


def test_search_sends_key_header_only_when_configured(monkeypatch):
    class _Resp:
        def json(self):
            return _load()
    monkeypatch.setattr(config, "S2_API_KEY", "")
    with patch("agents.cache.get", return_value=None), patch("agents.cache.put"), \
         patch("agents.http.get_with_retry", return_value=_Resp()) as get:
        papers = s2_client.search("agents", per_page=2)
    assert len(papers) == 2 and get.call_args.kwargs["headers"] is None
    assert get.call_args.kwargs["params"] == {"query": "agents", "limit": 2, "fields": s2_client._FIELDS}

    monkeypatch.setattr(config, "S2_API_KEY", "k")
    with patch("agents.cache.get", return_value=None), patch("agents.cache.put"), \
         patch("agents.http.get_with_retry", return_value=_Resp()) as get:
        s2_client.search("agents")
    assert get.call_args.kwargs["headers"] == {"x-api-key": "k"}


def test_search_uses_the_cache_before_the_network():
    with patch("agents.cache.get", return_value=_load()), patch("agents.http.get_with_retry") as get:
        papers = s2_client.search("agents")
    assert len(papers) == 2 and not get.called
