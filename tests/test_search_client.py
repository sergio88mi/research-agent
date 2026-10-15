"""Stage 1 tests: OpenAlex parsing, using a recorded response - no network."""
import json
import os

from unittest.mock import patch

import config
from agents import search_client
from agents.search_client import _normalise, reconstruct_abstract

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "openalex_sample.json")


def _load():
    with open(FIXTURE, encoding="utf-8") as fh:
        return json.load(fh)["results"]


def test_abstract_is_rebuilt_in_word_order():
    inverted = {"world": [1], "hello": [0], "again": [2]}
    assert reconstruct_abstract(inverted) == "hello world again"


def test_missing_abstract_stays_none():
    # Why: None is what the retrieval threshold counts as "unusable"; it must not become "".
    assert reconstruct_abstract(None) is None


def test_normalise_strips_doi_prefix_and_keeps_authors():
    paper = _normalise(_load()[0])
    assert paper.doi == "10.1007/s11704-024-40231-1"
    assert paper.authors == ["Lei Wang", "Chen Ma"]
    assert paper.year == 2024
    assert paper.abstract == "Autonomous agents use LLMs"


def test_normalise_never_invents_a_doi():
    paper = _normalise(_load()[1])
    assert paper.doi is None and paper.abstract is None


class _Resp:
    def json(self):
        return {"results": _load()}


def test_search_sends_email_and_key_only_when_configured(monkeypatch):
    # Why: OpenAlex asks for a contact address and gives a free key a larger daily budget
    # (remediation #10); neither may be sent as an empty parameter.
    monkeypatch.setattr(config, "OPENALEX_EMAIL", "")
    monkeypatch.setattr(config, "OPENALEX_API_KEY", "")
    with patch("agents.http.get_with_retry", return_value=_Resp()) as get:
        papers = search_client.search("agents", per_page=5)
    assert len(papers) == len(_load()) and all(p.source == "openalex" for p in papers)
    assert get.call_args.kwargs["params"] == {"search": "agents", "per-page": 5, "select": search_client._FIELDS}

    monkeypatch.setattr(config, "OPENALEX_EMAIL", "me@example.org")
    monkeypatch.setattr(config, "OPENALEX_API_KEY", "k")
    with patch("agents.http.get_with_retry", return_value=_Resp()) as get:
        search_client.search("other query", per_page=5)
    assert get.call_args.kwargs["params"]["mailto"] == "me@example.org"
    assert get.call_args.kwargs["params"]["api_key"] == "k"


def test_search_is_answered_from_the_cache_the_second_time():
    with patch("agents.http.get_with_retry", return_value=_Resp()) as get:
        first = search_client.search("agents")
        second = search_client.search("agents")
    assert get.call_count == 1 and [p.title for p in first] == [p.title for p in second]
