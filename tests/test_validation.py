"""Stage 2 tests: de-duplication, title matching, and verification outcomes - no network."""
import json
import os
from unittest.mock import patch

import requests

from agents import validation
from agents.models import Paper

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "crossref_sample.json")


def _paper(doi, title="A survey on large language model based autonomous agents"):
    return Paper(doi=doi, title=title)


def test_deduplicate_keeps_first_and_keeps_doiless():
    ps = [_paper("10.1/a"), _paper("10.1/A"), _paper(None, "no doi"), _paper("10.1/b")]
    out = validation.deduplicate_by_doi(ps)
    assert [p.doi for p in out] == ["10.1/a", None, "10.1/b"]   # case-insensitive dedupe


def test_titles_match_ignores_case_and_punctuation():
    assert validation.titles_match("A Survey on LLM-Based Agents!", "a survey on llm based agents")
    assert not validation.titles_match("Completely different title", "a survey on llm based agents")


def test_verified_when_crossref_title_matches():
    record = json.load(open(FIX, encoding="utf-8"))
    with patch("agents.crossref_client.lookup", return_value=record):
        [p] = validation.validate_dois_and_metadata([_paper("10.1007/s11704-024-40231-1")])
    assert p.doi_verified is True


def test_flagged_when_doi_missing_or_unregistered():
    with patch("agents.crossref_client.lookup", return_value=None):
        a, b = validation.validate_dois_and_metadata([_paper(None), _paper("10.9999/nope")])
    assert a.doi_verified is False and b.doi_verified is False


def test_unchecked_when_network_fails():
    # Why: "could not check" must stay distinguishable from "checked and wrong".
    with patch("agents.crossref_client.lookup", side_effect=requests.ConnectionError("down")):
        [p] = validation.validate_dois_and_metadata([_paper("10.1/x")])
    assert p.doi_verified is None
