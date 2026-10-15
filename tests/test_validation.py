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


def test_titles_in_other_scripts_are_compared_not_emptied():
    # Why: remediation #14. With letters outside a-z stripped, two different Chinese titles both
    # became "" and "matched", so a wrong DOI would have been reported as verified.
    assert not validation.titles_match("大型语言模型智能体综述", "图像描述评价指标研究")
    assert validation.titles_match("大型语言模型智能体综述", "大型语言模型智能体综述")
    assert validation.titles_match("Обзор агентов на основе LLM", "обзор агентов на основе llm!")


def test_a_missing_title_never_matches():
    assert not validation.titles_match("", "")
    assert not validation.titles_match(None, "a survey on llm based agents")
    assert not validation.titles_match("!!!", "???")


def test_flagged_when_a_non_latin_title_differs_from_the_registry(monkeypatch):
    p = _paper("10.1/zh", "大型语言模型智能体综述")
    with patch("agents.crossref_client.lookup", return_value={"title": ["图像描述评价指标研究"]}):
        validation.validate_dois_and_metadata([p])
    assert p.doi_verified is False


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
