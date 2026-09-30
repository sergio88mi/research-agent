"""Stage 7 tests: several sources per search, automatic failover, and the run-level tally - no network."""
from unittest.mock import patch

import pytest
import requests

import config
from agents.models import Paper, SubQuestion
from agents.retrieval_agent import RetrievalAgent

SQ = SubQuestion(id=3, text="t", search_query="q")


def _papers(prefix, n, source):
    return [Paper(doi=f"10.1/{prefix}{i}", title=f"{prefix} {i}", abstract="a", source=source) for i in range(n)]


def test_every_configured_source_is_searched_and_results_are_merged(monkeypatch):
    monkeypatch.setattr(config, "SOURCES", ["openalex", "semanticscholar"])
    with patch("agents.search_client.search", return_value=_papers("oa", 3, "openalex")) as oa, \
         patch("agents.s2_client.search", return_value=_papers("s2", 2, "semanticscholar")) as s2:
        agent = RetrievalAgent()
        found = agent.retrieve_evidence(SQ)
    assert oa.call_args.args == ("q",) and s2.call_args.args == ("q",)
    assert [p.source for p in found] == ["openalex"] * 3 + ["semanticscholar"] * 2
    assert all(p.subquestion_id == 3 for p in found)
    assert agent.source_stats == {"openalex": {"searches": 1, "failed": 0, "records": 3},
                                  "semanticscholar": {"searches": 1, "failed": 0, "records": 2}}
    assert agent.failed_sources() == {}


def test_one_source_failing_does_not_stop_the_search(monkeypatch):
    # Why: this is the failover the tutor's feedback asked for - a dead or rate-limited API
    # must cost coverage for that query, not the whole run.
    monkeypatch.setattr(config, "SOURCES", ["openalex", "semanticscholar"])
    with patch("agents.search_client.search", side_effect=requests.HTTPError("openalex unavailable after 4 retries")), \
         patch("agents.s2_client.search", return_value=_papers("s2", 4, "semanticscholar")):
        agent = RetrievalAgent()
        found = agent.retrieve_evidence(SQ)
    assert len(found) == 4 and all(p.source == "semanticscholar" for p in found)
    assert agent.failed_sources() == {"openalex": 1}
    assert "openalex: 0 of 1 searches answered, 0 records" in agent.sources_summary()
    assert "semanticscholar: 1 of 1 searches answered, 4 records" in agent.sources_summary()


def test_a_failed_source_is_tried_again_on_the_next_query(monkeypatch):
    # Why: an outage may clear between sub-questions; the agent must not write a source off for the run.
    monkeypatch.setattr(config, "SOURCES", ["openalex", "semanticscholar"])
    with patch("agents.search_client.search", side_effect=[requests.ConnectionError("down"), _papers("oa", 2, "openalex")]) as oa, \
         patch("agents.s2_client.search", return_value=_papers("s2", 1, "semanticscholar")):
        agent = RetrievalAgent()
        first = agent.retrieve_evidence(SQ)
        second = agent.retrieve_evidence(SubQuestion(id=4, text="t", search_query="q2"))
    assert oa.call_count == 2 and len(first) == 1 and len(second) == 3
    assert agent.source_stats["openalex"] == {"searches": 2, "failed": 1, "records": 2}


def test_all_sources_failing_stops_the_run_with_the_reasons(monkeypatch):
    monkeypatch.setattr(config, "SOURCES", ["openalex", "semanticscholar"])
    with patch("agents.search_client.search", side_effect=requests.HTTPError("HTTP 503")), \
         patch("agents.s2_client.search", side_effect=requests.ConnectionError("refused")):
        agent = RetrievalAgent()
        with pytest.raises(RuntimeError, match="every retrieval source failed for sub-question 3.*openalex.*semanticscholar"):
            agent.retrieve_evidence(SQ)


def test_unknown_source_name_is_a_configuration_error(monkeypatch):
    monkeypatch.setattr(config, "SOURCES", ["openalex", "scopus"])
    with pytest.raises(ValueError, match="scopus"):
        RetrievalAgent()


def test_non_network_errors_are_not_swallowed(monkeypatch):
    # Why: failover is for services being unavailable; a bug in a client must still surface.
    monkeypatch.setattr(config, "SOURCES", ["openalex", "semanticscholar"])
    with patch("agents.search_client.search", side_effect=KeyError("results")), \
         patch("agents.s2_client.search", return_value=[]):
        with pytest.raises(KeyError):
            RetrievalAgent().retrieve_evidence(SQ)


def test_brief_records_failover_when_a_source_failed(monkeypatch):
    # Why: the design's traceability rule - a brief must say what it could not cover. A source
    # that failed mid-run narrows coverage, so it has to appear in the limitations.
    import json, os
    from agents.orchestrator import Orchestrator
    from agents.researcher_interface import ResearcherInterface
    plan = open(os.path.join(os.path.dirname(__file__), "fixtures", "gemini_plan.json"), encoding="utf-8").read()

    def fake(prompt, **kw):
        if "research planning assistant" in prompt:
            return plan
        n = prompt.count("] Title:")
        if "evidence notes for a literature review" in prompt:
            return json.dumps({"summaries": [{"index": i + 1, "summary": "s"} for i in range(n)]})
        if "synthesising the evidence" in prompt:
            return json.dumps({"themes": [{"statement": "t1", "paper_numbers": [1]},
                                          {"statement": "t2", "paper_numbers": [1]}], "gaps": ["g"]})
        return json.dumps({"scores": [{"index": i + 1, "relevance_score": 4, "reason": "r"} for i in range(n)]})

    monkeypatch.setattr(config, "SOURCES", ["openalex", "semanticscholar"])
    with patch("agents.llm_client.generate", side_effect=fake), \
         patch("agents.search_client.search", side_effect=lambda q, per_page=None: _papers(q[:3], 6, "openalex")), \
         patch("agents.s2_client.search", side_effect=requests.HTTPError("semanticscholar unavailable after 4 retries")), \
         patch("agents.crossref_client.lookup", return_value=None), \
         patch("agents.orchestrator.save_brief", return_value=("m", "j")):
        brief = Orchestrator(interface=ResearcherInterface(scripted=["approve", "approve"])).run("q")
    joined = " ".join(brief.limitations)
    assert "openalex: 3 of 3 searches answered, 18 records" in joined
    assert "semanticscholar: 0 of 3 searches answered, 0 records" in joined
    assert "Automatic failover was used: semanticscholar failed for 3 search(es)" in joined
