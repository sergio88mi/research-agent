"""Stage 4 tests: relevance scoring, per-batch schema check, cut-off filtering, brief output - no network."""
import json
import os
from unittest.mock import patch

import pytest

import config
from agents import llm_client
from agents.evaluation_agent import EvaluationSynthesisAgent, _response_model_for, select_papers
from agents.llm_client import LLMError
from agents.models import Brief, Paper, SubQuestion
from agents.storage import save_brief

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "gemini_scores.json")
GOOD = open(FIX, encoding="utf-8").read()

SQ = SubQuestion(id=1, text="What reduces hallucination in LLM agents?", search_query="hallucination llm agents")
PAPERS = [
    Paper(doi="10.1/a", title="Retrieval-augmented agents", abstract="We reduce hallucination...", subquestion_id=1),
    Paper(doi="10.1/b", title="Image captioning metrics", abstract="We evaluate captions...", subquestion_id=1),
    Paper(doi=None, title="A survey of grounding techniques", abstract=None, subquestion_id=1),
]


def test_scores_map_back_to_papers_and_apply_cutoff():
    with patch("agents.llm_client.generate", return_value=GOOD):
        assessments = EvaluationSynthesisAgent().evaluate_evidence("q", [SQ], PAPERS)
    assert [a.relevance_score for a in assessments] == [5, 2, 3]
    assert [a.selected for a in assessments] == [True, False, True]     # cut-off is 3
    assert assessments[2].paper_doi is None and assessments[2].paper_title == PAPERS[2].title


def test_select_papers_keeps_order_and_splits():
    with patch("agents.llm_client.generate", return_value=GOOD):
        assessments = EvaluationSynthesisAgent().evaluate_evidence("q", [SQ], PAPERS)
    selected, dropped = select_papers(PAPERS, assessments)
    assert [p.title for p in selected] == [PAPERS[0].title, PAPERS[2].title]
    assert [p.title for p in dropped] == [PAPERS[1].title]


def test_missing_index_is_a_schema_error_and_gets_repaired():
    # Why: the model must score every paper. Two of three is invalid -> one repair prompt -> accepted.
    partial = json.loads(GOOD); partial["scores"] = partial["scores"][:2]
    with patch("agents.llm_client.generate", side_effect=[json.dumps(partial), GOOD]) as gen:
        parsed = llm_client.generate_json("p", _response_model_for(3))
    assert gen.call_count == 2 and len(parsed.scores) == 3


def test_duplicate_index_is_rejected():
    dup = json.loads(GOOD); dup["scores"][1]["index"] = 1
    with patch("agents.llm_client.generate", side_effect=[json.dumps(dup), json.dumps(dup)]):
        with pytest.raises(LLMError):
            llm_client.generate_json("p", _response_model_for(3))


def test_score_out_of_range_is_rejected():
    bad = json.loads(GOOD); bad["scores"][0]["relevance_score"] = 7
    with patch("agents.llm_client.generate", side_effect=[json.dumps(bad), json.dumps(bad)]):
        with pytest.raises(LLMError):
            llm_client.generate_json("p", _response_model_for(3))


def test_papers_are_batched_per_subquestion(monkeypatch):
    # Why: 25 papers with batch size 10 must produce 3 LLM calls of 10, 10, 5.
    monkeypatch.setattr(config, "EVAL_BATCH_SIZE", 10)
    many = [Paper(doi=f"10.1/{i}", title=f"P{i}", abstract="x", subquestion_id=1) for i in range(25)]
    sizes = []

    def fake_generate(prompt, **kw):
        n = prompt.count("] Title:")
        sizes.append(n)
        return json.dumps({"scores": [{"index": i + 1, "relevance_score": 4, "reason": "r"} for i in range(n)]})

    with patch("agents.llm_client.generate", side_effect=fake_generate):
        assessments = EvaluationSynthesisAgent().evaluate_evidence("q", [SQ], many)
    assert sizes == [10, 10, 5] and len(assessments) == 25


class _Resp:
    def __init__(self, status, text='{"candidates":[{"content":{"parts":[{"text":"ok"}]}}]}', headers=None):
        self.status_code, self.text, self.headers = status, text, headers or {}
    def json(self):
        return json.loads(self.text)


def test_429_uses_retry_after_header_and_logs_quota_message(monkeypatch, caplog):
    # Why: remediation #6 - a 429 must show Google's quota message and honour its Retry-After advice.
    monkeypatch.setattr(config, "GEMINI_API_KEY", "test-key")
    body = '{"error": {"code": 429, "message": "Quota exceeded for requests per minute"}}'
    waits = []
    with patch("agents.llm_client.requests.post",
               side_effect=[_Resp(429, body, {"Retry-After": "7"}), _Resp(200)]), \
         patch("agents.llm_client.time.sleep", side_effect=waits.append), \
         patch("agents.cache.get", return_value=None), patch("agents.cache.put"):
        assert llm_client.generate("p") == "ok"
    assert waits == [7]
    assert "Quota exceeded for requests per minute" in caplog.text


def test_transient_503_is_retried_then_succeeds(monkeypatch):
    # Why: the live Stage 4 run died on a 503 "high demand" (remediation log). Two 503s then a 200 must succeed.
    monkeypatch.setattr(config, "GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(config, "CACHE_DIR", "/nonexistent-cache-dir-for-test")
    waits = []
    with patch("agents.llm_client.requests.post", side_effect=[_Resp(503), _Resp(503), _Resp(200)]) as post, \
         patch("agents.llm_client.time.sleep", side_effect=waits.append), \
         patch("agents.cache.get", return_value=None), patch("agents.cache.put"):
        assert llm_client.generate("p") == "ok"
    assert post.call_count == 3 and waits == [2, 4]        # exponential back-off


def test_read_timeout_is_retried_then_succeeds(monkeypatch):
    # Why: remediation #4 - the service took >60s under load; a timeout is transient, not fatal.
    import requests
    monkeypatch.setattr(config, "GEMINI_API_KEY", "test-key")
    with patch("agents.llm_client.requests.post", side_effect=[requests.ReadTimeout("slow"), _Resp(200)]) as post, \
         patch("agents.llm_client.time.sleep"), \
         patch("agents.cache.get", return_value=None), patch("agents.cache.put"):
        assert llm_client.generate("p") == "ok"
    assert post.call_count == 2


def test_gives_up_after_configured_transient_retries(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(config, "LLM_TRANSIENT_RETRIES", 3)     # independent of the live setting
    with patch("agents.llm_client.requests.post", return_value=_Resp(503)) as post, \
         patch("agents.llm_client.time.sleep"), \
         patch("agents.cache.get", return_value=None), patch("agents.cache.put"):
        with pytest.raises(LLMError, match="after 3 retries"):
            llm_client.generate("p")
    assert post.call_count == 4                              # 1 attempt + 3 retries


def test_daily_quota_stops_immediately_with_clear_message(monkeypatch):
    # Why: remediation #7 - a per-day quota cannot recover in a back-off window; do not burn a minute retrying.
    monkeypatch.setattr(config, "GEMINI_API_KEY", "test-key")
    body = json.dumps({"error": {"code": 429, "message": "You exceeded your current quota", "details": [
        {"@type": "type.googleapis.com/google.rpc.QuotaFailure",
         "violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}]}})
    with patch("agents.llm_client.requests.post", return_value=_Resp(429, body)) as post, \
         patch("agents.llm_client.time.sleep") as sleep, \
         patch("agents.cache.get", return_value=None), patch("agents.cache.put"):
        with pytest.raises(LLMError, match="daily free-tier quota"):
            llm_client.generate("p")
    assert post.call_count == 1 and not sleep.called


def test_non_transient_error_is_not_retried(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "test-key")
    with patch("agents.llm_client.requests.post", return_value=_Resp(404, '{"error":"no such model"}')) as post, \
         patch("agents.llm_client.time.sleep") as sleep, \
         patch("agents.cache.get", return_value=None), patch("agents.cache.put"):
        with pytest.raises(LLMError):
            llm_client.generate("p")
    assert post.call_count == 1 and not sleep.called


def test_brief_shows_reason_for_kept_and_lists_dropped(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path))
    with patch("agents.llm_client.generate", return_value=GOOD):
        assessments = EvaluationSynthesisAgent().evaluate_evidence("q", [SQ], PAPERS)
    selected, _ = select_papers(PAPERS, assessments)
    md_path, _ = save_brief(Brief(research_question="q", subquestions=[SQ],
                                  selected_papers=selected, assessments=assessments))
    text = open(md_path, encoding="utf-8").read()
    assert "relevance 5/5" in text and "Excluded after relevance screening (1)" in text
    assert "Image captioning metrics" in text.split("## Excluded")[1]
