"""Stage 5 tests: retrieval threshold, one-reformulation cap, review 2 (approve / refine / reject scope) - no network."""
import json
import os
from unittest.mock import patch

import config
from agents.models import Decision, Paper, SubQuestion
from agents.orchestrator import Orchestrator
from agents.researcher_interface import ResearcherInterface
from agents.retrieval_agent import RetrievalAgent

PLAN = open(os.path.join(os.path.dirname(__file__), "fixtures", "gemini_plan.json"), encoding="utf-8").read()
SQ = SubQuestion(id=1, text="t", search_query="q")


def _papers(n, prefix, with_abstract=True):
    return [Paper(doi=f"10.1/{prefix}{i}", title=f"{prefix} {i}", abstract="a" if with_abstract else None,
                  year=2024, authors=["A"]) for i in range(n)]


def _scores(prompt, score):
    n = prompt.count("] Title:")
    return json.dumps({"scores": [{"index": i + 1, "relevance_score": score, "reason": "r"} for i in range(n)]})


def _llm(score_by_call):
    """Fake Gemini: plan JSON, a new query for reformulation prompts, scores for scoring prompts."""
    def fake(prompt, **kw):
        if "research planning assistant" in prompt:
            return PLAN
        if "refining one scholarly search query" in prompt:
            return json.dumps({"search_query": "better query"})
        return _scores(prompt, score_by_call(prompt))
    return fake


def test_threshold_is_a_count_of_records_with_abstracts():
    agent = RetrievalAgent()
    assert agent.assess_threshold(_papers(config.RETRIEVAL_THRESHOLD, "x")) is True
    assert agent.assess_threshold(_papers(config.RETRIEVAL_THRESHOLD - 1, "x")) is False
    assert agent.assess_threshold(_papers(20, "x", with_abstract=False)) is False


def test_trigger1_reformulates_once_when_too_few_abstracts():
    # Why: sub-question 1's first search returns 2 usable records -> one reformulation, then no more.
    calls = []
    first_query = json.loads(PLAN)["subquestions"][0]["search_query"]
    def fake_search(query, per_page=None):
        calls.append(query)
        if query == first_query:
            return _papers(2, "few")                       # below RETRIEVAL_THRESHOLD
        if query == "better query":
            return _papers(8, "more")
        return _papers(8, query[:3])                       # sub-questions 2 and 3 are fine
    ui = ResearcherInterface(scripted=["approve", "approve"])
    with patch("agents.llm_client.generate", side_effect=_llm(lambda p: 4)), \
         patch("agents.search_client.search", side_effect=fake_search), \
         patch("agents.crossref_client.lookup", return_value=None), \
         patch("agents.orchestrator.save_brief", return_value=("m", "j")):
        brief = Orchestrator(interface=ui).run("q")
    sq1 = brief.subquestions[0]
    assert sq1.retried_once and sq1.previous_query == first_query and sq1.search_query == "better query"
    assert calls.count("better query") == 1                # reformulated exactly once
    assert all(not sq.previous_query for sq in brief.subquestions[1:])   # others untouched
    assert len(brief.selected_papers) == 2 + 8 + 8 + 8     # both searches for sq1 are kept


def test_trigger2_reformulates_when_too_few_relevant_and_respects_cap():
    # Why: 20 abstracts pass trigger 1, but every paper scores 1 -> trigger 2 fires once per sub-question,
    # and even though the reformulated search is *also* all-irrelevant, there is no second retry.
    reformulations = []
    def fake(prompt, **kw):
        if "refining one scholarly search query" in prompt:
            reformulations.append(prompt)
            assert "typical rejections" in prompt         # the drop reasons were fed back
            return json.dumps({"search_query": "better query"})
        return _llm(lambda p: 1)(prompt)
    ui = ResearcherInterface(scripted=["approve", "approve"])
    with patch("agents.llm_client.generate", side_effect=fake), \
         patch("agents.search_client.search", side_effect=lambda q, per_page=None: _papers(20, q[:3])), \
         patch("agents.crossref_client.lookup", return_value=None), \
         patch("agents.orchestrator.save_brief", return_value=("m", "j")):
        orch = Orchestrator(interface=ui)
        brief = orch.run("q")
    assert len(reformulations) == 3                        # one per sub-question, never two
    assert all(sq.retried_once for sq in brief.subquestions)
    assert sum(orch.retry_counts.values()) == 3 and brief.selected_papers == []


def test_review2_refine_removes_numbered_papers():
    ui = ResearcherInterface(scripted=["approve", "drop 1, 3", "approve"])
    with patch("agents.llm_client.generate", side_effect=_llm(lambda p: 4)), \
         patch("agents.search_client.search", side_effect=lambda q, per_page=None: _papers(6, q[:3])), \
         patch("agents.crossref_client.lookup", return_value=None), \
         patch("agents.orchestrator.save_brief", return_value=("m", "j")):
        brief = Orchestrator(interface=ui).run("q")
    assert len(brief.selected_papers) == 18 - 2
    removed = [a for a in brief.assessments if "removed by the researcher" in a.reason]
    assert len(removed) == 2 and not any(a.selected for a in removed)


def test_review2_reject_scope_replans_once_then_option_disappears():
    # Script: approve plan -> reject evidence (+feedback) -> approve revised plan -> approve evidence.
    ui = ResearcherInterface(scripted=["approve", "reject", "focus on healthcare", "approve", "approve"])
    plans = []
    def fake(prompt, **kw):
        if "research planning assistant" in prompt:
            plans.append(prompt)
            return PLAN
        return _llm(lambda p: 4)(prompt)
    with patch("agents.llm_client.generate", side_effect=fake), \
         patch("agents.search_client.search", side_effect=lambda q, per_page=None: _papers(6, q[:3])), \
         patch("agents.crossref_client.lookup", return_value=None), \
         patch("agents.orchestrator.save_brief", return_value=("m", "j")):
        orch = Orchestrator(interface=ui)
        brief = orch.run("q")
    assert orch.scope_revision_count == 1
    assert len(plans) == 2 and "focus on healthcare" in plans[1]      # feedback reached the planner
    assert any("revised the scope once" in l for l in brief.limitations)


def test_review2_reject_refused_when_budget_used(capsys):
    ui = ResearcherInterface(scripted=["reject", "approve"])
    decision, _ = ui.review_evidence([SQ], [], [], can_reject_scope=False)
    assert decision is Decision.APPROVE
    assert "has been used" in capsys.readouterr().out


def test_review2_numbering_matches_display_order():
    # Why: 'drop 2' must remove the paper printed as "2." - the orchestrator relies on the same order.
    ui = ResearcherInterface(scripted=["drop 2"])
    papers = [Paper(doi=f"10.1/p{i}", title=f"P{i}", subquestion_id=1) for i in range(3)]
    decision, fb = ui.review_evidence([SQ], papers, [], can_reject_scope=True)
    assert decision is Decision.REFINE and Orchestrator._parse_numbers(fb) == [2]
