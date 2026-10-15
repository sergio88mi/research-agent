"""Stage 3 tests: LLM JSON validation + repair, the 3-5 rule, and the review-1 loop - no network."""
import json
import os
from unittest.mock import patch

import pytest

from agents import llm_client
from agents.llm_client import LLMError
from agents.models import Decision
from agents.orchestrator import Orchestrator
from agents.planning_agent import PlanningAgent, PlanResponse
from agents.researcher_interface import ResearcherInterface

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "gemini_plan.json")
GOOD = open(FIX, encoding="utf-8").read()


def test_plan_parses_recorded_response():
    with patch("agents.llm_client.generate", return_value=GOOD):
        plan = PlanningAgent().plan_research("How can hallucination be reduced?")
    assert len(plan.subquestions) == 3 and plan.subquestions[0].id == 1
    assert plan.subquestions[0].approved is False


def test_repair_retry_recovers_from_bad_json_once():
    # Why: the first reply is broken; the client must send ONE repair prompt and accept the fix.
    with patch("agents.llm_client.generate", side_effect=["{not json", GOOD]) as gen:
        plan = llm_client.generate_json("p", PlanResponse)
    assert gen.call_count == 2 and len(plan.subquestions) == 3


def test_repair_gives_up_after_configured_retries():
    with patch("agents.llm_client.generate", side_effect=["{bad", "{still bad"]):
        with pytest.raises(LLMError):
            llm_client.generate_json("p", PlanResponse)


def test_too_few_subquestions_is_rejected():
    two = json.loads(GOOD); two["subquestions"] = two["subquestions"][:2]
    with patch("agents.llm_client.generate", side_effect=[json.dumps(two), json.dumps(two)]):
        with pytest.raises(LLMError):
            llm_client.generate_json("p", PlanResponse)


def test_review_loop_revises_then_approves():
    ui = ResearcherInterface(scripted=["drop the benchmark question", "approve"])
    with patch("agents.llm_client.generate", return_value=GOOD) as gen:
        orch = Orchestrator(interface=ui)
        plan = orch._approved_plan("q")
    assert gen.call_count == 2                      # initial plan + one revision
    assert all(sq.approved for sq in plan.subquestions)


def test_scripted_interface_decisions():
    ui = ResearcherInterface(scripted=["make it about healthcare"])
    decision, fb = ui.review_subquestions("i", [])
    assert decision is Decision.REVISE and fb == "make it about healthcare"


def test_constraints_are_written_into_every_planning_and_rewriting_prompt():
    # Why: a limit the user set must not depend on the model remembering it. It is held by the
    # agent and goes into the first plan, a revised plan and a reformulated query alike.
    from agents.models import SubQuestion
    prompts = []
    def fake(prompt, **kw):
        prompts.append(prompt)
        return json.dumps({"search_query": "new query"}) if "refining one scholarly search query" in prompt else GOOD
    agent = PlanningAgent("healthcare settings only")
    with patch("agents.llm_client.generate", side_effect=fake):
        agent.plan_research("q")
        agent.revise_plan("q", "drop the benchmark question")
        agent.reformulate_query(SubQuestion(id=1, text="t", search_query="old query"), "too few records", "q")
    assert len(prompts) == 3
    assert all("User constraints (must be respected): healthcare settings only" in p for p in prompts)


def test_without_constraints_the_prompt_says_nothing_about_them():
    # Why: the cached replies of earlier runs are keyed by the prompt text; an empty constraint must not change it.
    assert "User constraints" not in PlanningAgent()._prompt("q")


def test_a_revision_shows_the_model_the_plan_being_revised():
    # Why: every model call starts from nothing. "merge sub-questions 2 and 3" means nothing unless the
    # numbered plan the researcher was looking at is in the prompt (remediation #18).
    prompts = []
    def fake(prompt, **kw):
        prompts.append(prompt)
        return GOOD
    ui = ResearcherInterface(scripted=["merge sub-questions 2 and 3", "approve"])
    with patch("agents.llm_client.generate", side_effect=fake):
        Orchestrator(interface=ui)._approved_plan("q")
    first, revision = prompts
    assert "The plan the researcher reviewed" not in first
    shown = json.loads(GOOD)["subquestions"]
    for number, sq in enumerate(shown, start=1):
        assert f"{number}. {sq['text'].strip()} (search query: {sq['search_query'].strip()})" in revision
    assert "asked for changes: merge sub-questions 2 and 3" in revision


def test_a_constraint_is_reported_in_the_brief(monkeypatch):
    # Why: the brief has to show the scope it ran under, and that the constraint was a steer, not a filter.
    from agents.models import Paper
    def fake(prompt, **kw):
        if "research planning assistant" in prompt:
            return GOOD
        n = prompt.count("] Title:")
        if "evidence notes for a literature review" in prompt:
            return json.dumps({"summaries": [{"index": i + 1, "summary": "s"} for i in range(n)]})
        if "synthesising the evidence" in prompt:
            return json.dumps({"themes": [{"statement": "t1", "paper_numbers": [1]},
                                          {"statement": "t2", "paper_numbers": [2]}], "gaps": ["g"]})
        return json.dumps({"scores": [{"index": i + 1, "relevance_score": 4, "reason": "r"} for i in range(n)]})
    papers = lambda q, per_page=None: [Paper(doi=f"10.1/{q[:3]}{i}", title=f"{q[:3]} {i}", abstract="a") for i in range(6)]
    with patch("agents.llm_client.generate", side_effect=fake), \
         patch("agents.search_client.search", side_effect=papers), \
         patch("agents.crossref_client.lookup", return_value=None), \
         patch("agents.orchestrator.save_brief", return_value=("m", "j")):
        brief = Orchestrator(interface=ResearcherInterface(scripted=["approve", "approve"]),
                             constraints="healthcare settings only").run("q")
    assert "given the constraint 'healthcare settings only'" in brief.limitations[0]
    assert "not applied as a filter" in brief.limitations[0]
