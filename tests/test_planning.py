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
