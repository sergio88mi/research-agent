"""
Planning Agent - decomposition and query generation.

Why this agent uses the LLM: turning a research question into good
sub-questions and search terms is a judgement task with no exact answer.
Why the user's constraints are passed on every call rather than remembered by
the model: we hold them outside the LLM context and re-apply them, so a date
range or topic boundary cannot be "forgotten" between turns (proposal, section 5).
"""
import logging
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

import config
from agents import llm_client
from agents.models import SubQuestion

log = logging.getLogger("research_agent.planning")


class _PlannedSubQuestion(BaseModel):
    text: str
    search_query: str


class PlanResponse(BaseModel):
    """The JSON shape the model must return. Validation enforces the 3-5 rule."""
    interpretation: str
    subquestions: List[_PlannedSubQuestion]

    @field_validator("subquestions")
    @classmethod
    def _count(cls, v):
        if not (config.MIN_SUBQUESTIONS <= len(v) <= config.MAX_SUBQUESTIONS):
            raise ValueError(f"need {config.MIN_SUBQUESTIONS}-{config.MAX_SUBQUESTIONS} sub-questions, got {len(v)}")
        return v


class ResearchPlan(BaseModel):
    interpretation: str
    subquestions: List[SubQuestion]


class QueryResponse(BaseModel):
    """JSON shape for a reformulated query."""
    search_query: str


_REFORMULATE_PROMPT = """You are refining one scholarly search query for a literature review.

Research question: "{question}"
Sub-question: "{subquestion}"
Previous search query: "{previous}"
Why it was not good enough: {reason}
{constraints}
Write ONE improved search query (5-10 keywords, no quotation marks, no boolean operators) that targets
the sub-question more precisely. Use more specific or different terms than the previous query.

Return ONLY JSON in this exact shape:
{{"search_query": "..."}}"""


_PLAN_PROMPT = """You are a research planning assistant helping a student begin a literature search.

Research question: "{question}"
{constraints}
Tasks:
1. Write one sentence stating how you interpret the question.
2. Break it into {min_n} to {max_n} focused sub-questions that together cover the question.
3. For each sub-question write a concise academic search query (5-10 keywords, no quotation marks,
   no boolean operators) suitable for a scholarly search engine.

Return ONLY JSON in this exact shape:
{{"interpretation": "...", "subquestions": [{{"text": "...", "search_query": "..."}}]}}
{feedback}"""


class PlanningAgent:
    def __init__(self, constraints: Optional[str] = None):
        self.constraints = constraints or ""

    def _prompt(self, question: str, feedback: str = "") -> str:
        cons = f"User constraints (must be respected): {self.constraints}\n" if self.constraints else ""
        fb = f"\nThe researcher reviewed a previous plan and asked for changes: {feedback}\nProduce a revised plan." if feedback else ""
        return _PLAN_PROMPT.format(question=question, constraints=cons, feedback=fb,
                                   min_n=config.MIN_SUBQUESTIONS, max_n=config.MAX_SUBQUESTIONS)

    def plan_research(self, question: str, feedback: str = "") -> ResearchPlan:
        log.info("planning | question=%r | feedback=%r", question[:80], feedback[:80])
        parsed = llm_client.generate_json(self._prompt(question, feedback), PlanResponse)
        subqs = [SubQuestion(id=i + 1, text=s.text.strip(), search_query=s.search_query.strip())
                 for i, s in enumerate(parsed.subquestions)]
        for sq in subqs:
            log.info("planned | %d | %s | query=%r", sq.id, sq.text, sq.search_query)
        return ResearchPlan(interpretation=parsed.interpretation.strip(), subquestions=subqs)

    def revise_plan(self, question: str, feedback: str) -> ResearchPlan:
        """Review point 1 allows unlimited revision: correcting the decomposition
        before retrieval costs no search quota (proposal, section 4)."""
        return self.plan_research(question, feedback=feedback)

    def reformulate_query(self, subquestion: SubQuestion, reason: str, question: str = "") -> SubQuestion:
        """One improved query for a sub-question whose search fell short.

        Returns a copy with the new query, the old one kept in `previous_query`,
        and `retried_once=True` - the flag the orchestrator checks so that no
        sub-question is reformulated twice (bounded autonomy, proposal section 4).
        The caller supplies `reason`; the model only sees why the last query failed.
        """
        cons = f"User constraints (must be respected): {self.constraints}\n" if self.constraints else ""
        prompt = _REFORMULATE_PROMPT.format(question=question, subquestion=subquestion.text,
                                            previous=subquestion.search_query, reason=reason, constraints=cons)
        new_query = llm_client.generate_json(prompt, QueryResponse).search_query.strip()
        log.info("reformulated | subquestion=%d | old=%r | new=%r", subquestion.id, subquestion.search_query, new_query)
        return subquestion.model_copy(update={
            "search_query": new_query,
            "previous_query": subquestion.search_query,
            "retried_once": True,
        })
