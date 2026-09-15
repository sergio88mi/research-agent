"""
Orchestrator - deterministic coordinator. Not an agent.

Why fixed sequencing rather than agents negotiating: academic research needs
controllable, auditable results, and errors early in an unconstrained agent
chain compound downstream (proposal, section 4). The orchestrator calls each
agent in turn, enforces the review points and retry limits, and does the
exact-answer work itself (de-duplication, verification, storage).

Stage 3 wiring: plan -> review 1 (loop until approved) -> retrieve per
sub-question -> dedupe -> verify -> save.
"""
import logging
from typing import List, Optional

from agents import validation
from agents.models import Brief, Decision, Paper, SubQuestion
from agents.planning_agent import PlanningAgent
from agents.researcher_interface import ResearcherInterface
from agents.retrieval_agent import RetrievalAgent
from agents.storage import save_brief

log = logging.getLogger("research_agent.orchestrator")


class Orchestrator:
    def __init__(self, interface: Optional[ResearcherInterface] = None, constraints: Optional[str] = None):
        self.ui = interface or ResearcherInterface()
        self.planner = PlanningAgent(constraints)
        self.retriever = RetrievalAgent()
        self.scope_revision_count = 0
        self.retry_counts = {}

    def _approved_plan(self, question: str):
        plan = self.planner.plan_research(question)
        while True:
            decision, feedback = self.ui.review_subquestions(plan.interpretation, plan.subquestions)
            log.info("review1 | decision=%s | feedback=%r", decision.value, feedback)
            if decision is Decision.APPROVE:
                for sq in plan.subquestions:
                    sq.approved = True
                return plan
            plan = self.planner.revise_plan(question, feedback)   # unlimited: no quota spent yet

    def run(self, question: str) -> Brief:
        log.info("run | question=%r", question)
        plan = self._approved_plan(question)

        papers: List[Paper] = []
        for sq in plan.subquestions:
            papers.extend(self.retriever.retrieve_evidence(sq))
        papers = validation.deduplicate_by_doi(papers)
        papers = validation.validate_dois_and_metadata(papers)

        brief = Brief(
            research_question=question,
            interpretation=plan.interpretation,
            subquestions=plan.subquestions,
            selected_papers=papers,
            limitations=[
                "Stage 3 output: papers are retrieved and verified but not yet scored for relevance.",
                "Single source (OpenAlex); abstracts only.",
            ],
        )
        md_path, json_path = save_brief(brief)
        log.info("saved | %s | %s", md_path, json_path)
        print(f"\n{len(papers)} unique papers across {len(plan.subquestions)} sub-questions "
              f"| {sum(1 for p in papers if p.doi_verified)} DOIs verified")
        print(f"Saved: {md_path}")
        return brief
