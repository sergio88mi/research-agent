"""
Orchestrator - deterministic coordinator. Not an agent.

Why fixed sequencing rather than agents negotiating: academic research needs
controllable, auditable results, and errors early in an unconstrained agent
chain compound downstream (proposal, section 4). The orchestrator calls each
agent in turn, enforces the review points and retry limits, and does the
exact-answer work itself (de-duplication, verification, storage).

Stage 4 wiring: plan -> review 1 (loop until approved) -> retrieve per
sub-question -> dedupe -> verify -> score relevance -> apply cut-off -> save.
"""
import logging
from typing import List, Optional

import config
from agents import validation
from agents.evaluation_agent import EvaluationSynthesisAgent, select_papers
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
        self.evaluator = EvaluationSynthesisAgent()
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

        # The agent proposes scores; the fixed cut-off (config) decides. Keeping
        # the decision here means the filter is auditable and not model-dependent.
        assessments = self.evaluator.evaluate_evidence(question, plan.subquestions, papers)
        selected, dropped = select_papers(papers, assessments)

        brief = Brief(
            research_question=question,
            interpretation=plan.interpretation,
            subquestions=plan.subquestions,
            selected_papers=selected,
            assessments=assessments,
            limitations=[
                "Stage 4 output: papers are screened for relevance but not yet summarised or synthesised.",
                f"Relevance was judged from title and abstract only; {len(dropped)} of {len(papers)} "
                f"retrieved papers scored below {config.RELEVANCE_CUTOFF}/5 and were excluded.",
                "Single source (OpenAlex); abstracts only.",
            ],
        )
        md_path, json_path = save_brief(brief)
        log.info("saved | %s | %s", md_path, json_path)
        print(f"\n{len(papers)} unique papers across {len(plan.subquestions)} sub-questions "
              f"| {sum(1 for p in papers if p.doi_verified)} DOIs verified "
              f"| {len(selected)} selected, {len(dropped)} dropped (cut-off {config.RELEVANCE_CUTOFF}/5)")
        print(f"Saved: {md_path}")
        return brief
