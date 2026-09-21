"""
Orchestrator - deterministic coordinator. Not an agent.

Why fixed sequencing rather than agents negotiating: academic research needs
controllable, auditable results, and errors early in an unconstrained agent
chain compound downstream (proposal, section 4). The orchestrator calls each
agent in turn, enforces the review points and retry limits, and does the
exact-answer work itself (de-duplication, verification, storage).

Stage 5 wiring:
  plan -> review 1 (loop until approved)
  -> for each sub-question: retrieve -> [threshold miss? one reformulation]
       -> dedupe -> verify -> score -> [too few relevant? one reformulation]
  -> review 2: approve | refine (drop papers, re-show) | reject scope (once: back to planning)
  -> save.

The two bounded-autonomy limits from the design live here and nowhere else:
one query reformulation per sub-question (SubQuestion.retried_once) and one
scope revision per run (self.scope_revision_count).
"""
import logging
import re
from typing import Dict, List, Optional, Set, Tuple

import config
from agents import validation
from agents.evaluation_agent import EvaluationSynthesisAgent, select_papers
from agents.models import Assessment, Brief, Decision, Paper, SubQuestion
from agents.planning_agent import PlanningAgent, ResearchPlan
from agents.researcher_interface import ResearcherInterface
from agents.retrieval_agent import RetrievalAgent
from agents.storage import save_brief

log = logging.getLogger("research_agent.orchestrator")


def _key(p: Paper) -> str:
    return (p.doi or p.title).strip().lower()


class Orchestrator:
    def __init__(self, interface: Optional[ResearcherInterface] = None, constraints: Optional[str] = None):
        self.ui = interface or ResearcherInterface()
        self.planner = PlanningAgent(constraints)
        self.retriever = RetrievalAgent()
        self.evaluator = EvaluationSynthesisAgent()
        self.scope_revision_count = 0
        self.retry_counts: Dict[int, int] = {}      # sub-question id -> reformulations used

    # ------------------------------------------------------------------ review 1
    def _approved_plan(self, question: str, feedback: str = "") -> ResearchPlan:
        plan = self.planner.revise_plan(question, feedback) if feedback else self.planner.plan_research(question)
        while True:
            decision, fb = self.ui.review_subquestions(plan.interpretation, plan.subquestions)
            log.info("review1 | decision=%s | feedback=%r", decision.value, fb)
            if decision is Decision.APPROVE:
                for sq in plan.subquestions:
                    sq.approved = True
                return plan
            plan = self.planner.revise_plan(question, fb)   # unlimited: no quota spent yet

    # ---------------------------------------------------------------- retrieval
    def _reformulate(self, plan: ResearchPlan, idx: int, reason: str, question: str) -> SubQuestion:
        """Spend the sub-question's single reformulation. Caller must check retried_once first."""
        sq = plan.subquestions[idx]
        log.info("threshold | subquestion=%d | %s | reformulating (1 of %d allowed)",
                 sq.id, reason, config.MAX_REFORMULATIONS)
        new_sq = self.planner.reformulate_query(sq, reason, question)
        plan.subquestions[idx] = new_sq
        self.retry_counts[sq.id] = self.retry_counts.get(sq.id, 0) + 1
        return new_sq

    def _novel(self, papers: List[Paper], seen: Set[str]) -> List[Paper]:
        """De-duplicate within this batch and against everything already gathered.

        A paper retrieved by two sub-questions is scored once, against the
        sub-question that found it first - the same rule as Stage 2's global
        de-duplication, applied incrementally so the threshold check can run
        per sub-question.
        """
        out = []
        for p in validation.deduplicate_by_doi(papers):
            if _key(p) not in seen:
                seen.add(_key(p))
                out.append(p)
        return out

    def _gather_evidence(self, question: str, plan: ResearchPlan) -> Tuple[List[Paper], List[Assessment]]:
        seen: Set[str] = set()
        papers: List[Paper] = []
        assessments: List[Assessment] = []

        for idx in range(len(plan.subquestions)):
            sq = plan.subquestions[idx]
            found = self.retriever.retrieve_evidence(sq)

            # Trigger 1 - the Retrieval Agent's own goal check: enough usable records?
            if not self.retriever.assess_threshold(found) and not sq.retried_once:
                sq = self._reformulate(plan, idx, self.retriever.reformulation_reason(found), question)
                found += self.retriever.retrieve_evidence(sq)

            new = validation.validate_dois_and_metadata(self._novel(found, seen))
            assessed = self.evaluator.evaluate_evidence(question, [sq], new)

            # Trigger 2 - the orchestrator's check after screening: enough *relevant* records?
            # Stage 4's live run showed why this matters: a query can return 20 abstracts of
            # which only one answers the sub-question. Same single-retry budget.
            relevant = sum(1 for a in assessed if a.selected)
            if relevant < config.RETRIEVAL_THRESHOLD and not sq.retried_once:
                rejections = " | ".join([a.reason[:90] for a in assessed if not a.selected][:3])
                reason = (f"the query returned {len(new)} records but only {relevant} were relevant "
                          f"(minimum {config.RETRIEVAL_THRESHOLD}); typical rejections: {rejections}")
                sq = self._reformulate(plan, idx, reason, question)
                extra = validation.validate_dois_and_metadata(self._novel(self.retriever.retrieve_evidence(sq), seen))
                extra_assessed = self.evaluator.evaluate_evidence(question, [sq], extra)
                log.info("threshold | subquestion=%d | after reformulation: %d new records, %d relevant",
                         sq.id, len(extra), sum(1 for a in extra_assessed if a.selected))
                new += extra
                assessed += extra_assessed

            papers += new
            assessments += assessed
        return papers, assessments

    # ------------------------------------------------------------------ review 2
    @staticmethod
    def _parse_numbers(text: str) -> List[int]:
        return [int(n) for n in re.findall(r"\d+", text)]

    def _review_evidence(self, plan: ResearchPlan, selected: List[Paper],
                         assessments: List[Assessment]) -> Tuple[Decision, List[Paper], str]:
        """Loop on REFINE (cheap, no API calls); return on APPROVE or REJECT_SCOPE."""
        while True:
            can_reject = self.scope_revision_count < config.MAX_SCOPE_REVISIONS
            decision, feedback = self.ui.review_evidence(plan.subquestions, selected, assessments, can_reject)
            log.info("review2 | decision=%s | feedback=%r", decision.value, feedback)
            if decision is Decision.APPROVE:
                return decision, selected, ""
            if decision is Decision.REJECT_SCOPE:
                return decision, selected, feedback
            # REFINE: numbers refer to the 1..N order the interface displayed (= `selected` order)
            numbers = [n for n in self._parse_numbers(feedback) if 1 <= n <= len(selected)]
            removed = {_key(selected[n - 1]) for n in numbers}
            for a in assessments:
                if (a.paper_doi or a.paper_title).strip().lower() in removed and a.selected:
                    a.selected = False
                    a.reason += " [removed by the researcher at review 2]"
            selected = [p for p in selected if _key(p) not in removed]
            log.info("review2 | refined | removed=%d | remaining=%d", len(removed), len(selected))
            print(f"Removed {len(removed)} paper(s); {len(selected)} remain.")

    # ---------------------------------------------------------------------- run
    def run(self, question: str) -> Brief:
        log.info("run | question=%r", question)
        plan = self._approved_plan(question)
        limitations: List[str] = []

        while True:
            papers, assessments = self._gather_evidence(question, plan)
            selected, dropped = select_papers(papers, assessments)
            decision, selected, feedback = self._review_evidence(plan, selected, assessments)
            if decision is Decision.APPROVE:
                break
            # REJECT_SCOPE - the interface only offers it while the budget allows.
            self.scope_revision_count += 1
            log.info("scope revision %d/%d | feedback=%r", self.scope_revision_count, config.MAX_SCOPE_REVISIONS, feedback)
            limitations.append(f"The researcher rejected the first evidence set and revised the scope once: {feedback!r}.")
            plan = self._approved_plan(question, feedback=feedback)

        reformulated = [sq for sq in plan.subquestions if sq.previous_query]
        if reformulated:
            limitations.append("Queries reformulated once after a threshold miss: sub-question(s) "
                               + ", ".join(str(sq.id) for sq in reformulated) + ".")
        n_dropped = sum(1 for a in assessments if not a.selected)
        brief = Brief(
            research_question=question,
            interpretation=plan.interpretation,
            subquestions=plan.subquestions,
            selected_papers=selected,
            assessments=assessments,
            limitations=limitations + [
                "Stage 5 output: evidence is screened and approved by the researcher but not yet summarised or synthesised.",
                f"Relevance was judged from title and abstract only; {n_dropped} of {len(papers)} "
                f"retrieved papers were excluded (score below {config.RELEVANCE_CUTOFF}/5 or removed at review 2).",
                "Single source (OpenAlex); abstracts only.",
            ],
        )
        md_path, json_path = save_brief(brief)
        log.info("saved | %s | %s", md_path, json_path)
        print(f"\n{len(papers)} unique papers across {len(plan.subquestions)} sub-questions "
              f"| {sum(1 for p in papers if p.doi_verified)} DOIs verified "
              f"| {len(selected)} approved, {n_dropped} excluded "
              f"| reformulations: {sum(self.retry_counts.values())}, scope revisions: {self.scope_revision_count}")
        print(f"Saved: {md_path}")
        return brief
