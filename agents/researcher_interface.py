"""
Researcher boundary - the «boundary» class in Diagram 1.

Why a class rather than input() calls scattered through the orchestrator:
all human interaction passes through one object. In the terminal it asks the
researcher; in tests it can be given scripted answers, so the review loops
are testable without a person present.
"""
from typing import Dict, List, Optional, Tuple

from agents.models import Assessment, Decision, Paper, SubQuestion

_APPROVE_WORDS = ("approve", "a", "yes", "y", "")


class ResearcherInterface:
    def __init__(self, scripted: Optional[List[str]] = None):
        self._script = list(scripted) if scripted is not None else None

    def _ask(self, prompt: str) -> str:
        if self._script is not None:
            return self._script.pop(0) if self._script else "approve"
        return input(prompt).strip()

    def submit_question(self) -> str:
        return self._ask("Research question: ")

    def review_subquestions(self, interpretation: str, subqs: List[SubQuestion]) -> Tuple[Decision, str]:
        """Human review 1. Returns (APPROVE, "") or (REVISE, feedback)."""
        print("\n=== Review 1: proposed research plan ===")
        print(f"Interpretation: {interpretation}\n")
        for sq in subqs:
            print(f"  {sq.id}. {sq.text}\n     query: {sq.search_query}")
        while True:
            print("\nType 'approve' to search, or describe what to change (e.g. 'focus on healthcare, drop #3').")
            answer = self._ask("> ")
            low = answer.lower().strip()
            if low in _APPROVE_WORDS:
                return Decision.APPROVE, ""
            # A typo like 'approv' must not be sent to the planner as feedback - that
            # costs a model call and a re-plan (seen live, functional test FT-03).
            # Real feedback is a sentence; anything under 3 words is re-asked.
            if len(low.split()) < 3:
                print(f"Not understood: {answer!r}. Type 'approve', or describe the change in a sentence.")
                continue
            return Decision.REVISE, answer

    def review_evidence(self, subqs: List[SubQuestion], selected: List[Paper],
                        assessments: List[Assessment], can_reject_scope: bool) -> Tuple[Decision, str]:
        """Human review 2. Returns (APPROVE, ""), (REFINE, "3, 7") or (REJECT_SCOPE, feedback).

        Papers are numbered 1..N in the order given; the orchestrator uses the
        same order to apply a REFINE. `can_reject_scope` is decided by the
        orchestrator (one scope revision per run) - the interface only shows
        the option when it is still available.
        """
        by_key: Dict[str, Assessment] = {(a.paper_doi or a.paper_title).strip().lower(): a for a in assessments}
        dropped = sum(1 for a in assessments if not a.selected)

        print("\n=== Review 2: selected evidence ===")
        n = 0
        for sq in subqs:
            mine = [p for p in selected if p.subquestion_id == sq.id]
            note = f" (query reformulated once: {sq.previous_query!r} -> {sq.search_query!r})" if sq.previous_query else ""
            print(f"\nSub-question {sq.id}: {sq.text}{note}")
            if not mine:
                print("  (no relevant papers found)")
            for p in mine:
                n += 1
                a = by_key.get((p.doi or p.title).strip().lower())
                score = f"{a.relevance_score}/5" if a else "?"
                flag = {True: "verified", False: "UNVERIFIED", None: "unchecked"}[p.doi_verified]
                print(f"  {n:2d}. [{score}] {p.title} ({p.year}) - DOI {flag}")
                if a and a.reason:
                    print(f"      {a.reason[:140]}")
        print(f"\n{n} papers selected; {dropped} screened out (reasons are recorded in the brief).")

        while True:
            print("\nType 'approve' to write the brief, or 'drop 3, 7' to remove papers by number"
                  + (", or 'reject' to revise the scope of the plan (one scope revision per run)." if can_reject_scope else "."))
            answer = self._ask("> ")
            low = answer.lower().strip()
            if low in _APPROVE_WORDS:
                return Decision.APPROVE, ""
            if low.startswith("drop") or low[:1].isdigit():
                return Decision.REFINE, answer
            if low.startswith("reject"):
                if not can_reject_scope:
                    print("The one scope revision for this run has been used - approve or drop instead.")
                    continue
                feedback = self._ask("What should change about the scope? > ")
                return Decision.REJECT_SCOPE, feedback
            print("Not understood.")
