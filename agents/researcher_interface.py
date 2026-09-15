"""
Researcher boundary - the «boundary» class in Diagram 1.

Why a class rather than input() calls scattered through the orchestrator:
all human interaction passes through one object. In the terminal it asks the
researcher; in tests it can be given scripted answers, so the review loops
are testable without a person present.
"""
from typing import List, Optional, Tuple

from agents.models import Decision, SubQuestion


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
        print("\nType 'approve' to search, or describe what to change (e.g. 'focus on healthcare, drop #3').")
        answer = self._ask("> ")
        if answer.lower() in ("approve", "a", "yes", "y", ""):
            return Decision.APPROVE, ""
        return Decision.REVISE, answer
