"""
Entry point.

The complete pipeline. The Planning Agent decomposes the question; the
researcher approves the plan (review 1); each sub-question is searched,
verified and screened for relevance, with one automatic query reformulation
if a search falls short; the researcher approves, trims, or rejects the
evidence set (review 2 - one scope revision per run); the approved papers are
summarised from their abstracts, themes and gaps are drawn across them, and a
Markdown + JSON brief with a computed limitations section and a reference
list is written to output/.

Usage:  python main.py "your research question"
        python main.py                      (you will be prompted)
"""
import sys

from agents.logging_setup import setup
from agents.orchestrator import Orchestrator
from agents.researcher_interface import ResearcherInterface

if __name__ == "__main__":
    setup()
    ui = ResearcherInterface()
    question = " ".join(sys.argv[1:]).strip() or ui.submit_question()
    Orchestrator(interface=ui).run(question)
    print("\nStage 6 complete - brief written.")
