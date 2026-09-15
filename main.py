"""
Entry point.

Stage 3: the full front half of the design - the Planning Agent (Gemini)
decomposes the question, the researcher approves or revises the plan, and
only then does any searching happen.

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
    print("\nStage 3 complete.")
