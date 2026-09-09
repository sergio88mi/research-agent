"""
Entry point. Stage 0: skeleton only - later stages wire in the orchestrator.

Usage (once stages are built):  python main.py "your research question"
"""
import sys

from agents.logging_setup import setup

if __name__ == "__main__":
    log = setup()
    question = " ".join(sys.argv[1:]).strip() or "no question supplied"
    log.info("research_agent skeleton running | question=%r", question)
    log.info("Stage 0 complete: package imports, configuration and logging verified.")
