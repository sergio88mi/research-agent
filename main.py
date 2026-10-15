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
        python main.py "question" --constraints "healthcare settings only; empirical studies, not surveys"

Why the constraints are an option here and not something typed at a review:
whatever is given is held by the Planning Agent and written into every
planning and query-rewriting prompt, so a boundary set at the start is still
in the prompt when a query is reformulated ten minutes later. It steers what
is searched for; it is not a filter on what comes back, and the brief says so.

Why the failures are caught here and nowhere lower down: the agents raise as
soon as something cannot be done honestly (no key, a daily quota used up,
every source failing), and each of those exceptions already carries a message
written for the person running the tool. This is the one place that turns
them into a line on the screen, a line in output/run.log and an exit status,
instead of a traceback. Those deliberate stops are all RuntimeErrors (or the
network errors of `requests`), so that is what is caught. Anything else is
deliberately not caught: a bug should still show where it happened. (The catch
cannot be narrower than RuntimeError, so one of Python's own - a
RecursionError, say - would also end in one line; its text goes to the log.)
"""
import argparse
import logging
import sys
from typing import List, Optional

import requests

from agents.logging_setup import setup
from agents.orchestrator import Orchestrator
from agents.researcher_interface import ResearcherInterface

log = logging.getLogger("research_agent.main")


def main(argv: Optional[List[str]] = None) -> int:
    """Run one research question. Returns the exit status: 0 brief written, 1 run stopped, 130 interrupted."""
    parser = argparse.ArgumentParser(description="Turn a research question into a checked evidence brief.")
    parser.add_argument("question", nargs="*", help="the research question (asked for if left out)")
    parser.add_argument("-c", "--constraints", default="",
                        help="a boundary written into every planning and query-rewriting prompt, "
                             "e.g. 'healthcare settings only'")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    setup()
    ui = ResearcherInterface()
    try:
        question = " ".join(args.question).strip() or ui.submit_question()
        if not question:
            print("No research question given - nothing to do.")
            return 1
        if args.constraints:
            log.info("run | constraints=%r", args.constraints)
        Orchestrator(interface=ui, constraints=args.constraints or None).run(question)
    except (KeyboardInterrupt, EOFError):       # Ctrl-C anywhere, or Ctrl-D at one of the prompts
        log.info("run | interrupted by the researcher")
        print("\nStopped - no brief written for this run.")
        return 130
    except (RuntimeError, requests.RequestException) as exc:
        # LLMError is a RuntimeError; so are "every source failed" and a wrong
        # source name in the configuration.
        log.error("run | stopped | %s", exc)
        print(f"\nRun stopped: {exc}")
        return 1
    print("\nRun complete - brief written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
