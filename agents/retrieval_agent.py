"""
Retrieval Agent.

Its local goal (proposal, section 2: it "judges whether it met a predefined
record threshold"): obtain enough usable scholarly records for each
sub-question. It can search (Stage 1), it can judge whether a search
met its goal (Stage 5), and since Stage 7 it searches several sources for
every sub-question and carries on when one of them fails.

Why this agent has no LLM: deciding whether a search returned enough usable
records is a count, not a judgement. Agency here comes from having a goal and
acting on its own assessment of success - the autonomy and pro-activeness in
Wooldridge and Jennings' (1995) account of an agent - not from a language model.

Why it does not reformulate the query itself: writing a better query is a
language task, so the request goes to the Planning Agent (via the
orchestrator, which also enforces the one-retry cap).

Why several sources, and why failover lives here (Stage 7): the design had
one primary API with a second held back as a fallback. The tutor's feedback
on that design was to use several sources as part of the normal process,
with automatic failover, for wider coverage and less exposure to any one
API's downtime or rate limits. Which sources to use is configuration
(config.SOURCES); this agent runs each of them for every query, merges the
records (the orchestrator de-duplicates by DOI), and if one source raises
after the shared retry policy has given up, it logs the failure, remembers it
for the brief's limitations, and continues with the remaining sources. Only
when every source fails for a query does the run stop - there is then
nothing honest to continue with.
"""
import logging
from typing import Dict, List

import requests

import config
from agents import s2_client, search_client
from agents.models import Paper, SubQuestion

log = logging.getLogger("research_agent.retrieval")

# Registry of search clients, keyed by the names used in config.SOURCES.
# Adding a source = one client module + one line here. The module (not its
# function) is stored so the call resolves `search` at run time - which is what
# lets the tests substitute a recorded response for either source.
SOURCE_CLIENTS = {
    "openalex": search_client,
    "semanticscholar": s2_client,
}


class RetrievalAgent:
    """Searches every configured source for a sub-question and judges whether it found enough."""

    def __init__(self) -> None:
        unknown = [s for s in config.SOURCES if s not in SOURCE_CLIENTS]
        if unknown or not config.SOURCES:
            # RuntimeError like every other "this run cannot go ahead" condition, so main() reports it in one line
            raise RuntimeError(f"config.SOURCES must name one or more of {sorted(SOURCE_CLIENTS)}; got {config.SOURCES}")
        self.source_stats: Dict[str, Dict[str, int]] = {}
        self.last_errors: Dict[str, str] = {}
        self.reset_stats()

    def reset_stats(self) -> None:
        """Start the per-source tallies again.

        The orchestrator reads them when it writes the brief's limitations: how
        many searches each source answered or failed. It calls this at the
        start of every evidence-gathering pass, so that after a scope revision
        the brief counts the searches behind the evidence it actually reports,
        not those of the rejected first pass as well (remediation #15).
        """
        self.source_stats = {name: {"searches": 0, "failed": 0, "records": 0} for name in config.SOURCES}
        self.last_errors = {}

    def retrieve_evidence(self, subquestion: SubQuestion) -> List[Paper]:
        """Search every configured source for one sub-question; tag every result with its origin.

        A source that fails (after agents/http.py has retried the transient
        cases) is skipped for this query and tried again on the next one - an
        outage or rate limit may have cleared by then.
        """
        log.info("retrieving | subquestion=%d | query=%r | sources=%s",
                 subquestion.id, subquestion.search_query, ",".join(config.SOURCES))
        papers: List[Paper] = []
        answered: List[str] = []
        for name in config.SOURCES:
            stats = self.source_stats[name]
            stats["searches"] += 1
            try:
                found = SOURCE_CLIENTS[name].search(subquestion.search_query)
            except requests.RequestException as exc:
                stats["failed"] += 1
                self.last_errors[name] = str(exc)
                log.warning("retrieval | subquestion=%d | source=%s failed (%s) - continuing with remaining sources",
                            subquestion.id, name, exc)
                continue
            for p in found:
                p.subquestion_id = subquestion.id
            stats["records"] += len(found)
            answered.append(name)
            log.info("retrieved | subquestion=%d | source=%s | records=%d | with_abstract=%d",
                     subquestion.id, name, len(found), self.usable_count(found))
            papers += found
        if not answered:
            raise RuntimeError(f"every retrieval source failed for sub-question {subquestion.id}: "
                               + "; ".join(f"{k}: {v}" for k, v in self.last_errors.items()))
        log.info("retrieved | subquestion=%d | sources_answered=%s | records=%d | with_abstract=%d",
                 subquestion.id, ",".join(answered), len(papers), self.usable_count(papers))
        return papers

    def sources_summary(self) -> str:
        """One line for the brief: what each source contributed to this run."""
        return "; ".join(
            f"{name}: {s['searches'] - s['failed']} of {s['searches']} searches answered, {s['records']} records"
            for name, s in self.source_stats.items())

    def failed_sources(self) -> Dict[str, int]:
        """Sources that failed at least one search in this pass, with the number of failures."""
        return {name: s["failed"] for name, s in self.source_stats.items() if s["failed"]}

    @staticmethod
    def usable_count(papers: List[Paper]) -> int:
        """A record without an abstract cannot be screened for relevance, so it is not usable."""
        return sum(1 for p in papers if p.abstract)

    @staticmethod
    def distinct_usable_count(papers: List[Paper]) -> int:
        """Usable records, counting a paper once however many sources returned it.

        With two sources the same paper often comes back twice. The goal is
        "enough different papers to screen", so the threshold must not be met
        by one paper counted twice (remediation #15).
        """
        seen = set()
        for p in papers:
            if p.abstract:
                seen.add((p.doi or p.title).strip().lower())
        return len(seen)

    def assess_threshold(self, papers: List[Paper]) -> bool:
        """Did the search meet the agent's goal? (config.RETRIEVAL_THRESHOLD distinct usable records)"""
        return self.distinct_usable_count(papers) >= config.RETRIEVAL_THRESHOLD

    def reformulation_reason(self, papers: List[Paper]) -> str:
        """The message the agent sends when requesting a new query."""
        return (f"the search returned {len(papers)} records but only {self.distinct_usable_count(papers)} "
                f"different papers with an abstract (minimum {config.RETRIEVAL_THRESHOLD})")
