"""
Retrieval Agent.

Its local goal (from the proposal): obtain enough usable scholarly records for
each sub-question. It can search (Stage 1) and it can judge whether a search
met its goal (Stage 5); when it did not, it asks for a reformulated query.

Why this agent has no LLM: deciding whether a search returned enough usable
records is a count, not a judgement. Agency here comes from having a goal and
acting on its own assessment of success - not from a language model.

Why it does not reformulate the query itself: writing a better query is a
language task, so the request goes to the Planning Agent (via the
orchestrator, which also enforces the one-retry cap on SubQuestion.retried_once).
"""
import logging
from typing import List

import config
from agents import search_client
from agents.models import Paper, SubQuestion

log = logging.getLogger("research_agent.retrieval")


class RetrievalAgent:
    def retrieve_evidence(self, subquestion: SubQuestion) -> List[Paper]:
        """Search for one sub-question and tag every result with its origin."""
        log.info("retrieving | subquestion=%d | query=%r", subquestion.id, subquestion.search_query)
        papers = search_client.search(subquestion.search_query)
        for p in papers:
            p.subquestion_id = subquestion.id
        log.info("retrieved | subquestion=%d | records=%d | with_abstract=%d",
                 subquestion.id, len(papers), self.usable_count(papers))
        return papers

    @staticmethod
    def usable_count(papers: List[Paper]) -> int:
        """A record without an abstract cannot be screened for relevance, so it is not usable."""
        return sum(1 for p in papers if p.abstract)

    def assess_threshold(self, papers: List[Paper]) -> bool:
        """Did the search meet the agent's goal? (config.RETRIEVAL_THRESHOLD usable records)"""
        return self.usable_count(papers) >= config.RETRIEVAL_THRESHOLD

    def reformulation_reason(self, papers: List[Paper]) -> str:
        """The message the agent sends when requesting a new query."""
        return (f"the search returned {len(papers)} records but only {self.usable_count(papers)} "
                f"had abstracts (minimum {config.RETRIEVAL_THRESHOLD})")
