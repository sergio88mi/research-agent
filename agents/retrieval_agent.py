"""
Retrieval Agent.

Its local goal (from the proposal): obtain enough usable scholarly records for
each sub-question. Stage 1 gives it the ability to search; the threshold test
and the request-for-reformulation arrive in Stage 5.

Why this agent has no LLM: deciding whether a search returned enough usable
records is a count, not a judgement. Agency here comes from having a goal and
acting on its own assessment of success - not from a language model.
"""
import logging
from typing import List

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
        usable = sum(1 for p in papers if p.abstract)
        log.info("retrieved | subquestion=%d | records=%d | with_abstract=%d",
                 subquestion.id, len(papers), usable)
        return papers
