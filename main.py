"""
Entry point.

Stage 2: retrieval -> de-duplication -> Crossref verification -> saved brief.
Still no LLM: this proves the deterministic spine (search, validate, store)
works end to end before any judgement is layered on top (build order follows
risk - proposal section 6).

Usage:  python main.py            (built-in query)
        python main.py "query"
"""
import sys

from agents import validation
from agents.logging_setup import setup
from agents.models import Brief, SubQuestion
from agents.retrieval_agent import RetrievalAgent
from agents.storage import save_brief

STAGE_QUERY = "hallucination reduction in large language model agents"

if __name__ == "__main__":
    log = setup()
    query = " ".join(sys.argv[1:]).strip() or STAGE_QUERY
    log.info("Stage 2 run | query=%r", query)

    sq = SubQuestion(id=1, text=query, search_query=query, approved=True)
    papers = RetrievalAgent().retrieve_evidence(sq)
    papers = validation.deduplicate_by_doi(papers)
    papers = validation.validate_dois_and_metadata(papers)

    brief = Brief(
        research_question=query,
        subquestions=[sq],
        selected_papers=papers,
        limitations=[
            "Stage 2 output: no relevance scoring yet - every retrieved record is listed.",
            "Single source (OpenAlex); abstracts only, no full text.",
        ],
    )
    md_path, json_path = save_brief(brief)

    verified = sum(1 for p in papers if p.doi_verified)
    flagged = sum(1 for p in papers if p.doi_verified is False)
    print(f"\n{len(papers)} unique papers | {verified} DOIs verified | {flagged} flagged unverified")
    print(f"Saved: {md_path}\n       {json_path}")
    print("\nStage 2 complete.")
