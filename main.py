"""
Entry point.

Stage 1: a hard-coded query goes to OpenAlex through the Retrieval Agent and
the results are printed. Nothing is planned or judged yet - the point of
building this first is to prove the pipeline's plumbing before any LLM is
involved (build order follows risk, proposal section 6).

Usage:  python main.py            (uses the built-in Stage 1 query)
        python main.py "query"    (search for your own words)
"""
import sys

from agents.logging_setup import setup
from agents.models import SubQuestion
from agents.retrieval_agent import RetrievalAgent

STAGE1_QUERY = "hallucination reduction in large language model agents"

if __name__ == "__main__":
    log = setup()
    query = " ".join(sys.argv[1:]).strip() or STAGE1_QUERY
    log.info("Stage 1 run | query=%r", query)

    sq = SubQuestion(id=1, text=query, search_query=query)
    papers = RetrievalAgent().retrieve_evidence(sq)

    print(f"\n{len(papers)} results for: {query}\n")
    for i, p in enumerate(papers, 1):
        first_author = p.authors[0] if p.authors else "unknown"
        has_abstract = "abstract" if p.abstract else "NO abstract"
        print(f"{i:2d}. [{p.year}] {p.title[:80]}")
        print(f"     {first_author} | DOI: {p.doi or 'none'} | {has_abstract}")
    print("\nStage 1 complete.")
