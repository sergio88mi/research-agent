"""
OpenAlex search client - the Retrieval Agent's connection to the outside world.

Why OpenAlex: it needs no API key (a contact email gets you the faster "polite
pool"), has strong abstract coverage, and is the proposal's named fallback
source. This module returns normalised Paper records, so adding a source
changes one function, not any agent - exactly the decoupling the design
promised. Stage 7 used that: Semantic Scholar (s2_client.py) now runs
alongside this client for every search, and the Retrieval Agent fails over
between them.
"""
from typing import Any, Dict, List, Optional


import config
from agents import cache, http
from agents.models import Paper

# Ask OpenAlex for only the fields we use. Why: smaller responses, faster calls,
# and the cache files stay readable when we inspect them during debugging.
_FIELDS = "id,doi,title,authorships,publication_year,abstract_inverted_index"


def reconstruct_abstract(inverted: Optional[Dict[str, List[int]]]) -> Optional[str]:
    """
    OpenAlex stores abstracts as an inverted index: {word: [positions]}.
    Rebuild the plain text by placing each word at its positions.
    Returns None when no abstract exists - callers treat None as "unusable
    record", which is what the retrieval threshold counts.
    """
    if not inverted:
        return None
    positions: Dict[int, str] = {}
    for word, idxs in inverted.items():
        for i in idxs:
            positions[i] = word
    return " ".join(positions[i] for i in sorted(positions))


def _normalise(work: Dict[str, Any]) -> Paper:
    """Map one OpenAlex work onto our Paper model. Missing values become None,
    never invented - a missing DOI must stay missing so validation can flag it."""
    doi = work.get("doi")
    if doi and doi.startswith("https://doi.org/"):
        doi = doi[len("https://doi.org/"):]          # store the bare DOI, e.g. 10.1038/...
    authors = [
        (a.get("author") or {}).get("display_name", "")
        for a in work.get("authorships", [])
    ]
    return Paper(
        doi=doi,
        title=work.get("title") or "(untitled)",
        authors=[a for a in authors if a],
        year=work.get("publication_year"),
        abstract=reconstruct_abstract(work.get("abstract_inverted_index")),
        source="openalex",
    )


def search(query: str, per_page: int = config.RESULTS_PER_QUERY) -> List[Paper]:
    """Run one literature search and return normalised Paper records."""
    params = {"search": query, "per-page": per_page, "select": _FIELDS}
    if config.OPENALEX_EMAIL:
        params["mailto"] = config.OPENALEX_EMAIL   # contact address, as OpenAlex asks
    if config.OPENALEX_API_KEY:
        params["api_key"] = config.OPENALEX_API_KEY  # free key = 10x the keyless daily budget

    cache_key = f"{query}|{per_page}"
    data = cache.get("openalex", cache_key)
    if data is None:
        response = http.get_with_retry(config.OPENALEX_ENDPOINT, params=params, timeout=30, name="openalex")
        data = response.json()
        cache.put("openalex", cache_key, data)     # replayable later, and quota-safe

    return [_normalise(w) for w in data.get("results", [])]
