"""
Semantic Scholar search client - the second literature source (Stage 7).

Why a second source at all: the tutor's feedback on the design proposal was
that depending on one API for the main retrieval, with another only as a
fallback, limits coverage and makes every run hostage to that API's downtime
and rate limits. Semantic Scholar is the source the proposal itself named
first; OpenAlex and Semantic Scholar index the literature differently, so
searching both and de-duplicating by DOI widens recall at no cost in code.

Why it looks like search_client.py: same signature, same Paper output, same
cache and retry policy. The Retrieval Agent therefore treats the two sources
identically and can drop either one when it fails.

API notes (api.semanticscholar.org, Graph API v1, checked 30 Sept 2026):
paper/search returns {"total", "offset", "next", "data": [...]}; the DOI sits
in externalIds; abstracts are plain text (unlike OpenAlex's inverted index).
Unauthenticated callers share one public rate pool; a free key (S2_API_KEY)
gives a dedicated 1 request/second. 429/5xx are retried by agents/http.py.
"""
from typing import Any, Dict, List, Optional

import config
from agents import cache, http
from agents.models import Paper

# Only the fields we use - same reasoning as the OpenAlex client.
_FIELDS = "title,abstract,year,authors,externalIds"


def _clean_abstract(text: Optional[str]) -> Optional[str]:
    """Semantic Scholar abstracts sometimes carry stray newlines and padding
    (structured abstracts pasted from publishers). Collapse whitespace; an
    empty abstract stays None so the retrieval threshold does not count it."""
    if not text:
        return None
    cleaned = " ".join(text.split())
    return cleaned or None


def _normalise(item: Dict[str, Any]) -> Paper:
    """Map one Semantic Scholar record onto our Paper model. Missing values stay
    missing (never invented) so validation can flag them."""
    doi = (item.get("externalIds") or {}).get("DOI")
    authors = [a.get("name", "") for a in item.get("authors") or []]
    return Paper(
        doi=doi or None,
        title=item.get("title") or "(untitled)",
        authors=[a for a in authors if a],
        year=item.get("year"),
        abstract=_clean_abstract(item.get("abstract")),
        source="semanticscholar",
    )


def search(query: str, per_page: int = config.RESULTS_PER_QUERY) -> List[Paper]:
    """Run one literature search and return normalised Paper records."""
    params = {"query": query, "limit": per_page, "fields": _FIELDS}
    headers = {"x-api-key": config.S2_API_KEY} if config.S2_API_KEY else None

    cache_key = f"{query}|{per_page}"
    data = cache.get("semanticscholar", cache_key)
    if data is None:
        response = http.get_with_retry(config.S2_ENDPOINT, params=params, headers=headers,
                                       timeout=30, name="semanticscholar")
        data = response.json()
        cache.put("semanticscholar", cache_key, data)

    return [_normalise(item) for item in data.get("data", [])]
