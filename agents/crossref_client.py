"""
Crossref lookup - citation integrity checking, not search.

Why a separate registry: the search source told us a paper exists; Crossref
is the authority that registers DOIs. Looking each DOI up independently
confirms the citation refers to a real scholarly record and that the title
registered under that DOI matches the one retrieved (validation.py does the
comparison; the year is not compared, because a preprint and its published
version often carry different years). This is the defence against fabricated
or garbled citations (proposal, section 5; Walters and Wilder, 2023).
It says nothing about whether a summary is faithful - that is a different
mitigation, handled in Stage 6.
"""
import logging
from typing import Any, Dict, Optional
from urllib.parse import quote

import config
from agents import cache, http

log = logging.getLogger("research_agent.crossref")


def lookup(doi: str) -> Optional[Dict[str, Any]]:
    """
    Return Crossref's record for a DOI, or None if the DOI is not registered.
    Raises on network failure so the caller can distinguish "not found"
    (a fact about the DOI) from "could not check" (a fact about the network).
    """
    cached = cache.get("crossref", doi)
    if cached is not None:
        return cached or None            # {} is cached for "not found"

    contact = f" (mailto:{config.OPENALEX_EMAIL})" if config.OPENALEX_EMAIL else ""   # sent only when one is configured
    headers = {"User-Agent": "research-agent/0.1" + contact}
    # A DOI may contain characters that mean something else in a URL ('#', '?', '<', '>');
    # they are escaped so the whole DOI stays in the path. The usual ones are left as they are.
    response = http.get_with_retry(f"{config.CROSSREF_ENDPOINT}/{quote(doi, safe='/:;()')}", headers=headers,
                                   timeout=config.CROSSREF_TIMEOUT_SECONDS,
                                   ok_statuses=(200, 404), name="crossref")
    if response.status_code == 404:
        cache.put("crossref", doi, {})   # remember the miss too - it is a real finding
        return None
    record = response.json().get("message", {})
    cache.put("crossref", doi, record)
    return record
