"""
Crossref lookup - citation integrity checking, not search.

Why a separate registry: OpenAlex told us a paper exists; Crossref is the
authority that registers DOIs. Looking each DOI up independently confirms the
citation refers to a real scholarly record and that title/year match. This is
the defence against fabricated or garbled citations (proposal, section 5).
It says nothing about whether a summary is faithful - that is a different
mitigation, handled in Stage 6.
"""
import logging
from typing import Any, Dict, Optional

import requests

import config
from agents import cache

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

    headers = {"User-Agent": f"research-agent/0.1 (mailto:{config.OPENALEX_EMAIL})"}
    response = requests.get(f"{config.CROSSREF_ENDPOINT}/{doi}", headers=headers, timeout=20)
    if response.status_code == 404:
        cache.put("crossref", doi, {})   # remember the miss too - it is a real finding
        return None
    response.raise_for_status()
    record = response.json().get("message", {})
    cache.put("crossref", doi, record)
    return record
