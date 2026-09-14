"""
Deterministic checks the orchestrator runs between retrieval and evaluation:
de-duplication by DOI and DOI/metadata verification against Crossref.

Why these are plain functions and not an agent: each has an exactly right
answer. Routing a registry lookup through an LLM would add cost and introduce
error without adding capability (proposal, section 4).
"""
import logging
import re
from difflib import SequenceMatcher
from typing import List, Optional

import requests

from agents import crossref_client
from agents.models import Paper

log = logging.getLogger("research_agent.validation")

TITLE_MATCH_THRESHOLD = 0.6   # fuzzy ratio; titles differ in casing, punctuation, subtitles


def deduplicate_by_doi(papers: List[Paper]) -> List[Paper]:
    """Keep the first occurrence of each DOI. Papers without a DOI are kept
    (they cannot be proven duplicates) but will be flagged unverified later."""
    seen, unique = set(), []
    for p in papers:
        key = p.doi.lower() if p.doi else None
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        unique.append(p)
    log.info("deduplicate | in=%d | out=%d", len(papers), len(unique))
    return unique


def _clean(text: Optional[str]) -> str:
    return re.sub(r"[^a-z0-9 ]", "", (text or "").lower()).strip()


def titles_match(a: Optional[str], b: Optional[str]) -> bool:
    return SequenceMatcher(None, _clean(a), _clean(b)).ratio() >= TITLE_MATCH_THRESHOLD


def validate_dois_and_metadata(papers: List[Paper]) -> List[Paper]:
    """
    Set paper.doi_verified for every paper:
      True  - DOI is registered and the title matches
      False - no DOI, DOI not registered, or title does not match (flagged, kept)
      None  - could not check (network error) - kept, still distinguishable
    Records are flagged, never silently dropped: the design promised
    "unverified rather than invalid", because registry coverage is incomplete.
    """
    for p in papers:
        if not p.doi:
            p.doi_verified = False
            log.info("verify | no DOI | %r", p.title[:60])
            continue
        try:
            record = crossref_client.lookup(p.doi)
        except requests.RequestException as exc:
            p.doi_verified = None
            log.warning("verify | could not check %s | %s", p.doi, exc)
            continue
        if record is None:
            p.doi_verified = False
            log.info("verify | not in Crossref | %s", p.doi)
            continue
        cr_title = (record.get("title") or [""])[0]
        p.doi_verified = titles_match(p.title, cr_title)
        log.info("verify | %s | %s", "match" if p.doi_verified else "TITLE MISMATCH", p.doi)
    summary = {"verified": 0, "flagged": 0, "unchecked": 0}
    for p in papers:
        summary["verified" if p.doi_verified else "unchecked" if p.doi_verified is None else "flagged"] += 1
    log.info("verify summary | %s", summary)
    return papers
