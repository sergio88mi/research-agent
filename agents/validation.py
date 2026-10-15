"""
Deterministic checks the orchestrator runs between retrieval and evaluation:
de-duplication by DOI and DOI/metadata verification against Crossref.

Why these are plain functions and not an agent: each has an exactly right
answer. Routing a registry lookup through an LLM would add cost and introduce
error without adding capability (proposal, section 4).

Why the check is independent of the search: language models are known to
fabricate references and to garble real ones (Walters and Wilder, 2023). The
records here come from a scholarly index rather than from the model, but the
brief still only calls a citation "verified" when the DOI registry itself
holds a record under that DOI with a matching title.
"""
import logging
from difflib import SequenceMatcher
from typing import List, Optional

import requests

import config
from agents import crossref_client
from agents.models import Paper

log = logging.getLogger("research_agent.validation")


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
    """Lower-case and keep only letters, digits and spaces - in any script.

    The first version kept a-z and 0-9 only, which reduced a title written in
    Chinese, Cyrillic or Arabic to an empty string; two empty strings compare
    as identical, so any such title "matched" whatever Crossref held
    (remediation #14).
    """
    return "".join(ch for ch in (text or "").lower() if ch.isalnum() or ch == " ").strip()


def titles_match(a: Optional[str], b: Optional[str]) -> bool:
    """True when two titles are close enough to be the same paper (config.TITLE_MATCH_THRESHOLD).

    A missing title never matches: with nothing to compare there is no evidence either way.
    """
    left, right = _clean(a), _clean(b)
    if not left or not right:
        return False
    return SequenceMatcher(None, left, right).ratio() >= config.TITLE_MATCH_THRESHOLD


def validate_dois_and_metadata(papers: List[Paper]) -> List[Paper]:
    """
    Set paper.doi_verified for every paper:
      True  - DOI is registered and the title matches
      False - no DOI, DOI not registered, or title does not match (flagged, kept)
      None  - could not check (network error) - kept, still distinguishable
    Records are flagged, never silently dropped: the proposal's mitigation is
    "flagging unmatched records" (section 5), and a missing Crossref record
    is not proof that a paper is invalid - registry coverage is incomplete.
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
