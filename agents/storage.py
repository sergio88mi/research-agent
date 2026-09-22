"""
Persist the brief as Markdown (for people) and JSON (for machines).

Why both: Markdown is what a researcher reads a week later; JSON is what the
tests compare and what any later tool can ingest. Recording what the search
could NOT cover (limitations) is what makes the output traceable rather than
merely plausible (proposal, section 5 of the blueprint).
"""
import json
import os
import re
from datetime import datetime
from typing import Tuple

import config
from agents.models import Brief, Paper


def _slug(text: str, n: int = 40) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:n]


def _reference(p: Paper) -> str:
    """Author-date reference built from the record's own metadata - nothing invented.

    Deliberately deterministic: the reference list is the one part of a
    research brief that must never be model-generated (fabricated citations
    are the failure this whole system exists to avoid).
    """
    if not p.authors:
        who = "Unknown author(s)"
    elif len(p.authors) <= 3:
        who = ", ".join(p.authors)
    else:
        who = f"{p.authors[0]} et al."
    year = p.year or "n.d."
    doi = f" https://doi.org/{p.doi}" if p.doi else ""
    return f"{who} ({year}) *{p.title}*.{doi}"


def save_brief(brief: Brief) -> Tuple[str, str]:
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = os.path.join(config.OUTPUT_DIR, f"brief-{stamp}-{_slug(brief.research_question)}")
    json_path, md_path = base + ".json", base + ".md"

    with open(json_path, "w", encoding="utf-8") as fh:
        fh.write(brief.model_dump_json(indent=2))

    lines = [f"# Research brief", "", f"**Question:** {brief.research_question}", ""]
    if brief.interpretation:
        lines += [f"**Interpretation:** {brief.interpretation}", ""]
    if brief.subquestions:
        lines += ["## Sub-questions", ""]
        for sq in brief.subquestions:
            lines.append(f"{sq.id}. {sq.text}  \n   *query:* `{sq.search_query}`")
            if sq.previous_query:
                lines.append(f"   *reformulated once from:* `{sq.previous_query}`")
        lines.append("")
    # Assessments are joined to papers by DOI (or title when there is none), so the
    # brief can show *why* each paper was kept - the score alone is not explainable.
    by_key = {(a.paper_doi or a.paper_title).strip().lower(): a for a in brief.assessments}

    # Papers are numbered [1..N] in this order; summaries, themes and references
    # use the same numbers, so a theme's "[3, 7]" can be followed to the paper.
    lines += [f"## Selected papers ({len(brief.selected_papers)})", ""]
    for k, p in enumerate(brief.selected_papers, start=1):
        flag = {True: "verified", False: "UNVERIFIED", None: "unchecked"}[p.doi_verified]
        authors = ", ".join(p.authors[:3]) + (" et al." if len(p.authors) > 3 else "")
        doi = f"https://doi.org/{p.doi}" if p.doi else "no DOI"
        line = f"{k}. **{p.title}** ({p.year}) — {authors}. {doi} — DOI {flag}"
        a = by_key.get((p.doi or p.title).strip().lower())
        if a:
            line += f"  \n   *sub-question {a.subquestion_id}, relevance {a.relevance_score}/5:* {a.reason}"
        lines.append(line)
    lines.append("")

    dropped = [a for a in brief.assessments if not a.selected]
    if dropped:
        lines += [f"## Excluded after relevance screening ({len(dropped)})", ""]
        for a in dropped:
            lines.append(f"- {a.paper_title} — sub-question {a.subquestion_id}, "
                         f"score {a.relevance_score}/5: {a.reason}")
        lines.append("")
    if brief.summaries:
        lines += ["## Evidence notes (one per approved paper, from its abstract)", ""]
        lines += [f"- {s}" for s in brief.summaries] + [""]
    if brief.themes:
        lines += ["## Themes across the evidence", ""] + [f"- {t}" for t in brief.themes] + [""]
    if brief.gaps:
        lines += ["## Gaps", ""] + [f"- {g}" for g in brief.gaps] + [""]
    lines += ["## Limitations of this search", ""] + [f"- {l}" for l in brief.limitations] + [""]
    if brief.selected_papers:
        lines += ["## References", ""] + [f"{k}. {_reference(p)}" for k, p in enumerate(brief.selected_papers, start=1)] + [""]

    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return md_path, json_path
