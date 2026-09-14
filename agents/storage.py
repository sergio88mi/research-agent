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
from agents.models import Brief


def _slug(text: str, n: int = 40) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:n]


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
        lines.append("")
    lines += ["## Selected papers", ""]
    for p in brief.selected_papers:
        flag = {True: "verified", False: "UNVERIFIED", None: "unchecked"}[p.doi_verified]
        authors = ", ".join(p.authors[:3]) + (" et al." if len(p.authors) > 3 else "")
        doi = f"https://doi.org/{p.doi}" if p.doi else "no DOI"
        lines.append(f"- **{p.title}** ({p.year}) — {authors}. {doi} — DOI {flag}")
    lines.append("")
    if brief.summaries:
        lines += ["## Summaries", ""] + [f"- {s}" for s in brief.summaries] + [""]
    if brief.themes:
        lines += ["## Themes", ""] + [f"- {t}" for t in brief.themes] + [""]
    if brief.gaps:
        lines += ["## Gaps", ""] + [f"- {g}" for g in brief.gaps] + [""]
    lines += ["## Limitations of this search", ""] + [f"- {l}" for l in brief.limitations] + [""]

    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return md_path, json_path
