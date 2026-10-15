"""
Data classes passed between the agents.

These four classes are the ones drawn in Diagram 1 (SubQuestion, Paper,
Assessment, Brief) plus the Decision enum from the researcher boundary. They
are the typed messages of the design: the records one agent produces reach
the next as these objects, and always by way of the orchestrator.

Why Pydantic rather than plain dataclasses: the LLM returns JSON, and Pydantic
gives us validation for free - a malformed model response fails loudly at the
boundary instead of propagating a half-formed object through the pipeline.
The proposal names Pydantic for exactly this ("validates LLM output", section
3); the single repair retry around it was added in the build (llm_client.py).
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class Decision(str, Enum):
    """Outcomes of the two human review points (see the sequence diagram)."""
    APPROVE = "approve"
    REVISE = "revise"          # review 1: regenerate the sub-questions
    REFINE = "refine"          # review 2: drop some papers, then continue
    REJECT_SCOPE = "reject"    # review 2: back to planning (once per run)


class SubQuestion(BaseModel):
    """One part of the research question, with the query used to search for it."""
    id: int
    text: str
    search_query: str
    approved: bool = False
    retried_once: bool = False           # set when the query has been reformulated (the cap itself is counted by the orchestrator)
    previous_query: Optional[str] = None  # kept when reformulated, so the brief can show both


class Paper(BaseModel):
    """One retrieved record, normalised so that every source looks the same to the agents.

    A paper is identified by its DOI, or by its title when it has none; that
    key is what joins a paper to its assessment and removes duplicates.
    """
    doi: Optional[str] = None
    title: str
    authors: List[str] = Field(default_factory=list)
    year: Optional[int] = None
    abstract: Optional[str] = None
    source: str = "openalex"
    doi_verified: Optional[bool] = None   # None = not checked, False = flagged unverified
    subquestion_id: Optional[int] = None  # which sub-question retrieved it


class Assessment(BaseModel):
    """The relevance judgement for one paper: score, the reason given, and whether it was kept."""
    paper_doi: Optional[str]
    paper_title: str
    subquestion_id: int
    relevance_score: int = Field(ge=1, le=5)
    reason: str = ""
    selected: bool = False


class Brief(BaseModel):
    """Everything the run produced, in the form that is saved as Markdown and JSON."""
    research_question: str
    interpretation: str = ""
    subquestions: List[SubQuestion] = Field(default_factory=list)
    selected_papers: List[Paper] = Field(default_factory=list)
    assessments: List[Assessment] = Field(default_factory=list)
    summaries: List[str] = Field(default_factory=list)
    themes: List[str] = Field(default_factory=list)
    gaps: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
