"""
Data classes passed between the agents.

These four classes are the ones drawn in Diagram 1 (SubQuestion, Paper,
Assessment, Brief) plus the Decision enum from the researcher boundary.

Why Pydantic rather than plain dataclasses: the LLM returns JSON, and Pydantic
gives us validation for free - a malformed model response fails loudly at the
boundary instead of propagating a half-formed object through the pipeline.
That is the "schema validation with repair retry" mitigation from the proposal.
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
    id: int
    text: str
    search_query: str
    approved: bool = False
    retried_once: bool = False   # enforces the one-reformulation cap


class Paper(BaseModel):
    doi: Optional[str] = None
    title: str
    authors: List[str] = Field(default_factory=list)
    year: Optional[int] = None
    abstract: Optional[str] = None
    source: str = "openalex"
    doi_verified: Optional[bool] = None   # None = not checked, False = flagged unverified
    subquestion_id: Optional[int] = None  # which sub-question retrieved it


class Assessment(BaseModel):
    paper_doi: Optional[str]
    paper_title: str
    subquestion_id: int
    relevance_score: int = Field(ge=1, le=5)
    reason: str = ""
    selected: bool = False


class Brief(BaseModel):
    research_question: str
    interpretation: str = ""
    subquestions: List[SubQuestion] = Field(default_factory=list)
    selected_papers: List[Paper] = Field(default_factory=list)
    assessments: List[Assessment] = Field(default_factory=list)
    summaries: List[str] = Field(default_factory=list)
    themes: List[str] = Field(default_factory=list)
    gaps: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
