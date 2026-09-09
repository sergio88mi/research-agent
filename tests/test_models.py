"""Stage 0 tests: the data classes validate as the design requires."""
import pytest
from pydantic import ValidationError

from agents.models import Assessment, Brief, Decision, Paper, SubQuestion


def test_subquestion_defaults():
    sq = SubQuestion(id=1, text="What is X?", search_query="X")
    assert sq.approved is False and sq.retried_once is False


def test_assessment_rejects_out_of_range_score():
    # Why: relevance is a 1-5 scale; anything else is an LLM formatting error we must catch.
    with pytest.raises(ValidationError):
        Assessment(paper_doi=None, paper_title="t", subquestion_id=1, relevance_score=7)


def test_paper_doi_verified_is_tristate():
    p = Paper(title="t")
    assert p.doi_verified is None      # not yet checked - distinct from False (flagged)


def test_brief_round_trips_through_json():
    b = Brief(research_question="q", themes=["a"], gaps=["b"])
    assert Brief.model_validate_json(b.model_dump_json()) == b


def test_decision_values_match_sequence_diagram():
    assert {d.value for d in Decision} == {"approve", "revise", "refine", "reject"}
