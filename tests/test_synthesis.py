"""Stage 6 tests: grounded summaries, themes that cite real papers, gaps, limitations, references - no network."""
import json
import os
from unittest.mock import patch

import pytest

import config
from agents import llm_client
from agents.evaluation_agent import EvaluationSynthesisAgent, _synthesis_model_for
from agents.llm_client import LLMError
from agents.models import Assessment, Brief, Paper, SubQuestion
from agents.orchestrator import Orchestrator
from agents.researcher_interface import ResearcherInterface
from agents.storage import _reference, save_brief

FIX = os.path.join(os.path.dirname(__file__), "fixtures")
PLAN = open(os.path.join(FIX, "gemini_plan.json"), encoding="utf-8").read()
SUMMARIES = open(os.path.join(FIX, "gemini_summaries.json"), encoding="utf-8").read()
SYNTHESIS = open(os.path.join(FIX, "gemini_synthesis.json"), encoding="utf-8").read()

SQ = SubQuestion(id=1, text="What reduces hallucination in LLM agents?", search_query="q")
PAPERS = [
    Paper(doi="10.1/a", title="Retrieval-augmented agents", abstract="We reduce hallucination...", year=2024,
          authors=["A. Author", "B. Author"], subquestion_id=1),
    Paper(doi="10.1/b", title="Image captioning metrics", abstract="We evaluate captions...", year=2023,
          authors=["C. Author"], subquestion_id=1),
    Paper(doi=None, title="A survey of grounding techniques", abstract=None, subquestion_id=1),
]


def test_summaries_follow_paper_order_and_numbering():
    with patch("agents.llm_client.generate", return_value=SUMMARIES):
        out = EvaluationSynthesisAgent().summarise("q", [SQ], PAPERS)
    assert len(out) == 3
    assert out[0].startswith("[1] Retrieval-augmented agents (2024) - The paper proposes")
    assert out[2].startswith("[3] A survey of grounding techniques (n.d.) - No abstract available")


def test_themes_and_gaps_parsed_and_numbered():
    with patch("agents.llm_client.generate", return_value=SYNTHESIS):
        themes, gaps = EvaluationSynthesisAgent().synthesise("q", [SQ], PAPERS, ["[1] ..", "[2] ..", "[3] .."])
    assert len(themes) == 2 and themes[1].endswith("[1, 3]")
    assert len(gaps) == 2


def test_theme_citing_nonexistent_paper_is_rejected_then_repaired():
    # Why: a theme that cites paper 9 of 3 is a fabricated citation - schema error, one repair, then accepted.
    bad = json.loads(SYNTHESIS); bad["themes"][0]["paper_numbers"] = [9]
    with patch("agents.llm_client.generate", side_effect=[json.dumps(bad), SYNTHESIS]) as gen:
        parsed = llm_client.generate_json("p", _synthesis_model_for(3))
    assert gen.call_count == 2 and len(parsed.themes) == 2


def test_theme_without_any_citation_is_rejected():
    bad = json.loads(SYNTHESIS); bad["themes"][0]["paper_numbers"] = []
    with patch("agents.llm_client.generate", side_effect=[json.dumps(bad), json.dumps(bad)]):
        with pytest.raises(LLMError):
            llm_client.generate_json("p", _synthesis_model_for(3))


def test_too_many_themes_is_rejected(monkeypatch):
    monkeypatch.setattr(config, "MAX_THEMES", 1)
    with patch("agents.llm_client.generate", side_effect=[SYNTHESIS, SYNTHESIS]):
        with pytest.raises(LLMError):
            llm_client.generate_json("p", _synthesis_model_for(3))


def test_no_approved_papers_skips_synthesis_with_a_gap():
    with patch("agents.llm_client.generate") as gen:
        themes, gaps = EvaluationSynthesisAgent().synthesise("q", [SQ], [], [])
    assert themes == [] and "No papers were approved" in gaps[0] and not gen.called


def test_reference_is_built_from_metadata_only():
    assert _reference(PAPERS[0]) == "A. Author, B. Author (2024) *Retrieval-augmented agents*. https://doi.org/10.1/a"
    assert _reference(PAPERS[2]) == "Unknown author(s) (n.d.) *A survey of grounding techniques*."
    many = Paper(title="T", year=2020, authors=["W", "X", "Y", "Z"], doi="10.1/m")
    assert _reference(many).startswith("W et al. (2020)")


def test_brief_markdown_has_all_sections_with_shared_numbering(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path))
    brief = Brief(research_question="q", subquestions=[SQ], selected_papers=PAPERS,
                  assessments=[Assessment(paper_doi=p.doi, paper_title=p.title, subquestion_id=1,
                                          relevance_score=3, reason="r", selected=True) for p in PAPERS],
                  summaries=["[1] Retrieval-augmented agents (2024) - s1", "[2] Image captioning metrics (2023) - s2",
                             "[3] A survey of grounding techniques (n.d.) - s3"],
                  themes=["Grounding helps. [1, 3]"], gaps=["No long-horizon benchmark."],
                  limitations=["abstracts only"])
    md_path, _ = save_brief(brief)
    text = open(md_path, encoding="utf-8").read()
    for heading in ["## Selected papers (3)", "## Evidence notes", "## Themes across the evidence",
                    "## Gaps", "## Limitations of this search", "## References"]:
        assert heading in text
    refs = text.split("## References")[1]
    assert "1. A. Author, B. Author (2024)" in refs and "3. Unknown author(s) (n.d.)" in refs


def test_full_run_produces_summaries_themes_gaps_and_computed_limitations():
    # End-to-end with every external call mocked: plan -> approve -> retrieve -> score -> approve -> synthesise.
    def fake(prompt, **kw):
        if "research planning assistant" in prompt:
            return PLAN
        if "evidence notes for a literature review" in prompt:
            n = prompt.count("] Title:")
            return json.dumps({"summaries": [{"index": i + 1, "summary": f"s{i+1}"} for i in range(n)]})
        if "synthesising the evidence" in prompt:
            return json.dumps({"themes": [{"statement": "t", "paper_numbers": [1, 2]}], "gaps": ["g"]}) \
                if config.MIN_THEMES <= 1 else \
                json.dumps({"themes": [{"statement": "t1", "paper_numbers": [1]}, {"statement": "t2", "paper_numbers": [2]}], "gaps": ["g"]})
        n = prompt.count("] Title:")
        return json.dumps({"scores": [{"index": i + 1, "relevance_score": 4, "reason": "r"} for i in range(n)]})

    ui = ResearcherInterface(scripted=["approve", "approve"])
    papers = lambda q, per_page=None: [Paper(doi=f"10.1/{q[:3]}{i}", title=f"{q[:3]} {i}", abstract="a", year=2024,
                                             authors=["A"]) for i in range(6)]
    with patch("agents.llm_client.generate", side_effect=fake), \
         patch("agents.search_client.search", side_effect=papers), \
         patch("agents.crossref_client.lookup", return_value=None), \
         patch("agents.orchestrator.save_brief", return_value=("m", "j")):
        brief = Orchestrator(interface=ui).run("q")
    assert len(brief.summaries) == 18 and brief.summaries[0].startswith("[1] ")
    assert brief.themes and brief.gaps == ["g"]
    joined = " ".join(brief.limitations)
    assert "abstract" in joined and "OpenAlex" in joined and "checked against the full texts" in joined
    assert "18 of 18 approved papers have DOIs not registered" in joined     # all mocked as not-in-Crossref
