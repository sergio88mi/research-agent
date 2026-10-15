"""Stage 2 tests: the brief is written as both Markdown and JSON and round-trips."""
import json

import config
from agents.models import Brief, Paper
from agents.storage import save_brief


def test_save_brief_writes_md_and_json(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path))
    brief = Brief(research_question="Test question?",
                  selected_papers=[Paper(doi="10.1/x", title="T", year=2024, doi_verified=False)],
                  limitations=["one source"])
    md, js = save_brief(brief)
    md_text = open(md, encoding="utf-8").read()
    assert "Test question?" in md_text and "UNVERIFIED" in md_text and "one source" in md_text
    assert Brief.model_validate(json.load(open(js, encoding="utf-8"))) == brief


def test_a_paper_without_a_year_is_shown_as_undated(tmp_path, monkeypatch):
    # Why: some records carry no publication year; the brief used to print "(None)" for them.
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path))
    brief = Brief(research_question="q", selected_papers=[Paper(doi="10.1/x", title="Undated", year=None)])
    md, _ = save_brief(brief)
    md_text = open(md, encoding="utf-8").read()
    assert "**Undated** (n.d.)" in md_text and "(None)" not in md_text
