"""Stage 5 tests: retrieval threshold, one-reformulation cap, review 2 (approve / refine / reject scope) - no network.

The tests from "the same paper from two sources" down were added after the
pre-submission read-through (remediations #15 and #16)."""
import json
import os
from unittest.mock import patch

import config
from agents.models import Decision, Paper, SubQuestion
from agents.orchestrator import Orchestrator
from agents.planning_agent import ResearchPlan
from agents.researcher_interface import ResearcherInterface
from agents.retrieval_agent import RetrievalAgent

PLAN = open(os.path.join(os.path.dirname(__file__), "fixtures", "gemini_plan.json"), encoding="utf-8").read()
SQ = SubQuestion(id=1, text="t", search_query="q")


def _papers(n, prefix, with_abstract=True):
    return [Paper(doi=f"10.1/{prefix}{i}", title=f"{prefix} {i}", abstract="a" if with_abstract else None,
                  year=2024, authors=["A"]) for i in range(n)]


def _scores(prompt, score):
    n = prompt.count("] Title:")
    return json.dumps({"scores": [{"index": i + 1, "relevance_score": score, "reason": "r"} for i in range(n)]})


def _llm(score_by_call):
    """Fake Gemini: plan JSON, a new query for reformulation prompts, scores for scoring prompts,
    and (since Stage 6) minimal summaries / themes so the run can finish."""
    def fake(prompt, **kw):
        if "research planning assistant" in prompt:
            return PLAN
        if "refining one scholarly search query" in prompt:
            return json.dumps({"search_query": "better query"})
        if "evidence notes for a literature review" in prompt:
            n = prompt.count("] Title:")
            return json.dumps({"summaries": [{"index": i + 1, "summary": "s"} for i in range(n)]})
        if "synthesising the evidence" in prompt:
            return json.dumps({"themes": [{"statement": "t1", "paper_numbers": [1]},
                                          {"statement": "t2", "paper_numbers": [1]}], "gaps": ["g"]})
        return _scores(prompt, score_by_call(prompt))
    return fake


def test_threshold_is_a_count_of_records_with_abstracts():
    agent = RetrievalAgent()
    assert agent.assess_threshold(_papers(config.RETRIEVAL_THRESHOLD, "x")) is True
    assert agent.assess_threshold(_papers(config.RETRIEVAL_THRESHOLD - 1, "x")) is False
    assert agent.assess_threshold(_papers(20, "x", with_abstract=False)) is False


def test_trigger1_reformulates_once_when_too_few_abstracts():
    # Why: sub-question 1's first search returns 2 usable records -> one reformulation, then no more.
    calls = []
    first_query = json.loads(PLAN)["subquestions"][0]["search_query"]
    def fake_search(query, per_page=None):
        calls.append(query)
        if query == first_query:
            return _papers(2, "few")                       # below RETRIEVAL_THRESHOLD
        if query == "better query":
            return _papers(8, "more")
        return _papers(8, query[:3])                       # sub-questions 2 and 3 are fine
    ui = ResearcherInterface(scripted=["approve", "approve"])
    with patch("agents.llm_client.generate", side_effect=_llm(lambda p: 4)), \
         patch("agents.search_client.search", side_effect=fake_search), \
         patch("agents.crossref_client.lookup", return_value=None), \
         patch("agents.orchestrator.save_brief", return_value=("m", "j")):
        brief = Orchestrator(interface=ui).run("q")
    sq1 = brief.subquestions[0]
    assert sq1.retried_once and sq1.previous_query == first_query and sq1.search_query == "better query"
    assert calls.count("better query") == 1                # reformulated exactly once
    assert all(not sq.previous_query for sq in brief.subquestions[1:])   # others untouched
    assert len(brief.selected_papers) == 2 + 8 + 8 + 8     # both searches for sq1 are kept


def test_trigger2_reformulates_when_too_few_relevant_and_respects_cap():
    # Why: 20 abstracts pass trigger 1, but every paper scores 1 -> trigger 2 fires once per sub-question,
    # and even though the reformulated search is *also* all-irrelevant, there is no second retry.
    reformulations = []
    def fake(prompt, **kw):
        if "refining one scholarly search query" in prompt:
            reformulations.append(prompt)
            assert "typical rejections" in prompt         # the drop reasons were fed back
            return json.dumps({"search_query": "better query"})
        return _llm(lambda p: 1)(prompt)
    ui = ResearcherInterface(scripted=["approve", "approve"])
    with patch("agents.llm_client.generate", side_effect=fake), \
         patch("agents.search_client.search", side_effect=lambda q, per_page=None: _papers(20, q[:3])), \
         patch("agents.crossref_client.lookup", return_value=None), \
         patch("agents.orchestrator.save_brief", return_value=("m", "j")):
        orch = Orchestrator(interface=ui)
        brief = orch.run("q")
    assert len(reformulations) == 3                        # one per sub-question, never two
    assert all(sq.retried_once for sq in brief.subquestions)
    assert sum(orch.retry_counts.values()) == 3 and brief.selected_papers == []


def test_review2_refine_removes_numbered_papers():
    ui = ResearcherInterface(scripted=["approve", "drop 1, 3", "approve"])
    with patch("agents.llm_client.generate", side_effect=_llm(lambda p: 4)), \
         patch("agents.search_client.search", side_effect=lambda q, per_page=None: _papers(6, q[:3])), \
         patch("agents.crossref_client.lookup", return_value=None), \
         patch("agents.orchestrator.save_brief", return_value=("m", "j")):
        brief = Orchestrator(interface=ui).run("q")
    assert len(brief.selected_papers) == 18 - 2
    removed = [a for a in brief.assessments if "removed by the researcher" in a.reason]
    assert len(removed) == 2 and not any(a.selected for a in removed)


def test_review2_reject_scope_replans_once_then_option_disappears(capsys):
    # Script: approve plan -> reject evidence (+feedback) -> approve revised plan -> approve evidence.
    ui = ResearcherInterface(scripted=["approve", "reject", "focus on healthcare", "approve", "approve"])
    plans = []
    def fake(prompt, **kw):
        if "research planning assistant" in prompt:
            plans.append(prompt)
            return PLAN
        return _llm(lambda p: 4)(prompt)
    with patch("agents.llm_client.generate", side_effect=fake), \
         patch("agents.search_client.search", side_effect=lambda q, per_page=None: _papers(6, q[:3])), \
         patch("agents.crossref_client.lookup", return_value=None), \
         patch("agents.orchestrator.save_brief", return_value=("m", "j")):
        orch = Orchestrator(interface=ui)
        brief = orch.run("q")
    assert orch.scope_revision_count == 1
    assert len(plans) == 2 and "focus on healthcare" in plans[1]      # feedback reached the planner
    assert "The plan the researcher reviewed:\n1. " in plans[1]       # together with the plan it rejects
    assert any("revised the scope once" in l for l in brief.limitations)
    out = capsys.readouterr().out
    assert out.count("=== Review 2") == 2 and out.count("'reject' to revise the scope") == 1   # offered once only


def test_review2_reject_refused_when_budget_used(capsys):
    ui = ResearcherInterface(scripted=["reject", "approve"])
    decision, _ = ui.review_evidence([SQ], [], [], can_reject_scope=False)
    assert decision is Decision.APPROVE
    assert "has been used" in capsys.readouterr().out


def test_review2_numbering_matches_display_order(capsys):
    # Why: 'drop 2' must remove the paper printed as "2." - the orchestrator relies on the same order.
    papers = [Paper(doi=f"10.1/p{i}", title=f"P{i}", subquestion_id=1) for i in range(3)]
    orch = Orchestrator(interface=ResearcherInterface(scripted=["drop 2", "approve"]))
    decision, kept, _ = orch._review_evidence(ResearchPlan(interpretation="i", subquestions=[SQ]), papers, [])
    assert " 2. [?] P1 (n.d.)" in capsys.readouterr().out
    assert decision is Decision.APPROVE and [p.title for p in kept] == ["P0", "P2"]


def test_review1_typo_is_reasked_not_sent_as_feedback(capsys):
    # Why: 'approv' (seen live) must not trigger a re-plan; a real sentence still counts as feedback.
    ui = ResearcherInterface(scripted=["approv", "approve"])
    decision, fb = ui.review_subquestions("i", [])
    assert decision is Decision.APPROVE and "Not understood" in capsys.readouterr().out
    ui = ResearcherInterface(scripted=["focus on healthcare applications only"])
    decision, fb = ui.review_subquestions("i", [])
    assert decision is Decision.REVISE and fb.startswith("focus on")


def test_the_same_paper_from_two_sources_counts_once_towards_the_threshold():
    # Why: with two sources a paper often comes back twice; five copies of one paper are not five papers.
    agent = RetrievalAgent()
    four = _papers(config.RETRIEVAL_THRESHOLD - 1, "x")
    assert agent.usable_count(four + four) == 2 * len(four)          # what the log line reports
    assert agent.assess_threshold(four + four) is False              # what the goal check uses
    assert agent.assess_threshold(four + _papers(1, "y")) is True
    assert f"returned {2 * len(four)} records but only {len(four)} different papers with an abstract" \
        in agent.reformulation_reason(four + four)


def test_no_reformulation_at_all_when_the_cap_is_zero(monkeypatch):
    # Why: config.MAX_REFORMULATIONS has to be the number the orchestrator obeys, not just the one it prints.
    monkeypatch.setattr(config, "MAX_REFORMULATIONS", 0)
    reformulations = []
    def fake(prompt, **kw):
        if "refining one scholarly search query" in prompt:
            reformulations.append(prompt)
        return _llm(lambda p: 1)(prompt)
    with patch("agents.llm_client.generate", side_effect=fake), \
         patch("agents.search_client.search", side_effect=lambda q, per_page=None: _papers(2, q[:3])), \
         patch("agents.crossref_client.lookup", return_value=None), \
         patch("agents.orchestrator.save_brief", return_value=("m", "j")):
        orch = Orchestrator(interface=ResearcherInterface(scripted=["approve", "approve"]))
        brief = orch.run("q")
    assert reformulations == [] and sum(orch.retry_counts.values()) == 0
    assert not any(sq.previous_query for sq in brief.subquestions)


def test_after_a_scope_revision_the_brief_counts_the_second_pass_only():
    # Why: the limitations line says what each source answered. After a rejected first pass it used to
    # report both passes together (6 searches, 36 records) beside an evidence set built from one.
    ui = ResearcherInterface(scripted=["approve", "reject", "focus on healthcare", "approve", "approve"])
    with patch("agents.llm_client.generate", side_effect=_llm(lambda p: 4)), \
         patch("agents.search_client.search", side_effect=lambda q, per_page=None: _papers(6, q[:3])), \
         patch("agents.crossref_client.lookup", return_value=None), \
         patch("agents.orchestrator.save_brief", return_value=("m", "j")):
        brief = Orchestrator(interface=ui).run("q")
    joined = " ".join(brief.limitations)
    assert "openalex: 3 of 3 searches answered, 18 records" in joined
    assert "revised the scope once" in joined


def test_a_new_plan_after_a_scope_revision_gets_its_own_reformulation_budget():
    # Why: the revised plan's sub-questions are numbered 1..n again. A retry spent on the old
    # sub-question 1 must not use up the budget of the new one.
    ui = ResearcherInterface(scripted=["approve", "reject", "focus on healthcare", "approve", "approve"])
    reformulations = []
    def fake(prompt, **kw):
        if "refining one scholarly search query" in prompt:
            reformulations.append(prompt)
        return _llm(lambda p: 1)(prompt)
    with patch("agents.llm_client.generate", side_effect=fake), \
         patch("agents.search_client.search", side_effect=lambda q, per_page=None: _papers(20, q[:3])), \
         patch("agents.crossref_client.lookup", return_value=None), \
         patch("agents.orchestrator.save_brief", return_value=("m", "j")):
        orch = Orchestrator(interface=ui)
        orch.run("q")
    assert len(reformulations) == 6                         # three sub-questions, once each, in both passes
    assert sum(orch.retry_counts.values()) == 6


def test_reject_without_saying_what_to_change_does_not_spend_the_scope_revision(capsys):
    ui = ResearcherInterface(scripted=["reject", "   ", "approve"])
    decision, feedback = ui.review_evidence([SQ], [], [], can_reject_scope=True)
    assert decision is Decision.APPROVE and feedback == ""
    assert "Nothing to change was given" in capsys.readouterr().out


def test_review2_shows_an_undated_paper_as_undated(capsys):
    ui = ResearcherInterface(scripted=["approve"])
    ui.review_evidence([SQ], [Paper(doi="10.1/u", title="Undated", year=None, subquestion_id=1)], [], True)
    out = capsys.readouterr().out
    assert "Undated (n.d.)" in out and "(None)" not in out
