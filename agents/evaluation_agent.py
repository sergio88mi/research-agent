"""
Evaluation & Synthesis Agent - relevance scoring (Stage 4); synthesis follows in Stage 6.

This is the EvaluationSynthesisAgent from Diagram 1. Its local goal: decide,
for every retrieved paper, whether it actually bears on the sub-question that
found it. Keyword search returns papers that share words with the query, not
papers that answer it, so this is the step that turns "results" into "evidence".

Why the LLM does this and not a formula: judging whether an abstract addresses
a question is a reading task with no exact answer, which is the proposal's
criterion for using the model. Everything with an exact answer (de-duplication,
DOI checks, applying the cut-off) stays deterministic in the orchestrator.

Why the model sees only title + abstract: scoring must be grounded in the
record we hold, not in whatever the model remembers about a paper. The prompt
says so explicitly, and the reason it returns has to point at the abstract.

Why papers are scored in batches per sub-question: one call per paper would
mean ~80 calls a run and quickly hit free-tier rate limits; one call for all
papers risks truncated output. EVAL_BATCH_SIZE (config) is the middle ground.
The response schema is built per batch so validation can insist on exactly one
score per paper - a missing or duplicated index is a schema error, and the
existing one-shot repair in llm_client handles it.
"""
import logging
from typing import Dict, List, Sequence, Tuple, Type

from pydantic import BaseModel, Field, model_validator

import config
from agents import llm_client
from agents.models import Assessment, Paper, SubQuestion

log = logging.getLogger("research_agent.evaluation")


class _Score(BaseModel):
    index: int
    relevance_score: int = Field(ge=1, le=5)
    reason: str = ""


class ScoreResponse(BaseModel):
    scores: List[_Score]


def _response_model_for(n: int) -> Type[ScoreResponse]:
    """A ScoreResponse that also demands indices 1..n, each exactly once.

    Built per batch because the expected count is only known at call time;
    putting the check in the schema means the LLM repair retry covers it.
    """
    class _Exact(ScoreResponse):
        @model_validator(mode="after")
        def _covers_every_paper(self):
            got = sorted(s.index for s in self.scores)
            if got != list(range(1, n + 1)):
                raise ValueError(f"expected one score for each index 1..{n}, got {got}")
            return self
    _Exact.__name__ = "ScoreResponse"
    return _Exact


_SCORE_PROMPT = """You are screening papers for a literature review. Rate each paper's relevance to the SUB-QUESTION.

Research question: "{question}"
Sub-question: "{subquestion}"

Scale:
5 = directly addresses the sub-question
4 = strongly related; clearly useful evidence
3 = partially relevant or useful background
2 = tangential; shares vocabulary but not the question
1 = not relevant

Rules:
- Judge ONLY from the title and abstract given below. Do not use anything you may know about the paper from elsewhere.
- If the abstract is missing, rate from the title alone and say so in the reason.
- The reason must be one sentence that points to what the abstract does or does not cover.

Papers:
{papers}

Return ONLY JSON in this exact shape, with exactly one entry for every index 1 to {n}:
{{"scores": [{{"index": 1, "relevance_score": 3, "reason": "..."}}]}}"""


def _key(paper: Paper) -> str:
    """Identity used to join papers and assessments: DOI when present, else title."""
    return (paper.doi or paper.title).strip().lower()


class EvaluationSynthesisAgent:
    def _format_papers(self, papers: Sequence[Paper]) -> str:
        blocks = []
        for i, p in enumerate(papers, start=1):
            abstract = (p.abstract or "").strip()
            if len(abstract) > config.EVAL_ABSTRACT_CHARS:
                abstract = abstract[: config.EVAL_ABSTRACT_CHARS].rstrip() + " [...]"
            blocks.append(f"[{i}] Title: {p.title}\n    Year: {p.year or 'unknown'}\n"
                          f"    Abstract: {abstract or '(no abstract available)'}")
        return "\n\n".join(blocks)

    def _score_batch(self, question: str, sq: SubQuestion, papers: Sequence[Paper]) -> List[Assessment]:
        prompt = _SCORE_PROMPT.format(question=question, subquestion=sq.text,
                                      papers=self._format_papers(papers), n=len(papers))
        parsed = llm_client.generate_json(prompt, _response_model_for(len(papers)))
        by_index: Dict[int, _Score] = {s.index: s for s in parsed.scores}
        out = []
        for i, p in enumerate(papers, start=1):
            s = by_index[i]
            out.append(Assessment(
                paper_doi=p.doi, paper_title=p.title, subquestion_id=sq.id,
                relevance_score=s.relevance_score, reason=s.reason.strip(),
                selected=s.relevance_score >= config.RELEVANCE_CUTOFF,
            ))
        return out

    def evaluate_evidence(self, question: str, subquestions: Sequence[SubQuestion],
                          papers: Sequence[Paper]) -> List[Assessment]:
        """Score every paper against the sub-question that retrieved it.

        Returns one Assessment per paper. `selected` applies config.RELEVANCE_CUTOFF;
        the orchestrator uses it to filter, so the threshold lives in one place.
        """
        assessments: List[Assessment] = []
        for sq in subquestions:
            mine = [p for p in papers if p.subquestion_id == sq.id]
            if not mine:
                log.warning("evaluate | subquestion=%d | no papers to score", sq.id)
                continue
            batches = [mine[i:i + config.EVAL_BATCH_SIZE] for i in range(0, len(mine), config.EVAL_BATCH_SIZE)]
            for b, batch in enumerate(batches, start=1):
                log.info("evaluating | subquestion=%d | batch=%d/%d | papers=%d", sq.id, b, len(batches), len(batch))
                scored = self._score_batch(question, sq, batch)
                for a in scored:
                    log.info("score | %d | %s | %s | %s", a.relevance_score,
                             "keep" if a.selected else "drop", a.paper_doi or a.paper_title[:60], a.reason[:100])
                assessments.extend(scored)

        orphans = [p for p in papers if p.subquestion_id is None]
        if orphans:
            log.warning("evaluate | %d papers had no sub-question id and were not scored", len(orphans))
        kept = sum(1 for a in assessments if a.selected)
        log.info("evaluate summary | scored=%d | selected=%d | dropped=%d | cutoff=%d",
                 len(assessments), kept, len(assessments) - kept, config.RELEVANCE_CUTOFF)
        return assessments


def select_papers(papers: Sequence[Paper], assessments: Sequence[Assessment]) -> Tuple[List[Paper], List[Paper]]:
    """Apply the assessments: returns (selected, dropped) preserving retrieval order.

    Deterministic, so it lives outside the agent - the model proposes scores,
    the fixed cut-off decides. Papers with no assessment are treated as dropped.
    """
    keep = {(a.paper_doi or a.paper_title).strip().lower() for a in assessments if a.selected}
    selected = [p for p in papers if _key(p) in keep]
    dropped = [p for p in papers if _key(p) not in keep]
    return selected, dropped
