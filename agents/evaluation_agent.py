"""
Evaluation & Synthesis Agent - relevance scoring (Stage 4) and synthesis (Stage 6).

This is the EvaluationSynthesisAgent from Diagram 1. Its local goal: decide,
for every retrieved paper, whether it actually bears on the sub-question that
found it, and then turn the approved evidence into a brief the researcher can
check. Keyword search returns papers that share words with the query, not
papers that answer it, so scoring is the step that turns "results" into
"evidence"; synthesis is the step that turns evidence into an answer.

Why synthesis is grounded paper-by-paper first: a summary written from one
abstract can be checked against that abstract. Themes and gaps are then
written from those summaries only, and every theme must cite paper numbers
that exist - the schema rejects a theme that points at a paper not in the set.
That is the "traceable rather than plausible" requirement from the proposal.

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


def _exact_index_model(base: Type[BaseModel], field: str, n: int) -> Type[BaseModel]:
    """A copy of `base` whose list `field` must contain indices 1..n, each exactly once.

    Built per batch because the expected count is only known at call time;
    putting the check in the schema means the LLM repair retry covers it.
    """
    class _Exact(base):
        @model_validator(mode="after")
        def _covers_every_paper(self):
            got = sorted(item.index for item in getattr(self, field))
            if got != list(range(1, n + 1)):
                raise ValueError(f"expected one entry for each index 1..{n}, got {got}")
            return self
    _Exact.__name__ = base.__name__
    return _Exact


def _response_model_for(n: int) -> Type[ScoreResponse]:
    return _exact_index_model(ScoreResponse, "scores", n)


class _Summary(BaseModel):
    index: int
    summary: str


class SummaryResponse(BaseModel):
    summaries: List[_Summary]


class _Theme(BaseModel):
    statement: str
    paper_numbers: List[int]


class SynthesisResponse(BaseModel):
    themes: List[_Theme]
    gaps: List[str]


def _synthesis_model_for(n_papers: int) -> Type[SynthesisResponse]:
    """Themes must cite only papers 1..n_papers and stay within the configured count."""
    class _Checked(SynthesisResponse):
        @model_validator(mode="after")
        def _grounded(self):
            if not (config.MIN_THEMES <= len(self.themes) <= config.MAX_THEMES):
                raise ValueError(f"need {config.MIN_THEMES}-{config.MAX_THEMES} themes, got {len(self.themes)}")
            for t in self.themes:
                bad = [k for k in t.paper_numbers if not 1 <= k <= n_papers]
                if bad or not t.paper_numbers:
                    raise ValueError(f"theme cites papers outside 1..{n_papers} or none: {t.paper_numbers}")
            return self
    _Checked.__name__ = "SynthesisResponse"
    return _Checked


_SUMMARY_PROMPT = """You are writing evidence notes for a literature review.

Research question: "{question}"
Sub-question: "{subquestion}"

For each paper below write a summary of 2-3 sentences: what the paper does, what it reports, and how
it bears on the sub-question.

Rules:
- Use ONLY the title and abstract given. Do not add findings, numbers, methods or claims that are not
  in the abstract, and do not use anything you may know about the paper from elsewhere.
- If the abstract says a result was observed, report it as the abstract states it; do not strengthen it.
- If the abstract is missing, write exactly: "No abstract available; the title suggests: ..." and stop.

Papers:
{papers}

Return ONLY JSON in this exact shape, with exactly one entry for every index 1 to {n}:
{{"summaries": [{{"index": 1, "summary": "..."}}]}}"""


_SYNTHESIS_PROMPT = """You are synthesising the evidence collected for a literature review.

Research question: "{question}"

Sub-questions:
{subquestions}

Numbered evidence notes (each written from one paper's abstract):
{notes}

Tasks:
1. Identify {min_t} to {max_t} themes that run across the notes. Each theme is one or two sentences and
   must list the numbers of the papers that support it. Cite only numbers from the list above.
2. List the gaps: which sub-questions or aspects of the research question the notes do NOT answer, or
   answer only with general surveys rather than direct studies. Be specific about what is missing.

Rules:
- Use ONLY the notes above. Do not introduce papers, findings or claims that are not in them.
- Where a sub-question has few or no papers, say so as a gap.

Return ONLY JSON in this exact shape:
{{"themes": [{{"statement": "...", "paper_numbers": [1, 4]}}], "gaps": ["...", "..."]}}"""


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

    # ------------------------------------------------------------ synthesis
    def summarise(self, question: str, subquestions: Sequence[SubQuestion],
                  papers: Sequence[Paper]) -> List[str]:
        """One grounded summary per approved paper, in the same order as `papers`.

        Batched per sub-question like scoring, so each prompt carries the
        sub-question the paper was found for. Returned strings are numbered
        `[k] Title (year) - summary` where k is the paper's position in the
        approved list, so the brief, the themes and the reference list all use
        one numbering.
        """
        position = {_key(p): i + 1 for i, p in enumerate(papers)}
        text_by_key: Dict[str, str] = {}
        for sq in subquestions:
            mine = [p for p in papers if p.subquestion_id == sq.id]
            batches = [mine[i:i + config.EVAL_BATCH_SIZE] for i in range(0, len(mine), config.EVAL_BATCH_SIZE)]
            for b, batch in enumerate(batches, start=1):
                log.info("summarising | subquestion=%d | batch=%d/%d | papers=%d", sq.id, b, len(batches), len(batch))
                prompt = _SUMMARY_PROMPT.format(question=question, subquestion=sq.text,
                                                papers=self._format_papers(batch), n=len(batch))
                parsed = llm_client.generate_json(prompt, _exact_index_model(SummaryResponse, "summaries", len(batch)))
                by_index = {s.index: s.summary.strip() for s in parsed.summaries}
                for i, p in enumerate(batch, start=1):
                    text_by_key[_key(p)] = by_index[i]
        out = []
        for p in papers:
            k = position[_key(p)]
            out.append(f"[{k}] {p.title} ({p.year or 'n.d.'}) - {text_by_key.get(_key(p), 'No summary produced.')}")
        log.info("summarised | papers=%d", len(out))
        return out

    def synthesise(self, question: str, subquestions: Sequence[SubQuestion],
                   papers: Sequence[Paper], summaries: Sequence[str]) -> Tuple[List[str], List[str]]:
        """Themes and gaps across the approved evidence, written from the summaries only.

        Returns (themes, gaps). Each theme string ends with the paper numbers
        that support it; the schema has already rejected any number outside
        1..len(papers), so every citation in the brief points at a real paper.
        """
        if not papers:
            log.warning("synthesise | no approved papers - themes skipped")
            return [], ["No papers were approved, so no themes could be drawn."]
        sq_lines = "\n".join(f"{sq.id}. {sq.text}" for sq in subquestions)
        prompt = _SYNTHESIS_PROMPT.format(question=question, subquestions=sq_lines,
                                          notes="\n".join(summaries),
                                          min_t=config.MIN_THEMES, max_t=config.MAX_THEMES)
        parsed = llm_client.generate_json(prompt, _synthesis_model_for(len(papers)))
        themes = [f"{t.statement.strip()} [{', '.join(str(k) for k in sorted(set(t.paper_numbers)))}]"
                  for t in parsed.themes]
        gaps = [g.strip() for g in parsed.gaps if g.strip()]
        log.info("synthesised | themes=%d | gaps=%d", len(themes), len(gaps))
        return themes, gaps


def select_papers(papers: Sequence[Paper], assessments: Sequence[Assessment]) -> Tuple[List[Paper], List[Paper]]:
    """Apply the assessments: returns (selected, dropped) preserving retrieval order.

    Deterministic, so it lives outside the agent - the model proposes scores,
    the fixed cut-off decides. Papers with no assessment are treated as dropped.
    """
    keep = {(a.paper_doi or a.paper_title).strip().lower() for a in assessments if a.selected}
    selected = [p for p in papers if _key(p) in keep]
    dropped = [p for p in papers if _key(p) not in keep]
    return selected, dropped
