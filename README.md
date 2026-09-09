# Agent-Based Academic Research System

An LLM-powered planning agent that turns a research question into a traceable
evidence brief: it decomposes the question into sub-questions, retrieves
scholarly literature, verifies citations, scores relevance, and writes a
Markdown/JSON brief - with the researcher approving the plan and the evidence
at two review points.

Individual implementation (Sergei Misjura) of the Group D design proposal,
Intelligent Agents, University of Essex Online, 2026.

*README is completed as the build progresses; this is the Stage 0 skeleton.*

## Status

| Stage | Deliverable | Status |
|---|---|---|
| 0 | Skeleton, configuration, data models, logging | done |
| 1 | OpenAlex retrieval | pending |
| 2 | DOI de-duplication, Crossref validation, storage | pending |
| 3 | Planning agent + human review 1 | pending |
| 4 | Relevance scoring and filtering | pending |
| 5 | Retrieval threshold, one-retry reformulation, evidence review | pending |
| 6 | Synthesis and final brief | pending |

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then add your Gemini key and email
python main.py "your research question"
pytest
```
