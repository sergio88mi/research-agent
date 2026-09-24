# Research Agent – an LLM-powered planning agent for academic literature search

Individual implementation for the Intelligent Agents module (MSc, University of
Essex Online), built from the Group D design proposal. Author: Sergei Misjura.

## What it does

You give it a research question. It breaks the question into three to five
sub-questions, searches the scholarly literature for each one, checks that the
papers it found are real, decides which of them actually answer the question,
and writes a short evidence brief: one note per paper, the themes that run
across them, the gaps it could not cover, and a reference list.

Two things make it different from just asking a chatbot. First, it stops twice
and asks you: once to approve the plan before any searching happens, and once
to approve the evidence before anything is written. Second, every claim in the
brief can be traced back: each note comes from one paper's abstract, each
theme cites paper numbers, every DOI has been checked against Crossref, and
the reference list is built from the records themselves rather than written by
the model. The brief also states what it could not do.

## How it works

```
question ─► Planning Agent ─► [Review 1] ─► Retrieval Agent ─► de-duplicate ─► verify DOIs
                 ▲                              │  (threshold miss → one reformulation)
                 │                              ▼
          revise / reject scope ◄── [Review 2] ◄── Evaluation & Synthesis Agent (score)
                                        │ approve
                                        ▼
                     Evaluation & Synthesis Agent (notes, themes, gaps) ─► brief.md + brief.json
```

There are three agents and one coordinator, matching the class diagram in the
proposal (`agents/models.py` holds the data classes from that diagram):

| Component | File | Uses the LLM? | Job |
|---|---|---|---|
| Orchestrator | `agents/orchestrator.py` | no | Runs the steps in a fixed order, enforces the retry and review limits, does the exact work (de-duplication, verification, storage) itself |
| Planning Agent | `agents/planning_agent.py` | yes | Turns the question into sub-questions and search queries; rewrites a query once if a search falls short |
| Retrieval Agent | `agents/retrieval_agent.py` | no | Searches OpenAlex for each sub-question and judges whether it got enough usable records |
| Evaluation & Synthesis Agent | `agents/evaluation_agent.py` | yes | Scores each paper's relevance 1–5 with a reason, then writes the notes, themes and gaps from the approved set |
| Researcher interface | `agents/researcher_interface.py` | no | The two review points; can be scripted so the loops are testable |

The autonomy is deliberately bounded: at most one automatic query
reformulation per sub-question and one scope revision per run. Those numbers,
and every other tunable, live in `config.py`.

The model is used only for judgement tasks (decomposing, scoring, summarising).
Anything with an exact answer is ordinary code. Every model reply is parsed
against a schema; if it does not fit, the error is sent back once for repair,
and if that fails the run stops rather than continuing on bad data.

## Setup

Python 3.11 or newer.

```bash
git clone https://github.com/sergio88mi/research-agent.git
cd research-agent
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Then open `.env` and fill in:

| Variable | What | Where to get it |
|---|---|---|
| `GEMINI_API_KEY` | Google Gemini API key (free tier is enough) | https://aistudio.google.com/apikey |
| `GEMINI_MODEL` | Model name; default `gemini-3.5-flash-lite` | see note below |
| `OPENALEX_EMAIL` | Your email, sent with OpenAlex requests as they ask | – |
| `OPENALEX_API_KEY` | Free OpenAlex key (raises the daily search budget 10×) | https://openalex.org/settings/api |

Model note: the free tier of the full Flash models allows about 20 requests a
day, which is roughly one run. The Flash-Lite models allow 500. If you have a
paid key any Gemini text model works; only the name changes.

## Running it

```bash
python main.py "What approaches reduce hallucination in LLM-based autonomous agents?"
```

or just `python main.py` and type the question when asked. A run looks like this:

1. The plan is printed. Type `approve`, or describe what to change in a sentence
   (e.g. `merge sub-questions 3 and 4 into one about evaluation`) and a revised
   plan is shown.
2. Each sub-question is searched, verified and scored. This takes a minute or
   two. If a sub-question yields fewer than five relevant papers the query is
   rewritten once and searched again.
3. The selected papers are listed with their scores and reasons. Type `approve`,
   `drop 3, 7` to remove papers by number, or `reject` to send feedback and
   re-plan (once per run).
4. The brief is written to `output/brief-<timestamp>-<question>.md` and `.json`,
   and a log of the run goes to `output/run.log`.

Responses from all three services are cached in `cache/`, so re-running the
same question is fast and costs nothing until you change something.

## Tests

```bash
pytest
```

56 unit tests, no network needed: the API replies they rely on are recorded in
`tests/fixtures/`. They cover the data models, abstract reconstruction,
de-duplication and DOI checks, the schema-validation-and-repair path, the 3–5
sub-question rule, the review loops, the threshold and reformulation caps, the
relevance cut-off, batching, themes that cite non-existent papers, the retry
policies, and the brief layout.

Live functional tests are recorded in `evidence/functional_tests.md` (nine
scenarios, each with the command, expected and observed result). Things that
went wrong during development and what was done about them are in
`evidence/remediation_log.md` (eleven entries). Screenshots and log captures
from the build stages are in `evidence/`, and `notes/stage0.md` to `stage6.md`
explain what each stage added and why it was built that way.

## Project layout

```
main.py                    entry point
config.py                  all tunables in one place
agents/
  models.py                SubQuestion, Paper, Assessment, Brief, Decision (from the class diagram)
  orchestrator.py          deterministic coordinator, review loops, limits, limitations section
  planning_agent.py        decomposition, revision, query reformulation
  retrieval_agent.py       search + threshold check
  evaluation_agent.py      relevance scoring, evidence notes, themes and gaps
  researcher_interface.py  the two human review points
  search_client.py         OpenAlex works API, abstract reconstruction
  crossref_client.py       DOI lookup
  validation.py            de-duplication, DOI/title verification
  llm_client.py            Gemini over HTTP, schema validation, repair retry, transient-error retry
  http.py                  shared retry policy for the scholarly APIs
  storage.py               Markdown + JSON brief, deterministic reference list
  cache.py, logging_setup.py
tests/                     pytest suite and recorded fixtures
notes/                     one explanation per build stage
evidence/                  screenshots, logs, functional test log, remediation log
output/                    briefs from the live runs and run.log
```

## Known limitations

- One source (OpenAlex), 20 results per query, abstracts only – no full text,
  no citation chasing, no date filters. The brief says so in its last section.
- Relevance and summaries are model judgements from abstracts. They are
  traceable and checkable, not authoritative; the brief tells you to check the
  full text before relying on them.
- Papers whose DOIs are registered with DataCite rather than Crossref (mostly
  arXiv preprints) are flagged "unverified", not dropped.
- One automatic query rewrite is a modest safety net. In the live runs it
  lifted a thin sub-question from one relevant paper to three, not to five.
  That is what the reject-scope option at Review 2 is for.
- The free-tier quotas of the model and search providers shape how often you
  can run it; see the model note above and remediation entries 5–10.

## Sources and libraries

Design: Group D, *Proposed Team Project: Agent-Based Academic Research
System* (Assessment 1, Team Design Proposal), Intelligent Agents, University of
Essex Online, 2026 – in particular sections 3 (agent architecture), 4
(workflow), 5 (expected output), 8 (design decisions) and 9 (challenges and
mitigations), and the class, sequence and activity diagrams. The reasons behind individual design
decisions are given in the module docstrings and comments where they apply.

Services: Google Gemini API (`generativelanguage.googleapis.com`, model set in
`.env`); OpenAlex works API (https://docs.openalex.org); Crossref REST API
(https://api.crossref.org).

Python libraries (see `requirements.txt`): `requests` (HTTP), `pydantic`
(data models and schema validation), `python-dotenv` (configuration), `pytest`
(tests). No agent framework is used; the orchestration is plain Python so that
every step can be read and tested.
