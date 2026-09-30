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
                 ▲                              │  (OpenAlex + Semantic Scholar, failover;
                 │                              ▼   threshold miss → one reformulation)
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
| Retrieval Agent | `agents/retrieval_agent.py` | no | Searches every configured source (OpenAlex and Semantic Scholar) for each sub-question, carries on if one source fails, and judges whether it got enough usable records |
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
| `SOURCES` | Literature sources searched for every sub-question, in order; default `openalex,semanticscholar` | – |
| `S2_API_KEY` | Optional free Semantic Scholar key: a dedicated 1 request/second instead of the shared public pool | https://www.semanticscholar.org/product/api#api-key |

Model note: the free tier of the full Flash models allows about 20 requests a
day, which is roughly one run. The Flash-Lite models allow 500. If you have a
paid key any Gemini text model works; only the name changes. The reasons
behind the default model are set out under *Why this model* below.

## Running it

```bash
python main.py "What approaches reduce hallucination in LLM-based autonomous agents?"
```

or just `python main.py` and type the question when asked. A run looks like this:

1. The plan is printed. Type `approve`, or describe what to change in a sentence
   (e.g. `merge sub-questions 3 and 4 into one about evaluation`) and a revised
   plan is shown.
2. Each sub-question is searched on both sources, the records are merged and
   de-duplicated, verified and scored. This takes a few minutes. If one source
   is down or rate-limited the run carries on with the other and the brief says
   so. If a sub-question yields fewer than five relevant papers the query is
   rewritten once and searched again.
3. The selected papers are listed with their scores and reasons. Type `approve`,
   `drop 3, 7` to remove papers by number, or `reject` to send feedback and
   re-plan (once per run).
4. The brief is written to `output/brief-<timestamp>-<question>.md` and `.json`,
   and a log of the run goes to `output/run.log`.

Responses from all four services are cached in `cache/`, so re-running the
same question is fast and costs nothing until you change something.

## Tests

```bash
pytest
```

68 unit tests, no network needed: the API replies they rely on are recorded in
`tests/fixtures/`. They cover the data models, abstract reconstruction,
de-duplication and DOI checks, the schema-validation-and-repair path, the 3–5
sub-question rule, the review loops, the threshold and reformulation caps, the
relevance cut-off, batching, themes that cite non-existent papers, the retry
policies, the brief layout, and (since Stage 7) the second source, the
merging of results and the failover between sources.

Live functional tests are recorded in `evidence/functional_tests.md` (eleven
scenarios, each with the command, expected and observed result). Things that
went wrong during development and what was done about them are in
`evidence/remediation_log.md` (eleven entries). Screenshots and log captures
from the build stages are in `evidence/`, and `notes/stage0.md` to `stage7.md`
explain what each stage added and why it was built that way.

### What the tests measure, and what they do not

The tutor's feedback on the design proposal asked for clear targets for
retrieval and synthesis quality – recall, precision, citation accuracy,
de-duplication – and for the dataset they would be measured on. Here is the
honest position.

*Measured now.*

- **De-duplication.** Unit-tested (first occurrence of a DOI kept, records
  without a DOI kept and flagged) and visible in every run as
  `deduplicate | in=N | out=M`. Its known miss is also measured: the same
  paper under two DOIs (an arXiv preprint and its published version) is not
  merged – one such pair in 77 approved papers on 30 Sept, and the same pair
  was what the `drop` command was used on in FT-02.
- **Citation accuracy.** Structural, not statistical: every DOI is looked up
  in Crossref (55 of 78 registered on 22 Sept; 78 of 117 on 30 Sept; the rest
  flagged, mostly arXiv DOIs registered with DataCite), the reference list is
  built from the records' own metadata, and a theme can only cite numbers
  1..N of the approved list – the schema rejects anything else (unit-tested).
  So a citation to a paper that does not exist cannot appear in the brief.
- **Grounding of the notes.** Spot-checked, not systematic: in the 22 Sept
  run every number quoted in the evidence notes (five figures) was found in
  the paper's own abstract (`notes/stage6.md`).

*Not measured.*

- **Recall and precision** of retrieval and of relevance screening. There is
  no labelled dataset for the research question, so any figure would be
  invented. The one number available is a disagreement, not an accuracy:
  on the same question the Flash-Lite model kept 42 papers where the full
  Flash model kept 26, which says the cut-off needs calibrating per model but
  not which model is right.

*What a real evaluation would need.* A gold set: for one research question,
two people judge every retrieved record (about 120 on 30 Sept) against the
sub-questions as relevant or not, recording their agreement. Precision of the
`≥ 3/5` cut-off and agreement between model and people (Cohen's κ) follow
from that. Recall needs an independent list of what *should* be found – for
example the included studies of a published systematic review on the same
topic – and the fraction of them the agent retrieves. Only with that set in
hand would targets (say precision ≥ 0.8 at the cut-off, recall ≥ 0.6 against
the review's list) mean anything; none are claimed here without it.

## Project layout

```
main.py                    entry point
config.py                  all tunables in one place
agents/
  models.py                SubQuestion, Paper, Assessment, Brief, Decision (from the class diagram)
  orchestrator.py          deterministic coordinator, review loops, limits, limitations section
  planning_agent.py        decomposition, revision, query reformulation
  retrieval_agent.py       search on every source with failover + threshold check
  evaluation_agent.py      relevance scoring, evidence notes, themes and gaps
  researcher_interface.py  the two human review points
  search_client.py         OpenAlex works API, abstract reconstruction
  s2_client.py             Semantic Scholar Graph API (Stage 7)
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

## Changes after the tutor's feedback on the design proposal

The team design proposal was marked on 29 September (64%, Merit). The
feedback made three points; each is answered here, and by the code where it
could be.

1. **"Not enough evidence to explain why that model was chosen – include
   benchmarks, context limits, function-calling performance and pricing or
   quota information."** Answered under *Why this model* below.
2. **"The targets for measuring retrieval and synthesis quality are still
   quite broad – set clear targets for recall, precision, citation accuracy
   and de-duplication, and explain what dataset you will use."** Answered
   under *What the tests measure, and what they do not* above: what is
   measured, what is not, and what a real evaluation would need. No target is
   stated without a dataset to measure it on.
3. **"The main retrieval process depends heavily on the Semantic Scholar API,
   with OpenAlex only used as a fallback – use several sources as part of the
   normal retrieval process, with automatic failover."** Built in Stage 7:
   every sub-question is searched on OpenAlex *and* Semantic Scholar, results
   are merged and de-duplicated by DOI, and a source that fails after its
   retries is skipped for that query and tried again on the next, with the
   failure recorded in the brief. On the standard question this took the
   unique papers found from 78 to 117 (`evidence/functional_tests.md`,
   FT-10), and it kept the run alive when Semantic Scholar's public pool
   rate-limited three of five searches. Details in `notes/stage7.md`.

## Why this model

The proposal named a general-purpose Gemini Flash model without saying why.
This is the reasoning behind the default, `gemini-3.5-flash-lite`, with the
provider's own figures (Google AI for Developers pages and the model card,
read 30 September 2026).

- **What the model has to be good at here.** The model does three things:
  turn a question into 3–5 sub-questions with search queries; score up to ten
  abstracts at a time for relevance and give a one-sentence reason each; write
  a short note per abstract and then themes and gaps across up to ~80 notes.
  All three are short-context, JSON-structured, high-volume judgement tasks.
  None needs long reasoning chains or tool use by the model itself – the
  orchestrator calls the tools (search, Crossref, storage) deterministically,
  so the model's *function-calling* ability is deliberately not relied on;
  what matters is that it returns valid, schema-conforming JSON.
- **Context limits.** Gemini 3.5 Flash-Lite accepts 1,048,576 input tokens and
  produces up to 65,536 output tokens. The largest prompt this system sends
  is the synthesis prompt – 48,059 characters (roughly 12,000 tokens) on
  30 Sept – and a scoring batch is about 14,000 characters. Context is
  therefore not a constraint; the batches of ten are for prompt discipline
  and cheaper repairs, not to fit a window.
- **Structured output.** The model page lists structured outputs and function
  calling as supported; the client sends `responseMimeType: application/json`
  and validates every reply against a Pydantic schema, with one repair retry.
  Observed: on the Lite model the reply occasionally came back as a bare JSON
  list instead of an object and was repaired on the single retry (FT-09,
  seven times in the 30 Sept run); on the full Flash model that path never
  fired. That difference is the measured cost of the cheaper model.
- **Quota and rate limits (free tier).** Rate limits are per project and are
  shown per model in AI Studio; when read on 22 September the full Flash
  models allowed 5 requests/minute and 20 requests/day, the Flash-Lite model
  15/minute and 500/day. A run costs roughly 30–40 requests (plan, scoring
  batches, notes, synthesis, plus repairs), so the full Flash free tier is one
  run a day and hit its daily cap mid-run twice (remediations #7–#8); the
  Lite tier is not a constraint. Requests-per-day quotas reset at midnight
  Pacific time, which is why the client stops at once on a per-day quota
  error rather than retrying.
- **Price (paid tier, per million tokens, USD).** Gemini 3.5 Flash-Lite: $0.30
  input, $2.50 output. Gemini 3.6 Flash: $0.75 input, $3.75 output until
  31 December 2026, then $1.50 / $7.50. A run sends on the order of 100,000
  input tokens and receives around 25,000, so a paid run on Lite would cost in
  the region of ten cents (an estimate from the logged prompt and response
  sizes, not a measured bill).
- **Provider's own guidance and lifecycle.** The models page says "for any
  new projects, use our latest models: 3.5 Flash-Lite or 3.8 Flash", lists
  `gemini-2.0-flash` – the model the design was written against – as shut
  down, and restricts the 2.5 family to existing users. Gemini 3.5 Flash-Lite
  is a stable release (July 2026, based on 3.1 Flash-Lite; knowledge cut-off
  March 2026).
- **Benchmarks.** The model card (results as of July 2026, Google's own
  figures) reports, for Gemini 3.5 Flash-Lite against the model it replaces
  (3.1 Flash-Lite) and two rivals in the same price class (GPT-5.4 mini,
  Claude Haiku 4.5): SWE-Bench Pro 54.2% (38.3% / 54.4% / 39.5%),
  Terminal-Bench 2.1 54.0% (31.0% / 59.2% / 44.2%), MLE-Bench 39.2% (22.0%
  / – / –), GDPVal-AA v2 1140 Elo (642 / 1171 / 907), OSWorld-Verified 74.0%
  (54.3% / 72.1% / 50.7%), CharXiv Reasoning 74.5% without tools (73.2% /
  80.3% / 61.7%), and GDM-MRCR v2 long-context 72.2% at 128k (60.1% / 42.7%
  / 35.3%). The same table prices the rivals at $0.75–$1.00 input and
  $4.50–$5.00 output per million tokens against Lite's $0.30 / $2.50. None of
  these benchmarks measures abstract screening, so they support the choice
  only indirectly – the model is in the top group of its price class on
  agentic and information-synthesis tasks; the measure that matters for this
  system is how often it returns valid, well-grounded JSON, and that is what
  the functional tests record.
- **Data use.** On the free tier the provider states that prompts are "used to
  improve our products"; on the paid tier they are not. The only user data
  this system sends is the research question and the researcher's review
  feedback; the abstracts are public records. Anyone using it on a
  confidential question should use a paid key.
- **What would change the choice.** The Lite model is a more generous judge
  than the full Flash model (42 papers kept against 26 on the same question),
  so the relevance cut-off should be calibrated per model before the
  brief's counts are compared across models. Because the model name is
  configuration, switching the run to `gemini-3.8-flash` is a one-line change
  in `.env` once its free quota or a paid key allows it; using a stronger
  model for scoring only and the Lite model for the rest would be a small
  code change, and is the first thing to try if screening quality, not cost,
  becomes the priority.

## Known limitations

- Two sources (OpenAlex and Semantic Scholar), 20 results per query per
  source, abstracts only – no full text, no citation chasing, no date filters.
  The brief says so in its last section. Without a Semantic Scholar key the
  public pool rate-limited three of five searches on 30 Sept; failover kept
  the run going, but coverage for those sub-questions was OpenAlex only.
- A paper that appears under two DOIs (arXiv preprint and published version)
  is not merged; the researcher can drop one at Review 2.
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
`.env`); OpenAlex works API (https://docs.openalex.org); Semantic Scholar
Academic Graph API (https://api.semanticscholar.org/api-docs/, `graph/v1`,
rate-limit notes at https://www.semanticscholar.org/product/api/tutorial);
Crossref REST API (https://api.crossref.org).

Provider documentation used for *Why this model* (all read 30 September
2026): Google AI for Developers – Models
(https://ai.google.dev/gemini-api/docs/models), Gemini 3.5 Flash-Lite model
page (https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite),
Pricing (https://ai.google.dev/gemini-api/docs/pricing), Rate limits
(https://ai.google.dev/gemini-api/docs/rate-limits); Google DeepMind, *Gemini
3.5 Flash-Lite Model Card* (July 2026,
https://deepmind.google/models/model-cards/gemini-3-5-flash-lite/) and its
evaluation methodology note. Free-tier per-model limits were read from the
AI Studio rate-limit page on 22 September 2026.

Python libraries (see `requirements.txt`): `requests` (HTTP), `pydantic`
(data models and schema validation), `python-dotenv` (configuration), `pytest`
(tests). No agent framework is used; the orchestration is plain Python so that
every step can be read and tested.
