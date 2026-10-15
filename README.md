# Research Agent – an LLM-powered planning agent for academic literature search

Individual implementation for the Intelligent Agents module (MSc, University of
Essex Online), built from the Group D design proposal.

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
question ─► Planning Agent ⇄ [Review 1] ─► Retrieval Agent ─► de-duplicate ─► verify DOIs
                 ▲        (revise until         │  (OpenAlex + Semantic Scholar, failover;
                 │         approved)            ▼   threshold miss → one reformulation)
          reject scope (once) ◄── [Review 2] ◄── Evaluation & Synthesis Agent (score)
                                        │ approve (after dropping papers, if wanted)
                                        ▼
                     Evaluation & Synthesis Agent (notes, themes, gaps) ─► brief.md + brief.json
```

There are three agents and one coordinator, matching the class diagram in the
proposal (Group D, 2026; `agents/models.py` holds the data classes from that
diagram, and the differences from the drawing are listed under *Where the
build departs from the proposal*):

| Component | File | Uses the LLM? | Job |
|---|---|---|---|
| Orchestrator | `agents/orchestrator.py` | no | Runs the steps in a fixed order, enforces the retry and review limits, does the exact work (de-duplication, verification, storage) itself |
| Planning Agent | `agents/planning_agent.py` | yes | Turns the question into sub-questions and search queries; rewrites a query once if a search falls short |
| Retrieval Agent | `agents/retrieval_agent.py` | no | Searches every configured source (OpenAlex and Semantic Scholar) for each sub-question, carries on if one source fails, and judges whether it got enough usable records |
| Evaluation & Synthesis Agent | `agents/evaluation_agent.py` | yes | Scores each paper's relevance 1–5 with a reason, then writes the notes, themes and gaps from the approved set |
| Researcher interface | `agents/researcher_interface.py` | no | The two review points; can be scripted so the loops are testable |

The autonomy is deliberately bounded: at most one automatic query
reformulation per sub-question and one scope revision per run. Those numbers,
and the other settings (time-outs, back-off, batch sizes, the title-match
ratio), live in `config.py`.

The model is used only for judgement tasks (decomposing, scoring, summarising).
Anything with an exact answer is ordinary code. Every model reply is parsed
against a schema; if it does not fit, the error is sent back once for repair,
and if that fails the run stops rather than continuing on bad data.

### Why it is built this way

The reasoning starts from the proposal's (Group D, 2026, sections 2 and 4).
This is the short form, with the sources it rests on; three of them
(Wooldridge, 2009; Russell and Norvig, 2021; Bratman, Israel and Pollack,
1988) are module reading that the proposal did not use.

- **Three agents, each with a goal of its own.** An agent is a system that
  acts on its own to meet its objectives rather than doing only what it is
  told step by step (Wooldridge, 2009); the properties usually asked of one
  are autonomy, social ability, reactivity and pro-activeness (Wooldridge and
  Jennings, 1995). Here each agent has one goal and a test for it: a plan of
  three to five sub-questions that covers the question, enough different
  usable records, evidence that answers the sub-question. Only the Retrieval
  Agent applies its test entirely by itself (it is a count). Whether the plan
  covers the question is judged by the researcher at Review 1, and whether
  enough of the evidence is relevant is checked by the orchestrator from the
  Evaluation Agent's scores. Measured against that list they are limited
  agents: they pursue a goal and react to what a search returns, but their
  social ability is minimal. They do not negotiate, and everything that
  passes between them – the plan, the records, a request for a new query –
  goes through the orchestrator. The split by role
  follows the Gaia idea of designing a system as a set of roles with
  responsibilities (Wooldridge, Jennings and Kinny, 2000), and it means a
  failure can be traced to the role that caused it (Hong et al., 2024).
- **A fixed coordinator instead of agents negotiating.** More agents and more
  coordination do not reliably give better results: the gain depends on the
  task and the architecture, and designs without central verification pass
  errors on more strongly (Kim et al., 2026). So the agents never call each
  other; everything goes through the orchestrator, which is ordinary code.
- **The model only where there is no exact answer.** Language models produce
  fluent output that nothing supports (Klesel and Wittmann, 2025), fabricate
  references and get the details of real ones wrong (Walters and Wilder,
  2023), and write
  summaries that are unfaithful to their source (Maynez et al., 2020). The
  records therefore come from scholarly indexes, not from the model; every DOI
  is checked against Crossref; each note is written from one abstract and can
  be checked against it; and the reference list is assembled by code.
- **Two human reviews and hard limits.** The search environment is only partly
  observable and not deterministic (Russell and Norvig, 2021): what a query
  will return cannot be known in advance, and a source can refuse or fail, as
  Semantic Scholar did for three of five searches on 30 Sept. So the agent is
  not left to iterate on its own. The researcher approves the plan
  before any search (planning with feedback: Wang et al., 2024) and the
  evidence before anything is written, which keeps the evaluative decisions
  with the person and the legwork with the system (Zhu et al., 2026; Bolaños
  et al., 2024). An approved plan is then kept, not reopened at every step,
  and the retries are capped – the way an agent with limited resources avoids
  deliberating without end (Bratman, Israel and Pollack, 1988).
- **Abstracts, not full texts.** Screening on titles and abstracts is how
  tool-assisted systematic reviews already work (van de Schoot et al., 2021);
  it keeps the system feasible on free tiers and the brief says plainly that
  the full texts still have to be read. The prompts use the first 1,200
  characters of each abstract.

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
| `OPENALEX_EMAIL` | Optional. Your email, sent with OpenAlex and Crossref requests as a contact address, as they ask | – |
| `OPENALEX_API_KEY` | Optional but worth having: a free OpenAlex key raises the daily search budget 10× (remediation #10) | https://openalex.org/settings/api |
| `SOURCES` | Literature sources searched for every sub-question, in order; default `openalex,semanticscholar` | – |
| `S2_API_KEY` | Optional free Semantic Scholar key: a dedicated 1 request/second instead of the shared public pool | https://www.semanticscholar.org/product/api#api-key-form |
| `S2_ENDPOINT` | Optional. The Semantic Scholar search address; only ever changed to make that source fail on purpose (functional test FT-11) | – |

Only `GEMINI_API_KEY` is required. Leave an optional line empty rather than
filling it with placeholder text: whatever is there is sent.

Model note: when checked on 23 September the free tier of the full Flash
models allowed 20 requests a day – less than one fresh run – and the
Flash-Lite model 500. If you have a paid key any Gemini text model works; only
the name changes. The reasons behind the default model are set out under *Why
this model* below.

## Running it

```bash
python main.py "What approaches reduce hallucination in LLM-based autonomous agents?"
```

or just `python main.py` and type the question when asked. A boundary for the
plan and the queries can be added with `--constraints`, for example
`--constraints "healthcare settings only; empirical studies, not surveys"`.
It is written into every planning and query-rewriting prompt and recorded in
the brief. It steers what is searched for; it is not a filter on what comes
back (a date limit, for instance, cannot be enforced this way). A run looks
like this:

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

113 unit tests, no network needed: the API replies they rely on are recorded in
`tests/fixtures/`, and every test gets a throw-away cache folder so it can
neither read nor write the real one. They cover the data models, abstract
reconstruction, de-duplication and DOI checks, the schema-validation-and-repair
path, the 3–5 sub-question rule, the review loops, the threshold and
reformulation caps, the relevance cut-off, batching, themes that cite
non-existent papers, the retry policies, the brief layout, the second source,
the merging of results and the failover between sources (Stage 7), and – added
in the read-through before submission – the disk cache (including a damaged
file), what the model client does and does not keep in it, the OpenAlex
request, logging, the constraints option, what an error message may contain,
the plan being shown to the model when it is revised, and how a run ends
(brief written, stopped with a message, interrupted). The output of the run on
the submitted code is `evidence/pytest_113_passed.txt`.

`pytest --cov=agents --cov=main --cov=config` (needs `pytest-cov`, which is in
`requirements.txt`) reports which lines the tests run.

Live functional tests are recorded in `evidence/functional_tests.md`: sixteen
scenarios with the steps, the expected and the observed result. Eleven were
run during development (four of those are observations made during other
runs). Five were run on 2 Oct on the submitted code with
`evidence/live_tests_20261002.sh`; what the terminal showed is in
`evidence/ft12_...log` to `ft16_...log`. They are followed by an offline
replay of the 30 Sept run, which checks that the corrections of 1 Oct left its
result unchanged (`evidence/replay_offline.py`, result in
`evidence/replay_offline.log`).
Things that went wrong during development and what was done about them are in
`evidence/remediation_log.md` (twenty-one entries). Screenshots and log captures
from the build stages are in `evidence/`, and `notes/stage0.md` to `stage7.md`
explain what each stage added and why it was built that way.

The relevance screening of one run was also checked from outside the
pipeline. `evidence/judge_relevance.py` shows a person every paper of a run,
one at a time, from the text the model scored, and hides the agent's
decisions until every paper has an answer. It was started on the telemedicine
run (FT-13) and set aside after four papers, because judging all 65 by hand
would have taken too long; those four answers are not included.
`evidence/judge_with_model.py` puts the person's question ("Would you keep
this paper for this sub-question?"), with the same definitions of yes and no, to
the full Flash model (`gemini-3.6-flash`), without the agent's
scores. Its comparison for FT-13 is in
`evidence/model_judge_20261002-090832_results.md` and is summarised below.
Both programs read the search replies saved in `cache/`, which is not in the
repository, so they can only be repeated on the machine that made the run;
the second model's answers, comparison and log are the evidence.

### What the tests measure, and what they do not

The proposal said success would be measured by retrieval relevance, citation
accuracy, goal preservation and successful storage. The tutor's feedback asked
for clear targets for retrieval and synthesis quality – recall, precision,
citation accuracy, de-duplication – and for the dataset they would be measured
on. Here is the honest position.

*Measured now.*

- **De-duplication.** Unit-tested (first occurrence of a DOI kept, records
  without a DOI kept and flagged) and visible in every run as
  `deduplicate | in=N | out=M`. Its known miss is also measured: the same
  paper under two DOIs is not merged – four such pairs among the 77 approved
  papers on 30 Sept (three are an arXiv preprint and its published version,
  one is the same article under two publishers' DOIs), the same four among the
  42 approved on 23 Sept, where FT-02 used `drop` on one of them. The same
  miss showed on a new topic on 2 Oct: in FT-13 the 2022 ADA/EASD consensus
  report on hyperglycaemia, published in both *Diabetes Care* and
  *Diabetologia*, was approved twice.
- **Citation accuracy.** Structural, not statistical: every DOI is looked up
  in Crossref (55 of 78 registered on 22 Sept; 78 of 117 on 30 Sept; the rest
  flagged, mostly arXiv DOIs registered with DataCite; on other topics nearly
  all verify – 65 of 65, 101 of 104 and 75 of 77 in FT-13 to FT-15), the
  reference list is
  built from the records' own metadata, and a theme can only cite numbers
  1..N of the approved list – the schema rejects anything else (unit-tested).
  So a citation to a paper that does not exist cannot appear in the brief.
- **Grounding of the notes.** Spot-checked, not systematic: in the 22 Sept
  run every number quoted in the evidence notes (percentages, study counts
  and dataset sizes, in five of the 26 notes) was found in the paper's own
  abstract (`notes/stage6.md`).
- **Storage.** Unit-tested (the saved JSON reads back as the same brief) and
  seen in every completed run: a Markdown and a JSON file per brief in
  `output/`.

*Not measured.*

- **Recall and precision** of retrieval and of relevance screening. There is
  no labelled dataset for the research question, so any figure would be
  invented. Two comparisons between models are available; neither measures
  accuracy. On the same question the Flash-Lite model kept 42 of 77 papers
  where the full Flash model kept 26 of 78 – each on the plan it had written
  itself, so the two sets of papers are not the same. And on the telemedicine
  run (FT-13) the same full Flash model, `gemini-3.6-flash`, judged the
  agent's 65 papers from the same text without seeing the agent's scores
  (`evidence/judge_with_model.py`). Of the 35 kept papers it gave a clear
  answer on, it would also keep 27 (0.77); of the 28 dropped papers it gave a
  clear answer on, it would keep 3, all scored 2. Over all its clear answers,
  Cohen's κ between the two was 0.65. It would keep every paper the agent
  scored 4 or 5 (11 of 11). All 8 kept papers it would not keep had a score
  of exactly 3, the cut-off, and for each of them the agent's own reason
  already said the paper was broader than the sub-question or about
  something else (type 1 instead of type 2 diabetes, no telemedicine, not
  specific to diabetes, clinical outcomes instead of barriers and acceptance). Both
  are Gemini models and may share blind spots, so this says where the
  cut-off is soft, not which papers are right.
- **Goal preservation** – whether the finished brief still answers the
  question that was asked. It is guarded, not measured: the researcher sees
  the plan before any search and the evidence before any writing. FT-12 shows
  a plan corrected at Review 1 much as asked – two sub-questions kept, two
  combined (FT-03, on the earlier prompt, did not) – and FT-04 a scope
  corrected at Review 2. There is no score for it.
- **Variety of questions.** The proposal planned end-to-end tests on a fixed
  set of questions chosen for variety. That was done late and only in part.
  Until 2 Oct every completed live run used one research question (in two
  wordings). On 2 Oct three more were run once each (FT-13 to FT-15): one on
  telemedicine for diabetes, one on food waste in retail supply chains with a
  constraint, and a deliberately vague one, "is remote work good". Each
  produced a plan on its topic and a complete brief with no change to code or
  settings. That shows the pipeline is not tied to one topic. It does not
  show that those briefs are good: no person judged them, and for the
  telemedicine brief only the screening was checked, by a second model
  (above).
  (The word "test", typed for FT-01, started one run by mistake on 22 Sept;
  it ended on the daily quota without a brief.)

*Targets, and the dataset each needs.* These are the targets the system
should be held to. None of the first three has been measured, because the
datasets have not been built; no result is claimed for them. The
second-model check above does not stand in for the gold set: it measures
agreement between two models, not precision.

| Measure | Target | Dataset and method |
|---|---|---|
| Precision of screening | ≥ 0.8 of the papers kept at the `≥ 3/5` cut-off judged relevant by people | Gold set: for one research question, two people judge every retrieved record (about 120 on 30 Sept) against the sub-questions, recording their agreement |
| Agreement of model and people | Reported as Cohen's κ; no threshold set | The same gold set |
| Recall of retrieval | ≥ 0.6 of an independent list found | The included studies of a published systematic review on the same topic |
| Citation accuracy | Every reference in a brief resolves to the paper it names | Each brief's reference list; today enforced by construction, as described above, and not yet checked by hand |
| De-duplication | No paper twice in the approved list | Each brief's approved list; **not met** – four pairs among 77 on 30 Sept; on 2 Oct three among 57 in FT-12 and one among 36 in FT-13 |

## Project layout

```
main.py                    entry point
config.py                  the settings, in one place
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
evidence/                  screenshots, logs, test output, functional test log, remediation log,
                           the offline replay script and its log, the live-test script of 2 Oct and its logs,
                           the two programs that check the screening (by a person, by a second model)
                           and the second model's answers, comparison and log for FT-13
output/                    briefs from the live runs and run.log
presentation/              the slide deck and the narration transcript
```

## Changes after the tutor's feedback on the design proposal

The team design proposal received 64% (Merit). I read the feedback on
29 September. It made three points; each is answered here, and by the code
where it could be.

1. **"... there is not enough evidence to explain why that model was chosen.
   Include relevant benchmarks, context limits, function-calling performance
   and pricing or quota information to support your choice."** Answered under
   *Why this model* below.
2. **"... the targets for measuring retrieval and synthesis quality are still
   quite broad. Set clear targets for things such as recall, precision,
   citation accuracy and deduplication, and explain what dataset you will use
   to test them."** Answered under *What the tests measure, and what they do
   not* above: what is measured now, the targets, and the dataset each one
   needs. The targets are stated; the datasets to measure the first three on
   have not been built. The screening of one run has since been checked by a
   second model, as described there; that is agreement between two models,
   not the measurement the feedback asked for.
3. **"The main retrieval process depends heavily on the Semantic Scholar API,
   with OpenAlex only used as a fallback. Consider using several sources as
   part of the normal retrieval process, with automatic failover ..."** Built
   in Stage 7:
   every sub-question is searched on OpenAlex *and* Semantic Scholar, results
   are merged and de-duplicated by DOI, and a source that fails after its
   retries is skipped for that query and tried again on the next, with the
   failure recorded in the brief. With the same plan and the same model this
   took the unique papers found from 77 (one source, 23 Sept) to 117 (two
   sources, 30 Sept; `evidence/functional_tests.md`, FT-10), and it kept the
   run alive when Semantic Scholar's public pool rate-limited three of five
   searches on 30 Sept. On 2 Oct that pool refused every live search (16 of
   16, FT-12 to FT-15): the runs completed and their briefs say so, but
   without a key the second source added nothing new that day. The design is
   described in `notes/stage7.md`, the runs in
   `evidence/functional_tests.md`.

## Where the build departs from the proposal

- **Which source came first.** The proposal named Semantic Scholar as the
  primary source and OpenAlex as the fallback, one source at a time. Stages 1
  to 6 were built and tested on OpenAlex alone, because it needs no key
  (`notes/stage1.md`); Semantic Scholar was added in Stage 7, and since then
  both are searched for every sub-question.
- **How it was tested end to end.** One research question for the whole of
  the development; three more were run once each on 2 Oct (FT-13 to FT-15).
  That is not the fixed set of varied questions, used throughout, that the
  proposal planned (see above).
- **Identifiers.** The proposal spoke of run and paper identifiers. There is
  no separate id for either: a run is identified by the timestamp in its
  brief's file name and in `output/run.log`, a paper by its DOI, or by its
  title when it has none.
- **A second scope rejection.** In Diagrams 2 and 3 a second rejection stops
  the run and recommends starting again with a rephrased question. The build
  does not offer `reject` again once the one revision is used and refuses it
  if typed (FT-04); the researcher can approve, drop papers, or stop the run
  with Ctrl-C.
- **What Crossref checks.** The proposal has Crossref verify "each retrieved
  DOI and its metadata". The build checks that the DOI is registered and that
  the title registered under it matches the one retrieved. Authors and year
  are not compared (a preprint and its published version often differ in
  year).
- **Order of the steps, and no ranking.** In Diagram 2 everything is retrieved
  first, then de-duplicated, verified and evaluated once, and the model
  returns "assessments, ranking, themes, gaps" before Review 2. The build
  de-duplicates, verifies and scores one sub-question at a time, so that the
  second rewrite trigger below can act on the scores; it does not rank (papers
  are listed by sub-question with their scores); and themes and gaps are
  written only after Review 2, from the approved papers.
- **Classes that differ from Diagram 1.** `Assessment` holds the paper's DOI
  and title and the sub-question's id instead of references to the objects,
  an integer score from 1 to 5 instead of a float, and the reason. `Paper`
  gained `source` and `subquestion_id`, and its `doi_verified` has three
  states (verified, flagged, not checked) instead of two. `Brief` gained the
  interpretation and the assessments. `RetrievalAgent` has no `searchBudget`,
  and its `requestReformulation` became a reason it hands to the
  orchestrator. `synthesise` returns themes and gaps; the orchestrator
  assembles the brief. Constraints are a line of text, not a class, given
  to the orchestrator when it is created rather than to `run` or
  `submitQuestion`. `revise_plan` takes the question and the plan under
  review as well as the feedback (remediation #18). The evaluation agent has
  a separate `summarise`. De-duplication, verification and saving are
  functions in `validation.py` and `storage.py` that the orchestrator calls,
  and the retry counts are kept by sub-question id.
  `SubQuestion.approved` is set at Review 1 and nothing reads it: the
  "validated states" of section 5 are the schema checks, not a state machine.
- **Three smaller ones.** A second trigger for the single query rewrite,
  after scoring, when too few of the records turn out to be relevant (the
  proposal had only the count of records); one shared retry policy for the
  scholarly APIs, added when they turned out to rate-limit (remediation #9);
  and the replaced query kept on the sub-question so the brief can show both.

## Why this model

The proposal asked only for "a general-purpose LLM, accessed via a free-tier
hosted API (e.g. Google Gemini Flash)" (section 3) and gave no reasons for a
particular model, which is what the feedback picked up. This is the reasoning
behind the default, `gemini-3.5-flash-lite`, with the provider's own figures
(Google, 2026a, 2026b, 2026c, 2026d; Google DeepMind, 2026a, 2026b).

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
  produces up to 65,536 output tokens (Google, 2026b). The largest prompt this system sends
  is the synthesis prompt – 48,059 characters (roughly 12,000 tokens) on
  30 Sept – and a scoring batch is about 14,000 characters. Context is
  therefore not a constraint; the batches of ten are for prompt discipline
  and cheaper repairs, not to fit a window.
- **Structured output.** The model page lists structured outputs and function
  calling as supported (Google, 2026b); the client sends `responseMimeType: application/json`
  and validates every reply against a Pydantic schema, with one repair retry.
  Observed: on the Lite model a reply often comes back as a bare JSON list
  instead of an object and is repaired on the single retry. FT-09 records the
  first cases; in all it happened four times in the 21 requests of the first
  full run on 23 Sept, eight more in the FT-03/FT-04 run that day, three
  times in the 16 requests of 30 Sept, and for 18 of the 56 first replies in
  the four runs of 2 Oct. No reply has needed a second repair. On the full Flash model that path never fired. That difference is the measured cost of the cheaper model.
- **Quota and rate limits (free tier).** Rate limits are per project and are
  shown per model in AI Studio (Google, 2026d); when read on 23 September the
  full Flash models allowed 5 requests/minute and 20 requests/day, the
  Flash-Lite model 15/minute and 500/day. The first complete run on the Lite
  model, with one source, made 21 requests (23 Sept 10:57: nine scoring
  batches, one rewrite, six note batches, the synthesis and four repairs; the
  plan had been made in an earlier attempt and came from the cache). With two
  sources there are half as many papers again to score and nearly twice as
  many to summarise, so a fresh two-source run is estimated at 30–40 requests;
  a repeat run pays only for what is new (16 on 30 Sept). The three fresh runs
  of 2 Oct, on which only OpenAlex answered, made 18, 28 and 22. The full Flash free tier is
  therefore less than one fresh two-source run a day, and with one source it
  hit its daily cap mid-run twice (remediations #7–#8); the Lite tier is not
  a constraint. Requests-per-day quotas reset at midnight Pacific time
  (Google, 2026d), which is why the client stops at once on a per-day quota
  error rather than retrying.
- **Price (paid tier, per million tokens, USD; Google, 2026c).** Gemini 3.5
  Flash-Lite: $0.30 input, $2.50 output. Gemini 3.6 Flash, the full Flash
  model Stages 3 to 6 ran on: $0.75 input, $3.75 output until 31 December
  2026, then $1.50 / $7.50. A run sends on the order of 100,000
  input tokens and receives around 25,000, so a paid run on Lite would cost in
  the region of ten cents (an estimate from the logged prompt and response
  sizes, not a measured bill).
- **Provider's own guidance and lifecycle.** The models page says "For any
  new projects, use our latest models: 3.5 Flash-Lite or 3.8 Flash" (Google,
  2026a), lists
  `gemini-2.0-flash` – the model the first build was configured for
  (remediation #2) – as shut down, and restricts the 2.5 family to existing
  users. Gemini 3.5 Flash-Lite was published in July 2026, is based on 3.1
  Flash-Lite and has a knowledge cut-off of March 2026 (Google DeepMind,
  2026b).
- **Benchmarks.** Google DeepMind's page for the model (2026a; Google's own
  figures) reports, for Gemini 3.5 Flash-Lite against the model it replaces
  (3.1 Flash-Lite) and two small models from other providers (GPT-5.4 mini,
  Claude Haiku 4.5): SWE-Bench Pro 54.2% (38.3% / 54.4% / 39.5%),
  Terminal-Bench 2.1 54.0% (31.0% / 59.2% / 44.2%), MLE-Bench 39.2% (22.0%
  / – / –), GDPVal-AA v2 1140 Elo (642 / 1171 / 907), OSWorld-Verified 74.0%
  (54.3% / 72.1% / 50.7%), CharXiv Reasoning 74.5% without tools (73.2% /
  80.3% / 61.7%), and GDM-MRCR v2 long-context 72.2% at 128k (60.1% / 42.7%
  / 35.3%). The same table prices the other two at $0.75–$1.00 input and
  $4.50–$5.00 output per million tokens against Lite's $0.30 / $2.50. None of
  these benchmarks measures abstract screening, so they support the choice
  only indirectly – of the three small models it is first or second on every
  row where all three are reported, at the lowest price; the measure that matters for this
  system is how often it returns valid, well-grounded JSON, and that is what
  the functional tests record.
- **Data use.** On the free tier the provider states that content is "used to
  improve our products"; on the paid tier it is not (Google, 2026c). The only
  user data this system sends to the model is the research question, any
  constraint given with it and the researcher's review feedback; the
  abstracts are public records. (The
  contact email in `.env` goes to OpenAlex and Crossref with each request, as
  those services ask; the search queries go to OpenAlex and Semantic
  Scholar.) Anyone using it on a confidential question should use a paid key.
- **What would change the choice.** The Lite model looks like a more generous
  judge than the full Flash model (42 of 77 papers kept against 26 of 78, each
  on its own plan). On the same 65 papers of FT-13, `gemini-3.6-flash` would
  not keep 8 that Lite kept, all of them scored 3 by Lite, and would keep 3
  that Lite dropped. So the relevance cut-off should be calibrated per model before the
  brief's counts are compared across models. Because the model name is
  configuration, switching the run to a full Flash model (`gemini-3.8-flash`
  is the one the provider now recommends; its price and quota were not looked
  up) is a one-line change in `.env` once a free quota or a paid key allows
  it; using a stronger
  model for scoring only and the Lite model for the rest would be a small
  code change, and is the first thing to try if screening quality, not cost,
  becomes the priority.

## Known limitations

- Two sources (OpenAlex and Semantic Scholar), 20 results per query per
  source, abstracts only – no full text, no citation chasing, no date filters.
  The brief says so in its Limitations section. Without a Semantic Scholar key the
  public pool rate-limited three of five searches on 30 Sept and all 16 live
  searches on 2 Oct; failover kept every run going, but coverage for those
  searches was OpenAlex only. In practice the second source needs a key
  (`S2_API_KEY`).
- A paper that appears under two DOIs (usually an arXiv preprint and its
  published version) is not merged; the researcher can drop one at Review 2. Four of the 77
  papers in the 30 Sept brief are such second copies; in FT-13 a consensus
  report published in two journals was approved twice.
- The development and the first eleven live tests used one research question.
  Three other questions were run once each on 2 Oct and completed, but no
  person has judged those briefs. A second model checked the screening of
  one of them (FT-13; see *What the tests measure*): it agreed on most papers,
  and every disagreement was at a score of 2 or 3, either side of the
  cut-off. How well the scoring and the cut-off carry
  over to other topics is still not measured.
- Relevance and summaries are model judgements from abstracts. They are
  traceable and checkable, not authoritative; the brief tells you to check the
  full text before relying on them. The prompts use the first 1,200 characters
  of an abstract (`EVAL_ABSTRACT_CHARS`); 59 of the 77 approved papers on
  30 Sept had a longer one, so anything said only near its end was not seen.
- The fixes of 1 October (remediations #12–#21) were run live on 2 Oct only in
  part (FT-12 to FT-16: the stop messages, the single-rewrite cap, the
  constraints option, the revision prompt at Review 1). Covered by unit
  tests only: #14 and #15, whose cases did not arise; the error text of #19;
  the interrupted cache write of #20; Ctrl-C in the middle of a run; and a
  revision after a scope rejection. FT-01 to FT-11 were not repeated. The
  full account is in the notes of `evidence/functional_tests.md`.
- A constraint steers the plan and the queries only. Nothing checks that the
  model kept to it. In FT-14 the constraint "empirical studies only, not
  reviews" showed in the plan's reading of the question and in both rewritten
  queries, not in the sub-questions or their first queries, and 7 of the 32
  approved papers were still reviews or overviews by their titles.
- Papers whose DOIs are registered with DataCite rather than Crossref (mostly
  arXiv preprints) are flagged "unverified", not dropped.
- One automatic query rewrite is a modest safety net, and what it achieves
  varies. For the thin sub-question on evaluation it took the relevant papers
  from one to three on the full Flash model (21–22 Sept), from two to six on
  the Lite model with one source (23 Sept), and from two to twenty-two with
  two sources (30 Sept). On another topic (FT-14, 2 Oct) it took one
  sub-question from three relevant papers to four, still below the threshold,
  and another from two to seven. When it is not enough, the reject-scope
  option at Review 2 is what is left.
- The free-tier quotas of the model and search providers shape how often you
  can run it; see the model note above and remediation entries 5–10
  (overload, rate limits and quotas).

## Sources and libraries

Design: Group D (2026), the team's design proposal for Assessment 1 – in
particular sections 2 (proposed architecture and agent roles), 3 (system
requirements), 4 (design decisions and technical rationale), 5 (challenges,
risks and mitigations) and 6 (development approach and evaluation), and the
class, sequence and activity diagrams (Diagrams 1 to 3). The reasons behind
individual design decisions are given in the module docstrings and comments
where they apply; they cite the literature by author and date, as above, and
the proposal by section. The full references are at the end of this file.

Services: Google Gemini API (`generativelanguage.googleapis.com`, model set in
`.env`); OpenAlex works API (https://docs.openalex.org); Semantic Scholar
Academic Graph API (https://api.semanticscholar.org/api-docs/, `graph/v1`,
rate limits at https://www.semanticscholar.org/product/api);
Crossref REST API (https://api.crossref.org).

Provider documentation used for *Why this model*: the Gemini API pages for
the models, the Flash-Lite model, pricing and rate limits (Google, 2026a to
2026d) and Google DeepMind's model page and model card (2026a, 2026b), all in
the reference list. Free-tier per-model limits were read from the AI Studio
rate-limit page (https://aistudio.google.com/rate-limit, behind a login) on
23 September 2026.

Python libraries (see `requirements.txt`): `requests` (HTTP), `pydantic`
(data models and schema validation), `python-dotenv` (configuration), `pytest`
and `pytest-cov` (tests and coverage). No agent framework is used; the
orchestration is plain Python so that every step can be read and tested.

## References

Bolaños, F. et al. (2024) 'Artificial intelligence for literature reviews:
opportunities and challenges', *Artificial Intelligence Review*, 57(10),
article 259. Available at: https://doi.org/10.1007/s10462-024-10902-3

Bratman, M.E., Israel, D.J. and Pollack, M.E. (1988) 'Plans and
resource-bounded practical reasoning', *Computational Intelligence*, 4(3),
pp. 349–355. Available at: https://doi.org/10.1111/j.1467-8640.1988.tb00284.x

Google (2026a) *Models*. Gemini API documentation. Available at:
https://ai.google.dev/gemini-api/docs/models (Accessed: 30 September 2026).

Google (2026b) *Gemini 3.5 Flash-Lite*. Gemini API documentation. Available
at: https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite
(Accessed: 30 September 2026).

Google (2026c) *Gemini Developer API pricing*. Gemini API documentation.
Available at:
https://ai.google.dev/gemini-api/docs/pricing (Accessed: 30 September 2026).

Google (2026d) *Rate limits*. Gemini API documentation. Available at:
https://ai.google.dev/gemini-api/docs/rate-limits (Accessed: 30 September
2026).

Google DeepMind (2026a) *Gemini 3.5 Flash-Lite*. Available at:
https://deepmind.google/models/gemini/flash-lite/ (Accessed: 1 October 2026).

Google DeepMind (2026b) *Gemini 3.5 Flash-Lite – model card*. Available at:
https://deepmind.google/models/model-cards/gemini-3-5-flash-lite/ (Accessed:
30 September 2026).

Group D (2026) *Project report*. Team design proposal (Assessment 1),
Intelligent Agents, University of Essex Online. Unpublished.

Hong, S. et al. (2024) 'MetaGPT: meta programming for a multi-agent
collaborative framework', *The Twelfth International Conference on Learning
Representations*, Vienna, Austria, 7–11 May 2024. OpenReview.net. Available
at: https://openreview.net/forum?id=VtmBAGCN7o (Accessed: 24 August 2026).

Kim, Y. et al. (2026) 'Capable language models can outgrow the benefits of
collaboration', *Nature Machine Intelligence*, 8(7), pp. 1157–1172. Available at:
https://doi.org/10.1038/s42256-026-01268-y

Klesel, M. and Wittmann, H.F. (2025) 'Retrieval-augmented generation (RAG)',
*Business & Information Systems Engineering*, 67(4), pp. 551–561. Available at:
https://doi.org/10.1007/s12599-025-00945-3

Maynez, J. et al. (2020) 'On faithfulness and factuality in abstractive
summarization', *Proceedings of the 58th Annual Meeting of the Association for
Computational Linguistics*, Online, July 2020. Association for Computational
Linguistics, pp. 1906–1919. Available at: https://doi.org/10.18653/v1/2020.acl-main.173

Russell, S. and Norvig, P. (2021) *Artificial intelligence: a modern
approach*. 4th edn. Pearson Education.

van de Schoot, R. et al. (2021) 'An open source machine learning framework for
efficient and transparent systematic reviews', *Nature Machine Intelligence*,
3(2), pp. 125–133. Available at: https://doi.org/10.1038/s42256-020-00287-7

Walters, W.H. and Wilder, E.I. (2023) 'Fabrication and errors in the
bibliographic citations generated by ChatGPT', *Scientific Reports*, 13(1),
article 14045. Available at: https://doi.org/10.1038/s41598-023-41032-5

Wang, L. et al. (2024) 'A survey on large language model based autonomous
agents', *Frontiers of Computer Science*, 18(6), article 186345. Available at:
https://doi.org/10.1007/s11704-024-40231-1

Wooldridge, M. (2009) *An introduction to multiagent systems*. 2nd edn. John
Wiley & Sons.

Wooldridge, M. and Jennings, N.R. (1995) 'Intelligent agents: theory and
practice', *The Knowledge Engineering Review*, 10(2), pp. 115–152. Available
at: https://doi.org/10.1017/S0269888900008122

Wooldridge, M., Jennings, N.R. and Kinny, D. (2000) 'The Gaia methodology for
agent-oriented analysis and design', *Autonomous Agents and Multi-Agent
Systems*, 3(3), pp. 285–312. Available at:
https://doi.org/10.1023/A:1010071910869

Zhu, L. et al. (2026) 'Designing meaningful human oversight in AI', *AI and
Ethics*, 6(3), article 286. Available at:
https://doi.org/10.1007/s43681-026-01147-7
