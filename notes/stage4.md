# Stage 4 - Evaluation Agent (relevance scoring and filtering)

**What was built:** `agents/evaluation_agent.py` - the `EvaluationSynthesisAgent`
from Diagram 1 (the synthesis half comes in Stage 6). After retrieval and DOI
verification, every paper is sent to Gemini together with the sub-question that
found it, and comes back with a relevance score 1-5 and a one-sentence reason.
`select_papers()` then applies the cut-off from `config.RELEVANCE_CUTOFF` (3):
score 3 or more is kept, below is dropped. The orchestrator stores both the
kept papers and *all* assessments in the brief; the Markdown shows the score
and reason under each kept paper and lists the dropped ones with their reasons.

**Why the LLM does this step:** keyword search returns papers that share words
with the query, not papers that answer the question. Deciding whether an
abstract addresses a question is a reading judgement with no exact answer -
the proposal's criterion for using the model. Applying the cut-off is exact,
so that stays deterministic in the orchestrator.

**Why the model only sees title + abstract:** the score has to be grounded in
the record we hold, not in what the model remembers about a paper. The prompt
says so, and the reason must point at what the abstract does or does not cover.
If the abstract is missing the model is told to rate from the title and say so.

**Why batches of 10 per sub-question:** one call per paper would be ~80 calls
a run and hit free-tier rate limits; one call for everything risks a truncated
reply. `EVAL_BATCH_SIZE` in config is the compromise (about 8 calls per run).

**Why the response schema is built per batch:** the schema demands exactly one
score for each index 1..N. A missing or duplicated index is therefore a
validation error, and the same one-shot repair retry from Stage 3 handles it -
no new error-handling code was needed.

**Why dropped papers are still recorded:** the brief lists what was excluded
and why, so a researcher can disagree with the agent. That is the traceability
the proposal promises and the basis for Human Review 2 in Stage 5.

**What went wrong on the live runs (remediation #3 and #4):** the first run
scored three batches, then Gemini answered HTTP 503 "high demand" and the run
aborted, because the client treated every non-200 status as fatal. Fix:
`_post_with_retry` in `llm_client.py` retries 429/503 up to three times with
2/4/8 s back-off. The second run proved that fix (one retry, then success) and
then hit a *read timeout* - the same overloaded service, but as an exception
rather than a status code, so it slipped past the loop. Fix: timeouts and
dropped connections are retried the same way, and the timeout is 90 s because
a 10-paper scoring prompt takes longer than a planning prompt. Other codes
(400, 403, 404) still fail at once - a wrong key or model name must not be
hidden behind retries. Because every scored batch is cached, each re-run only
paid for the batches that had not yet been scored.

**What the live run showed (17 Sept):** 65 papers scored, 24 kept, 41
dropped. Per sub-question the kept counts were 9, 9, 5 and 1: the fourth
query ("benchmarking detecting hallucination evaluation metrics ...") pulled
back mostly general LLM surveys, and the agent correctly rejected 11 of 12.
That is the case Stage 5 is built for - the retrieval threshold, the single
automatic query reformulation, and the researcher's Review 2.

**Evidence:** `python main.py "question"` now ends with
`N unique papers | V DOIs verified | S selected, D dropped (cut-off 3/5)`;
`pytest` -> 32 passed (11 new: mapping, cut-off, batching, missing/duplicate
index repair, out-of-range score rejection, brief output, 503 retry, timeout
retry, give-up after 3, 404 no-retry).
