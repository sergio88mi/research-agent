# Stage 5 - Retrieval threshold, one-retry reformulation, Human Review 2

**What was built:** the two bounded-autonomy mechanisms from the design and
the second human review point.

1. **Threshold check** (`RetrievalAgent.assess_threshold`): after a search the
   Retrieval Agent counts records that have an abstract. Fewer than
   `RETRIEVAL_THRESHOLD` (5) means its goal was not met, and it asks for a
   better query. This is the agent's own judgement of success - a count, no
   LLM - which is what makes it an agent rather than a function.
2. **A second trigger after screening** (in the orchestrator): a search can
   return 20 abstracts of which only one answers the sub-question - exactly
   what happened to sub-question 4 in the Stage 4 live run. So if fewer than
   5 papers survive relevance screening, the same reformulation is requested,
   and the reasons the papers were rejected are sent along so the Planning
   Agent knows *why* the query missed.
3. **One reformulation per sub-question** (`PlanningAgent.reformulate_query`):
   the Planning Agent writes one improved query; the old one is kept in
   `SubQuestion.previous_query` and `retried_once` is set. The orchestrator
   checks that flag before either trigger, so a sub-question can never be
   reformulated twice, whichever trigger fired first.
4. **Human Review 2** (`ResearcherInterface.review_evidence`): the researcher
   sees every selected paper, numbered, with its score and one-line reason,
   grouped by sub-question, and can `approve`, `drop 3, 7` (refine - remove
   papers by number; the list is re-shown, no API calls), or `reject` (revise
   the scope - feedback goes back to the Planning Agent and the run returns to
   Review 1). The interface only offers `reject` while the budget allows.
5. **One scope revision per run** (`Orchestrator.scope_revision_count`):
   after it is used, `reject` is refused with a message.

**Why de-duplication became incremental:** Stage 4 de-duplicated all 80
records at once. The threshold check needs to run per sub-question, so the
orchestrator now keeps a running `seen` set and scores only papers not already
gathered - same rule (a paper is scored against the sub-question that found
it first), applied one sub-question at a time.

**Why papers removed at Review 2 stay in the brief:** their assessment is
marked `selected=False` with "[removed by the researcher at review 2]" added
to the reason, so the brief still records that the paper was found and who
excluded it. Nothing is silently deleted.

**What the live runs showed (17 Sept, remediation #5-#7):** the threshold
fired exactly where Stage 4 predicted - sub-question 4, 1 relevant of 12 -
and the Planning Agent rewrote the query from
`benchmarking detecting hallucination evaluation metrics autonomous language model agents`
to `hallucination benchmark evaluation long horizon autonomous agent trajectory detection`;
OpenAlex returned 20 new records (9 DOIs verified). Scoring them then hit
HTTP 429. Three lessons went into `llm_client.py`: the retry budget was
raised to 5 (2/4/8/16/32 s) after four 503s in a row; Google's error message
and quota id are now logged (the message text alone does not say which
quota); and a *per-day* quota now stops the run immediately with a plain
message instead of retrying, because it cannot recover within a back-off
window. The quota that ran out was
`GenerateRequestsPerDayPerProjectPerModel-FreeTier`: every retry counts as a
request, so the day's budget went on retries as much as on work.

**Live completion (21 Sept, after the quota reset):** the reformulated
query for sub-question 4 returned 13 new records of which only 2 were
relevant - so the retry lifted that sub-question from 1 kept paper to 3, not
to the threshold of 5. The cap held (no second attempt), Review 2 listed 26
papers with scores and reasons, and the approved brief records 78 unique
papers, 55 DOIs verified, 26 approved, 52 excluded, 1 reformulation. Honest
reading: one automatic reformulation is a modest safety net, not a fix for a
sub-question the corpus covers thinly; the researcher's reject-scope option
exists for that case.

**Evidence:** `python main.py "question"` now stops at Review 2; the summary
line reports `reformulations: N, scope revisions: M`; the brief lists any
reformulated query under its sub-question. `pytest` -> 41 passed (9 new:
threshold count, trigger 1 once, trigger 2 once with cap, refine removes
numbered papers, reject re-plans once and feedback reaches the planner,
reject refused when budget used, numbering matches display order, 429 with
Retry-After, per-day quota stops immediately).
