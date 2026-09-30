# Stage 7 - A second source and automatic failover (after tutor feedback)

**Why this stage exists:** the tutor's feedback on the team design proposal
(mark released 29 Sept) made three points. One of them was about retrieval:
the design leaned on one API for the main search and kept the other only as
a fallback, which limits coverage and makes every run hostage to that API's
downtime and rate limits. The advice was to use several sources as part of
the normal retrieval process, with automatic failover. The built system had
exactly that weakness - a single source (OpenAlex) - and the 23 Sept OpenAlex
rate-limit episode (remediation #9) had already shown the cost. So this stage
changes the retrieval design rather than adding a feature.

**What was built:**

1. **`agents/s2_client.py`** - a Semantic Scholar client with the same
   signature and output as the OpenAlex client: `search(query) -> List[Paper]`,
   same cache, same retry policy (`agents/http.py`). The DOI comes from
   `externalIds`, abstracts are plain text (whitespace collapsed; empty stays
   `None` so the threshold does not count it), every record is tagged
   `source="semanticscholar"`.
2. **`config.SOURCES`** - the list of sources to search, default
   `openalex,semanticscholar`, settable in `.env`. Adding a source is one
   client module and one line in the Retrieval Agent's registry; nothing in
   the orchestrator or the other agents knows how many sources there are.
3. **Retrieval Agent** - for every sub-question it now runs every configured
   source and merges the records. If a source raises after the retry policy
   has given up (HTTP 429/5xx exhausted, timeout, connection error, or a
   non-transient HTTP error), it logs
   `source=<name> failed (...) - continuing with remaining sources`, counts
   the failure, and carries on. The failed source is tried again on the next
   sub-question, because an outage may have cleared. Only if *every* source
   fails for a query does the run stop, with all the reasons in the message.
   Errors that are not network errors (a bug in a client) are not swallowed.
4. **The brief** - the limitations section now states what each source
   contributed (`openalex: 5 of 5 searches answered, 100 records; ...`) and,
   if failover happened, which source failed how many times and that coverage
   for those sub-questions may be narrower. Computed by the orchestrator from
   the agent's tallies, as with every other limitation.
5. **De-duplication** needed no change: the orchestrator already merges each
   sub-question's records by DOI (first occurrence kept, so an overlap keeps
   the OpenAlex copy) and against everything gathered so far.

**Why the threshold got easier to meet:** the count of usable records is now
taken over both sources, so a sub-question that would have triggered a
reformulation on OpenAlex alone may not need one. That is the coverage gain
the feedback was after; the one-reformulation cap is unchanged.

**Why the source list is configuration:** it makes the failover demonstrable
without touching code - `S2_ENDPOINT` can be pointed at a non-existent path
for one run so that Semantic Scholar fails at once (HTTP 404, not retried)
and the log and brief show the run continuing on OpenAlex. That is
functional test FT-11.

**Tests:** `pytest` -> 68 passed (12 new). Semantic Scholar parsing from a
recorded response (DOI from externalIds, whitespace, no invented DOI, key
header only when configured, cache before network); several sources searched
and merged; one source failing does not stop the search; a failed source is
retried on the next query; all sources failing stops the run with the
reasons; an unknown source name is a configuration error; non-network errors
still surface; a full mocked run writes the failover into the limitations.
The earlier tests keep their single-source assumptions through a fixture in
`tests/conftest.py`.
