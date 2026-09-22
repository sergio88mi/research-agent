# Stage 6 - Synthesis and the finished brief

**What was built:** the second half of `EvaluationSynthesisAgent`
(`agents/evaluation_agent.py`) and the final shape of the brief.

1. **`summarise()`** - one 2-3 sentence evidence note per approved paper,
   written from that paper's title and abstract only, batched per sub-question
   like scoring. Each note is numbered `[k]` by the paper's position in the
   approved list. The prompt forbids adding findings not in the abstract and
   tells the model to say "No abstract available" rather than guess.
2. **`synthesise()`** - one call that reads all the notes and returns 2-6
   themes plus a list of gaps. Every theme must cite paper numbers, and the
   schema rejects a number outside 1..N or a theme with no citation - so a
   fabricated citation is a validation error, repaired once or the run stops.
   The gaps prompt asks specifically which sub-questions are answered only by
   surveys or not at all.
3. **Limitations** are *computed* by the orchestrator, not written by the
   model: reformulated sub-questions, sub-questions that ended below the
   threshold, papers removed at review 2, DOIs not in Crossref, the exclusion
   count, the model name, and the single-source/abstract-only scope.
4. **References** are built deterministically from each record's own
   metadata (authors, year, title, DOI) in `storage.py`. The reference list is
   the one part of a research brief that must never be model-generated.

**Why grounding is done paper-by-paper before any cross-paper writing:** a
note written from one abstract can be checked against that abstract in
seconds. Themes are then written only from those notes, so the chain
abstract -> note -> theme -> reference is traceable end to end, which is the
proposal's "traceable rather than plausible" requirement.

**Why synthesis runs only after review 2:** the agent never writes about a
paper the researcher has not seen and accepted. Summarising 78 papers before
screening would also cost three times the API calls for evidence that is
mostly discarded.

**The brief now contains:** question and interpretation; sub-questions (with
any reformulated query); the numbered selected papers with score and reason;
the excluded papers with reasons; evidence notes; themes with citations;
gaps; limitations; references - as Markdown for reading and JSON for machines.

**Live run (22 Sept):** 26 approved papers summarised in 4 calls (one per
sub-question), themes and gaps in 1 call; the whole pipeline finished in about
four minutes with 7 transient 503s absorbed by the retry logic (one batch
succeeded on its fifth and last retry). Spot-check of grounding: every number
quoted in the notes (17-33% residual hallucination in legal RAG tools, 57.18%
Recall@1, 2,919 diseases, 30 and 90 reviewed studies) appears in the paper's
own abstract. The three themes cite 6, 5 and 8 papers respectively; the gaps
name sub-questions 3 and 4 as thin and the set as survey-heavy - the same
verdict the per-sub-question counts give. The brief is 6,387 words including
the 52 exclusion reasons and the reference list.

**Evidence:** `python main.py "question"` runs the whole pipeline and ends
with `... | T themes, G gaps | reformulations: N, scope revisions: M`.
`pytest` -> 50 passed (9 new: note numbering, theme/gap parsing, fabricated
citation rejected then repaired, uncited theme rejected, theme count bound,
empty evidence handled, references from metadata, all brief sections with
shared numbering, full mocked run with computed limitations).
