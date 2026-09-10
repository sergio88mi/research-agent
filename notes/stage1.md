# Stage 1 - Retrieval from OpenAlex

**What was built:** `agents/search_client.py` (talks to OpenAlex, rebuilds
abstracts from its inverted-index format, normalises results into `Paper`
records, caches responses to disk) and `agents/retrieval_agent.py` (the
Retrieval Agent: given a SubQuestion, runs the search and logs how many usable
records came back). `main.py` now runs one hard-coded query and prints results.

**Why OpenAlex:** no API key needed, good abstract coverage, and the proposal's
named fallback. Because results are normalised into our own `Paper` model,
swapping to Semantic Scholar later would change one function only.

**Why no LLM here:** the Retrieval Agent's goal - "did I get enough usable
records?" - is a count. The design reserves the LLM for judgement tasks.

**Why the cache:** free-tier friendliness and reproducibility - a repeated query
returns the identical records, which the functional tests rely on.

**Evidence:** `python main.py` lists ~20 real papers with DOIs;
`pytest` -> 9 passed (5 model tests + 4 parsing tests on a recorded response).
