# Stage 2 - De-duplication, Crossref verification, storage

**What was built:** `agents/crossref_client.py` (looks a DOI up in the Crossref
registry), `agents/validation.py` (removes duplicate DOIs; sets each paper's
`doi_verified` to True / False / None), `agents/storage.py` (writes the brief
as Markdown and JSON). `main.py` now runs search -> dedupe -> verify -> save.

**Why Crossref:** OpenAlex says a paper exists; Crossref is the authority that
registers DOIs. Checking each one independently is the defence against
fabricated citations - the failure mode of LLMs the proposal singles out.

**Why three states, not two:** "checked and wrong" (False) and "could not
check" (None) are different facts. Collapsing them would hide network failures
inside data errors. Records are flagged, never dropped - registry coverage is
incomplete, so absence from Crossref is not proof a paper is fake.

**Why no LLM:** every operation here has an exactly right answer.

**Evidence:** `python main.py` reports "N unique papers | N DOIs verified |
N flagged" and writes `output/brief-<timestamp>-....md/.json`;
`pytest` -> 15 passed.
