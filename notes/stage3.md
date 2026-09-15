# Stage 3 - Planning Agent (Gemini) and Human Review 1

**What was built:** `agents/llm_client.py` (talks to Gemini over plain HTTP;
validates every reply against a Pydantic schema; one repair retry),
`agents/planning_agent.py` (turns the question into an interpretation plus 3-5
sub-questions with search queries), `agents/researcher_interface.py` (the
«boundary» class from Diagram 1 - all human interaction in one place, and it
can be scripted for tests), `agents/orchestrator.py` (the manager: plan ->
review loop -> retrieve each sub-question -> dedupe -> verify -> save).

**Why review 1 allows unlimited revision:** correcting the decomposition
before any search spends no API quota, so iteration is cheapest here.

**Why the LLM's output is validated, not trusted:** the schema check enforces
the 3-5 sub-question rule and rejects malformed JSON; the model gets exactly
one chance to repair, then the run stops loudly.

**Why constraints travel with every prompt:** they are stored outside the
model's context and re-applied each call, so they cannot be "forgotten".

**Evidence:** `python main.py "question"` shows the plan, waits for approval,
then searches every sub-question; `pytest` -> 21 passed (6 new).
