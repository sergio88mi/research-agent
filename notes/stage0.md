# Stage 0 - Project skeleton

**What was built:** the folder structure, `config.py` (all the design's numbers in
one place), `agents/models.py` (the four data classes from Diagram 1 plus the
Decision enum), `cache.py`, `logging_setup.py`, a placeholder `main.py`, and
five tests for the data models.

**Why first:** every later stage passes these objects around. Getting the data
shapes right - and validated by Pydantic - means a malformed LLM response fails
at the boundary instead of corrupting the pipeline.

**How to reproduce by hand:** mkdir + venv + requirements.txt + write the files +
`git init`, `git add .`, `git commit`.

**Evidence:** `python main.py` prints "Stage 0 complete"; `pytest` -> 5 passed.
