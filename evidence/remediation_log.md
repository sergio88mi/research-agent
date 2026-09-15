# Testing and remediation log

| # | Date | Stage | Issue observed | Cause | Fix | Verified by |
|---|---|---|---|---|---|---|
| 1 | 2026-09-14 | 2 | 7 of 20 DOIs flagged "not in Crossref" | All 7 were arXiv preprints (10.48550/arxiv.*); arXiv registers DOIs with DataCite, not Crossref | No code change - system behaved as designed (flag, never drop). Recorded as a known limitation of single-registry verification; DataCite lookup noted as a possible extension | Log lines in Stage 2 run; brief shows papers marked UNVERIFIED rather than missing |
| 2 | 2026-09-14 | 3 | First Gemini call failed: HTTP 404 "model gemini-2.0-flash is no longer available" | Provider retired the model between design (Aug) and implementation (Sept) | Model name moved to configuration in Stage 0 for exactly this case; changed GEMINI_MODEL to gemini-3.6-flash in .env / .env.example / config default | Re-run of Stage 3 succeeds (see Stage 3 evidence screenshot) |
