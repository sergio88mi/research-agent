"""
Central configuration for the research agent.

Why a single module: the limits the design proposal sets (three to five
sub-questions, one query reformulation, one scope revision) live here,
together with the numbers the proposal left open and the build had to choose
(the retrieval threshold, results per query, the relevance cut-off, time-outs,
back-off, batch sizes, the title-match ratio), so that the behaviour the
report promises can be checked - and changed - in one place rather than
hunted through the agents.
"""
import os
from dotenv import load_dotenv

load_dotenv()  # reads .env if present; environment variables still win

# --- LLM ---------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")  # free tier: 500 requests/day vs 20 for the full Flash models (remediation #8)
# Provider endpoint is kept here (not in the client) so a different hosted model
# can be substituted by changing configuration: the proposal names Gemini Flash
# only as an example of a free-tier hosted API (section 3).
GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)
LLM_REPAIR_RETRIES = 1  # one repair attempt when the model returns invalid JSON
LLM_TRANSIENT_RETRIES = 5       # retries on HTTP 429/503, timeouts and dropped connections (raised from 3, remediation #5)
LLM_BACKOFF_SECONDS = 2         # wait 2, 4, 8, 16, 32 s between those retries (~1 min total)
LLM_TIMEOUT_SECONDS = 90        # a 10-paper scoring prompt can take >60s when the service is busy
LLM_MAX_RETRY_AFTER_SECONDS = 60  # cap on a server-supplied Retry-After wait (also applied to the scholarly APIs, agents/http.py)
LLM_TEMPERATURE = 0.2           # low on purpose: the model screens and summarises, it is not asked to be inventive

# --- Literature search --------------------------------------------------------
OPENALEX_EMAIL = os.getenv("OPENALEX_EMAIL", "")
OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY", "")  # free account key: $1/day (~1,000 searches) vs $0.10 keyless (remediation #10)
OPENALEX_ENDPOINT = "https://api.openalex.org/works"
RESULTS_PER_QUERY = 20          # per query, per source: four times the threshold below, and two scoring prompts' worth
CROSSREF_ENDPOINT = "https://api.crossref.org/works"
API_TRANSIENT_RETRIES = 4       # OpenAlex/Crossref/Semantic Scholar: retries on 429/502/503/504 and time-outs (remediation #9)
API_BACKOFF_SECONDS = 3         # wait 3, 6, 12, 24 s
API_TIMEOUT_SECONDS = 30        # one search request
CROSSREF_TIMEOUT_SECONDS = 20   # one DOI lookup - a small record, so a shorter wait
TITLE_MATCH_THRESHOLD = 0.6     # similarity ratio above which a retrieved title and the Crossref title count as
                                # the same paper; loose because casing, punctuation and subtitles differ

# Stage 7 - several sources as part of the normal retrieval process, with automatic failover.
# The team-project feedback pointed out that leaning on one API (with a second only as a
# fallback) narrows coverage and makes the run hostage to that API's downtime and rate
# limits. Every sub-question is now searched on every source listed here; if one source
# fails after its retries the run carries on with the rest and the brief says so. The
# list is configuration, not code, so a source can be dropped or reordered without a
# change to any agent (order matters only for which copy of a duplicate is kept: the
# first source's).
SOURCES = [s.strip() for s in os.getenv("SOURCES", "openalex,semanticscholar").split(",") if s.strip()]
S2_ENDPOINT = os.getenv("S2_ENDPOINT", "https://api.semanticscholar.org/graph/v1/paper/search")  # overridable so failover can be demonstrated live
S2_API_KEY = os.getenv("S2_API_KEY", "")  # optional: a key gives a dedicated 1 request/s; without one, requests share the public pool

# --- Agent behaviour ----------------------------------------------------------
# From the design: 3-5 sub-questions (Diagram 1), one reformulation and one scope
# revision (section 4, Diagram 3). Chosen in the build: the threshold value (the
# design says only "configured threshold") and the relevance cut-off.
MIN_SUBQUESTIONS = 3
MAX_SUBQUESTIONS = 5
RETRIEVAL_THRESHOLD = 5         # minimum different usable records (with abstracts) per sub-question; also the minimum relevant ones
MAX_REFORMULATIONS = 1          # automatic query reformulations per sub-question. Two checks can ask for one (too few
                                # records; too few relevant records) and they share this single budget.
MAX_SCOPE_REVISIONS = 1         # one researcher-triggered scope revision per run
RELEVANCE_CUTOFF = 3            # keep papers scored >= 3 on a 1-5 scale
EVAL_BATCH_SIZE = 10            # papers scored (or summarised) per LLM call (bounded prompt size, fewer calls)
EVAL_ABSTRACT_CHARS = 1200      # abstract is truncated to this many chars in the scoring/summary prompts
MIN_THEMES = 2                  # synthesis must find at least this many cross-paper themes ...
MAX_THEMES = 6                  # ... and at most this many, each tied to numbered papers

# --- Files --------------------------------------------------------------------
OUTPUT_DIR = "output"
CACHE_DIR = "cache"
LOG_FILE = "output/run.log"
