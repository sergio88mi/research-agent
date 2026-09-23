"""
Central configuration for the research agent.

Why a single module: every tunable number in the design proposal (retrieval
threshold, retry caps, results per query) lives here so that the behaviour the
report promises can be checked - and changed - in one place rather than hunted
through the agents.
"""
import os
from dotenv import load_dotenv

load_dotenv()  # reads .env if present; environment variables still win

# --- LLM ---------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")  # free tier: 500 requests/day vs 20 for the full Flash models (remediation #8)
# Provider endpoint is kept here (not in the client) so a different hosted model
# could be substituted by changing configuration, as the proposal requires.
GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)
LLM_REPAIR_RETRIES = 1  # one repair attempt when the model returns invalid JSON
LLM_TRANSIENT_RETRIES = 5       # retries on HTTP 429/503, timeouts and dropped connections (raised from 3, remediation #5)
LLM_BACKOFF_SECONDS = 2         # wait 2, 4, 8, 16, 32 s between those retries (~1 min total)
LLM_TIMEOUT_SECONDS = 90        # a 10-paper scoring prompt can take >60s when the service is busy
LLM_MAX_RETRY_AFTER_SECONDS = 60  # cap on a provider-supplied Retry-After wait

# --- Literature search --------------------------------------------------------
OPENALEX_EMAIL = os.getenv("OPENALEX_EMAIL", "")
OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY", "")  # free account key: $1/day (~1,000 searches) vs $0.10 keyless (remediation #10)
OPENALEX_ENDPOINT = "https://api.openalex.org/works"
RESULTS_PER_QUERY = 20          # ~20 results per sub-question (proposal, section 6)
CROSSREF_ENDPOINT = "https://api.crossref.org/works"
API_TRANSIENT_RETRIES = 4       # OpenAlex/Crossref: retries on 429/5xx, timeouts (remediation #9)
API_BACKOFF_SECONDS = 3         # wait 3, 6, 12, 24 s

# --- Agent behaviour (the bounded-autonomy numbers from the design) ----------
MIN_SUBQUESTIONS = 3
MAX_SUBQUESTIONS = 5
RETRIEVAL_THRESHOLD = 5         # minimum usable records (with abstracts) per sub-question
MAX_REFORMULATIONS = 1          # one automatic query reformulation per sub-question
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
