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
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
# Provider endpoint is kept here (not in the client) so a different hosted model
# could be substituted by changing configuration, as the proposal requires.
GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)
LLM_REPAIR_RETRIES = 1  # one repair attempt when the model returns invalid JSON

# --- Literature search --------------------------------------------------------
OPENALEX_EMAIL = os.getenv("OPENALEX_EMAIL", "")
OPENALEX_ENDPOINT = "https://api.openalex.org/works"
RESULTS_PER_QUERY = 20          # ~20 results per sub-question (proposal, section 6)
CROSSREF_ENDPOINT = "https://api.crossref.org/works"

# --- Agent behaviour (the bounded-autonomy numbers from the design) ----------
MIN_SUBQUESTIONS = 3
MAX_SUBQUESTIONS = 5
RETRIEVAL_THRESHOLD = 5         # minimum usable records (with abstracts) per sub-question
MAX_REFORMULATIONS = 1          # one automatic query reformulation per sub-question
MAX_SCOPE_REVISIONS = 1         # one researcher-triggered scope revision per run
RELEVANCE_CUTOFF = 3            # keep papers scored >= 3 on a 1-5 scale

# --- Files --------------------------------------------------------------------
OUTPUT_DIR = "output"
CACHE_DIR = "cache"
LOG_FILE = "output/run.log"
