"""
One retry policy for the two scholarly APIs (OpenAlex, Crossref).

Why this exists: the Gemini client learned the hard way (remediation #3-#5)
that a busy service answers 429/503 or times out, and that treating those as
fatal loses a whole run. On 23 Sept OpenAlex did the same (remediation #9),
so the same policy - retry transient failures with exponential back-off,
honour Retry-After, fail fast on everything else - now covers all three
external services from one place instead of being copied into each client.
"""
import logging
import time

import requests

import config

log = logging.getLogger("research_agent.http")

_TRANSIENT_STATUSES = {429, 502, 503, 504}
_TRANSIENT_EXCEPTIONS = (requests.Timeout, requests.ConnectionError)


def get_with_retry(url: str, *, params=None, headers=None, timeout: int = 30,
                   ok_statuses=(200,), name: str = "api") -> requests.Response:
    """GET `url`; retry transient failures; return the response for any status in ok_statuses.

    `ok_statuses` lets a caller treat e.g. 404 as a legitimate answer (Crossref:
    "this DOI is not registered") rather than an error.
    """
    problem, wait = None, 0
    for attempt in range(config.API_TRANSIENT_RETRIES + 1):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=timeout)
            if response.status_code in ok_statuses:
                return response
            if response.status_code not in _TRANSIENT_STATUSES:
                response.raise_for_status()       # real error - fail at once
            problem = f"HTTP {response.status_code}"
            retry_after = response.headers.get("Retry-After", "")
            wait = min(int(float(retry_after)), config.LLM_MAX_RETRY_AFTER_SECONDS) if retry_after.replace(".", "", 1).isdigit() else 0
        except _TRANSIENT_EXCEPTIONS as exc:
            problem, wait = type(exc).__name__, 0
        if attempt < config.API_TRANSIENT_RETRIES:
            wait = wait or config.API_BACKOFF_SECONDS * (2 ** attempt)
            log.warning("%s | %s (transient) - retry %d/%d in %ds", name, problem, attempt + 1,
                        config.API_TRANSIENT_RETRIES, wait)
            time.sleep(wait)
    raise requests.HTTPError(f"{name} unavailable after {config.API_TRANSIENT_RETRIES} retries (last problem: {problem})")
