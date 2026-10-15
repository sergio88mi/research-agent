"""
One retry policy for the three scholarly APIs (OpenAlex, Semantic Scholar,
Crossref).

Why this exists: the Gemini client learned the hard way (remediation #3-#5)
that a busy service answers 429/503 or times out, and that treating those as
fatal loses a whole run. On 23 Sept OpenAlex did the same (remediation #9),
so the same policy - retry transient failures with exponential back-off,
honour Retry-After, fail fast on everything else - is applied to the
scholarly APIs from one place instead of being copied into each client. (The
Gemini client keeps its own loop in llm_client.py because it also has to read
the quota name out of a 429 and stop at once on a daily limit.)

What counts as transient: 429, 502, 503 and 504, a time-out, a dropped
connection. Any other status - a 400, 403, 404 or a plain 500 - is treated as
a real answer about the request and is not retried.

Why the error text is written here and not taken from `raise_for_status()`:
that message quotes the full request URL, query string included. For OpenAlex
the query string carries the API key and the contact email, and the message
is logged to output/run.log, which is kept in the repository. The text built
here names the service, the status and the address without its query
(remediation #19).
"""
import logging
import time
from typing import Optional

import requests

import config

log = logging.getLogger("research_agent.http")

_TRANSIENT_STATUSES = {429, 502, 503, 504}
_TRANSIENT_EXCEPTIONS = (requests.Timeout, requests.ConnectionError)


def get_with_retry(url: str, *, params=None, headers=None, timeout: Optional[int] = None,
                   ok_statuses=(200,), name: str = "api") -> requests.Response:
    """GET `url`; retry transient failures; return the response for any status in ok_statuses.

    `ok_statuses` lets a caller treat e.g. 404 as a legitimate answer (Crossref:
    "this DOI is not registered") rather than an error. `timeout` defaults to
    config.API_TIMEOUT_SECONDS.
    """
    timeout = config.API_TIMEOUT_SECONDS if timeout is None else timeout
    problem, wait = None, 0
    for attempt in range(config.API_TRANSIENT_RETRIES + 1):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=timeout)
            if response.status_code in ok_statuses:
                return response
            if response.status_code not in _TRANSIENT_STATUSES:
                # real error - fail at once; `url` is the address without its query string
                raise requests.HTTPError(f"{name} returned HTTP {response.status_code} for {url}", response=response)
            problem = f"HTTP {response.status_code}"
            retry_after = response.headers.get("Retry-After", "")
            # the cap on a server-supplied wait is the one setting shared with the Gemini client
            wait = min(int(float(retry_after)), config.LLM_MAX_RETRY_AFTER_SECONDS) if retry_after.replace(".", "", 1).isdigit() else 0
        except _TRANSIENT_EXCEPTIONS as exc:
            problem, wait = type(exc).__name__, 0
        if attempt < config.API_TRANSIENT_RETRIES:
            wait = wait or config.API_BACKOFF_SECONDS * (2 ** attempt)
            log.warning("%s | %s (transient) - retry %d/%d in %ds", name, problem, attempt + 1,
                        config.API_TRANSIENT_RETRIES, wait)
            time.sleep(wait)
    raise requests.HTTPError(f"{name} unavailable after {config.API_TRANSIENT_RETRIES} retries (last problem: {problem})")
