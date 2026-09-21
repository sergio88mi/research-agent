"""
Gemini client: the only place the system talks to a language model.

Why plain HTTP rather than a vendor SDK: the proposal lists `requests` and
`pydantic` as the stack and keeps the provider in configuration. A ~40-line
client makes every call inspectable - the prompt goes in, text comes out - and
swapping providers means changing an endpoint and a payload shape, not a
dependency tree.

Why `generate_json` validates and retries once: the LLM is the least
predictable component. Every model response is parsed into a Pydantic model;
if that fails, the error is sent back to the model for ONE repair attempt
(config.LLM_REPAIR_RETRIES). Repeated failure raises, so the pipeline stops
loudly instead of continuing on a broken object.
"""
import json
import logging
import time
from typing import Type, TypeVar

import requests
from pydantic import BaseModel, ValidationError

import config
from agents import cache

log = logging.getLogger("research_agent.llm")
T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


# Failures that mean "the service is busy", not "the request is wrong".
# Found the hard way during Stage 4 (see evidence/remediation_log.md #3, #4):
# one run died on HTTP 503 "high demand", the next on a read timeout while the
# same overloaded service took >60s to reply. Both are retried with a short
# back-off. Anything else (400, 403, 404) is a real error and still fails
# immediately, so a bad key or model name is never silently retried.
_TRANSIENT_STATUSES = {429, 503}
_TRANSIENT_EXCEPTIONS = (requests.Timeout, requests.ConnectionError)


def _post_with_retry(payload: dict) -> requests.Response:
    """POST to Gemini, retrying transient failures with exponential back-off."""
    for attempt in range(config.LLM_TRANSIENT_RETRIES + 1):
        problem, server_wait, detail = None, None, ""
        try:
            response = requests.post(
                config.GEMINI_ENDPOINT,
                params={"key": config.GEMINI_API_KEY},
                json=payload,
                timeout=config.LLM_TIMEOUT_SECONDS,
            )
            if response.status_code == 200:
                return response
            if response.status_code not in _TRANSIENT_STATUSES:
                raise LLMError(f"Gemini returned HTTP {response.status_code}: {response.text[:300]}")
            problem = f"HTTP {response.status_code}"
            detail = _error_message(response)     # e.g. which quota a 429 refers to
            if "PerDay" in detail:
                # A daily quota will not recover within any back-off window - stop now, say so plainly.
                raise LLMError("Gemini daily free-tier quota is exhausted for this model; it resets at "
                               f"midnight Pacific time. Provider message: {detail}")
            server_wait = _retry_after(response)  # provider's own advice, if given
        except _TRANSIENT_EXCEPTIONS as exc:
            problem = type(exc).__name__          # e.g. ReadTimeout, ConnectionError

        if attempt < config.LLM_TRANSIENT_RETRIES:
            wait = server_wait or config.LLM_BACKOFF_SECONDS * (2 ** attempt)
            log.warning("llm | %s (transient) - retry %d/%d in %ds%s",
                        problem, attempt + 1, config.LLM_TRANSIENT_RETRIES, wait,
                        f" | {detail}" if detail else "")
            time.sleep(wait)
    raise LLMError(f"Gemini unavailable after {config.LLM_TRANSIENT_RETRIES} retries "
                   f"(last problem: {problem}{' - ' + detail if detail else ''})")


def _error_message(response: requests.Response) -> str:
    """Google's reason, plus the quota name(s) from error.details when it is a 429.

    The message text is the same for the per-minute and per-day limits; only the
    `quotaId` inside details (e.g. ...PerDayPerProjectPerModel-FreeTier) tells
    which one was hit - and that decides whether to wait a minute or a day.
    """
    try:
        err = response.json()["error"]
        quotas = [v.get("quotaId", "") for d in err.get("details", [])
                  for v in d.get("violations", []) if isinstance(v, dict)]
        msg = str(err.get("message", ""))[:160]
        return msg + (f" [quota: {', '.join(q for q in quotas if q)}]" if any(quotas) else "")
    except Exception:
        return response.text[:200].replace("\n", " ")


def _retry_after(response: requests.Response) -> int:
    """Seconds the provider asked us to wait (Retry-After header), capped; 0 if absent."""
    value = response.headers.get("Retry-After", "")
    try:
        return min(int(float(value)), config.LLM_MAX_RETRY_AFTER_SECONDS) if value else 0
    except ValueError:
        return 0


def generate(prompt: str, json_mode: bool = True, temperature: float = 0.2) -> str:
    """Send one prompt to Gemini and return the text of the first candidate."""
    if not config.GEMINI_API_KEY:
        raise LLMError("GEMINI_API_KEY is not set - copy .env.example to .env and add your key.")

    cache_key = f"{config.GEMINI_MODEL}|{json_mode}|{temperature}|{prompt}"
    cached = cache.get("gemini", cache_key)
    if cached is not None:
        return cached["text"]

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature},
    }
    if json_mode:
        # Ask the model to emit JSON only - reduces (does not eliminate) parse failures.
        payload["generationConfig"]["responseMimeType"] = "application/json"

    response = _post_with_retry(payload)
    data = response.json()
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as exc:
        raise LLMError(f"Unexpected Gemini response shape: {json.dumps(data)[:300]}") from exc

    cache.put("gemini", cache_key, {"text": text})
    log.info("llm | prompt_chars=%d | response_chars=%d", len(prompt), len(text))
    return text


def generate_json(prompt: str, model: Type[T]) -> T:
    """Generate, validate against `model`, and repair once if invalid."""
    text = generate(prompt)
    attempts = 0
    while True:
        try:
            return model.model_validate_json(text)
        except (ValidationError, ValueError) as exc:
            attempts += 1
            log.warning("llm | invalid JSON for %s (attempt %d): %s", model.__name__, attempts, str(exc)[:200])
            if attempts > config.LLM_REPAIR_RETRIES:
                raise LLMError(f"Model output did not match {model.__name__} after repair: {str(exc)[:300]}")
            repair_prompt = (
                f"{prompt}\n\nYour previous reply was rejected because it did not match the required "
                f"JSON schema. Error: {str(exc)[:400]}\nReturn ONLY valid JSON that fixes this."
            )
            text = generate(repair_prompt)
