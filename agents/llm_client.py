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
from typing import Type, TypeVar

import requests
from pydantic import BaseModel, ValidationError

import config
from agents import cache

log = logging.getLogger("research_agent.llm")
T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


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

    response = requests.post(
        config.GEMINI_ENDPOINT,
        params={"key": config.GEMINI_API_KEY},
        json=payload,
        timeout=60,
    )
    if response.status_code != 200:
        raise LLMError(f"Gemini returned HTTP {response.status_code}: {response.text[:300]}")
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
