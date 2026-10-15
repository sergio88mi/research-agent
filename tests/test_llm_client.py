"""Gemini client tests: the missing-key stop, and what is and is not kept in the cache - no network."""
import json
from unittest.mock import patch

import pytest
from pydantic import BaseModel

import config
from agents import llm_client
from agents.llm_client import LLMError


class _Answer(BaseModel):
    value: int


class _Resp:
    """A 200 reply from Gemini carrying `text` as the first candidate."""
    status_code = 200
    headers = {}

    def __init__(self, text):
        self._text = text

    def json(self):
        return {"candidates": [{"content": {"parts": [{"text": self._text}]}}]}


def test_missing_key_stops_before_any_request(monkeypatch):
    # Why: functional test FT-01 - the message has to name the variable and the file to copy.
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")
    with patch("agents.llm_client.requests.post") as post:
        with pytest.raises(LLMError, match="GEMINI_API_KEY is not set") as err:
            llm_client.generate_json("p", _Answer)
    assert ".env.example" in str(err.value) and not post.called


def test_a_valid_reply_is_cached_and_the_second_call_makes_no_request(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "test-key")
    with patch("agents.llm_client.requests.post", return_value=_Resp('{"value": 7}')) as post:
        assert llm_client.generate_json("p", _Answer).value == 7
        assert llm_client.generate_json("p", _Answer).value == 7
    assert post.call_count == 1


def test_temperature_comes_from_configuration(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(config, "LLM_TEMPERATURE", 0.5)
    with patch("agents.llm_client.requests.post", return_value=_Resp("ok")) as post:
        llm_client.generate("p", json_mode=False)
    assert post.call_args.kwargs["json"]["generationConfig"] == {"temperature": 0.5}


def test_a_rejected_reply_is_not_cached_and_the_repaired_one_is(monkeypatch, caplog):
    # Why: remediation #13. The first reply is a bare list (seen live on the Lite model); the repair is valid.
    # On the next run of the same prompt the good answer must come straight from the cache:
    # no request, and no "invalid JSON" warning repeated from a stored bad reply.
    monkeypatch.setattr(config, "GEMINI_API_KEY", "test-key")
    with patch("agents.llm_client.requests.post",
               side_effect=[_Resp("[1, 2]"), _Resp('{"value": 3}')]) as post:
        assert llm_client.generate_json("p", _Answer).value == 3
    assert post.call_count == 2 and caplog.text.count("invalid JSON") == 1

    caplog.clear()
    with patch("agents.llm_client.requests.post") as post:
        assert llm_client.generate_json("p", _Answer).value == 3
    assert not post.called and "invalid JSON" not in caplog.text


def test_when_the_repair_fails_too_nothing_is_cached(monkeypatch):
    # Why: a reply that never validated must not be handed back on the next run, or that
    # prompt could never succeed again without clearing the cache by hand.
    monkeypatch.setattr(config, "GEMINI_API_KEY", "test-key")
    with patch("agents.llm_client.requests.post", side_effect=[_Resp("{bad"), _Resp("{still bad")]):
        with pytest.raises(LLMError, match="after repair"):
            llm_client.generate_json("p", _Answer)
    with patch("agents.llm_client.requests.post", return_value=_Resp(json.dumps({"value": 1}))) as post:
        assert llm_client.generate_json("p", _Answer).value == 1
    assert post.call_count == 1
