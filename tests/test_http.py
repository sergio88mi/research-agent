"""Remediation #9: OpenAlex/Crossref transient-error retry - no network."""
import json
from unittest.mock import patch

import pytest
import requests

import config
from agents import crossref_client, http


class _Resp:
    def __init__(self, status, body="{}", headers=None):
        self.status_code, self.text, self.headers = status, body, headers or {}
    def json(self):
        return json.loads(self.text)
    def raise_for_status(self):
        raise requests.HTTPError(f"{self.status_code}")


def test_429_then_200_is_retried_with_backoff(monkeypatch):
    waits = []
    with patch("agents.http.requests.get", side_effect=[_Resp(429), _Resp(503), _Resp(200)]) as get, \
         patch("agents.http.time.sleep", side_effect=waits.append):
        r = http.get_with_retry("u", name="t")
    assert r.status_code == 200 and get.call_count == 3
    assert waits == [config.API_BACKOFF_SECONDS, config.API_BACKOFF_SECONDS * 2]


def test_retry_after_header_is_honoured():
    waits = []
    with patch("agents.http.requests.get", side_effect=[_Resp(429, headers={"Retry-After": "9"}), _Resp(200)]), \
         patch("agents.http.time.sleep", side_effect=waits.append):
        http.get_with_retry("u", name="t")
    assert waits == [9]


def test_non_transient_status_fails_immediately():
    with patch("agents.http.requests.get", return_value=_Resp(400)) as get, patch("agents.http.time.sleep") as sleep:
        with pytest.raises(requests.HTTPError):
            http.get_with_retry("u", name="t")
    assert get.call_count == 1 and not sleep.called


def test_gives_up_after_configured_retries(monkeypatch):
    monkeypatch.setattr(config, "API_TRANSIENT_RETRIES", 2)
    with patch("agents.http.requests.get", return_value=_Resp(503)) as get, patch("agents.http.time.sleep"):
        with pytest.raises(requests.HTTPError, match="after 2 retries"):
            http.get_with_retry("u", name="t")
    assert get.call_count == 3


def test_crossref_404_is_still_a_miss_not_a_retry():
    # Why: for Crossref, 404 is the answer "not registered" - it must not be retried or raised.
    with patch("agents.cache.get", return_value=None), patch("agents.cache.put") as put, \
         patch("agents.http.requests.get", return_value=_Resp(404)) as get, patch("agents.http.time.sleep") as sleep:
        assert crossref_client.lookup("10.1/none") is None
    assert get.call_count == 1 and not sleep.called and put.called
