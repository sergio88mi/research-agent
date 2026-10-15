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


def test_crossref_record_is_fetched_once_then_served_from_the_cache():
    # Why: a DOI is looked up on every run that finds the paper; only the first lookup may cost a request.
    with patch("agents.http.requests.get", return_value=_Resp(200, json.dumps({"message": {"title": ["A title"]}}))) as get:
        first = crossref_client.lookup("10.1/cached")
        second = crossref_client.lookup("10.1/cached")
    assert get.call_count == 1 and first == second == {"title": ["A title"]}


def test_crossref_miss_is_remembered_as_a_miss():
    with patch("agents.http.requests.get", return_value=_Resp(404)) as get:
        assert crossref_client.lookup("10.1/unregistered") is None
        assert crossref_client.lookup("10.1/unregistered") is None
    assert get.call_count == 1


def test_an_error_message_never_carries_the_query_string():
    # Why: the OpenAlex key and the contact email travel in the query string, and the message is
    # written to output/run.log, which is kept in the repository (remediation #19).
    secret = {"api_key": "SECRET-KEY", "mailto": "someone@example.org", "search": "agents"}
    with patch("agents.http.requests.get", return_value=_Resp(403)):
        with pytest.raises(requests.HTTPError) as caught:
            http.get_with_retry("https://api.openalex.org/works", params=secret, name="openalex")
    text = str(caught.value)
    assert text == "openalex returned HTTP 403 for https://api.openalex.org/works"
    assert "SECRET-KEY" not in text and "someone@example.org" not in text


def test_a_doi_with_url_characters_stays_in_the_path():
    # Why: '#' or '?' inside a DOI would otherwise cut the address short and the wrong record would be looked up.
    doi = "10.1002/(SICI)1097-4571(199806)49:8<693::AID-ASI4>3.0.CO;2-N#x?y"
    with patch("agents.http.requests.get", return_value=_Resp(404)) as get:
        crossref_client.lookup(doi)
    url = get.call_args.args[0]
    assert url == config.CROSSREF_ENDPOINT + "/10.1002/(SICI)1097-4571(199806)49:8%3C693::AID-ASI4%3E3.0.CO;2-N%23x%3Fy"
