"""Stage 1 tests: OpenAlex parsing, using a recorded response - no network."""
import json
import os

from agents.search_client import _normalise, reconstruct_abstract

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "openalex_sample.json")


def _load():
    with open(FIXTURE, encoding="utf-8") as fh:
        return json.load(fh)["results"]


def test_abstract_is_rebuilt_in_word_order():
    inverted = {"world": [1], "hello": [0], "again": [2]}
    assert reconstruct_abstract(inverted) == "hello world again"


def test_missing_abstract_stays_none():
    # Why: None is what the retrieval threshold counts as "unusable"; it must not become "".
    assert reconstruct_abstract(None) is None


def test_normalise_strips_doi_prefix_and_keeps_authors():
    paper = _normalise(_load()[0])
    assert paper.doi == "10.1007/s11704-024-40231-1"
    assert paper.authors == ["Lei Wang", "Chen Ma"]
    assert paper.year == 2024
    assert paper.abstract == "Autonomous agents use LLMs"


def test_normalise_never_invents_a_doi():
    paper = _normalise(_load()[1])
    assert paper.doi is None and paper.abstract is None
