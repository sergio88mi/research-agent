"""Entry-point tests: how a run ends - brief written, stopped with a message, or interrupted. No network.

`main.setup` is replaced throughout so that a test run never writes to the
real output/run.log, which is part of the evidence.
"""
from unittest.mock import patch

import pytest
import requests

import config
import main
from agents.llm_client import LLMError


@pytest.fixture(autouse=True)
def no_log_file(monkeypatch):
    monkeypatch.setattr(main, "setup", lambda: None)


def test_missing_key_ends_with_the_message_and_status_1(monkeypatch, capsys):
    # Why: functional test FT-01 as a unit test - nothing is searched, the message names the variable.
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")
    with patch("agents.search_client.search") as search:
        status = main.main(["test"])
    out = capsys.readouterr().out
    assert status == 1 and not search.called
    assert "Run stopped: GEMINI_API_KEY is not set" in out and ".env.example" in out


def test_a_completed_run_ends_with_status_0(capsys):
    with patch("main.Orchestrator") as orch:
        status = main.main(["a", "question"])
    assert status == 0 and "Run complete - brief written." in capsys.readouterr().out
    orch.return_value.run.assert_called_once_with("a question")
    assert orch.call_args.kwargs["constraints"] is None


def test_constraints_given_on_the_command_line_reach_the_planner():
    # The real Orchestrator is built; only its run() is replaced, so the planner it holds can be inspected.
    with patch.object(main.Orchestrator, "run", autospec=True) as run:
        main.main(["a question", "--constraints", "healthcare settings only"])
    orchestrator, question = run.call_args.args
    assert question == "a question" and orchestrator.planner.constraints == "healthcare settings only"


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, EOFError])      # Ctrl-C, or Ctrl-D at a prompt
def test_an_interruption_ends_quietly_with_status_130(interruption, capsys):
    with patch("main.Orchestrator") as orch:
        orch.return_value.run.side_effect = interruption
        status = main.main(["q"])
    assert status == 130 and "no brief written" in capsys.readouterr().out


@pytest.mark.parametrize("error", [
    LLMError("Gemini daily free-tier quota is exhausted for this model"),
    RuntimeError("every retrieval source failed for sub-question 1"),
    requests.ConnectionError("no route to host"),
])
def test_an_operational_failure_is_reported_in_one_line(error, capsys, caplog):
    with patch("main.Orchestrator") as orch:
        orch.return_value.run.side_effect = error
        status = main.main(["q"])
    assert status == 1 and f"Run stopped: {error}" in capsys.readouterr().out
    assert "run | stopped" in caplog.text                 # the reason also goes to the log


def test_a_wrong_source_name_is_reported_not_traced(monkeypatch, capsys):
    monkeypatch.setattr(config, "SOURCES", ["openalex", "scopus"])
    assert main.main(["q"]) == 1
    assert "config.SOURCES must name" in capsys.readouterr().out


@pytest.mark.parametrize("bug", [TypeError("unexpected"), ValueError("unexpected")])
def test_a_bug_is_not_hidden(bug):
    # Why: only the failures the agents raise on purpose (RuntimeErrors) are turned into a message.
    # A ValueError is the family a schema or JSON error belongs to, so it must surface too.
    with patch("main.Orchestrator") as orch:
        orch.return_value.run.side_effect = bug
        with pytest.raises(type(bug)):
            main.main(["q"])


def test_no_question_given_does_nothing(capsys):
    with patch("builtins.input", return_value=""), patch("main.Orchestrator") as orch:
        status = main.main([])
    assert status == 1 and not orch.called
    assert "No research question given" in capsys.readouterr().out
