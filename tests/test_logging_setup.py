"""Logging tests: one console handler, one file handler, however often setup() is called."""
import logging
import os

import pytest

import config
from agents import logging_setup


@pytest.fixture
def clean_logger(monkeypatch, tmp_path):
    """Give setup() an empty logger and a log file in a temporary folder; put everything back afterwards."""
    logger = logging.getLogger("research_agent")
    saved, logger.handlers = logger.handlers, []
    monkeypatch.setattr(config, "LOG_FILE", str(tmp_path / "out" / "run.log"))
    yield logger
    for handler in logger.handlers:
        handler.close()
    logger.handlers = saved


def test_setup_creates_the_folder_and_writes_every_agent_line_to_the_file(clean_logger):
    logging_setup.setup()
    logging.getLogger("research_agent.retrieval").info("retrieved | subquestion=1 | records=20")
    for handler in clean_logger.handlers:
        handler.flush()
    text = open(config.LOG_FILE, encoding="utf-8").read()
    assert os.path.isdir(os.path.dirname(config.LOG_FILE))
    assert "| INFO | research_agent.retrieval | retrieved | subquestion=1 | records=20" in text


def test_setup_twice_does_not_double_the_handlers(clean_logger):
    logging_setup.setup()
    logging_setup.setup()
    assert len(clean_logger.handlers) == 2          # console and file, once each
