"""
Logging configuration.

The design argument rests on traceability: every query, score and decision
must be inspectable after the run (proposal, section 5 - explainability
mitigation). So logging is set up once, to both console and a file, and every
agent logs through the standard library rather than printing.
"""
import logging
import os

import config


def setup() -> logging.Logger:
    os.makedirs(os.path.dirname(config.LOG_FILE), exist_ok=True)
    logger = logging.getLogger("research_agent")
    if logger.handlers:            # idempotent - safe to call more than once
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(config.LOG_FILE, encoding="utf-8")):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger
