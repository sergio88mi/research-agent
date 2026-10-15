"""
Offline replay of the run of 30 September 2026 - a regression check, not a live test.

Why it exists: remediations #12 onwards were made after the last live run. A new
live run would cost model quota and would not return the same records, so it
could not show whether the fixes changed anything they were not meant to. The
API replies saved in cache/ allow a stricter check: give the present code
exactly the replies the 30 September run worked from (its own and the earlier
ones it reused) and compare the brief it writes with the brief written then. Any field that differs is a change in
behaviour and has to be explained.

What it does: copies cache/ to a temporary folder, switches the network off
(so the three Semantic Scholar searches that were never answered on 30
September fail over to OpenAlex again, as they did then), runs the standard
question twice with both reviews approved, compares each brief field by field
with the one saved on 30 September, and writes the result and the log lines
of both passes to evidence/replay_offline.log. It reads cache/ and output/
and changes neither.

Why two passes: the first pass meets the replies the old code had stored
without checking them (remediation #13); the second shows what a later run
sees once only validated replies are kept.

It needs cache/ from the machine that made the 30 September run. That folder
holds API replies and is not in the repository, so the replay cannot be
repeated from a clone; the log is the evidence.

Usage, from the project folder:  python evidence/replay_offline.py
"""
import contextlib
import datetime
import io
import json
import logging
import os
import platform
import shutil
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUESTION = "What approaches reduce hallucination in LLM-based autonomous agents?"
REFERENCE = os.path.join("output", "brief-20260930-101857-what-approaches-reduce-hallucination-in-.json")
REPORT = os.path.join("evidence", "replay_offline.log")


def _no_network(*args, **kwargs):
    """Stands in for requests.get and requests.post: every call fails as if the machine were offline."""
    import requests
    raise requests.ConnectionError("network switched off for the replay")


def _differences(old, new, path=""):
    """Every place where two JSON values differ, as 'path: old -> new' lines."""
    if isinstance(old, dict) and isinstance(new, dict):
        out = []
        for key in sorted(set(old) | set(new)):
            out += _differences(old.get(key), new.get(key), f"{path}.{key}" if path else key)
        return out
    if isinstance(old, list) and isinstance(new, list):
        out = [] if len(old) == len(new) else [f"{path}: {len(old)} items -> {len(new)} items"]
        for i, (a, b) in enumerate(zip(old, new)):
            out += _differences(a, b, f"{path}[{i}]")
        return out
    return [] if old == new else [f"{path}:\n      30 Sept: {old!r}\n      replay:  {new!r}"]


def main() -> int:
    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    if not os.path.isdir("cache") or not os.path.exists(REFERENCE):
        print("Nothing to replay: this needs cache/ and " + REFERENCE + " from the machine that made the run.")
        return 1

    import requests
    import config

    work = tempfile.mkdtemp(prefix="replay-")
    shutil.copytree("cache", os.path.join(work, "cache"))
    config.CACHE_DIR = os.path.join(work, "cache")
    config.OUTPUT_DIR = os.path.join(work, "output")
    config.LOG_FILE = os.path.join(work, "output", "run.log")
    # The settings of 30 September; the model name is part of the key the model's replies are stored under.
    config.GEMINI_MODEL = "gemini-3.5-flash-lite"
    config.SOURCES = ["openalex", "semanticscholar"]
    config.GEMINI_API_KEY = config.GEMINI_API_KEY or "not-needed-offline"
    requests.get = requests.post = _no_network
    time.sleep = lambda seconds: None            # the retries of the unanswered searches need not really wait

    from agents.orchestrator import Orchestrator
    from agents.researcher_interface import ResearcherInterface

    lines = []
    handler = logging.Handler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    handler.emit = lambda record: lines.append(handler.format(record))
    logger = logging.getLogger("research_agent")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.propagate = False

    with open(REFERENCE, encoding="utf-8") as fh:
        reference = json.load(fh)

    report = [
        "Offline replay of the 30 September 2026 run",
        f"Made: {datetime.datetime.now().isoformat(timespec='seconds')} | Python {platform.python_version()} on {platform.system()}",
        f"Question: {QUESTION}",
        f"Compared with: {REFERENCE}",
        f"Cache files replayed: {len(os.listdir(config.CACHE_DIR))} (copied; cache/ itself is not written to)",
        "Network: switched off for every service",
        "",
    ]
    for number in (1, 2):
        start = len(lines)
        screen = io.StringIO()
        with contextlib.redirect_stdout(screen):
            brief = Orchestrator(interface=ResearcherInterface(scripted=["approve", "approve"])).run(QUESTION)
        mine = lines[start:]
        summary = [text for text in screen.getvalue().splitlines() if "unique papers across" in text]
        different = _differences(reference, json.loads(brief.model_dump_json()))
        report += [
            f"=== Pass {number} ===",
            "Summary line: " + (summary[-1] if summary else "(none)"),
            f"Model requests made: {sum(' | llm | prompt_chars=' in text for text in mine)}",
            f"Replies rejected by the schema check and repaired: {sum('invalid JSON' in text for text in mine)}",
            f"Searches that failed over: {sum('continuing with remaining sources' in text for text in mine)}",
            f"Fields that differ from the brief of 30 September: {len(different)}",
        ] + [f"  - {text}" for text in different] + [""]
        report += [f"--- log of pass {number} ({len(mine)} lines) ---"] + mine + [""]

    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(report) + "\n")
    shutil.rmtree(work, ignore_errors=True)
    print("\n".join(text for text in report if not text.startswith("20")))   # everything but the log lines
    print(f"Written: {REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
