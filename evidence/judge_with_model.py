"""
A second model checks the agent's relevance screening.

Why it exists: judging every paper of a run by hand (evidence/judge_relevance.py)
is slow, and it was set aside after four papers. This program puts the same
question to a second Gemini model, a full Flash model rather than the
Flash-Lite one the agent uses, for the same papers and from the same text,
without showing it the agent's scores, and compares its answers with the
agent's decisions.

What it can show and what it cannot: agreement between two models. It is not
the precision of screening in the README's targets table, which is defined by
people's judgements. Both models are Gemini models and may share blind spots,
so a paper both of them keep may still be the wrong one.

What the second model sees: the research question, the sub-question, and for
each paper the title, the year and the abstract cut where the agent's scoring
prompt cut it (the same text judge_relevance.py shows a person). The question
and the yes/no definitions are the ones the person is given, word for word.
Papers go ten to a request, per sub-question, in the shuffled order of
judge_relevance.py, so the batches are not the agent's.

Requests: one per batch (seven for a run of 65 papers), plus at most one
repair each. Replies are cached like the agent's, so running it again costs
nothing. If the daily free quota runs out part-way, the batches already
answered are kept and the same command carries on later.

What it writes, in evidence/:
  model_judge_<run>_answers.csv  the second model's answer and reason for every paper
  model_judge_<run>_results.md   the comparison with the agent
  model_judge_<run>.log          the requests made (the agent's own runs log to output/run.log)

Usage, from the project folder with the virtual environment active:
    python evidence/judge_with_model.py output/brief-<run>.json
    python evidence/judge_with_model.py output/brief-<run>.json --model gemini-3.6-flash
"""
import argparse
import csv
import datetime
import json
import logging
import os
import re
import sys
from typing import List, Literal

from pydantic import BaseModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The full Flash model the provider's models page recommended when it was read
# on 1 October (README, "Why this model"). Its free-tier quota was not looked up;
# gemini-3.6-flash allowed 20 requests a day on 23 September.
DEFAULT_MODEL = "gemini-3.8-flash"
BATCH = 10

PROMPT = """You are screening papers for a literature review.

Research question: "{question}"
Sub-question: "{subquestion}"

For each paper below, answer this question: {ask}
- "yes": it addresses the sub-question, or is useful background for it.
- "no": it only shares words with the sub-question, or is about something else.
- "unsure": the title and abstract do not allow a decision.

Rules:
- Judge ONLY from the title and abstract given below. Do not use anything you may know about the paper from elsewhere.
- If the abstract is missing, decide from the title alone.
- The reason must be one sentence.

Papers:
{papers}

Return ONLY JSON in this exact shape, with exactly one entry for every index 1 to {n}:
{{"answers": [{{"index": 1, "answer": "yes", "reason": "..."}}]}}"""


class _Answer(BaseModel):
    """One paper's answer as the second model returns it; `index` is its position in the prompt."""
    index: int
    answer: Literal["yes", "no", "unsure"]
    reason: str = ""


class AnswerResponse(BaseModel):
    """JSON shape of the second model's reply."""
    answers: List[_Answer]


def _format(papers, as_scored):
    """The papers as numbered title/year/abstract blocks, in the layout of the agent's prompts."""
    blocks = []
    for i, p in enumerate(papers, start=1):
        abstract = as_scored(p["abstract"])[0] if p["found"] else ""
        blocks.append(f"[{i}] Title: {p['title']}\n    Year: {p['year'] or 'unknown'}\n"
                      f"    Abstract: {abstract or '(no abstract available)'}")
    return "\n\n".join(blocks)


def _write_answers(path, papers, answers, model):
    partial = path + ".part"
    with open(partial, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["position", "subquestion", "doi", "title", "model", "answer", "reason"])
        writer.writeheader()
        for n, p in enumerate(papers, start=1):
            answer, reason = answers.get((p["sq"], p["key"]), ("", ""))
            writer.writerow({"position": n, "subquestion": p["sq"], "doi": p["doi"], "title": p["title"],
                             "model": model if answer else "", "answer": answer, "reason": reason})
    os.replace(partial, path)


def _f(x):
    return "-" if x is None else f"{x:.2f}"


def report(brief, brief_path, figures, reasons, model, agent_model, requests_made, day, ask):
    """The comparison as Markdown."""
    import config
    rows = figures["rows"]
    no_abstract = sum(1 for r in rows if r["found"] and not r["abstract"].strip())
    a, b, c = figures["a"], figures["b"], figures["c"]
    out = [
        f"# Relevance screening checked by a second model: {brief['research_question']}",
        "",
        f"- Brief: `{os.path.relpath(brief_path, ROOT)}`",
        f"- Checked on {day} by `{model}`. The agent scored with `{agent_model}`.",
        f"- Model requests made by this run of the check: {requests_made} (answers already in the cache cost none).",
        f"- Papers: {figures['total']} scored by the agent, {figures['kept_total']} of them kept "
        f"(score {config.RELEVANCE_CUTOFF} or more out of 5).",
        "- What the second model saw for each paper: the research question, the sub-question that found it, "
        f"the title, the year and the abstract as the agent's scoring prompt showed it (cut at "
        f"{config.EVAL_ABSTRACT_CHARS:,} characters). {no_abstract} papers had no abstract. "
        f"Ten papers to a request, per sub-question, in a shuffled order. It was not shown the agent's scores.",
        f"- Question put to it, as to a person judging by hand: \"{ask}\" Yes: it addresses the sub-question, "
        "or is useful background for it. No: it only shares words with the sub-question, or is about something "
        "else. Unsure: the title and abstract do not allow a decision.",
        "",
        "**These figures are agreement between two models.** They are not the precision of screening in the "
        "README's targets table, which is defined by people's judgements, and the two models, both Gemini, may "
        "share blind spots.",
        "",
        "## Result",
        "",
        "| | Second model: yes | Second model: no | Second model: unsure |",
        "|---|---|---|---|",
        f"| Agent kept | {a} | {b} | {figures['unsure_kept']} |",
        f"| Agent dropped | {c} | {figures['d']} | {figures['unsure_dropped']} |",
        "",
        f"- Kept papers the second model would also keep: {a} of the {a + b} it answered yes or no about "
        f"({_f(figures['precision'])}).",
        f"- Dropped papers the second model would keep: {c}.",
        f"- Agreement between the two, Cohen's kappa (unsure answers left out): {_f(figures['kappa'])}.",
        "",
        "## By sub-question",
        "",
        "| Sub-question | Papers | Agent kept | Kept, second model yes | Share | Dropped, second model yes |",
        "|---|---|---|---|---|---|",
    ]
    for sq in brief["subquestions"]:
        mine = [r for r in rows if r["sq"] == sq["id"]]
        kept = [r for r in mine if r["kept"]]
        yes, no = sum(r["answer"] == "yes" for r in kept), sum(r["answer"] == "no" for r in kept)
        missed = sum(1 for r in mine if not r["kept"] and r["answer"] == "yes")
        out.append(f"| {sq['id']} | {len(mine)} | {len(kept)} | {yes} | {_f(yes / (yes + no) if yes + no else None)} | {missed} |")

    def listing(title, chosen):
        out.extend(["", f"### {title} ({len(chosen)})", ""])
        out.extend([f"- Sub-question {r['sq']}, agent's score {r['score']}: {r['title']} ({r['year'] or 'year unknown'}). "
                    f"Agent: \"{r['reason']}\" Second model: \"{reasons.get((r['sq'], r['key']), '')}\""
                    for r in chosen] or ["None."])

    out += ["", "## Where the two models differ"]
    listing("Kept by the agent, second model says no", [r for r in rows if r["kept"] and r["answer"] == "no"])
    listing("Dropped by the agent, second model says yes", [r for r in rows if not r["kept"] and r["answer"] == "yes"])
    listing("Second model unsure", [r for r in rows if r["answer"] == "unsure"])
    out += [
        "",
        "## What this shows and what it does not",
        "",
        "- How often a second, larger model agrees with the agent's screening of the papers it found, judged from "
        "the same titles and abstracts. Not whether either model is right: that needs people.",
        "- Not the search: whether the right papers were found at all (recall of retrieval) needs an independent "
        "list, such as the included studies of a published systematic review.",
        "- One run on one research question.",
        "",
    ]
    return "\n".join(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Ask a second model about every paper one run scored, then compare.")
    parser.add_argument("brief", help="the brief's JSON file in output/")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"the Gemini model that judges (default {DEFAULT_MODEL})")
    args = parser.parse_args(argv)

    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.join(ROOT, "evidence"))
    # Before config is imported: the model name is part of the endpoint and of the cache key.
    os.environ["GEMINI_MODEL"] = args.model
    import config
    import judge_relevance as jr
    from agents import llm_client
    from agents.evaluation_agent import _exact_index_model
    if config.GEMINI_MODEL != args.model:
        print(f"Could not switch the model to {args.model} (config says {config.GEMINI_MODEL}).")
        return 1
    if not config.GEMINI_API_KEY:
        print("GEMINI_API_KEY is not set - the check needs the same key as the agent (.env).")
        return 1

    brief_path = os.path.abspath(args.brief)
    with open(brief_path, encoding="utf-8") as fh:
        brief = json.load(fh)
    run, _, _ = jr._paths(brief_path)
    folder = os.path.join(ROOT, "evidence")
    answers_path = os.path.join(folder, f"model_judge_{run}_answers.csv")
    results_path = os.path.join(folder, f"model_judge_{run}_results.md")
    log_path = os.path.join(folder, f"model_judge_{run}.log")
    found = re.search(r"generated by (\S+) from abstracts", " ".join(brief.get("limitations", [])))
    agent_model = found.group(1) if found else "the agent's model"

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger = logging.getLogger("research_agent")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.propagate = False
    requests_made = []
    counter = logging.Handler()
    counter.emit = lambda record: requests_made.append(1) if record.getMessage().startswith("llm | prompt_chars") else None
    logger.addHandler(counter)
    logging.getLogger("research_agent.judge").info("model judge | brief=%s | model=%s", os.path.basename(brief_path), args.model)

    papers = jr.blind_order(brief, jr.load_papers(brief))
    missing = sum(1 for p in papers if not p["found"])
    print(f"Checking the {len(papers)} papers of run {run} with {args.model}, ten to a request.")
    if missing:
        print(f"Note: {missing} records could not be found in cache/; for those the model sees only the title.")

    answers, reasons = {}, {}
    try:
        for sq in brief["subquestions"]:
            group = [p for p in papers if p["sq"] == sq["id"]]
            batches = [group[i:i + BATCH] for i in range(0, len(group), BATCH)]
            for number, batch in enumerate(batches, start=1):
                prompt = PROMPT.format(question=brief["research_question"], subquestion=sq["text"],
                                       ask=jr.QUESTION_TO_JUDGE, papers=_format(batch, jr._as_scored), n=len(batch))
                parsed = llm_client.generate_json(prompt, _exact_index_model(AnswerResponse, "answers", len(batch)))
                for item in parsed.answers:
                    p = batch[item.index - 1]
                    answers[(p["sq"], p["key"])] = (item.answer, "")
                    reasons[(p["sq"], p["key"])] = item.reason.strip()
                _write_answers(answers_path, papers, {k: (v[0], reasons[k]) for k, v in answers.items()}, args.model)
                print(f"  sub-question {sq['id']}: batch {number} of {len(batches)} answered")
    except llm_client.LLMError as exc:
        logging.getLogger("research_agent.judge").warning("model judge | stopped | %s", exc)
        print(f"\nStopped: {exc}")
        if "HTTP 404" in str(exc):
            print(f"The model name {args.model} was not found. Try another, e.g. --model gemini-3.6-flash")
        print(f"{len(answers)} of {len(papers)} papers answered and saved. Run the command again: "
              "the batches already answered come from the cache and cost no requests.")
        return 1

    figures = jr.compare(papers, answers)
    day = datetime.date.today().isoformat()
    text = report(brief, brief_path, figures, reasons, args.model, agent_model, len(requests_made), day,
                  jr.QUESTION_TO_JUDGE)
    with open(results_path, "w", encoding="utf-8") as fh:
        fh.write(text)
    print()
    print(text)
    print(f"Written: {os.path.relpath(results_path, ROOT)}, {os.path.relpath(answers_path, ROOT)}, "
          f"{os.path.relpath(log_path, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
