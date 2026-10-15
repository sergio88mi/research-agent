"""
Judging the agent's relevance screening by hand.

Why it exists: the agent keeps a paper when the model scores it 3 or more out
of 5 from its title and abstract. Nothing in the pipeline checks those scores
against a person, so the README lists "precision of screening" as a target
that has not been measured. This program puts the papers of one run in front
of a person, one at a time, and afterwards compares the person's answers with
the agent's decisions.

What the judge sees: for each paper, the research question, the sub-question
that found it, the title, the year and the abstract cut where the agent's
scoring prompt cut it (config.EVAL_ABSTRACT_CHARS; the rest can be shown on
request). The agent's score, reason and decision stay hidden until every paper
has an answer. Papers are grouped by sub-question and shuffled within each
group with a fixed seed, so kept and dropped papers are mixed and the order is
the same every time.

Where the texts come from: the search replies the run saved in cache/ (a
brief stores the abstracts of the kept papers only). Nothing is fetched from
the network. Like evidence/replay_offline.py it therefore needs the cache/
folder of the machine that made the run; the files it writes are the evidence.

What it writes, in evidence/:
  relevance_<run>_answers.csv  one row per paper, saved after every answer, so a
                               session can be stopped and carried on later
  relevance_<run>_results.md   the comparison, once every paper has an answer

Usage, from the project folder with the virtual environment active:
    python evidence/judge_relevance.py output/brief-<run>.json
    python evidence/judge_relevance.py output/brief-<run>.json --results
The second form asks nothing and shows the comparison again. There is no
comparison of part of the answers: it would reveal the agent's decisions
before the judging is over.
"""
import argparse
import csv
import datetime
import json
import os
import random
import re
import shutil
import sys
import textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 20261002                      # fixed, so the order of the papers can be repeated
REPLIES = {"y": "yes", "yes": "yes", "n": "no", "no": "no", "?": "unsure", "u": "unsure"}
QUESTION_TO_JUDGE = "Would you keep this paper for this sub-question?"
GUIDE = ("  y = yes: it addresses the sub-question, or is useful background for it\n"
         "  n = no: it only shares words with the sub-question, or is about something else\n"
         "  ? = cannot decide       f = show the whole abstract\n"
         "  b = back one paper      q = stop (every answer so far is saved)")
FIELDS = ["position", "subquestion", "doi", "title", "answer", "answered_at"]


def _key(doi, title):
    """Identity of a paper, as in the pipeline: its DOI, or its title when it has none."""
    return (doi or title).strip().lower()


def _as_scored(abstract):
    """The abstract as the scoring prompt showed it (same cut as EvaluationSynthesisAgent._format_papers)."""
    import config
    text = (abstract or "").strip()
    if len(text) > config.EVAL_ABSTRACT_CHARS:
        return text[: config.EVAL_ABSTRACT_CHARS].rstrip() + " [...]", True
    return text, False


def load_papers(brief):
    """Every paper the agent scored, with the agent's decision and the text it scored from.

    Each sub-question's searches are read back from cache/ in the order the run
    made them (the first query before a reformulated one, sources in the order
    of config.SOURCES), and the first record of each paper is taken - the
    record the orchestrator's de-duplication kept.
    """
    import config
    from agents import cache, s2_client, search_client
    readers = {"openalex": (search_client._normalise, "results"),
               "semanticscholar": (s2_client._normalise, "data")}
    sources = [s for s in config.SOURCES if s in readers] or list(readers)
    first = {}
    for sq in brief["subquestions"]:
        for query in [q for q in (sq.get("previous_query"), sq["search_query"]) if q]:
            for name in sources:
                normalise, field = readers[name]
                for item in (cache.get(name, f"{query}|{config.RESULTS_PER_QUERY}") or {}).get(field, []):
                    record = normalise(item)
                    first.setdefault((sq["id"], _key(record.doi, record.title)), record)

    papers = []
    for a in brief["assessments"]:
        record = first.get((a["subquestion_id"], _key(a["paper_doi"], a["paper_title"])))
        papers.append({
            "sq": a["subquestion_id"], "key": _key(a["paper_doi"], a["paper_title"]),
            "doi": a["paper_doi"] or "", "title": a["paper_title"],
            "year": record.year if record else None,
            "abstract": (record.abstract or "") if record else "",
            "found": record is not None,
            "kept": bool(a["selected"]), "score": a["relevance_score"], "reason": a["reason"],
        })
    return papers


def blind_order(brief, papers):
    """Grouped by sub-question, shuffled within each group with the fixed seed."""
    rng = random.Random(SEED)
    out = []
    for sq in brief["subquestions"]:
        group = [p for p in papers if p["sq"] == sq["id"]]
        rng.shuffle(group)
        out += group
    return out


def _paths(brief_path):
    found = re.search(r"brief-(\d{8}-\d{6})", os.path.basename(brief_path))
    run = found.group(1) if found else os.path.splitext(os.path.basename(brief_path))[0]
    folder = os.path.join(ROOT, "evidence")
    return (run, os.path.join(folder, f"relevance_{run}_answers.csv"),
            os.path.join(folder, f"relevance_{run}_results.md"))


def read_answers(path):
    """(sub-question, paper) -> (answer, time answered), from an earlier session if there was one."""
    if not os.path.exists(path):
        return {}
    with open(path, newline="", encoding="utf-8") as fh:
        return {(int(r["subquestion"]), _key(r["doi"], r["title"])): (r["answer"], r["answered_at"])
                for r in csv.DictReader(fh) if r["answer"]}


def write_answers(path, papers, answers):
    """All answers so far, in judging order; written to a temporary file and renamed, so never half-written."""
    partial = path + ".part"
    with open(partial, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for n, p in enumerate(papers, start=1):
            answer, when = answers.get((p["sq"], p["key"]), ("", ""))
            writer.writerow({"position": n, "subquestion": p["sq"], "doi": p["doi"], "title": p["title"],
                             "answer": answer, "answered_at": when})
    os.replace(partial, path)


def _wrap(text, indent="    "):
    width = min(shutil.get_terminal_size((100, 24)).columns, 100) - 1
    return textwrap.fill(text, width=width, initial_indent=indent, subsequent_indent=indent)


def _show(brief, p, n, total, previous_answer):
    import config
    sq = next(s for s in brief["subquestions"] if s["id"] == p["sq"])
    print("\n" + "=" * 78)
    print(f"Paper {n} of {total}  |  sub-question {p['sq']} of {len(brief['subquestions'])}")
    print("=" * 78)
    print(_wrap(f"Research question: {brief['research_question']}", ""))
    print(_wrap(f"Sub-question {p['sq']}: {sq['text']}", ""))
    print()
    print(_wrap(f"Title: {p['title']}", ""))
    print(f"Year:  {p['year'] or 'unknown'}")
    if not p["found"]:
        print("Abstract: (the record could not be found in cache/; judge from the title)")
    elif not p["abstract"].strip():
        print("Abstract: none - the agent judged this paper from its title alone")
    else:
        text, cut = _as_scored(p["abstract"])
        print("Abstract, as the agent saw it:")
        print(_wrap(text))
        if cut:
            print(f"    (the agent saw the first {config.EVAL_ABSTRACT_CHARS:,} characters; f shows the rest)")
    print()
    if previous_answer:
        print(f"Your answer: {previous_answer}. Type a new one, or press Enter to keep it.")
    print(QUESTION_TO_JUDGE)
    print(GUIDE)


def judge(brief, papers, answers, answers_path):
    """Ask about every paper not yet answered. True when every paper has an answer."""
    total = len(papers)
    position = next((i for i, p in enumerate(papers) if (p["sq"], p["key"]) not in answers), total)
    while position < total:
        p = papers[position]
        previous = answers.get((p["sq"], p["key"]), ("", ""))[0]
        _show(brief, p, position + 1, total, previous)
        while True:
            reply = input("> ").strip().lower()
            if reply in REPLIES:
                answers[(p["sq"], p["key"])] = (REPLIES[reply], datetime.datetime.now().isoformat(timespec="seconds"))
                write_answers(answers_path, papers, answers)
                position += 1
                break
            if reply == "" and previous:
                position += 1
                break
            if reply == "f":
                print("\nWhole abstract:")
                print(_wrap(p["abstract"] or "(none)"))
                print()
                continue
            if reply == "b":
                position = max(0, position - 1)
                break
            if reply == "q":
                return False
            print("Please type y, n, ?, f, b or q.")
    return True


# ------------------------------------------------------------------ comparison
def compare(papers, answers):
    """The figures of the comparison. Unsure answers are kept out of precision and agreement and reported beside them."""
    rows = [dict(p, answer=answers.get((p["sq"], p["key"]), ("", ""))[0]) for p in papers]

    def count(kept, answer):
        return sum(1 for r in rows if r["kept"] == kept and r["answer"] == answer)

    a, b, c, d = count(True, "yes"), count(True, "no"), count(False, "yes"), count(False, "no")
    unsure_kept, unsure_dropped = count(True, "unsure"), count(False, "unsure")
    kept_total = sum(1 for r in rows if r["kept"])
    n = a + b + c + d
    kappa = None
    if n:
        observed = (a + d) / n
        expected = ((a + b) * (a + c) + (c + d) * (b + d)) / (n * n)
        kappa = None if expected == 1 else (observed - expected) / (1 - expected)
    return {
        "rows": rows, "a": a, "b": b, "c": c, "d": d,
        "unsure_kept": unsure_kept, "unsure_dropped": unsure_dropped,
        "answered": sum(1 for r in rows if r["answer"]), "total": len(rows), "kept_total": kept_total,
        "precision": a / (a + b) if a + b else None,
        "precision_unsure_as_no": a / kept_total if kept_total else None,
        "precision_unsure_as_yes": (a + unsure_kept) / kept_total if kept_total else None,
        "kept_of_relevant": a / (a + c) if a + c else None,
        "kappa": kappa,
    }


def _f(x):
    return "-" if x is None else f"{x:.2f}"


def report(brief, brief_path, figures):
    """The comparison as Markdown."""
    import config
    rows = figures["rows"]
    days = sorted({stamp[:10] for stamp in figures.get("stamps", []) if stamp})
    when = (days[0] if len(days) == 1 else f"{days[0]} to {days[-1]}") if days else "-"
    removed = figures["kept_total"] - len(brief.get("selected_papers", []))
    no_abstract = sum(1 for r in rows if r["found"] and not r["abstract"].strip())
    missing = sum(1 for r in rows if not r["found"])
    target = 0.8                    # the README's target for precision of screening
    p = figures["precision"]
    out = [
        f"# Relevance screening judged by hand: {brief['research_question']}",
        "",
        f"- Brief: `{os.path.relpath(brief_path, ROOT)}`",
        f"- Judged on: {when}",
        "- Judges: one person",
        f"- Papers: {figures['total']} scored by the agent, {figures['kept_total']} of them kept "
        f"(score {config.RELEVANCE_CUTOFF} or more out of 5)"
        + (f"; {removed} of those were removed at review 2" if removed > 0 else ""),
        "- What the judge saw for each paper: the research question, the sub-question that found it, "
        f"the title, the year and the abstract as the agent's scoring prompt showed it (cut at "
        f"{config.EVAL_ABSTRACT_CHARS:,} characters; the rest could be shown on request). "
        f"{no_abstract} papers had no abstract." + (f" {missing} records could not be found in cache/." if missing else ""),
        "- Hidden until every paper had an answer: the agent's score, reason and decision.",
        f"- Order: grouped by sub-question, shuffled within each group (seed {SEED}).",
        f"- Question put to the judge: \"{QUESTION_TO_JUDGE}\" Yes: it addresses the sub-question, "
        "or is useful background for it. No: it only shares words with the sub-question, or is about something else. "
        "This mirrors the line the agent's scale draws between 3 (\"partially relevant or useful background\") "
        "and 2 (\"tangential; shares vocabulary but not the question\").",
        "",
        "## Result",
        "",
        "| | Judge: yes | Judge: no | Judge: unsure |",
        "|---|---|---|---|",
        f"| Agent kept | {figures['a']} | {figures['b']} | {figures['unsure_kept']} |",
        f"| Agent dropped | {figures['c']} | {figures['d']} | {figures['unsure_dropped']} |",
        "",
        f"- **Precision of screening: {_f(p)}**: of the {figures['a'] + figures['b']} kept papers given a yes or a no, "
        f"{figures['a']} were judged relevant. Target: at least {target}. "
        + ("Met." if p is not None and p >= target else "Not met." if p is not None else ""),
        f"- With the {figures['unsure_kept']} kept papers marked unsure counted as not relevant: "
        f"{_f(figures['precision_unsure_as_no'])}; counted as relevant: {_f(figures['precision_unsure_as_yes'])}.",
        f"- Relevant papers the agent dropped: {figures['c']}. Of the {figures['a'] + figures['c']} papers the judge "
        f"found relevant, the agent kept {figures['a']} ({_f(figures['kept_of_relevant'])}).",
        f"- Agreement between the agent and the judge, Cohen's kappa (unsure answers left out): {_f(figures['kappa'])}. "
        "No threshold was set for it.",
        "",
        "## By sub-question",
        "",
        "| Sub-question | Papers | Agent kept | Kept and judged relevant | Precision | Dropped but judged relevant |",
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
        out.extend([f"- Sub-question {r['sq']}, score {r['score']}: {r['title']} ({r['year'] or 'year unknown'}). "
                    f"Agent's reason: \"{r['reason']}\"" for r in chosen] or ["None."])

    out += ["", "## Where the judge and the agent differ"]
    listing("Kept by the agent, judged not relevant", [r for r in rows if r["kept"] and r["answer"] == "no"])
    listing("Dropped by the agent, judged relevant", [r for r in rows if not r["kept"] and r["answer"] == "yes"])
    listing("Judged unsure", [r for r in rows if r["answer"] == "unsure"])
    out += [
        "",
        "## What this shows and what it does not",
        "",
        "- It measures the screening: how well the agent sorted the papers it had found. It does not measure the "
        "search. Whether the right papers were found at all (recall of retrieval) needs an independent list, such as "
        "the included studies of a published systematic review.",
        "- One person judged, where the README's method asks for two and their agreement. The figures are one "
        "person's view of one run on one research question.",
        "- The judge saw what the agent saw: titles and abstracts, not full texts.",
        "",
    ]
    return "\n".join(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Judge by hand the papers one run scored, then compare with the agent.")
    parser.add_argument("brief", help="the brief's JSON file in output/")
    parser.add_argument("--results", action="store_true",
                        help="do not ask anything; show the comparison again (once every paper has an answer)")
    args = parser.parse_args(argv)

    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    brief_path = os.path.abspath(args.brief)
    with open(brief_path, encoding="utf-8") as fh:
        brief = json.load(fh)
    papers = blind_order(brief, load_papers(brief))
    run, answers_path, results_path = _paths(brief_path)
    answers = read_answers(answers_path)

    if not args.results:
        missing = sum(1 for p in papers if not p["found"])
        print(f"Judging the papers of run {run}: {len(papers)} papers, {len(answers)} already answered.")
        if missing:
            print(f"Note: {missing} records could not be found in cache/; for those only the title can be shown.")
        print(f"Answers are saved after every paper in {os.path.relpath(answers_path, ROOT)}.")
        print("The agent's decisions are shown only when every paper has an answer.")
        try:
            done = judge(brief, papers, answers, answers_path)
        except (KeyboardInterrupt, EOFError):
            done = False
        if not done:
            print(f"\nStopped with {len(answers)} of {len(papers)} answered. Run the same command to carry on.")
            return 0

    answered = sum(1 for p in papers if (p["sq"], p["key"]) in answers)
    if answered < len(papers):
        # No partial comparison: it would show the agent's decisions on papers already
        # judged, and with them hints about the papers still to come.
        print(f"{answered} of {len(papers)} papers have an answer. The comparison is shown "
              "only when every paper has one, so that the agent's decisions stay hidden until then.")
        return 0
    figures = compare(papers, answers)
    figures["stamps"] = [v[1] for v in answers.values()]
    text = report(brief, brief_path, figures)
    with open(results_path, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    print(f"Written: {os.path.relpath(results_path, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
