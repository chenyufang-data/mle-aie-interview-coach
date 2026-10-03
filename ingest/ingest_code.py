"""Build banks/rag_code (roadmap step 7, phase 2) from the author's practice
notebooks: the coding problems for practice and the mock coding round.

    .venv\\Scripts\\python ingest\\ingest_code.py --source "<notes folder>" --list
        Free: parse the notebooks, print what would go in the bank.
    .venv\\Scripts\\python ingest\\ingest_code.py --source "<notes folder>" --generate [--limit N]
        Dry run: how many records would be written, on which engine.
    .venv\\Scripts\\python ingest\\ingest_code.py --source "<notes folder>" --generate --confirm
        Write each missing record's rubric, hints and follow-ups through the
        user's signed-in Claude Code / Codex subscription (coach/cli_engine.py),
        then append it to banks/rag_code/all_chunks.jsonl. Idempotent by id.
    .venv\\Scripts\\python ingest\\ingest_code.py --source "<notes folder>" --check
        Re-run the LeetCode-text checks on the bank.

What goes in, by LeetCode's terms (no copying or republishing its content,
no scraping - docs/plan.md step 7):
- LeetCode problems: number, title, link, and what we write - approach
  labels, role, an estimated complexity, a rubric, hints, follow-ups. Some
  notebook cells hold LeetCode's own statement pasted in; from those only
  the number, the title line and the link are read. Pasted statement text is
  never written to the bank and never sent to a model; it is held in memory
  only to check that nothing generated overlaps it.
- The author's own exercises (ML-from-scratch, PyTorch): the statement and
  any starter code, since the text is the author's; answer keys inside
  <details> blocks are left out.
- Not included: the SQL drills (close paraphrases of LeetCode's SQL
  problems, same table and column names).

Records keep the bank chunk shape (id / interview / metadata) so
tools/review_bank.py can review them; the coding fields sit under "code".
The bank is not one of the practice track's corpora (coach/config.py
CORPUS_PATHS), so question retrieval never sees it.
"""

import argparse
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from coach import cli_engine, config  # noqa: E402

BANK_DIR = BASE_DIR / "banks" / "rag_code"
OUT_PATH = BANK_DIR / "all_chunks.jsonl"

DSA_GLOBS = ("week[1-6]_homework.ipynb", "additional_dsa_*.ipynb")
OWN_NOTEBOOKS = {"week7_homework.ipynb": "ml", "additional_ml_coding.ipynb": "ml",
                 "pytorch_practice.ipynb": "pytorch"}
LC_LINK = re.compile(r"leetcode\.com/problems/([a-z0-9-]+)")
LC_TITLED = re.compile(r"(?i)^leet\s*code\s*#?\s*(\d+)\s*\|\s*(.+)$")
LC_BARE = re.compile(r"(?i)^leet\s*code\s*#?\s*(\d+)\s*$")
OWN_HEAD = re.compile(r"^(ML|PT)\s+(\d+)\s*\|\s*(.+)$")
FIELD = re.compile(r"^\*\*(Difficulty|Pattern|Topic|Target):\*\*\s*(.+?)\s*$", re.M)
NOTEBOOK_GROUPS = {"week1_homework.ipynb": "Arrays and hashing warm-up"}
DETAILS = re.compile(r"<details>.*?</details>", re.S | re.I)
_lock = threading.Lock()


# ------------------------------------------------------------------ parsing

def title_from_slug(slug):
    return " ".join(part.capitalize() for part in slug.split("-"))


def parse_leetcode_cell(src):
    """(entry, pasted_text). The entry carries only what the bank may hold
    plus the author's own notes for generation; pasted_text is LeetCode's
    statement when the cell holds one (for the overlap check only)."""
    lines = [line.strip() for line in src.strip().splitlines()]
    slug = LC_LINK.search(src).group(1)
    first = lines[0] if lines else ""
    titled = LC_TITLED.match(first)
    bare = LC_BARE.match(first)
    if titled:
        number, title, pasted = int(titled.group(1)), titled.group(2).strip(), None
    else:
        number = int(bare.group(1)) if bare else int(re.search(r"(?i)leet\s*code\s*#?\s*(\d+)", src).group(1))
        candidate = next((l for l in lines[1:] if l), "")
        title = (candidate if 0 < len(candidate) <= 80 and not candidate.endswith(".")
                 and "leetcode.com" not in candidate else title_from_slug(slug))
        body_start = src.find(candidate) + len(candidate) if candidate else 0
        pasted = src[body_start:].strip() or None
    notes, difficulty = {}, ""
    if titled:
        for key, value in FIELD.findall(src):
            if key in ("Pattern", "Target"):
                notes[key.lower()] = value
            elif key == "Difficulty":
                difficulty = value.strip().lower()
    return ({"source": "leetcode", "number": number, "title": title, "slug": slug,
             "link": f"https://leetcode.com/problems/{slug}/", "notes": notes,
             "difficulty": difficulty}, pasted)


def parse_own_cell(src, family):
    first, _, rest = src.strip().partition("\n")
    head = OWN_HEAD.match(first.strip())
    fields = dict(FIELD.findall(rest))
    statement = DETAILS.sub("", rest)
    statement = FIELD.sub("", statement).strip()
    # the answer key it points to is not in the bank
    statement = re.sub(r"\s*Do not open the collapsed list[^.]*\.", "", statement)
    return {"source": "own", "family": family, "prefix": head.group(1),
            "number": int(head.group(2)), "title": head.group(3).strip(),
            "difficulty": fields.get("Difficulty", "").strip(),
            "topic": fields.get("Topic", "").strip(), "statement": statement}


def parse_notebook(path, family=None):
    """Problems in one notebook, with the section heading each sits under."""
    cells = json.loads(Path(path).read_text(encoding="utf-8"))["cells"]
    entries, pasted_texts, section = [], [], ""
    for i, cell in enumerate(cells):
        src = "".join(cell.get("source", ""))
        if cell.get("cell_type") != "markdown":
            continue
        stripped = src.strip()
        if stripped.startswith("## "):
            heading = stripped.splitlines()[0][3:].strip()
            if "speech script" not in heading.lower():
                section = heading
            continue
        if family is None and LC_LINK.search(src):
            entry, pasted = parse_leetcode_cell(src)
            entry.update(family="dsa", section=section or NOTEBOOK_GROUPS.get(Path(path).name, ""),
                         notebook=Path(path).name)
            entries.append(entry)
            if pasted:
                pasted_texts.append(pasted)
        elif family is not None and OWN_HEAD.match(stripped.splitlines()[0] if stripped else ""):
            entry = parse_own_cell(src, family)
            nxt = cells[i + 1] if i + 1 < len(cells) else None
            starter = "".join(nxt.get("source", "")) if nxt and nxt.get("cell_type") == "code" else ""
            entry.update(section=section, notebook=Path(path).name,
                         starter_code=starter.strip() or None)
            entries.append(entry)
    return entries, pasted_texts


def parse_folder(folder):
    folder = Path(folder)
    entries, pasted = [], []
    for pattern in DSA_GLOBS:
        for path in sorted(folder.glob(pattern)):
            e, p = parse_notebook(path)
            entries += e
            pasted += p
    for name, family in OWN_NOTEBOOKS.items():
        if (folder / name).exists():
            e, _ = parse_notebook(folder / name, family)
            entries += e
    seen, unique = set(), []
    for entry in entries:
        key = record_id(entry)
        if key not in seen:
            seen.add(key)
            unique.append(entry)
    return unique, pasted


def record_id(entry):
    words = re.sub(r"[^a-z0-9]+", "_", entry["title"].lower()).strip("_")[:48]
    if entry["source"] == "leetcode":
        return f"code_lc_{entry['number']:04d}_{words}"
    return f"code_{entry['prefix'].lower()}_{entry['number']:02d}_{words}"


# --------------------------------------------------------------- generation

SYSTEM = ("You write coaching material for coding interviews: a rubric, graded hints and "
          "follow-up questions for one problem. You never reproduce a problem statement, its "
          "examples, its constraints or an editorial from LeetCode or any other site, and you "
          "never write a full solution. Plain, precise English.")

_TEXTS = {"type": "array", "items": {"type": "string"}}
SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "approaches": _TEXTS, "role": {"type": "string", "enum": ["mle", "aie", "shared"]},
        "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
        "complexity": {"type": "object", "additionalProperties": False,
                       "properties": {"time": {"type": "string"}, "space": {"type": "string"}},
                       "required": ["time", "space"]},
        "key_points": _TEXTS, "edge_cases": _TEXTS, "pitfalls": _TEXTS,
        "code_quality": _TEXTS, "communication": _TEXTS, "followups": _TEXTS,
        "hints": {"type": "object", "additionalProperties": False,
                  "properties": {f"level{i}": {"type": "string"} for i in range(4)},
                  "required": [f"level{i}" for i in range(4)]},
    },
    "required": ["approaches", "role", "difficulty", "complexity", "key_points", "edge_cases", "pitfalls",
                 "code_quality", "communication", "followups", "hints"],
}

FIELDS_TEXT = """Write, in your own words:
- approaches: 1-3 labels of standard solution approaches, best first, each 1-4 words (e.g. "hash map", "two pointers", "BFS", "log-sum-exp trick"); a label, never a sentence.
- role: "shared" for general data-structures-and-algorithms problems (the default); "mle" for classic ML, statistics or data-pipeline work; "aie" for LLM, retrieval, RAG or inference work.
- difficulty: "easy", "medium" or "hard" for a typical interview.
- complexity: time and space of the best standard approach, in big-O.
- key_points: 3-6 rubric criteria that a strong solution and explanation meet.
- edge_cases: 2-6 inputs a candidate should handle or test.
- pitfalls: 2-5 common mistakes.
- code_quality: 2-4 criteria for clean code here.
- communication: 2-4 things a candidate should say out loud while solving it.
- followups: 2-4 questions an interviewer asks after a working solution.
- hints, a ladder from least to most revealing:
  level0: a clarifying question back to the candidate.
  level1: a concept nudge that names no data structure or algorithm.
  level2: the approach or data structure, in words, no code.
  level3: pseudocode for ONE key step only, at most 6 short lines - not the whole solution."""


def build_prompt(entry):
    if entry["source"] == "leetcode":
        notes = "; ".join(f"{k}: {v}" for k, v in entry["notes"].items())
        return (f"Problem: LeetCode {entry['number']} \"{entry['title']}\". Refer to it only by "
                f"number and title; do not restate its statement, examples or constraints.\n"
                f"Practice group in the candidate's notes: {entry['section'] or 'data structures and algorithms'}.\n"
                + (f"The candidate's own notes: {notes}.\n" if notes else "")
                + "\n" + FIELDS_TEXT)
    parts = [f"Exercise: {entry['prefix']} {entry['number']:02d} \"{entry['title']}\" "
             f"({entry['family']}, group: {entry['section']}).",
             f"Difficulty: {entry['difficulty'] or 'unstated'}. Topic: {entry['topic'] or 'unstated'}.",
             "Statement (written by the candidate's study plan):", entry["statement"]]
    if entry.get("starter_code"):
        parts += ["Starter code given with the exercise:", entry["starter_code"]]
    return "\n".join(parts) + "\n\n" + FIELDS_TEXT


def make_record(entry, generated, engine_label):
    lc = entry["source"] == "leetcode"
    hints = [{"level": i, "text": generated["hints"][f"level{i}"].strip()} for i in range(4)]
    display = (f"LeetCode {entry['number']} · {entry['title']}" if lc
               else f"{entry['prefix']} {entry['number']:02d} · {entry['title']}")
    difficulty = (entry.get("difficulty") or generated["difficulty"]).lower()
    code = {"source": entry["source"], "family": entry["family"], "number": entry["number"],
            "title": entry["title"], "approaches": generated["approaches"][:3],
            "role": generated["role"], "difficulty": difficulty,
            "complexity": {**generated["complexity"], "basis": "estimate" if lc else "expected"},
            "edge_cases": generated["edge_cases"], "code_quality": generated["code_quality"],
            "communication": generated["communication"], "hints": hints}
    if lc:
        code.update(slug=entry["slug"], link=entry["link"])
    else:
        code.update(statement=entry["statement"], starter_code=entry.get("starter_code"))
    metadata = {"module": f"{entry['family'].upper()}: {entry['section'] or 'general'}",
                "topic": ", ".join(code["approaches"]), "tags": code["approaches"],
                "difficulty": difficulty,
                "origin": "code", "group": entry["section"], "notebook": entry["notebook"],
                "generated_by": engine_label, "generated": date.today().isoformat(),
                "review": {"status": "unreviewed"}}
    if lc:
        metadata["source_url"] = entry["link"]
    return {"id": record_id(entry),
            "interview": {"question": display, "model_answer": "",
                          "key_points": generated["key_points"],
                          "common_mistakes": generated["pitfalls"],
                          "followups": generated["followups"]},
            "metadata": metadata, "code": code}


# ------------------------------------------------------------------- checks

CODE_LINE = re.compile(r"(?m)```|\bdef \w+\(|^\s*\w+(\[[^\]]*\])?\s*[-+*/]?=\s*\S"
                       r"|^\s*(for|while|if|elif|else)\b[^\n]*:\s*$")


def shingles(text, n=8):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def record_text(record):
    c, i = record["code"], record["interview"]
    parts = i["key_points"] + i["common_mistakes"] + i["followups"] + c["edge_cases"] \
        + c["code_quality"] + c["communication"] + [h["text"] for h in c["hints"]]
    return "\n".join(parts)


def qa_issues(record, pasted_shingles):
    """Problems a reviewer must see: LeetCode text, code in low hint levels."""
    issues = []
    text = record_text(record)
    if re.search(r"(?im)^\s*(example\s*\d*\s*:|input\s*:|output\s*:|constraints\s*:)", text):
        issues.append("looks like a copied example or constraints block")
    overlap = shingles(text) & pasted_shingles
    if overlap:
        issues.append(f"shares {len(overlap)} eight-word run(s) with a pasted LeetCode statement")
    for hint in record["code"]["hints"][:3]:
        if CODE_LINE.search(hint["text"]):
            issues.append(f"hint level {hint['level']} contains code")
    if len(record["code"]["hints"][3]["text"].splitlines()) > 8:
        issues.append("hint level 3 is longer than one step")
    if record["code"]["role"] not in ("mle", "aie", "shared"):
        issues.append("unknown role")
    if not record["code"]["approaches"]:
        issues.append("no approach label")
    if any(len(re.findall(r"[\w'-]+", a)) > 5 or a.rstrip().endswith(".")
           for a in record["code"]["approaches"]):
        issues.append("an approach label is a sentence")
    return issues


# --------------------------------------------------------------------- bank

def load_bank():
    if not OUT_PATH.exists():
        return {}
    return {r["id"]: r for r in (json.loads(l) for l in OUT_PATH.read_text(encoding="utf-8").splitlines() if l.strip())}


def write_bank(records):
    BANK_DIR.mkdir(parents=True, exist_ok=True)
    order = {"dsa": 0, "ml": 1, "pytorch": 2}
    rows = sorted(records.values(), key=lambda r: (order.get(r["code"]["family"], 9),
                                                   r["code"]["source"], r["code"]["number"]))
    OUT_PATH.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                        encoding="utf-8")


def setup_engine():
    """The user's signed-in subscription CLI, with their model and effort."""
    config.load_env_file()
    provider, note = cli_engine.auto_select("127.0.0.1", os.environ.get("LLM_CLI", "auto"))
    if not provider:
        raise SystemExit(f"No signed-in Claude Code or Codex subscription ({note or 'none found'}). "
                         "Sign in to one, or set LLM_CLI.")
    config.MODE, config.CLI_PROVIDER = "cli", provider
    config.CLI_MODEL = os.environ.get("LLM_CLI_MODEL", "").strip()
    config.CLI_EFFORT = os.environ.get("LLM_CLI_EFFORT", "low").strip() or "low"
    config.CLI_TIMEOUT_S = float(os.environ.get("LLM_CLI_TIMEOUT_S", "300"))
    return cli_engine.label()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", required=True, help="folder with the practice notebooks")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list", action="store_true")
    mode.add_argument("--generate", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--only", default="", help="comma-separated record ids to write")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)

    entries, pasted = parse_folder(args.source)
    pasted_shingles = set().union(*(shingles(t) for t in pasted)) if pasted else set()
    bank = load_bank()
    lc = [e for e in entries if e["source"] == "leetcode"]
    own = [e for e in entries if e["source"] == "own"]
    print(f"{len(lc)} LeetCode problems ({len(pasted)} cells with a pasted statement - never "
          f"stored), {len(own)} own exercises; bank holds {len(bank)} record(s)")

    if args.list:
        for e in entries:
            mark = "in bank" if record_id(e) in bank else "missing"
            label = (f"LeetCode {e['number']} {e['title']}" if e["source"] == "leetcode"
                     else f"{e['prefix']} {e['number']:02d} {e['title']}")
            print(f"  [{mark:7s}] {e['family']:7s} {label}  ({e['section']})")
        return 0

    if args.check:
        flagged = 0
        for record in bank.values():
            issues = qa_issues(record, pasted_shingles)
            if issues:
                flagged += 1
                print(f"  {record['id']}: {'; '.join(issues)}")
        print(f"{flagged} of {len(bank)} record(s) flagged")
        return 1 if flagged else 0

    todo = [e for e in entries if record_id(e) not in bank]
    if args.only:
        wanted = {w.strip() for w in args.only.split(",") if w.strip()}
        todo = [e for e in todo if record_id(e) in wanted]
    if args.limit:
        todo = todo[:args.limit]
    engine_label = setup_engine()
    print(f"{len(todo)} record(s) to write with {engine_label}")
    if not todo:
        return 0
    if not args.confirm:
        print("dry run only - add --confirm to write them (uses your plan's allowance)")
        return 0

    failures = []

    def one(entry):
        try:
            generated = cli_engine.complete(SYSTEM, build_prompt(entry), SCHEMA,
                                            effort=config.CLI_EFFORT)
            record = make_record(entry, generated, engine_label)
            issues = qa_issues(record, pasted_shingles)
            if issues:
                record["metadata"]["qa"] = issues
            with _lock:
                bank[record["id"]] = record
                write_bank(bank)
            print(f"  wrote {record['id']}" + (f"  QA: {'; '.join(issues)}" if issues else ""), flush=True)
        except Exception as exc:  # keep going; report at the end
            failures.append((record_id(entry), str(exc)[:200]))
            print(f"  FAILED {record_id(entry)}: {str(exc)[:200]}", flush=True)

    with ThreadPoolExecutor(max(1, args.workers)) as pool:
        list(pool.map(one, todo))
    print(f"done: {len(todo) - len(failures)} written, {len(failures)} failed; bank holds {len(bank)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
