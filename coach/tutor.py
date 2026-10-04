"""The coding tutor (roadmap step 7, phase 4): hints instead of answers.

One attempt = one problem worked in the coding page. The server keeps the
attempt (in memory; records are phase 6) and decides what a reply may
reveal - the prompt only asks:

- The hint ladder. Level 0 a clarifying question back; 1 a concept nudge;
  2 the approach in words; 3 pseudocode for one step; 4 the full solution.
  Levels climb per blocker - what the last run says is wrong: "start"
  before any run, the first failing test case, the exception type, a time
  limit, or "passing". Each help request may go at most one level above
  the highest already given on that blocker; a new blocker starts again at
  0. Level 4 needs the Show solution button and a confirmation - typing
  "just tell me" only makes the page offer that button.
- The output guard. Every reply below level 4 is checked for code beyond
  the level's line limit (a regex) and, when it is long and step-like, for
  a complete algorithm in prose (a short judge call). A failing reply is
  regenerated once with the reason quoted, then replaced by the bank's
  hint for that level (or a generic one).
- Watch-outs. On a run or check the tutor reviews the code silently and
  logs concrete risks (a loop bound that can overrun, a mutated input, an
  empty input not handled) for the report, with whether each was fixed.
- The coding report when the user is done: time to done, runs and failed
  runs, hints by level, watch-outs, the final complexity against the
  expected one (the tutor's estimate for LeetCode problems, labelled so),
  whether LeetCode accepted it (the user's word), and a review of their
  own code - never an editorial.

LeetCode problems are named by number and title only; the statement stays
on leetcode.com (docs/plan.md step 7).
"""

import re
import threading
import time
import uuid

from coach import code_tests

LEVEL_TEXT = {
    0: "a clarifying question back to the candidate - no hint about the method "
       "(at most 35 words)",
    1: "a concept nudge that names no data structure or algorithm (at most 45 words)",
    2: "the approach or key data structure: WHAT to track, in one or two sentences - not "
       "how to compute each step, not the formula, no code; leave the procedure to the "
       "candidate (at most 40 words)",
    3: "pseudocode for ONE key step only, at most 6 short lines, plus one sentence - not "
       "the whole solution",
    4: "the full solution: two or three sentences of explanation, then complete Python code",
}
LEVEL_NAMES = {0: "a question back", 1: "a nudge", 2: "the approach", 3: "one step",
               4: "the full solution"}
# lines of code a reply may carry at each level (quoting one line of the
# candidate's own code is fine; a level-3 step is at most six)
CODE_LIMIT = {0: 2, 1: 2, 2: 2, 3: 6}
GENERIC_HINTS = {
    0: "Before changing anything: what should your function return for the smallest "
       "valid input, and does your code do that?",
    1: "Think about what you are recomputing on every pass, and whether you could "
       "remember it instead.",
    2: "Write down the one piece of state you need to carry from one step to the next, "
       "then build the loop around keeping it correct.",
    3: "Take one input from your failing case and trace your code on it by hand, line by "
       "line, writing each variable's value; the first line where it differs from what "
       "you expect is the step to fix.",
}
JUST_TELL_ME = re.compile(
    r"\b(just|please)?\s*(tell|give|show)\s+me\s+(the\s+)?(answer|solution|code)\b"
    r"|\bi\s+give\s+up\b|\bwhat'?s\s+the\s+(answer|solution)\b", re.I)
CODE_LINE = re.compile(r"^\s*(?:\w+(?:\[[^\]]*\])?\s*[-+*/]?=\s*\S|def\s+\w+\(|class\s+\w+"
                       r"|(?:for|while|if|elif|else|try|except|with)\b[^\n]*:\s*$"
                       r"|return\b|import\s+\w+|from\s+\w+\s+import\b|\w+\([^)]*\)\s*$)")
STEP_WORDS = re.compile(r"\b(first|then|next|finally|after that|loop|iterate|for each|"
                        r"return|update|increment|decrement|while)\b", re.I)
ATTEMPT_TTL_S = 12 * 3600
MAX_MESSAGES = 60

_attempts = {}
_lock = threading.Lock()


class TutorError(Exception):
    """Shown to the user as is (HTTP 400/503)."""


# ----------------------------------------------------------------- attempts

def new_attempt(problem, user="anonymous"):
    attempt = {
        "id": uuid.uuid4().hex, "user": user, "problem": problem,
        "problem_key": problem.get("id") or problem.get("link") or problem.get("title"),
        "started": time.time(), "touched": time.time(),
        "messages": [], "ladder": {}, "hints_by_level": [0, 0, 0, 0, 0],
        "snapshots": [], "runs": [], "watch_outs": {}, "observed": {"code": None, "at": 0.0},
        "solution_shown": False, "guard_events": [], "report": None,
    }
    with _lock:
        now = time.time()
        for key in [k for k, a in _attempts.items() if now - a["touched"] > ATTEMPT_TTL_S]:
            del _attempts[key]
        _attempts[attempt["id"]] = attempt
    return attempt


def get_attempt(attempt_id):
    with _lock:
        attempt = _attempts.get(attempt_id or "")
        if attempt:
            attempt["touched"] = time.time()
        return attempt


def snapshot(attempt, code, reason):
    """Keep the code at a meaningful moment (a message, a run, a pause of
    about 20 s); identical consecutive snapshots are not repeated."""
    code = code or ""
    last = attempt["snapshots"][-1] if attempt["snapshots"] else None
    if last and last["code"] == code:
        return
    attempt["snapshots"].append({"t": round(time.time() - attempt["started"], 1),
                                 "reason": reason, "code": code})
    del attempt["snapshots"][:-200]


def error_type(stderr):
    for line in reversed((stderr or "").strip().splitlines()):
        match = re.match(r"^(\w+(?:Error|Exception|Interrupt|Exit))\b", line.strip())
        if match:
            return match.group(1)
    return "Error"


def record_run(attempt, code, result, kind="run"):
    """A Run (the file executed) or a Check (the tutor's tests) - the
    blocker and the report come from these."""
    snapshot(attempt, code, kind)
    entry = {"t": round(time.time() - attempt["started"], 1), "kind": kind,
             "timed_out": bool(result.get("timed_out"))}
    if kind == "check":
        failing = [c for c in result.get("cases", []) if not c.get("ok")]
        entry.update(passed=result.get("passed", 0), total=result.get("total", 0),
                     fatal=result.get("fatal"),
                     first_failure=(failing[0]["name"] if failing else None))
        entry["failed"] = bool(result.get("fatal")) or entry["passed"] < entry["total"]
        attempt["last_check_detail"] = failing[0] if failing else None
    else:
        entry.update(exit_code=result.get("exit_code"),
                     error=(error_type(result.get("stderr")) if result.get("exit_code") else None))
        entry["failed"] = entry["timed_out"] or bool(result.get("exit_code"))
        attempt["last_run_stderr"] = (result.get("stderr") or "")[-600:]
    attempt["runs"].append(entry)
    return entry


def blocker(attempt):
    """What the user is stuck on, from the latest informative run."""
    for run in reversed(attempt["runs"]):
        if run["kind"] == "check":
            if run.get("fatal"):
                return "load-error"
            if run["passed"] == run["total"]:
                return "passing"
            return f"case:{run['first_failure']}"
        if run["timed_out"]:
            return "timeout"
        if run.get("exit_code"):
            return f"error:{run['error']}"
    return "start"


def ladder_caps(attempt):
    """(level reached on this blocker, the most the next help request may
    use, the most a plain question may use)."""
    reached = attempt["ladder"].get(blocker(attempt), -1)
    return reached, min(reached + 1, 3), max(reached, 0)


# -------------------------------------------------------------------- guard

def code_lines(text):
    """Lines of code in a reply: everything inside ``` fences plus lines
    outside them that read as code. Inline `quotes` do not count."""
    count, fenced = 0, False
    for line in (text or "").splitlines():
        if line.strip().startswith("```"):
            fenced = not fenced
            continue
        if fenced:
            count += bool(line.strip())
        elif CODE_LINE.match(line):
            count += 1
    return count


def prose_suspect(text, level):
    """Worth a judge call: a long level-2 reply (the approach can slide into
    the whole procedure), or a long, step-like reply at levels 0-1."""
    words = len(re.findall(r"\w+", text or ""))
    if level == 2:
        return words > 55
    numbered = len(re.findall(r"(?m)^\s*(?:\d+[.)]|[-*])\s+", text or ""))
    return words > 60 and (numbered >= 3 or len(STEP_WORDS.findall(text or "")) >= 5)


def guard_issues(text, level, judge):
    """Why a reply reveals more than its level allows ([] when it is fine)."""
    if level >= 4:
        return []
    issues = []
    lines = code_lines(text)
    if lines > CODE_LIMIT[level]:
        issues.append(f"{lines} lines of code where level {level} allows {CODE_LIMIT[level]}")
    if level <= 2 and prose_suspect(text, level) and judge(text):
        issues.append("it spells out a complete algorithm in prose")
    return issues


def fallback_hint(problem, level):
    hints = {h["level"]: h["text"] for h in (problem.get("hints") or [])}
    return hints.get(level) or GENERIC_HINTS[min(level, 3)]


# ----------------------------------------------------------------- prompts

SYSTEM = """You are a coding-interview tutor beside a candidate who is solving a problem in
their own editor. Your job is to help them think, never to solve it for them.

The server sets the highest level your reply may use:
{levels}

Rules:
- Below level 4: no code beyond the level's limit (level 0-2: none, except quoting
  one line of the candidate's own code; level 3: at most six lines of pseudocode for
  one step). Never spell out the whole algorithm in prose.
- Ground the reply in the candidate's current code and last run: point at their own
  lines, e.g. "`seen` is reset inside the loop on line 7".
- Keep to the level's word budget, plain sentences, no headings, one idea per reply.
  Do not open with remarks about the editor or the code being empty.
- Never restate a LeetCode problem statement, its examples or its constraints.

Return JSON: "reply" (what the candidate reads), "level_used" (0-4, the level your
reply actually uses), "help_request" (true when the candidate asked for help getting
unstuck; false for a question you can answer without revealing more, or small talk)."""

REPLY_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"reply": {"type": "string"},
                   "level_used": {"type": "integer", "minimum": 0, "maximum": 4},
                   "help_request": {"type": "boolean"}},
    "required": ["reply", "level_used", "help_request"],
}

JUDGE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"complete_algorithm": {"type": "boolean"}, "why": {"type": "string"}},
    "required": ["complete_algorithm", "why"],
}
JUDGE_SYSTEM = ("You check a tutor's hint for a coding interview. Answer whether the hint "
                "gives away a complete algorithm - enough that the candidate could write the "
                "whole solution without further thinking. Naming an approach or one step is "
                "not complete.")

# Earlier watch-outs come back by their number in the prompt (a model does
# not reliably reuse free-form keys); new ones as text.
WATCH_PROPERTIES = {
    "earlier": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "properties": {"number": {"type": "integer"},
                       "status": {"type": "string", "enum": ["open", "fixed"]}},
        "required": ["number", "status"]}},
    "new": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "properties": {"text": {"type": "string"}, "line": {"type": ["integer", "null"]}},
        "required": ["text", "line"]}},
}
WATCH_SCHEMA = {"type": "object", "additionalProperties": False,
                "properties": WATCH_PROPERTIES, "required": ["earlier", "new"]}
WATCH_RULES = """Watch-outs are concrete risks in the code as written, for the problem's valid
inputs: a loop bound that can overrun, an input mutated that the caller may still
need, an empty or single-element input not handled, integer or float trouble, an
off-by-one, a wrong return type or shape, or work slower than the expected
complexity. Not style, and not inputs the problem rules out. One short sentence each,
with the line number when there is one; never the fix as code.
"earlier": every numbered earlier item, with "fixed" when the current code no longer
has it, else "open". "new": risks not already listed, at most three."""
WATCH_SYSTEM = ("You review a candidate's code silently while they solve a coding problem. "
                "You do not talk to them; you log watch-outs for their report.\n\n"
                + WATCH_RULES)

REVIEW_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "approach": {"type": "string"},
        "time": {"type": "string"}, "space": {"type": "string"},
        "expected_time": {"type": "string"}, "expected_space": {"type": "string"},
        "review": {"type": "array", "items": {"type": "string"}},
        **WATCH_PROPERTIES,
    },
    "required": ["approach", "time", "space", "expected_time", "expected_space", "review",
                 "earlier", "new"],
}
REVIEW_SYSTEM = """The candidate has finished a coding problem. Review their final code.

Return: "approach" (a label of a few words for the approach their code takes); "time"
and "space" (big-O of THEIR code); "expected_time" and "expected_space" (big-O of the
best standard approach); "review" (two to four short points on their code -
correctness, edge cases, clarity, complexity - specific to their lines; no full
rewrite); and the watch-outs in the final code.

""" + WATCH_RULES


def problem_text(problem, private=True):
    """How the problem is named to the model. LeetCode: number and title
    only - the model knows the problem; the statement is never in the app.
    Own exercises: the author's statement. `private` adds the tutor's notes
    (approaches, complexity, rubric, bank hints), never shown as such."""
    if problem.get("source") == "own":
        lines = [f"Problem: {problem.get('label')} (the candidate's own exercise).",
                 "Statement:", problem.get("statement") or ""]
    else:
        lines = [f"Problem: {problem.get('label') or problem.get('title')} - refer to it by "
                 "name; the candidate reads the statement on LeetCode."]
    if private and problem.get("approaches"):
        notes = [f"approaches: {', '.join(problem['approaches'])}"]
        if problem.get("complexity"):
            notes.append(f"expected complexity: time {problem['complexity']['time']}, "
                         f"space {problem['complexity']['space']}")
        rubric = problem.get("rubric") or {}
        if rubric.get("key_points"):
            notes.append("key points: " + "; ".join(rubric["key_points"]))
        hints = problem.get("hints") or []
        if hints:
            notes.append("bank hints by level: " + " | ".join(
                f"L{h['level']}: {h['text']}" for h in hints))
        lines.append("Tutor's private notes (reveal no more than the level allows): "
                     + ". ".join(notes))
    return "\n".join(lines)


def last_run_text(attempt):
    if not attempt["runs"]:
        return "No run yet."
    run = attempt["runs"][-1]
    if run["kind"] == "check":
        text = f"Last check: {run['passed']} of {run['total']} of the tutor's test cases passed"
        if run.get("fatal"):
            text += f"; the code did not load: {run['fatal']}"
        elif run.get("first_failure"):
            detail = attempt.get("last_check_detail") or {}
            text += (f"; first failure {run['first_failure']!r}: input {detail.get('args')}, "
                     f"expected {detail.get('expected')}, got {detail.get('got')}"
                     + (f", error {detail.get('error')}" if detail.get("error") else ""))
        return text + "."
    if run["timed_out"]:
        return "Last run: stopped at the time limit."
    if run.get("exit_code"):
        return f"Last run: exited with code {run['exit_code']} ({run['error']}). " + (
            attempt.get("last_run_stderr") or "")[-600:]
    return "Last run: finished without an error."


def conversation_text(attempt, limit=10):
    lines = []
    for m in attempt["messages"][-limit:]:
        who = "Candidate" if m["role"] == "user" else f"Tutor (level {m.get('level')})"
        lines.append(f"{who}: {m['text']}")
    return "\n".join(lines) or "(none yet)"


def reply_prompt(attempt, code, message, kind, caps):
    reached, next_cap, now_cap = caps
    given = [m for m in attempt["messages"] if m["role"] == "tutor"
             and m.get("blocker") == blocker(attempt) and m.get("level") is not None]
    if kind == "solution":
        request = ("The candidate confirmed they want the full solution. Answer at level 4: "
                   "two or three sentences of explanation, then the complete solution in Python, "
                   "keeping their function name and signature if they have one.")
    elif kind == "hint":
        request = (f"The candidate pressed Hint. Answer at exactly level {next_cap} "
                   f"({LEVEL_TEXT[next_cap]}).")
    else:
        request = (f"The candidate wrote: «{message}»\n"
                   f"If they are asking for help getting unstuck, you may use up to level "
                   f"{next_cap}; if it is a question you can answer without revealing more, stay "
                   f"at level {now_cap} or below.")
        if JUST_TELL_ME.search(message or ""):
            request += (" They are asking for the answer: do not give it. Answer at your level "
                        "and say the Show solution button is there if they want the full "
                        "solution.")
    return "\n\n".join([
        problem_text(attempt["problem"]),
        "Candidate's code (line numbers added):\n" + code_tests.numbered(code),
        last_run_text(attempt),
        f"Current blocker: {blocker(attempt)}. Hints already given on it: "
        + ("; ".join(f"L{m['level']}: {m['text']}" for m in given) or "none"),
        "Conversation so far (latest last):\n" + conversation_text(attempt),
        request,
    ])


def system_prompt():
    return SYSTEM.format(levels="\n".join(f"  level {k}: {v}" for k, v in LEVEL_TEXT.items()))


# ------------------------------------------------------------------ engines

def fake_reply(attempt, kind, caps, message):
    """--mock mode and the offline tests: the bank's hint for the level."""
    _reached, next_cap, now_cap = caps
    if kind == "solution":
        return {"reply": "Offline mode has no model to write a solution. Run the server with "
                         "your subscription or an API key.", "level_used": 4,
                "help_request": True}
    level = next_cap if kind == "hint" else now_cap
    return {"reply": fallback_hint(attempt["problem"], level), "level_used": level,
            "help_request": kind == "hint"}


def make_caller(engine):
    """call(system, prompt, schema, quick) on the chosen engine."""
    from coach.llm import call_model

    def call(system, prompt, schema, quick=False):
        return call_model(prompt, schema, engine, thinking=not quick, system=system,
                          quick=quick)
    return call


# -------------------------------------------------------------------- turns

def tutor_turn(attempt, code, message, kind, engine, confirmed=False):
    """One tutor reply. kind: "hint" (the Hint button), "message" (typed or
    spoken), "solution" (the Show solution button, after confirmation)."""
    if kind not in ("hint", "message", "solution"):
        raise TutorError("Unknown request.")
    if kind == "message" and not (message or "").strip():
        raise TutorError("Type a message first.")
    if kind == "solution" and not confirmed:
        return {"confirm_solution": True}
    snapshot(attempt, code, kind)
    caps = ladder_caps(attempt)
    reached, next_cap, now_cap = caps
    cap = 4 if kind == "solution" else next_cap
    started = time.perf_counter()
    guard = {"regenerated": False, "trimmed": False, "issues": []}
    if engine == "fake":
        out = fake_reply(attempt, kind, caps, message)
    else:
        call = make_caller(engine)

        verdicts = []

        def judge(text):
            verdict = call(JUDGE_SYSTEM, "Hint:\n" + text, JUDGE_SCHEMA, quick=True)
            verdicts.append(verdict.get("why", ""))
            return bool(verdict.get("complete_algorithm"))

        prompt = reply_prompt(attempt, code, message, kind, caps)
        out = call(system_prompt(), prompt, REPLY_SCHEMA, quick=kind != "solution")
        level = min(max(int(out.get("level_used", 0)), 0), cap)
        issues = guard_issues(out.get("reply", ""), level, judge)
        if issues:
            guard.update(regenerated=True, issues=issues)
            why = f" - {verdicts[-1]}" if verdicts else ""
            retry = (prompt + f"\n\nYour previous reply revealed too much ({'; '.join(issues)}{why}). "
                     f"Rewrite it at level {level}, shorter, within its limits: one idea, "
                     "and leave the working-out to the candidate.")
            out = call(system_prompt(), retry, REPLY_SCHEMA, quick=True)
            level = min(max(int(out.get("level_used", 0)), 0), cap)
            issues = guard_issues(out.get("reply", ""), level, judge)
            if issues:
                # replaced by the bank's hint for the rung this request was on
                guard.update(trimmed=True, issues=guard["issues"] + issues)
                rung = next_cap if kind == "hint" else min(level, 3)
                out = {"reply": fallback_hint(attempt["problem"], rung),
                       "level_used": rung, "help_request": out.get("help_request", True)}
        attempt["guard_events"].append({"t": round(time.time() - attempt["started"], 1),
                                        **guard})
    level = min(max(int(out.get("level_used", 0)), 0), cap)
    if kind == "hint":
        level = min(level, next_cap)
    block = blocker(attempt)
    helped = kind in ("hint", "solution") or bool(out.get("help_request"))
    if level <= 3 and level > reached:
        attempt["ladder"][block] = level
    if kind == "solution":
        attempt["solution_shown"] = True
    if helped:
        attempt["hints_by_level"][level] += 1
    seconds = round(time.perf_counter() - started, 2)
    now = round(time.time() - attempt["started"], 1)
    if kind != "hint":
        attempt["messages"].append({"role": "user", "text": message if kind == "message"
                                    else "(asked for the full solution)", "t": now})
    attempt["messages"].append({"role": "tutor", "text": out["reply"], "level": level,
                                "blocker": block, "t": now, "kind": kind})
    del attempt["messages"][:-MAX_MESSAGES]
    reached_now = attempt["ladder"].get(block, -1)
    return {"reply": out["reply"], "level": level, "level_name": LEVEL_NAMES[level],
            "blocker": block, "next_level": min(reached_now + 1, 3),
            "offer_solution": kind == "message" and bool(JUST_TELL_ME.search(message or "")),
            "guard": guard, "seconds": seconds}


def observe(attempt, code, engine, force=False):
    """The silent review after a run or check: watch-outs for the report.
    Skipped when the code has not changed since the last review, or the last
    one was under 25 s ago (unless forced)."""
    now = time.time()
    seen = attempt["observed"]
    if engine == "fake" or not (code or "").strip():
        return list(attempt["watch_outs"].values())
    if not force and (seen["code"] == code or now - seen["at"] < 25):
        return list(attempt["watch_outs"].values())
    seen.update(code=code, at=now)
    prompt = "\n\n".join([
        problem_text(attempt["problem"]),
        "Code (line numbers added):\n" + code_tests.numbered(code),
        last_run_text(attempt),
        earlier_text(attempt),
    ])
    out = make_caller(engine)(WATCH_SYSTEM, prompt, WATCH_SCHEMA, quick=True)
    merge_watch_outs(attempt, out)
    return list(attempt["watch_outs"].values())


def earlier_text(attempt):
    items = list(attempt["watch_outs"].values())
    if not items:
        return "Earlier watch-outs: none."
    return "Earlier watch-outs (by number):\n" + "\n".join(
        f"{i}. [{w['status']}] {w['text']}" + (f" (line {w['line']})" if w.get("line") else "")
        for i, w in enumerate(items, 1))


def merge_watch_outs(attempt, out):
    """Statuses for the numbered earlier items; new items join as open. The
    first wording of an item names the risk and is kept."""
    t = round(time.time() - attempt["started"], 1)
    items = list(attempt["watch_outs"].values())
    for update in out.get("earlier") or []:
        number = update.get("number")
        if isinstance(number, int) and 1 <= number <= len(items):
            items[number - 1].update(status=update.get("status", items[number - 1]["status"]),
                                     last_seen=t)
    for item in (out.get("new") or [])[:3]:
        text = str(item.get("text") or "").strip()
        key = re.sub(r"[^a-z0-9]+", "-", text[:40].lower()).strip("-")
        if not key or key in attempt["watch_outs"]:
            continue
        attempt["watch_outs"][key] = {"key": key, "text": text, "line": item.get("line"),
                                      "status": "open", "first_seen": t, "last_seen": t}


# ------------------------------------------------------------------- report

def finish(attempt, code, engine, leetcode=None):
    """The coding report. `leetcode`: "accepted" | "rejected" | "not-submitted"
    | None - the user's word, since LeetCode is never contacted."""
    snapshot(attempt, code, "done")
    problem = attempt["problem"]
    review = None
    if engine != "fake" and (code or "").strip():
        prompt = "\n\n".join([
            problem_text(problem),
            "Final code (line numbers added):\n" + code_tests.numbered(code),
            last_run_text(attempt),
            earlier_text(attempt),
        ])
        review = make_caller(engine)(REVIEW_SYSTEM, prompt, REVIEW_SCHEMA)
        merge_watch_outs(attempt, review)
    runs = [r for r in attempt["runs"] if r["kind"] == "run"]
    checks = [r for r in attempt["runs"] if r["kind"] == "check"]
    bank = problem.get("complexity")
    if bank:
        expected = {"time": bank["time"], "space": bank["space"],
                    "basis": "estimate" if problem.get("source") == "leetcode" else "expected"}
    elif review:
        expected = {"time": review["expected_time"], "space": review["expected_space"],
                    "basis": "estimate"}
    else:
        expected = None
    report = {
        "problem": {"label": problem.get("label") or problem.get("title"),
                    "link": problem.get("link"), "source": problem.get("source"),
                    "id": problem.get("id")},
        "seconds": round(time.time() - attempt["started"]),
        "runs": len(runs), "failed_runs": sum(r["failed"] for r in runs),
        "checks": len(checks),
        "last_check": ({"passed": checks[-1]["passed"], "total": checks[-1]["total"]}
                       if checks else None),
        "hints_by_level": list(attempt["hints_by_level"]),
        "solution_shown": attempt["solution_shown"],
        "watch_outs": sorted(list(attempt["watch_outs"].values()), key=lambda w: w["first_seen"]),
        "complexity": ({"time": review["time"], "space": review["space"]} if review else None),
        "expected": expected,
        "approach": review["approach"] if review else None,
        "review": review["review"] if review else [],
        "leetcode": leetcode if problem.get("source") == "leetcode" else None,
        "guard": {"regenerated": sum(e["regenerated"] for e in attempt["guard_events"]),
                  "trimmed": sum(e["trimmed"] for e in attempt["guard_events"])},
        "code": code,
    }
    attempt["report"] = report
    return report


def report_markdown(report):
    minutes, seconds = divmod(report["seconds"], 60)
    p = report["problem"]
    lines = [f"# Coding report: {p['label']}", ""]
    if p.get("link"):
        lines += [f"Problem: {p['link']}", ""]
    lines += [f"- Time to done: {minutes} min {seconds:02d} s",
              f"- Runs: {report['runs']} ({report['failed_runs']} failed)"]
    if report["last_check"]:
        lines.append(f"- Tutor's tests: {report['checks']} check(s); last "
                     f"{report['last_check']['passed']} of {report['last_check']['total']} passed")
    used = [f"level {i} ({LEVEL_NAMES[i]}) x{n}" for i, n in enumerate(report["hints_by_level"]) if n]
    lines.append("- Hints: " + (", ".join(used) if used else "none"))
    if report["leetcode"]:
        lines.append(f"- LeetCode (your word): {report['leetcode'].replace('-', ' ')}")
    if report["complexity"]:
        exp = report["expected"]
        lines.append(f"- Your complexity: time {report['complexity']['time']}, space "
                     f"{report['complexity']['space']}"
                     + (f" - expected time {exp['time']}, space {exp['space']} ({exp['basis']})"
                        if exp else ""))
    if report["approach"]:
        lines.append(f"- Approach: {report['approach']}")
    if report["watch_outs"]:
        lines += ["", "## Watch-outs", ""]
        lines += [f"- [{w['status']}] {w['text']}"
                  + (f" (line {w['line']})" if w.get("line") else "") for w in report["watch_outs"]]
    if report["review"]:
        lines += ["", "## Review of your code", ""] + [f"- {r}" for r in report["review"]]
    lines += ["", "## Your code", "", "```python", report["code"].rstrip(), "```", ""]
    return "\n".join(lines)
