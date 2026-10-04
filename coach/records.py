"""Coding records (roadmap step 7, phase 6): what a finished attempt leaves
behind, and the history page built from it.

Stored behind coach/store.py (data/records/ by default, Postgres when
DATABASE_URL is set): a *problem* per owner - source, number, title, our
approach labels, role, link, starred, first and last seen - and each
finished *attempt*: the mode (practice or the mock round, with or without
interview conditions), times, hints by level, the outcome, the solution
(the user's own code, the approach their code takes, its complexity, the
tutor's review), the coding report and, for a mock round, the
communication report. The plan's solutions / coding_reports /
communication_reports are fields of the attempt - one of each per attempt.

Consent: the local app keeps records for its one user ("local"); a shared
server (the online demo) keeps them only for an access key with logging on,
under a digest of the key, never for an anonymous visitor.

The review queue - a light spaced repetition, by the latest attempt: not
solved or the solution shown - due at once; solved only with a level-3 hint
(one step of pseudocode) - after two days; with smaller hints - after a
week; without hints - after three weeks. Due problems are listed in that
order, the oldest attempt first within each.
"""

from datetime import datetime, timedelta

from coach import config, store, users

DUE_AFTER_DAYS = {"unsolved": 0, "shown": 0, "heavy": 2, "hinted": 7, "clean": 21}
STATUS_ORDER = ("unsolved", "shown", "heavy", "hinted", "clean")
STATUS_TEXT = {"unsolved": "Not solved yet", "shown": "Solved with the solution shown",
               "heavy": "Solved with a one-step hint", "hinted": "Solved with small hints",
               "clean": "Solved without hints"}


def owner(user):
    """Whose records these are, or None when none may be kept."""
    if config.CODE_LOCAL:
        return "local"
    if user.get("key") and user.get("log", True):
        return users._usage_id(user["key"])
    return None


def outcome(report):
    """solved / shown / unsolved, from the user's own evidence: the tutor's
    tests all passing, or LeetCode accepting it (the user's word)."""
    if report.get("solution_shown"):
        return "shown"
    check = report.get("last_check")
    if report.get("leetcode") == "accepted" or (check and check["total"]
                                                and check["passed"] == check["total"]):
        return "solved"
    return "unsolved"


def hint_max(report):
    used = [level for level, n in enumerate(report.get("hints_by_level") or []) if n]
    return max(used) if used else None


def status(attempt):
    """What the latest attempt says about the problem, for the queue."""
    if attempt["outcome"] != "solved":
        return attempt["outcome"]
    top = attempt.get("hint_max")
    if top is None:
        return "clean"
    return "heavy" if top >= 3 else "hinted"


def save(attempt, report, user):
    """Store a finished attempt and its problem; returns the attempt id, or
    None when this user's records are not kept."""
    who = owner(user)
    if who is None:
        return None
    problem = attempt["problem"]
    key = attempt["problem_key"]
    st = store.current()
    st.upsert_problem(who, key, {
        "problem_id": problem.get("id"), "source": problem.get("source"),
        "number": problem.get("number"), "title": problem.get("title"),
        "label": problem.get("label") or problem.get("title"), "link": problem.get("link"),
        "role": problem.get("role"), "approaches": problem.get("approaches") or [],
        "difficulty": problem.get("difficulty"), "family": problem.get("family")})
    result = outcome(report)
    st.add_attempt({
        "id": attempt["id"], "owner": who, "problem_key": key,
        "mode": attempt.get("mode", "practice"), "strict": bool(attempt.get("strict")),
        "outcome": result, "hint_max": hint_max(report),
        "hints_by_level": report.get("hints_by_level"), "seconds": report.get("seconds"),
        "runs": report.get("runs"), "failed_runs": report.get("failed_runs"),
        "last_check": report.get("last_check"), "leetcode": report.get("leetcode"),
        "started_at": datetime.fromtimestamp(attempt["started"]).astimezone().isoformat(timespec="seconds"),
        "solution": {"code": report.get("code"), "approach": report.get("approach"),
                     "complexity": report.get("complexity"), "review": report.get("review")},
        "coding_report": {k: v for k, v in report.items() if k not in ("code", "communication")},
        "communication_report": report.get("communication"),
    })
    return attempt["id"]


def _when(stamp):
    try:
        moment = datetime.fromisoformat(stamp)
    except (TypeError, ValueError):
        return None
    return moment if moment.tzinfo else moment.astimezone()


def history(user, now=None):
    """Everything the history page shows for this user."""
    who = owner(user)
    if who is None:
        return {"kept": False, "problems": [], "queue": [], "stats": {}}
    st = store.current()
    now = now or datetime.now().astimezone()
    attempts = st.list_attempts(who)
    by_problem = {}
    for a in attempts:
        by_problem.setdefault(a["problem_key"], []).append(a)
    problems = []
    for p in st.list_problems(who):
        mine = by_problem.get(p["key"], [])
        if not mine:
            continue
        latest = mine[0]
        state = status(latest)
        solved = [a for a in mine if a["outcome"] == "solved"]
        last = _when(latest.get("finished_at"))
        due = (last + timedelta(days=DUE_AFTER_DAYS[state])) if last else now
        problems.append({
            **p,
            "attempts": len(mine), "mock_attempts": sum(1 for a in mine if a["mode"] == "mock"),
            "solved": bool(solved),
            "approach": next((a["solution"].get("approach") for a in solved
                              if (a.get("solution") or {}).get("approach")), None),
            "status": state, "status_text": STATUS_TEXT[state],
            "due": due.isoformat(timespec="seconds"), "is_due": due <= now,
            "latest": {k: latest.get(k) for k in ("id", "finished_at", "mode", "strict", "outcome",
                                                 "hint_max", "seconds", "last_check")},
            "history": [{k: a.get(k) for k in ("id", "finished_at", "mode", "strict", "outcome",
                                               "hint_max", "hints_by_level", "seconds", "runs",
                                               "failed_runs", "last_check", "leetcode")}
                        for a in mine],
        })
    queue = sorted([p for p in problems if p["is_due"] and p["status"] in ("unsolved", "shown", "heavy")]
                   + [p for p in problems if p["is_due"] and p["status"] in ("hinted", "clean")],
                   key=lambda p: (STATUS_ORDER.index(p["status"]), p["latest"]["finished_at"] or ""))
    problems.sort(key=lambda p: p["latest"]["finished_at"] or "", reverse=True)
    return {
        "kept": True,
        "stats": {"problems": len(problems), "solved": sum(1 for p in problems if p["solved"]),
                  "attempts": len(attempts),
                  "mock_rounds": sum(1 for a in attempts if a["mode"] == "mock"),
                  "starred": sum(1 for p in problems if p.get("starred"))},
        "queue": [p["key"] for p in queue],
        "problems": problems,
    }


def attempt_detail(user, attempt_id):
    who = owner(user)
    return store.current().get_attempt(who, attempt_id) if who else None


def star(user, key, starred):
    who = owner(user)
    return bool(who) and store.current().set_starred(who, key, starred)


def clear(user):
    who = owner(user)
    return store.current().clear_records(who) if who else 0
