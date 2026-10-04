"""Offline tests for coding records (coach/records.py): consent, outcomes,
the review queue (the light spaced repetition), and the routes - finish
saves, then history, one attempt, star, clear. File store in a temp folder;
the store contract itself runs on both backends in tests/test_store.py.

Run:  .venv\\Scripts\\python tests\\test_records.py
"""

import json
import os
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coach import config, records, store, tutor  # noqa: E402

PROBLEM = {"id": "code_lc_0001_two_sum", "label": "LeetCode 1 · Two Sum", "title": "Two Sum",
           "source": "leetcode", "number": 1, "link": "https://leetcode.com/problems/two-sum/",
           "approaches": ["hash map"], "role": "shared", "difficulty": "easy", "family": "dsa",
           "hints": [], "rubric": {}}


def fresh_store():
    os.environ["RECORDS_DIR"] = tempfile.mkdtemp(prefix="coach_records_")
    return store.use(store.FileStore())


def report(**over):
    out = {"seconds": 300, "runs": 2, "failed_runs": 1, "checks": 1,
           "last_check": {"passed": 9, "total": 9}, "hints_by_level": [0, 0, 0, 0, 0],
           "solution_shown": False, "leetcode": None, "approach": "one-pass hash map",
           "complexity": {"time": "O(n)", "space": "O(n)"}, "review": ["Clear."],
           "watch_outs": [], "expected": None, "code": "def f(): pass"}
    out.update(over)
    return out


def test_owner_follows_consent():
    saved = config.CODE_LOCAL
    try:
        config.CODE_LOCAL = True
        assert records.owner({"key": None, "log": True}) == "local"
        config.CODE_LOCAL = False
        assert records.owner({"key": None, "log": True}) is None, "never an anonymous visitor"
        assert records.owner({"key": "k", "log": False}) is None, "logging off"
        who = records.owner({"key": "secret-key", "log": True})
        assert who and "secret" not in who and len(who) == 16, "a digest, never the key"
    finally:
        config.CODE_LOCAL = saved


def test_outcome_status_and_hints():
    assert records.outcome(report()) == "solved"
    assert records.outcome(report(last_check={"passed": 8, "total": 9})) == "unsolved"
    assert records.outcome(report(last_check=None, leetcode="accepted")) == "solved"
    assert records.outcome(report(solution_shown=True)) == "shown"
    assert records.hint_max(report()) is None
    assert records.hint_max(report(hints_by_level=[1, 0, 1, 2, 0])) == 3
    assert records.status({"outcome": "solved", "hint_max": None}) == "clean"
    assert records.status({"outcome": "solved", "hint_max": 1}) == "hinted"
    assert records.status({"outcome": "solved", "hint_max": 3}) == "heavy"
    assert records.status({"outcome": "unsolved", "hint_max": None}) == "unsolved"


def finished(problem, rep, mode="practice", strict=False):
    attempt = tutor.new_attempt(dict(problem))
    attempt.update(mode=mode, strict=strict)
    return attempt, rep


def test_save_and_the_queue():
    saved_local = config.CODE_LOCAL
    config.CODE_LOCAL = True
    fresh_store()
    try:
        me = {"key": None, "log": True}
        clean = dict(PROBLEM)
        heavy = dict(PROBLEM, id="code_lc_0033_search", label="LeetCode 33 · Search", title="Search",
                     number=33, link="https://leetcode.com/problems/search-in-rotated-sorted-array/")
        failed = dict(PROBLEM, id=None, label="Word Search", title="Word Search", number=None,
                      link="https://leetcode.com/problems/word-search/", approaches=[])
        records.save(*finished(clean, report()), me)
        records.save(*finished(heavy, report(hints_by_level=[1, 1, 1, 1, 0])), me)
        attempt, rep = finished(failed, report(last_check={"passed": 3, "total": 9}, approach=None),
                                mode="mock", strict=True)
        attempt["problem_key"] = failed["link"]
        records.save(attempt, rep, me)
        data = records.history(me)
        assert data["kept"] and data["stats"] == {"problems": 3, "solved": 2, "attempts": 3,
                                                  "mock_rounds": 1, "starred": 0}
        keys = {p["title"]: p for p in data["problems"]}
        assert keys["Two Sum"]["approach"] == "one-pass hash map" and keys["Two Sum"]["status"] == "clean"
        assert keys["Search"]["status"] == "heavy" and keys["Word Search"]["status"] == "unsolved"
        assert keys["Word Search"]["mock_attempts"] == 1 and keys["Word Search"]["approach"] is None
        # due now: the unsolved one; the one-step-hint solve after two days (and
        # then ahead of the rest); the clean solve after three weeks
        assert data["queue"] == [failed["link"]]
        soon = records.history(me, now=datetime.now().astimezone() + timedelta(days=3))
        assert soon["queue"] == [failed["link"], heavy["id"]]
        later = records.history(me, now=datetime.now().astimezone() + timedelta(days=22))
        assert later["queue"] == [failed["link"], heavy["id"], clean["id"]]
        # one attempt in full, star, clear
        latest = keys["Two Sum"]["latest"]["id"]
        detail = records.attempt_detail(me, latest)
        assert detail["solution"]["code"] == "def f(): pass" and "code" not in detail["coding_report"]
        assert records.star(me, clean["id"], True) and not records.star(me, "nope", True)
        assert records.history(me)["stats"]["starred"] == 1
        assert records.clear(me) == 3 and records.history(me)["problems"] == []
    finally:
        config.CODE_LOCAL = saved_local


def test_nothing_is_kept_for_an_anonymous_visitor_on_a_shared_server():
    saved_local = config.CODE_LOCAL
    config.CODE_LOCAL = False
    fresh_store()
    try:
        visitor = {"key": None, "log": True}
        assert records.save(*finished(PROBLEM, report()), visitor) is None
        assert records.history(visitor) == {"kept": False, "problems": [], "queue": [], "stats": {}}
        assert store.current().list_attempts("local") == []
    finally:
        config.CODE_LOCAL = saved_local


def test_routes():
    from coach import http
    saved = (config.CODE_LOCAL, config.MODE)
    config.CODE_LOCAL, config.MODE = True, "mock"
    fresh_store()
    server = ThreadingHTTPServer(("127.0.0.1", 0), http.InterviewCoachHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def call(path, body=None):
        request = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json", "X-Coach-Local": "1"})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    try:
        assert call("/api/code/history")[1]["problems"] == []
        status, started = call("/api/code/attempt", {"problem": {"title": "Word Search",
                                                                 "link": "https://leetcode.com/problems/word-search/"}})
        aid = started["attempt_id"]
        call("/api/code/run", {"attempt_id": aid, "code": "print(1)"})
        status, done = call("/api/code/finish", {"attempt_id": aid, "code": "print(1)", "leetcode": "accepted"})
        assert status == 200 and done["saved"] is True
        status, history = call("/api/code/history")
        problem = history["problems"][0]
        assert problem["title"] == "Word Search" and problem["solved"] and problem["latest"]["id"] == aid
        assert call(f"/api/code/record?id={aid}")[1]["attempt"]["solution"]["code"] == "print(1)"
        assert call("/api/code/record?id=nope")[0] == 404
        key = problem["key"]
        assert call("/api/code/star", {"key": key, "starred": True})[1]["starred"] is True
        assert call("/api/code/star", {"key": "nope", "starred": True})[0] == 404
        assert call("/api/code/history/clear", {})[0] == 400, "needs the confirmation"
        assert call("/api/code/history/clear", {"confirm": True})[1]["deleted"] == 1
        assert call("/api/code/history")[1]["problems"] == []
    finally:
        server.shutdown()
        server.server_close()
        config.CODE_LOCAL, config.MODE = saved


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all records tests passed")
