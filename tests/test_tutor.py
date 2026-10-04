"""Offline tests for the coding tutor (coach/tutor.py) and the tutor's test
suites (coach/code_tests.py): the hint ladder the server enforces, the
output guard, watch-outs, the coding report, the harness, and the routes in
--mock mode. No model is called - a scripted stand-in plays it.

Run:  .venv\\Scripts\\python tests\\test_tutor.py
"""

import json
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coach import code_tests, coding, config, tutor  # noqa: E402

PROBLEM = {"id": "code_lc_0167_two_sum_ii", "label": "LeetCode 167 · Two Sum II", "source": "leetcode",
           "title": "Two Sum II", "link": "https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/",
           "approaches": ["two pointers"], "complexity": {"time": "O(n)", "space": "O(1)"},
           "hints": [{"level": i, "text": f"bank hint {i}"} for i in range(4)],
           "rubric": {"key_points": ["moves the pointer that fixes the sum"]}}

GOOD = ("class Solution:\n    def pairSum(self, nums, target):\n        lo, hi = 0, len(nums) - 1\n"
        "        while lo < hi:\n            s = nums[lo] + nums[hi]\n"
        "            if s == target:\n                return [lo, hi]\n"
        "            if s < target:\n                lo += 1\n            else:\n                hi -= 1\n"
        "        return []\n")
SUITE = {"entry": "Solution.pairSum", "setup": "", "compare": "exact", "reference": GOOD,
         "cases": [{"name": "two items", "args": "([1, 3], 4)", "why": "two values"},
                   {"name": "negatives", "args": "([-5, -1, 2, 8], 1)", "why": "negative values"},
                   {"name": "large", "args": "(list(range(200000)), 399997)", "why": "a long input"}]}


class Scripted:
    """Plays the model: returns the queued outputs in order, records calls."""

    def __init__(self, *outputs):
        self.outputs, self.calls = list(outputs), []

    def __call__(self, system, prompt, schema, quick=False):
        self.calls.append({"system": system, "prompt": prompt, "schema": schema, "quick": quick})
        return self.outputs.pop(0)


def with_model(script):
    saved = tutor.make_caller
    tutor.make_caller = lambda engine: script
    return saved


def reply(text="Think about the ends.", level=0, help_request=True):
    return {"reply": text, "level_used": level, "help_request": help_request}


# ------------------------------------------------------------------- ladder

def test_ladder_climbs_one_step_and_caps_the_model():
    attempt = tutor.new_attempt(dict(PROBLEM))
    script = Scripted(reply(level=3), reply(level=3), reply(level=2), reply(level=3),
                      reply(level=3), reply(level=4))
    saved = with_model(script)
    try:
        levels = [tutor.tutor_turn(attempt, "", "", "hint", "cli")["level"] for _ in range(6)]
    finally:
        tutor.make_caller = saved
    # the model asked for level 3 at once; the server allowed 0, 1, 2, 3, then held at 3
    assert levels == [0, 1, 2, 3, 3, 3], levels
    assert attempt["hints_by_level"][:4] == [1, 1, 1, 3]
    assert "exactly level 0" in script.calls[0]["prompt"] and script.calls[0]["quick"]


def test_new_blocker_restarts_and_questions_do_not_climb():
    attempt = tutor.new_attempt(dict(PROBLEM))
    script = Scripted(reply(level=0), reply(level=1), reply("What is s?", level=3, help_request=False),
                      reply(level=2))
    saved = with_model(script)
    try:
        tutor.tutor_turn(attempt, "", "", "hint", "cli")
        tutor.tutor_turn(attempt, "", "", "hint", "cli")
        assert tutor.blocker(attempt) == "start" and tutor.ladder_caps(attempt) == (1, 2, 1)
        question = tutor.tutor_turn(attempt, GOOD, "what does line 5 compute?", "message", "cli")
        # a plain question stays within what is unlocked (level 1 here), is not
        # a hint, and leaves the ladder where it was
        assert question["level"] == 1 and sum(attempt["hints_by_level"]) == 2
        assert tutor.ladder_caps(attempt) == (1, 2, 1)
        tutor.record_run(attempt, GOOD, {"cases": [{"name": "large", "ok": False}], "passed": 2,
                                         "total": 3}, kind="check")
        assert tutor.blocker(attempt) == "case:large"
        assert tutor.ladder_caps(attempt) == (-1, 0, 0)
        assert tutor.tutor_turn(attempt, GOOD, "", "hint", "cli")["level"] == 0
    finally:
        tutor.make_caller = saved


def test_blockers_from_runs():
    attempt = tutor.new_attempt(dict(PROBLEM))
    assert tutor.blocker(attempt) == "start"
    tutor.record_run(attempt, "x", {"exit_code": 0, "stderr": ""})
    assert tutor.blocker(attempt) == "start"  # a clean run says nothing about correctness
    tutor.record_run(attempt, "x", {"exit_code": 1, "stderr": "Traceback...\nIndexError: list index"})
    assert tutor.blocker(attempt) == "error:IndexError"
    tutor.record_run(attempt, "y", {"exit_code": None, "timed_out": True})
    assert tutor.blocker(attempt) == "timeout"
    tutor.record_run(attempt, "z", {"cases": [], "passed": 3, "total": 3}, kind="check")
    assert tutor.blocker(attempt) == "passing"
    assert [r["failed"] for r in attempt["runs"]] == [False, True, True, False]


def test_solution_needs_the_button_and_a_confirmation():
    attempt = tutor.new_attempt(dict(PROBLEM))
    script = Scripted(reply("Not yet - the Show solution button is there.", level=4),
                      reply("Here it is:\n```python\n" + GOOD + "```", level=4))
    saved = with_model(script)
    try:
        asked = tutor.tutor_turn(attempt, "", "just tell me the answer", "message", "cli")
        assert asked["offer_solution"] and asked["level"] == 0  # capped, no solution
        assert "do not give it" in script.calls[0]["prompt"]
        assert tutor.tutor_turn(attempt, "", "", "solution", "cli") == {"confirm_solution": True}
        shown = tutor.tutor_turn(attempt, "", "", "solution", "cli", confirmed=True)
        assert shown["level"] == 4 and attempt["solution_shown"] and not script.calls[1]["quick"]
    finally:
        tutor.make_caller = saved


def test_fake_engine_uses_the_bank_hints():
    attempt = tutor.new_attempt(dict(PROBLEM))
    first = tutor.tutor_turn(attempt, "", "", "hint", "fake")
    second = tutor.tutor_turn(attempt, "", "", "hint", "fake")
    assert (first["reply"], second["reply"]) == ("bank hint 0", "bank hint 1")
    assert second["next_level"] == 2


# -------------------------------------------------------------------- guard

def test_guard_counts_code_not_quotes():
    assert tutor.code_lines("Look at `seen` on line 7 - is it reset in the loop?") == 0
    assert tutor.code_lines("Try this:\n```python\nlo, hi = 0, n - 1\nwhile lo < hi:\n    pass\n```") == 3
    assert tutor.code_lines("lo = 0\nhi = len(nums) - 1\nreturn lo") == 3
    assert tutor.code_lines("Return the pair as soon as the sum matches.") == 0
    judge = lambda text: True  # noqa: E731
    assert tutor.guard_issues("lo = 0\nhi = 1\nmid = 2", 1, judge)
    assert not tutor.guard_issues("lo = 0\nhi = 1\nmid = 2", 3, judge)
    assert not tutor.guard_issues("```\n" + GOOD + "```", 4, judge)
    long_l2 = "Use two pointers " * 20
    assert tutor.guard_issues(long_l2, 2, judge) == ["it spells out a complete algorithm in prose"]
    assert tutor.guard_issues(long_l2, 2, lambda text: False) == []
    assert tutor.guard_issues("Short nudge.", 2, judge) == []


def test_guard_regenerates_then_trims():
    attempt = tutor.new_attempt(dict(PROBLEM))
    leaky = reply("Here:\n```python\n" + GOOD + "```", level=0)
    script = Scripted(leaky, reply("ok, smaller", level=0))
    saved = with_model(script)
    try:
        out = tutor.tutor_turn(attempt, "", "", "hint", "cli")
        assert out["reply"] == "ok, smaller" and out["guard"]["regenerated"] and not out["guard"]["trimmed"]
        assert "revealed too much" in script.calls[1]["prompt"]
        script.outputs = [leaky, leaky]
        out = tutor.tutor_turn(attempt, "", "", "hint", "cli")
        assert out["guard"]["trimmed"] and out["reply"] == "bank hint 1"
    finally:
        tutor.make_caller = saved


# --------------------------------------------------------------- watch-outs

def test_watch_outs_merge_by_number_and_keep_wording():
    attempt = tutor.new_attempt(dict(PROBLEM))
    script = Scripted({"earlier": [], "new": [{"text": "lo < hi skips the last element", "line": 4},
                                              {"text": "empty input not handled", "line": None}]},
                      {"earlier": [{"number": 1, "status": "fixed"}, {"number": 9, "status": "fixed"}],
                       "new": []})
    saved = with_model(script)
    try:
        items = tutor.observe(attempt, GOOD, "cli")
        assert [w["status"] for w in items] == ["open", "open"]
        assert tutor.observe(attempt, GOOD, "cli") == items  # same code: no second call
        attempt["observed"]["at"] = 0
        items = tutor.observe(attempt, GOOD + "\n# edited", "cli")
        assert [w["status"] for w in items] == ["fixed", "open"]
        assert items[0]["text"] == "lo < hi skips the last element"
        assert "1. [open] lo < hi skips the last element (line 4)" in script.calls[1]["prompt"]
    finally:
        tutor.make_caller = saved


# ------------------------------------------------------------------- report

def test_report():
    attempt = tutor.new_attempt(dict(PROBLEM))
    tutor.record_run(attempt, "x", {"exit_code": 1, "stderr": "ValueError: x"})
    tutor.record_run(attempt, GOOD, {"cases": [], "passed": 3, "total": 3}, kind="check")
    script = Scripted(reply(level=0),
                      {"approach": "two pointers", "time": "O(n)", "space": "O(1)",
                       "expected_time": "O(n)", "expected_space": "O(1)",
                       "review": ["Clear pointer names."], "earlier": [],
                       "new": [{"text": "no check for an empty list", "line": 3}]})
    saved = with_model(script)
    try:
        tutor.tutor_turn(attempt, GOOD, "", "hint", "cli")
        report = tutor.finish(attempt, GOOD, "cli", "accepted")
    finally:
        tutor.make_caller = saved
    assert report["runs"] == 1 and report["failed_runs"] == 1 and report["checks"] == 1
    assert report["last_check"] == {"passed": 3, "total": 3}
    assert report["hints_by_level"] == [1, 0, 0, 0, 0] and report["leetcode"] == "accepted"
    assert report["expected"]["basis"] == "estimate" and report["complexity"]["time"] == "O(n)"
    md = tutor.report_markdown(report)
    for part in ("# Coding report: LeetCode 167", "Runs: 1 (1 failed)", "accepted",
                 "[open] no check for an empty list", "## Review of your code", "```python"):
        assert part in md, part
    offline = tutor.finish(tutor.new_attempt(dict(PROBLEM, source="own")), GOOD, "fake", "accepted")
    assert offline["complexity"] is None and offline["leetcode"] is None
    assert offline["expected"]["basis"] == "expected"


# ------------------------------------------------------------------ harness

def test_harness_cases():
    assert code_tests.validate(SUITE) is None
    good = code_tests.run_suite(SUITE, GOOD)
    assert (good["passed"], good["total"], good["fatal"]) == (3, 3, None)
    bad = code_tests.run_suite(SUITE, GOOD.replace("return [lo, hi]", "return [lo, hi + 1]"))
    assert bad["passed"] == 0 and bad["cases"][0]["expected"] == "[0, 1]" and bad["cases"][0]["got"] == "[0, 2]"
    slow = ("class Solution:\n    def pairSum(self, nums, target):\n        for i in range(len(nums)):\n"
            "            for j in range(i + 1, len(nums)):\n"
            "                if nums[i] + nums[j] == target: return [i, j]\n        return []\n")
    timed = code_tests.run_suite(SUITE, slow, timeout=4)
    assert timed["timed_out"] and timed["passed"] == 2 and "timed out" in timed["cases"][2]["error"]
    missing = code_tests.run_suite(SUITE, "def other():\n    pass\n")
    assert "define Solution.pairSum" in missing["cases"][0]["error"]
    broken = code_tests.run_suite(SUITE, "class Solution:\n  def pairSum(self, nums, target)\n")
    assert broken["fatal"].startswith("your code failed to load: SyntaxError")
    exits = code_tests.run_suite(SUITE, "import sys\nclass Solution:\n    def pairSum(self, n, t):\n        sys.exit(3)\n")
    assert exits["passed"] == 0 and "SystemExit" in exits["cases"][0]["error"]
    noisy = code_tests.run_suite(SUITE, "print('debug')\n" + GOOD + "print(Solution().pairSum([1, 2], 3))\n")
    assert noisy["passed"] == 3
    inplace = {"entry": "reverse", "setup": "", "compare": "exact",
               "reference": "def reverse(a):\n    a.reverse()\n",
               "cases": [{"name": "three", "args": "([1, 2, 3],)", "why": "three values"}]}
    assert code_tests.run_suite(inplace, "def reverse(a):\n    a[:] = a[::-1]\n")["passed"] == 1
    floats = {"entry": "half", "setup": "", "compare": "float",
              "reference": "def half(xs):\n    return [x / 2 for x in xs]\n",
              "cases": [{"name": "thirds", "args": "([1.0, 2.0],)", "why": "two floats"}]}
    assert code_tests.run_suite(floats, "import numpy as np\ndef half(xs):\n    return np.array(xs) * 0.5\n")["passed"] == 1


def test_entry_point():
    assert code_tests.entry_point("from typing import List\n\nclass Solution:\n    def twoSum(self, numbers: List[int], target: int):\n        pass\n") == "Solution.twoSum"
    assert code_tests.entry_point("import numpy as np\n\ndef softmax(x):\n    pass\n") == "softmax"
    assert code_tests.entry_point("print(1)") is None


def test_build_suite_validates_retries_and_caches():
    saved = code_tests.CACHE_DIR
    with tempfile.TemporaryDirectory() as tmp:
        code_tests.CACHE_DIR = Path(tmp)
        try:
            broken = dict(SUITE, reference=GOOD.replace("return [lo, hi]", "return [1 / 0]"))
            script = Scripted(broken, dict(SUITE))
            suite = code_tests.build_suite(PROBLEM, PROBLEM["id"], GOOD, lambda s, p, sc: script(s, p, sc))
            assert suite["entry"] == "Solution.pairSum" and len(script.calls) == 2
            assert "rejected" in script.calls[1]["prompt"] and "ZeroDivisionError" in script.calls[1]["prompt"]
            assert "do not reuse the examples" in script.calls[0]["prompt"]
            again = code_tests.build_suite(PROBLEM, PROBLEM["id"], GOOD, lambda s, p, sc: 1 / 0)
            assert again["cases"] == SUITE["cases"]  # from the cache, no call
            code_tests.forget_suite(PROBLEM["id"], GOOD)
            assert code_tests.cached_suite(PROBLEM["id"], "Solution.pairSum") is None
            script = Scripted(broken, broken)
            try:
                code_tests.build_suite(PROBLEM, PROBLEM["id"], GOOD, lambda s, p, sc: script(s, p, sc))
                raise AssertionError("an invalid suite was accepted")
            except RuntimeError as exc:
                assert "could not write a working test suite" in str(exc)
        finally:
            code_tests.CACHE_DIR = saved


# ------------------------------------------------------------------- routes

def test_routes_in_mock_mode():
    from coach import http
    saved = (config.CODE_LOCAL, config.MODE)
    config.CODE_LOCAL, config.MODE = True, "mock"
    server = ThreadingHTTPServer(("127.0.0.1", 0), http.InterviewCoachHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def call(path, body):
        request = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json",
                                                  "X-Coach-Local": "1"})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    try:
        status, started = call("/api/code/attempt", {"problem": {"title": "Word Search",
                                                                 "link": "https://leetcode.com/problems/word-search/"}})
        assert status == 200 and started["checks"] is False and "offline" in started["tutor"]
        aid = started["attempt_id"]
        status, hint = call("/api/code/tutor", {"attempt_id": aid, "code": "", "kind": "hint"})
        assert status == 200 and hint["level"] == 0 and hint["reply"] == tutor.GENERIC_HINTS[0]
        status, ran = call("/api/code/run", {"attempt_id": aid, "code": "raise ValueError('x')"})
        assert status == 200 and ran["offer_hint"] and ran["next_level"] == 0
        assert call("/api/code/snapshot", {"attempt_id": aid, "code": "x = 1"})[0] == 200
        assert call("/api/code/check", {"attempt_id": aid, "code": "x = 1"})[0] == 403
        assert call("/api/code/tutor", {"attempt_id": "gone", "code": "", "kind": "hint"})[0] == 404
        assert call("/api/code/tutor", {"attempt_id": aid, "code": "", "kind": "message"})[0] == 400
        status, done = call("/api/code/finish", {"attempt_id": aid, "code": "x = 1", "leetcode": "rejected"})
        assert status == 200 and done["report"]["leetcode"] == "rejected"
        assert done["report"]["failed_runs"] == 1 and "# Coding report: Word Search" in done["markdown"]
        assert call("/api/code/finish", {"attempt_id": aid, "code": "", "leetcode": "maybe"})[0] == 400
        assert call("/api/code/speak", {"text": ""})[0] == 400
    finally:
        server.shutdown()
        server.server_close()
        config.CODE_LOCAL, config.MODE = saved


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all tutor tests passed")
