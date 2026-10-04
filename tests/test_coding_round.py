"""Offline tests for the mock coding round (coach/coding_round.py): phases in
code, the interview-conditions lock, the interviewer's help on the shared
ladder, check-ins, follow-ups, the communication report, and the routes in
--mock mode. A scripted stand-in plays the model.

Run:  .venv\\Scripts\\python tests\\test_coding_round.py
"""

import json
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coach import coding_round, config, tutor  # noqa: E402

LEETCODE = {"id": "code_lc_0001_two_sum", "label": "LeetCode 1 · Two Sum", "title": "Two Sum",
            "source": "leetcode", "link": "https://leetcode.com/problems/two-sum/",
            "approaches": ["hash map"], "complexity": {"time": "O(n)", "space": "O(n)"},
            "hints": [{"level": i, "text": f"bank hint {i}"} for i in range(4)],
            "rubric": {"followups": ["What if the input were sorted?", "What if there are many queries?",
                                     "A third one that is never asked"]}}
OWN = {"id": "code_ml_01_softmax", "label": "ML 01 · Softmax", "title": "Softmax", "source": "own",
       "statement": "Convert logits.", "hints": [], "rubric": {"followups": []}}
STARTER = "class Solution:\n    def twoSum(self, nums, target):\n        pass\n"
CODE = ("class Solution:\n    def twoSum(self, nums, target):\n        seen = {}\n"
        "        for i, x in enumerate(nums):\n            if target - x in seen:\n"
        "                return [seen[target - x], i]\n            seen[x] = i\n")


class Scripted:
    def __init__(self, *outputs):
        self.outputs, self.calls = list(outputs), []

    def __call__(self, system, prompt, schema, quick=False):
        self.calls.append({"system": system, "prompt": prompt, "schema": schema})
        return self.outputs.pop(0)


def with_model(script):
    saved = tutor.make_caller
    tutor.make_caller = lambda engine: script
    return saved


def say(text, level=0, help_request=False):
    return {"reply": text, "level_used": level, "help_request": help_request}


def test_opening_and_followups():
    attempt, opening = coding_round.start(dict(LEETCODE), "me", strict=True)
    assert attempt["mode"] == "mock" and attempt["phase"] == "discuss" and attempt["strict"]
    assert "LeetCode 1 · Two Sum" in opening and "Open it on LeetCode" in opening
    assert attempt["followups"] == LEETCODE["rubric"]["followups"][:2]
    own, opening = coding_round.start(dict(OWN), "me", strict=False)
    assert "statement is on the left" in opening and own["followups"] == [None, None]


def test_phase_advances_at_real_code_only():
    attempt, _ = coding_round.start(dict(LEETCODE), "me", strict=True)
    assert coding_round.real_code(STARTER) == 0
    coding_round.advance(attempt, STARTER + "        # think first\n")
    assert attempt["phase"] == "discuss" and coding_round.locked(attempt)
    coding_round.advance(attempt, CODE)
    assert attempt["phase"] == "coding" and attempt["coding_started"] is not None
    assert coding_round.locked(attempt)
    relaxed, _ = coding_round.start(dict(LEETCODE), "me", strict=False)
    assert not coding_round.locked(relaxed)


def test_discuss_answers_and_hints_on_the_ladder():
    attempt, _ = coding_round.start(dict(LEETCODE), "me", strict=True)
    script = Scripted(say("Yes, exactly one answer exists. What is your approach?", level=2),
                      say("Think about what you have already seen.", level=3, help_request=True),
                      say("What could you look up quickly?", level=3, help_request=True))
    saved = with_model(script)
    try:
        answer = coding_round.interviewer_turn(attempt, STARTER, "Is there always one answer?",
                                               "message", "cli", ms=4200)
        assert answer["phase"] == "discuss" and not answer["help"]
        assert attempt["hints_by_level"] == [0, 0, 0, 0, 0], "a plain answer is not a hint"
        assert "Phase: discuss" in script.calls[0]["prompt"] and "interviewer" in script.calls[0]["system"]
        first = coding_round.interviewer_turn(attempt, STARTER, "", "hint", "cli")
        second = coding_round.interviewer_turn(attempt, STARTER, "", "hint", "cli")
        assert (first["level"], second["level"]) == (0, 1), "the model asked for 3; the ladder allows 0 then 1"
        assert attempt["hints_by_level"][:2] == [1, 1]
        asked = [m for m in attempt["messages"] if m["role"] == "user"]
        assert asked[0]["ms"] == 4200 and asked[1]["text"] == "(asked for a hint)"
    finally:
        tutor.make_caller = saved


def test_check_ins_respect_conditions():
    strict, _ = coding_round.start(dict(LEETCODE), "me", strict=True)
    assert coding_round.interviewer_turn(strict, CODE, "", "checkin", "fake")["skipped"]
    relaxed, _ = coding_round.start(dict(LEETCODE), "me", strict=False)
    assert coding_round.interviewer_turn(relaxed, STARTER, "", "checkin", "fake")["skipped"]  # discuss
    out = coding_round.interviewer_turn(relaxed, CODE, "", "checkin", "fake")
    assert out["reply"] == coding_round.CHECK_IN and relaxed["checked_in"] == 1


def test_done_review_followups_and_close():
    attempt, _ = coding_round.start(dict(LEETCODE), "me", strict=True)
    done = coding_round.interviewer_turn(attempt, CODE, "", "done", "fake")
    assert done["reply"] == coding_round.WALKTHROUGH and done["unlocked"] and attempt["phase"] == "review"
    assert not coding_round.locked(attempt)
    try:
        coding_round.interviewer_turn(attempt, CODE, "", "done", "fake")
        raise AssertionError("done twice")
    except tutor.TutorError:
        pass
    script = Scripted(say("Okay. What if the input were sorted - what changes?"),
                      say("Got it. And with many queries?"))
    saved = with_model(script)
    try:
        q1 = coding_round.interviewer_turn(attempt, CODE, "On [2,7] with 9 it returns [0,1].", "message", "cli")
        q2 = coding_round.interviewer_turn(attempt, CODE, "Two pointers then.", "message", "cli")
        closing = coding_round.interviewer_turn(attempt, CODE, "Cache the map.", "message", "cli")
    finally:
        tutor.make_caller = saved
    assert "Next question: What if the input were sorted?" in script.calls[0]["prompt"]
    assert "Next question: What if there are many queries?" in script.calls[1]["prompt"]
    assert q1["phase"] == q2["phase"] == "review"
    assert closing["finished"] and closing["reply"] == coding_round.CLOSING and attempt["phase"] == "closed"
    try:
        coding_round.interviewer_turn(attempt, CODE, "more?", "message", "cli")
        raise AssertionError("talked after the close")
    except tutor.TutorError:
        pass


def test_followups_for_a_problem_outside_the_bank():
    attempt, _ = coding_round.start(dict(OWN), "me", strict=False)
    coding_round.interviewer_turn(attempt, CODE, "", "done", "fake")
    script = Scripted(say("Fine. How would this scale to a batch?"))
    saved = with_model(script)
    try:
        coding_round.interviewer_turn(attempt, CODE, "walked through", "message", "cli")
    finally:
        tutor.make_caller = saved
    assert "ask ONE follow-up of your own" in script.calls[0]["prompt"]


def test_communication_report():
    attempt, _ = coding_round.start(dict(LEETCODE), "me", strict=True)
    script = Scripted(say("Go on."))
    saved = with_model(script)
    try:
        coding_round.interviewer_turn(attempt, STARTER, "So I need two indices that add up, um, to the target?",
                                      "message", "cli", ms=6000, voice=True)
    finally:
        tutor.make_caller = saved
    coding_round.advance(attempt, CODE)
    coding_round.interviewer_turn(attempt, CODE, "", "done", "fake")
    script = Scripted({"verdicts": [
        {"criterion": "clarifying_questions", "verdict": "yes", "evidence": "[00:00] asked about two indices"},
        {"criterion": "clarifying_questions", "verdict": "no", "evidence": "duplicate"},
        {"criterion": "approach_before_code", "verdict": "no", "evidence": "coded first"}],
        "suggestions": ["State the approach first.", "Say the complexity.", "Walk a test.", "a fourth"]})
    saved = with_model(script)
    try:
        report = coding_round.communication(attempt, "cli")
    finally:
        tutor.make_caller = saved
    m = report["metrics"]
    assert m["answers"] == 1 and m["spoken_turns"] == 1 and m["filler_total"] == 1
    assert m["avg_answer_seconds"] == 6.0 and m["time_to_first_code_s"] is not None
    assert [v["criterion"] for v in report["verdicts"]] == ["clarifying_questions", "approach_before_code"]
    assert report["verdicts"][0]["label"] == coding_round.CRITERIA["clarifying_questions"]
    assert len(report["suggestions"]) == 3
    timeline = script.calls[0]["prompt"]
    assert "Candidate (spoken):" in timeline and "[coding started]" in timeline and "[said they were done coding]" in timeline
    md = coding_round.communication_markdown(report)
    assert "## Communication" in md and "**yes**" in md and "Suggestions:" in md
    offline = coding_round.communication(attempt, "fake")
    assert offline["verdicts"] == [] and offline["metrics"]["answers"] == 1


def test_routes_for_a_strict_round():
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
        status, started = call("/api/code/attempt", {"mode": "mock", "strict": True,
                                                     "problem": {"title": "Word Search",
                                                                 "link": "https://leetcode.com/problems/word-search/"}})
        assert status == 200 and started["mode"] == "mock" and started["phase"] == "discuss"
        assert "Word Search" in started["opening"]
        aid = started["attempt_id"]
        status, refused = call("/api/code/run", {"attempt_id": aid, "code": "print(1)"})
        assert status == 403 and "locked" in refused["error"]
        assert call("/api/code/check", {"attempt_id": aid, "code": "print(1)"})[0] == 403
        assert call("/api/code/tutor", {"attempt_id": aid, "code": "", "kind": "hint"})[0] == 400
        status, snap = call("/api/code/snapshot", {"attempt_id": aid, "code": "a = 1\nb = 2\n"})
        assert status == 200 and snap["phase"] == "coding"
        status, said = call("/api/code/interviewer", {"attempt_id": aid, "code": "a = 1\nb = 2\n",
                                                      "message": "I'll use DFS from each cell.",
                                                      "kind": "message", "ms": 5000})
        assert status == 200 and said["phase"] == "coding"
        status, done = call("/api/code/interviewer", {"attempt_id": aid, "code": "a = 1\nb = 2\n",
                                                      "kind": "done"})
        assert status == 200 and done["unlocked"] and done["phase"] == "review"
        status, ran = call("/api/code/run", {"attempt_id": aid, "code": "print(1)"})
        assert status == 200 and ran["stdout"].strip() == "1" and ran["phase"] == "review"
        for answer in ("It returns True on a 2x2 grid.", "O(m*n*4^L).", "Prune early."):
            status, reply = call("/api/code/interviewer", {"attempt_id": aid, "code": "x", "message": answer,
                                                           "kind": "message"})
            assert status == 200
        assert reply["finished"]
        status, done = call("/api/code/finish", {"attempt_id": aid, "code": "x", "leetcode": None})
        assert status == 200 and done["report"]["mode"] == "mock" and done["report"]["strict"]
        assert done["report"]["communication"]["metrics"]["answers"] == 4
        assert "## Communication" in done["markdown"]
    finally:
        server.shutdown()
        server.server_close()
        config.CODE_LOCAL, config.MODE = saved


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all coding round tests passed")
