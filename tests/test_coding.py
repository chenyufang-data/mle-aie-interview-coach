"""Offline tests for coding drills (coach/coding.py): the bank view, LeetCode
references without any online lookup, the local runner, the local-only
guard, and the bank-review notes - plus the routes over real HTTP.

Run:  .venv\\Scripts\\python tests\\test_coding.py
"""

import json
import os
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coach import coding, config  # noqa: E402
import ingest.ingest_code as ic  # noqa: E402


def generated(**over):
    out = {"approaches": ["two pointers"], "role": "shared", "difficulty": "easy",
           "complexity": {"time": "O(n)", "space": "O(1)"}, "key_points": ["k"],
           "edge_cases": ["e"], "pitfalls": ["p"], "code_quality": ["q"],
           "communication": ["c"], "followups": ["f"],
           "hints": {f"level{i}": f"hint {i}" for i in range(4)}}
    out.update(over)
    return out


ENTRIES = [
    {"source": "leetcode", "number": 167, "title": "Two Sum II - Input Array Is Sorted",
     "slug": "two-sum-ii-input-array-is-sorted",
     "link": "https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/",
     "notes": {}, "difficulty": "medium", "family": "dsa", "section": "Two pointers",
     "notebook": "week1_homework.ipynb"},
    {"source": "leetcode", "number": 69, "title": "Sqrt(x)", "slug": "sqrtx",
     "link": "https://leetcode.com/problems/sqrtx/", "notes": {}, "difficulty": "",
     "family": "dsa", "section": "Binary search", "notebook": "week2_homework.ipynb"},
    {"source": "own", "family": "ml", "prefix": "ML", "number": 1, "title": "Stable Softmax",
     "difficulty": "Easy", "topic": "overflow", "section": "NumPy",
     "statement": "Convert logits.\n\n**Signature:** `def softmax(x)`",
     "notebook": "week7_homework.ipynb", "starter_code": None},
]


class Bank:
    """A synthetic bank and review file for one test."""

    def __enter__(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        rows = [ic.make_record(e, generated(), "test") for e in ENTRIES]
        retired = ic.make_record({**ENTRIES[1], "number": 70, "title": "Climbing Stairs",
                                  "slug": "climbing-stairs"}, generated(), "test")
        retired["metadata"]["review"] = {"status": "retire"}
        rows.append(retired)
        bank = root / "all_chunks.jsonl"
        bank.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        self.saved = (config.CODE_BANK_PATH, config.CODE_REVIEW_PATH, config.CODE_LOCAL)
        config.CODE_BANK_PATH = bank
        config.CODE_REVIEW_PATH = root / "review" / "rag_code.decisions.json"
        config.CODE_LOCAL = True
        coding._cache.update(mtime=None, records=[])
        return self

    def __exit__(self, *exc):
        config.CODE_BANK_PATH, config.CODE_REVIEW_PATH, config.CODE_LOCAL = self.saved
        coding._cache.update(mtime=None, records=[])
        self.tmp.cleanup()


def test_slugify_follows_leetcode():
    cases = {"Sqrt(x)": "sqrtx", "Pow(x, n)": "powx-n", "3Sum": "3sum",
             "Two Sum II - Input Array Is Sorted": "two-sum-ii-input-array-is-sorted",
             "Implement Trie (Prefix Tree)": "implement-trie-prefix-tree",
             "Insert Delete GetRandom O(1)": "insert-delete-getrandom-o1"}
    for title, slug in cases.items():
        assert coding.slugify(title) == slug, (title, coding.slugify(title))


def test_bank_hides_retired_and_shows_no_statement_for_leetcode():
    with Bank():
        ids = [r["id"] for r in coding.records()]
        assert len(ids) == 3 and not any("climbing" in i for i in ids)
        lc = coding.detail(coding.by_id("code_lc_0167_two_sum_ii_input_array_is_sorted"))
        assert lc["statement"] is None and lc["link"].endswith("/two-sum-ii-input-array-is-sorted/")
        assert [h["level"] for h in lc["hints"]] == [0, 1, 2, 3]
        assert set(lc["rubric"]) == {"key_points", "edge_cases", "common_mistakes",
                                     "code_quality", "communication", "followups"}
        own = coding.detail(coding.by_id("code_ml_01_stable_softmax"))
        assert own["statement"].startswith("Convert logits") and own["link"] is None


def test_resolve_without_looking_anything_up():
    with Bank():
        assert coding.resolve("167")["id"].startswith("code_lc_0167")
        assert coding.resolve("LC #69")["title"] == "Sqrt(x)"
        assert coding.resolve("https://leetcode.com/problems/sqrtx/description/")["id"]
        assert coding.resolve("sqrt(x)")["id"]  # title -> slug -> the bank entry
        other = coding.resolve("https://leetcode.com/problems/word-search/")
        assert other["id"] is None and other["title"] == "Word Search" and other["rubric"] is None
        assert other["link"] == "https://leetcode.com/problems/word-search/"
        typed = coding.resolve("Word Ladder II")
        assert typed["link"] == "https://leetcode.com/problems/word-ladder-ii/" and typed["from_title"]
        for bad in ("", "   ", "!!!"):
            try:
                coding.resolve(bad)
                raise AssertionError(f"accepted {bad!r}")
            except ValueError:
                pass
        try:
            coding.resolve("1234")
            raise AssertionError("a number outside the bank must ask for the link")
        except ValueError as exc:
            assert "link" in str(exc)


def test_run_code():
    out = coding.run_code("import sys\nprint(sum(map(int, sys.stdin.read().split())))", "2 3 4")
    assert out["exit_code"] == 0 and out["stdout"].strip() == "9" and not out["timed_out"]
    err = coding.run_code("def f():\n    return 1 / 0\nf()\n")
    assert err["exit_code"] == 1 and "ZeroDivisionError" in err["stderr"]
    assert "solution.py" in err["stderr"] and tempfile.gettempdir() not in err["stderr"]
    saved = config.CODE_RUN_TIMEOUT_S
    config.CODE_RUN_TIMEOUT_S = 1.5
    try:
        slow = coding.run_code("while True:\n    pass\n")
        assert slow["timed_out"] and slow["exit_code"] is None
    finally:
        config.CODE_RUN_TIMEOUT_S = saved


def test_run_gets_no_api_keys():
    os.environ["ANTHROPIC_API_KEY"] = "sk-test-not-real"
    try:
        out = coding.run_code("import os\nprint(sorted(k for k in os.environ if 'KEY' in k))")
        assert out["stdout"].strip() == "[]", out
    finally:
        del os.environ["ANTHROPIC_API_KEY"]


def test_a_stale_server_knows_it():
    """A code file newer than the server's start means it runs old code -
    the pages then tell the user to restart (the author hit this: a server
    started before phase 6 had no history route and saved nothing)."""
    import time as clock
    saved = config.STARTED_AT
    try:
        config.STARTED_AT = 0.0
        assert coding.code_changed_since_start() is False, "unknown start: no claim"
        config.STARTED_AT = clock.time() + 3600
        assert coding.code_changed_since_start() is False
        config.STARTED_AT = 1.0  # started long before any file was written
        assert coding.code_changed_since_start() is True
    finally:
        config.STARTED_AT = saved


class FakeHandler:
    def __init__(self, client="127.0.0.1", headers=None):
        self.client_address = (client, 50000)
        self.headers = {"Host": "127.0.0.1:8001", "X-Coach-Local": "1", **(headers or {})}


def test_refusal_guards():
    with Bank():
        assert coding.refusal(FakeHandler()) is None
        assert coding.refusal(FakeHandler(headers={"Host": "localhost:8001"})) is None
        assert coding.refusal(FakeHandler(client="::1", headers={"Host": "[::1]:8001"})) is None
        assert coding.refusal(FakeHandler(headers={"Origin": "http://127.0.0.1:8001"})) is None
        assert coding.refusal(FakeHandler(client="10.0.0.5"))
        assert coding.refusal(FakeHandler(headers={"Host": "evil.example:8001"}))  # rebinding
        assert coding.refusal(FakeHandler(headers={"Origin": "https://evil.example"}))
        assert coding.refusal(FakeHandler(headers={"X-Coach-Local": ""}))
        config.CODE_LOCAL = False
        assert "local app" in coding.refusal(FakeHandler())


def test_review_notes():
    with Bank():
        rid = "code_lc_0167_two_sum_ii_input_array_is_sorted"
        assert coding.save_review(rid, "fix", " hint 2 gives it away ")["note"] == "hint 2 gives it away"
        coding.save_review("code_ml_01_stable_softmax", "keep", "")
        saved = json.loads(config.CODE_REVIEW_PATH.read_text(encoding="utf-8"))
        assert saved["bank"] == "rag_code" and [d["id"] for d in saved["decided"]] == \
            ["code_lc_0167_two_sum_ii_input_array_is_sorted", "code_ml_01_stable_softmax"]
        assert coding.detail(coding.by_id(rid))["review"]["status"] == "fix"
        coding.save_review(rid, "", "")
        assert rid not in coding.review_decisions()
        for bad in (("nope", "keep"), (rid, "maybe")):
            try:
                coding.save_review(bad[0], bad[1], "")
                raise AssertionError(bad)
            except ValueError:
                pass


def test_routes_over_http():
    from coach import http
    with Bank():
        server = ThreadingHTTPServer(("127.0.0.1", 0), http.InterviewCoachHandler)
        port = server.server_address[1]
        threading.Thread(target=server.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{port}"

        def call(path, body=None, local=True):
            headers = {"X-Coach-Local": "1"} if local else {}
            data = None
            if body is not None:
                data = json.dumps(body).encode()
                headers["Content-Type"] = "application/json"
            request = urllib.request.Request(base + path, data=data, headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=20) as response:
                    return response.status, json.loads(response.read())
            except urllib.error.HTTPError as exc:
                return exc.code, json.loads(exc.read())

        try:
            status, bank = call("/api/code/bank")
            assert status == 200 and len(bank["problems"]) == 3
            assert call("/api/code/bank", local=False)[0] == 403
            status, one = call("/api/code/problem?id=code_ml_01_stable_softmax")
            assert status == 200 and one["problem"]["statement"]
            assert call("/api/code/problem?id=nope")[0] == 404
            assert call("/api/code/resolve", {"query": "167"})[1]["problem"]["number"] == 167
            assert call("/api/code/resolve", {"query": "9999"})[0] == 400
            status, ran = call("/api/code/run", {"code": "print(6 * 7)"})
            assert status == 200 and ran["stdout"].strip() == "42"
            assert call("/api/code/run", {"code": "print(1)"}, local=False)[0] == 403
            assert call("/api/code/run", {"code": "  "})[0] == 400
            assert call("/api/code/review", {"id": "code_ml_01_stable_softmax", "status": "keep"})[0] == 200
            config.CODE_LOCAL = False
            assert call("/api/code/run", {"code": "print(1)"})[0] == 403
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all coding tests passed")
