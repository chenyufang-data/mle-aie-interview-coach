"""Offline tests for the coding bank builder (ingest/ingest_code.py) and
the committed bank (banks/rag_code).

Run:  .venv\\Scripts\\python tests\\test_ingest_code.py

Synthetic notebooks only - no notes folder, no subscription, no network.
The bank checks hold the LeetCode rules of docs/plan.md step 7: a LeetCode
record carries number, title, link and our own coaching text, never a
statement.
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ingest.ingest_code as ic  # noqa: E402

PASTED = ("Given a list of distinct integers and a target, find two positions whose values "
          "add up to the target and report them in increasing order of position.")


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.splitlines(keepends=True)}


def notebook(cells):
    return json.dumps({"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5})


def write_folder(root):
    root = Path(root)
    (root / "week1_homework.ipynb").write_text(notebook([
        md("Leetcode 167\nTwo Sum II - Input Array Is Sorted\n\n" + PASTED + "\n\n"
           "https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/description/"),
        code("class Solution:\n    pass"),
        md("## 🎤 Interview Speech Script — LC 167\nSay the invariant first."),
        md("Leetcode 641\nhttps://leetcode.com/problems/design-circular-deque/"),
    ]), encoding="utf-8")
    (root / "week2_homework.ipynb").write_text(notebook([
        md("## Rotated arrays"),
        md("Leetcode 33 | Search in Rotated Sorted Array\n\n**Difficulty:** Medium\n"
           "**Pattern:** Binary search with a sorted-half test\n\nOne-line paraphrase.\n\n"
           "**Target:** O(log n).\n\nhttps://leetcode.com/problems/search-in-rotated-sorted-array/"),
        # the same problem again later in the plan: one record
        md("Leetcode 33 | Search in Rotated Sorted Array\n\n"
           "https://leetcode.com/problems/search-in-rotated-sorted-array/"),
    ]), encoding="utf-8")
    (root / "pytorch_practice.ipynb").write_text(notebook([
        md("## Setup"),
        md("PT 04 | Fix the Broken Training Script\n\n**Difficulty:** Hard\n"
           "**Topic:** Debugging\n\nThe next cell trains badly. Fix it. Do not open the collapsed "
           "list below the cell until you have written down eight of your own.\n\n"
           "<details><summary>Answer key</summary>\n1. secret bug list\n</details>"),
        code("import torch\nmodel = None  # broken"),
    ]), encoding="utf-8")
    (root / "week7_homework.ipynb").write_text(notebook([
        md("## NumPy"),
        md("ML 01 | Numerically Stable Softmax\n\n**Difficulty:** Easy\n**Topic:** Overflow\n\n"
           "Convert logits to probabilities.\n\n**Signature:** `def softmax(x)`"),
        code(""),
    ]), encoding="utf-8")
    return root


def generated(**over):
    out = {"approaches": ["binary search"], "role": "shared", "difficulty": "medium",
           "complexity": {"time": "O(log n)", "space": "O(1)"},
           "key_points": ["States which half is sorted"], "edge_cases": ["no rotation"],
           "pitfalls": ["off-by-one at mid"], "code_quality": ["clear bounds"],
           "communication": ["say the invariant"], "followups": ["what about duplicates?"],
           "hints": {"level0": "What do you know about each half?",
                     "level1": "One half is always in order.",
                     "level2": "Binary search: test which half is sorted, then whether the target is in it.",
                     "level3": "if a[lo] <= a[mid]:\n    hi = mid - 1 if a[lo] <= t < a[mid] else ..."}}
    out.update(over)
    return out


def parsed():
    with tempfile.TemporaryDirectory() as tmp:
        return ic.parse_folder(write_folder(tmp))


def test_parse_keeps_only_number_title_link():
    entries, pasted = parsed()
    lc = {e["number"]: e for e in entries if e["source"] == "leetcode"}
    assert sorted(lc) == [33, 167, 641]
    assert lc[167]["title"] == "Two Sum II - Input Array Is Sorted"
    assert lc[641]["title"] == "Design Circular Deque"  # from the link's slug
    assert lc[33]["link"] == "https://leetcode.com/problems/search-in-rotated-sorted-array/"
    assert lc[33]["notes"] == {"pattern": "Binary search with a sorted-half test", "target": "O(log n)."}
    assert lc[33]["difficulty"] == "medium" and lc[33]["section"] == "Rotated arrays"
    # the speech-script heading is not a group; week 1 gets its own label
    assert lc[167]["section"] == ic.NOTEBOOK_GROUPS["week1_homework.ipynb"]
    # pasted LeetCode text is returned separately, for the overlap check only
    assert pasted == [PASTED + "\n\nhttps://leetcode.com/problems/two-sum-ii-input-array-is-sorted/description/"]
    assert "distinct integers" not in json.dumps(entries)


def test_prompt_never_carries_pasted_text():
    entries, _ = parsed()
    for entry in entries:
        prompt = ic.build_prompt(entry)
        assert "distinct integers" not in prompt
        if entry["source"] == "leetcode":
            assert f"LeetCode {entry['number']}" in prompt and "do not restate" in prompt


def test_own_exercise_drops_answer_key():
    entries, _ = parsed()
    pt = next(e for e in entries if e["source"] == "own" and e["prefix"] == "PT")
    assert pt["title"] == "Fix the Broken Training Script" and pt["difficulty"] == "Hard"
    assert "secret bug list" not in pt["statement"] and "collapsed list" not in pt["statement"]
    assert pt["starter_code"].startswith("import torch")
    ml = next(e for e in entries if e["source"] == "own" and e["prefix"] == "ML")
    assert ml["starter_code"] is None and "Convert logits" in ml["statement"]
    assert ic.record_id(ml) == "code_ml_01_numerically_stable_softmax"


def test_record_shape():
    entries, pasted = parsed()
    lc = next(e for e in entries if e.get("number") == 33)
    record = ic.make_record(lc, generated(difficulty="hard"), "test engine")
    assert record["id"] == "code_lc_0033_search_in_rotated_sorted_array"
    assert record["code"]["difficulty"] == "medium"  # the author's note wins
    assert record["code"]["link"] == lc["link"] and "statement" not in record["code"]
    assert [h["level"] for h in record["code"]["hints"]] == [0, 1, 2, 3]
    assert record["metadata"]["review"] == {"status": "unreviewed"}
    shingles = set().union(*(ic.shingles(t) for t in pasted))
    assert ic.qa_issues(record, shingles) == []
    pt = next(e for e in entries if e.get("prefix") == "PT")
    own = ic.make_record(pt, generated(), "test engine")
    assert own["code"]["statement"] == pt["statement"] and own["code"]["starter_code"]
    assert own["code"]["complexity"]["basis"] == "expected"


def test_qa_flags_leaks_and_code():
    entries, pasted = parsed()
    shingles = set().union(*(ic.shingles(t) for t in pasted))
    lc = next(e for e in entries if e.get("number") == 167)
    leak = ic.make_record(lc, generated(key_points=["Recall: " + PASTED]), "t")
    assert any("eight-word" in i for i in ic.qa_issues(leak, shingles))
    example = ic.make_record(lc, generated(edge_cases=["Example 1:", "Input: nums = [2,7]"]), "t")
    assert any("example" in i for i in ic.qa_issues(example, shingles))
    hints = generated()["hints"] | {"level1": "left = 0\nright = len(a) - 1"}
    coded = ic.make_record(lc, generated(hints=hints), "t")
    assert "hint level 1 contains code" in ic.qa_issues(coded, shingles)
    prose = generated()["hints"] | {"level2": "Return the pair as soon as the sum matches."}
    assert ic.qa_issues(ic.make_record(lc, generated(hints=prose), "t"), shingles) == []
    wordy = ic.make_record(lc, generated(approaches=["Run it, read the loss curve, then fix"]), "t")
    assert "an approach label is a sentence" in ic.qa_issues(wordy, shingles)


def test_generate_is_idempotent():
    calls = []
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "notes").mkdir()
        folder = write_folder(Path(tmp) / "notes")
        saved = ic.OUT_PATH, ic.BANK_DIR, ic.setup_engine, ic.cli_engine.complete
        ic.BANK_DIR = Path(tmp) / "bank"
        ic.OUT_PATH = ic.BANK_DIR / "all_chunks.jsonl"
        ic.setup_engine = lambda: "fake engine"
        ic.cli_engine.complete = lambda system, prompt, schema, effort=None: calls.append(prompt) or generated()
        try:
            assert ic.main(["--source", str(folder), "--generate", "--confirm", "--workers", "2"]) == 0
            assert len(calls) == 5
            assert all("distinct integers" not in p for p in calls)
            rows = [json.loads(l) for l in ic.OUT_PATH.read_text(encoding="utf-8").splitlines()]
            assert [r["code"]["family"] for r in rows] == ["dsa", "dsa", "dsa", "ml", "pytorch"]
            assert ic.main(["--source", str(folder), "--generate", "--confirm"]) == 0
            assert len(calls) == 5  # nothing re-written
            assert ic.main(["--source", str(folder), "--check"]) == 0
        finally:
            ic.OUT_PATH, ic.BANK_DIR, ic.setup_engine, ic.cli_engine.complete = saved


LC_CODE_KEYS = {"source", "family", "number", "title", "slug", "link", "approaches", "role",
                "difficulty", "complexity", "edge_cases", "code_quality", "communication", "hints"}


def test_committed_bank():
    """The shipped bank: LeetCode records hold no statement, every record
    passes the checks that need no notes folder."""
    path = ic.OUT_PATH
    if not path.exists():
        print("  (no banks/rag_code yet - skipped)")
        return
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids))
    for r in rows:
        c = r["code"]
        assert r["id"].startswith("code_") and c["role"] in ("mle", "aie", "shared")
        assert c["difficulty"] in ("easy", "medium", "hard")
        assert [h["level"] for h in c["hints"]] == [0, 1, 2, 3]
        if c["source"] == "leetcode":
            assert set(c) == LC_CODE_KEYS, r["id"]
            assert c["link"] == f"https://leetcode.com/problems/{c['slug']}/"
        else:
            assert c["statement"]
        issues = ic.qa_issues(r, set())
        assert not issues or r["metadata"].get("review", {}).get("status") in ("keep", "fix"), \
            (r["id"], issues)
    from coach import config
    assert all("rag_code" not in str(p) for p in config.CORPUS_PATHS.values())


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all ingest_code tests passed")
