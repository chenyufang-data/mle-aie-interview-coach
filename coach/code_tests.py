"""Tests the tutor writes for a coding problem, run on this machine (roadmap
step 7, phase 4).

LeetCode's own test cases are off limits (docs/plan.md step 7), so the
tutor writes a small suite per problem: a hidden reference solution, a few
inputs (the smallest valid ones, edge cases, typical ones, and one large
input a too-slow solution cannot finish), and how to compare outputs.
Expected outputs are never the model's arithmetic: the harness computes
them by running the reference. A suite is kept only when the reference
passes it; it is cached per problem and entry point under data/code_tests/
(local, never committed) and rewritten when the user's code names another
entry point. The reference stays on the server - the page sees each
input expression, the expected output and the user's output.

Model-written code (the reference, the helpers, the input expressions) runs
here exactly like the user's own: a subprocess of this Python in a
temporary folder, with a time limit and none of the server's API keys in
its environment (coach/coding.py).
"""

import hashlib
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from coach import config

CACHE_DIR = config.BASE_DIR / "data" / "code_tests"
CASE_TEXT = 300

SUITE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "entry": {"type": "string"},
        "setup": {"type": "string"},
        "reference": {"type": "string"},
        "compare": {"type": "string", "enum": ["exact", "unordered", "float", "custom"]},
        "cases": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "properties": {"name": {"type": "string"}, "args": {"type": "string"},
                           "why": {"type": "string"}},
            "required": ["name", "args", "why"]}},
    },
    "required": ["entry", "setup", "reference", "compare", "cases"],
}

SUITE_SYSTEM = ("You write unit tests for coding-interview practice. You never copy a "
                "problem's published examples or test cases; you invent your own inputs. "
                "Your reference solution must be correct and efficient. Python 3.12.")

SUITE_TEXT = """Write a test suite for the candidate's solution.

- entry: the callable the candidate's code must define, exactly as {entry_note}. Use
  "Solution.method" for a LeetCode-style class, or a plain function name.
- setup: Python helpers shared by the inputs and both solutions - e.g. a ListNode or
  TreeNode class and builders such as build_list(values) - or "" when none are needed.
  It may define to_plain(x), which turns an output into plain Python data for
  comparison (a linked list into a list), and same(expected, got) for compare "custom".
  typing names (List, Optional, ...) and numpy as np are already available.
- reference: a correct, efficient solution defining the same entry. It is never shown.
- compare: "exact"; "unordered" when the answer may list items in any order; "float"
  for numeric outputs (numpy allclose); "custom" with same() in setup.
- cases: 6 to 10 cases. Each "args" is ONE Python expression that evaluates to the
  TUPLE of positional arguments, e.g. "([3, 1, 2], 4)" or "(build_list([1, 2]),)" -
  keep it short: build large inputs with code, e.g. "(list(range(100000)), 7)".
  Cover the smallest valid inputs, edge cases (duplicates, negatives, empty where the
  problem allows it), typical inputs, and - when the expected complexity is better
  than quadratic - one large input sized so an efficient Python solution finishes in
  well under a second and a quadratic one does not. Inputs must be valid for the
  problem. "name" and "why" describe the INPUT in a few words ("one element, target
  present", "rotated at the last index") - never the method, the bug a case catches,
  or how to handle it: the candidate reads them while still solving.
  When the function changes its input in place and returns None, the harness compares
  the arguments after the call instead of the return value.

Problem: {problem}
{details}
The candidate's code so far (for its entry point and signature only):
{code}
"""

HARNESS = r'''import contextlib, io, json, math, sys, time, traceback
from typing import *
import numpy as np
import setup as _setup

SETUP = {k: getattr(_setup, k) for k in dir(_setup) if not k.startswith("__")}
BASE = dict(SETUP)
BASE.update({k: v for k, v in globals().items() if k[:1].isupper() and k not in ("SETUP", "BASE")})
BASE["np"] = np
ENTRY, COMPARE, CASES = __ENTRY__, __COMPARE__, __CASES__
CAP = __CAP__


def short(value):
    text = repr(value)
    return text if len(text) <= CAP else text[:CAP] + " ..."


def to_plain(x, depth=0):
    if "to_plain" in SETUP and depth == 0:
        x = SETUP["to_plain"](x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, (list, tuple)):
        return [to_plain(v, depth + 1) for v in x]
    if isinstance(x, (set, frozenset)):
        return sorted((to_plain(v, depth + 1) for v in x), key=repr)
    if isinstance(x, dict):
        return {k: to_plain(v, depth + 1) for k, v in x.items()}
    if hasattr(x, "val") and hasattr(x, "next"):
        out, seen = [], set()
        while x is not None and id(x) not in seen and len(out) < 100000:
            seen.add(id(x)); out.append(to_plain(x.val, depth + 1)); x = x.next
        return out
    return x


def same(expected, got):
    if COMPARE == "custom" and "same" in SETUP:
        return bool(SETUP["same"](expected, got))
    if COMPARE == "unordered" and isinstance(expected, list) and isinstance(got, list):
        return sorted(expected, key=repr) == sorted(got, key=repr)
    if COMPARE == "float":
        try:
            e, g = np.asarray(expected, dtype=float), np.asarray(got, dtype=float)
            return e.shape == g.shape and bool(np.allclose(e, g, rtol=1e-6, atol=1e-8, equal_nan=True))
        except (TypeError, ValueError):
            return expected == got
    return expected == got


def load(path, name):
    ns = dict(BASE)
    ns["__name__"] = name
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(open(path, encoding="utf-8").read(), path, "exec"), ns)
    return ns


def entry(ns):
    head, _, method = ENTRY.partition(".")
    if head not in ns:
        raise NameError(f"define {ENTRY}" if method else f"define a function named {ENTRY}")
    return getattr(ns[head](), method) if method else ns[head]


def call(ns, args):
    with contextlib.redirect_stdout(io.StringIO()):
        result = entry(ns)(*args)
    return result if result is not None else (args[0] if len(args) == 1 else list(args))


def where(exc_tb):
    frames = [f for f in traceback.extract_tb(exc_tb) if f.filename.endswith("solution.py")]
    return f" (line {frames[-1].lineno})" if frames else ""


def main(reference_path, candidate_path, out_path):
    out = open(out_path, "w", encoding="utf-8")
    try:
        ref = load(reference_path, "reference")
    except Exception as exc:
        out.write(json.dumps({"fatal": f"reference failed to load: {exc!r}"}) + "\n"); return
    try:
        cand = load(candidate_path, "solution")
    except BaseException as exc:
        out.write(json.dumps({"fatal": f"your code failed to load: {type(exc).__name__}: {exc}{where(exc.__traceback__)}"}) + "\n"); return
    for name, expr in CASES:
        row = {"name": name, "args": expr if len(expr) <= CAP else expr[:CAP] + " ..."}
        try:
            expected = to_plain(call(ref, eval(expr, dict(BASE))))
        except Exception as exc:
            row.update(ok=False, ref_error=f"{type(exc).__name__}: {exc}")
            out.write(json.dumps(row) + "\n"); out.flush(); continue
        row["expected"] = short(expected)
        try:
            args = eval(expr, dict(BASE))
            started = time.perf_counter()
            got = to_plain(call(cand, args))
            row.update(ok=same(expected, got), got=short(got),
                       seconds=round(time.perf_counter() - started, 4))
        except BaseException as exc:
            row.update(ok=False, error=f"{type(exc).__name__}: {exc}{where(exc.__traceback__)}")
        out.write(json.dumps(row) + "\n"); out.flush()


main(*sys.argv[1:4])
'''


def entry_point(code):
    """The callable the user's code defines: Solution.method, else the
    first top-level function, else None."""
    match = re.search(r"^class\s+Solution\b[^\n]*:\s*\n(?:[^\n]*\n)*?\s+def\s+(\w+)\s*\(\s*self",
                      code or "", re.M)
    if match:
        return f"Solution.{match.group(1)}"
    match = re.search(r"^def\s+(\w+)\s*\(", code or "", re.M)
    return match.group(1) if match else None


def numbered(code, limit=400):
    lines = (code or "").splitlines()[:limit]
    return "\n".join(f"{i:4d} | {line}" for i, line in enumerate(lines, 1)) or "(empty)"


def _cache_path(problem_key, entry):
    digest = hashlib.sha1(f"{problem_key}|{entry or ''}".encode("utf-8")).hexdigest()[:16]
    return CACHE_DIR / f"{digest}.json"


def cached_suite(problem_key, entry):
    try:
        return json.loads(_cache_path(problem_key, entry).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write_files(tmp, suite, candidate_code):
    tmp = Path(tmp)
    (tmp / "setup.py").write_text(
        "from typing import *\nimport numpy as np\n\n" + (suite.get("setup") or ""), encoding="utf-8")
    (tmp / "reference.py").write_text(suite["reference"], encoding="utf-8")
    (tmp / "solution.py").write_text(candidate_code, encoding="utf-8")
    cases = [(c["name"], c["args"]) for c in suite["cases"]]
    harness = (HARNESS.replace("__ENTRY__", json.dumps(suite["entry"]))
               .replace("__COMPARE__", json.dumps(suite["compare"]))
               .replace("__CASES__", json.dumps(cases))
               .replace("__CAP__", str(CASE_TEXT)))
    (tmp / "harness.py").write_text(harness, encoding="utf-8")


def run_suite(suite, candidate_code, timeout=None):
    """Run the candidate against the suite: one row per case, in order.
    A case that never reported (time limit hit) is marked as timed out and
    the rest as not run."""
    from coach.coding import child_env
    timeout = timeout or max(15.0, config.CODE_RUN_TIMEOUT_S * 1.5)
    with tempfile.TemporaryDirectory(prefix="coach-check-") as tmp:
        _write_files(tmp, suite, candidate_code)
        results = Path(tmp) / "results.jsonl"
        started = time.perf_counter()
        timed_out = False
        try:
            proc = subprocess.run([sys.executable, "harness.py", "reference.py", "solution.py",
                                   str(results)], cwd=tmp, env=child_env(), capture_output=True,
                                  text=True, encoding="utf-8", errors="replace", timeout=timeout)
            stderr = proc.stderr
        except subprocess.TimeoutExpired:
            timed_out, stderr = True, ""
        rows = []
        if results.exists():
            rows = [json.loads(line) for line in results.read_text(encoding="utf-8").splitlines()
                    if line.strip()]
        seconds = round(time.perf_counter() - started, 2)
    fatal = next((r["fatal"] for r in rows if "fatal" in r), None)
    rows = [r for r in rows if "fatal" not in r]
    if not fatal and not timed_out and len(rows) < len(suite["cases"]) and stderr.strip():
        fatal = stderr.strip().splitlines()[-1][:300]
    reported = {r["name"] for r in rows}
    missing = [c for c in suite["cases"] if c["name"] not in reported]
    if timed_out and missing:
        rows.append({"name": missing[0]["name"], "args": missing[0]["args"][:CASE_TEXT],
                     "ok": False, "error": f"timed out after {timeout:g} s - too slow, or an endless loop"})
        rows += [{"name": c["name"], "args": c["args"][:CASE_TEXT], "ok": False,
                  "error": "not run"} for c in missing[1:]]
    elif missing and not fatal:
        rows += [{"name": c["name"], "args": c["args"][:CASE_TEXT], "ok": False,
                  "error": "not run - the program stopped before this case"} for c in missing]
    for row in rows:
        why = next((c["why"] for c in suite["cases"] if c["name"] == row["name"]), "")
        row["why"] = why
    passed = sum(1 for r in rows if r.get("ok"))
    return {"cases": rows, "passed": passed, "total": len(suite["cases"]),
            "fatal": fatal, "timed_out": timed_out, "seconds": seconds,
            "ref_errors": [r["name"] for r in rows if r.get("ref_error")]}


def validate(suite):
    """A suite is usable when its reference passes every case."""
    if not suite.get("cases") or not suite.get("reference") or not suite.get("entry"):
        return "the suite is incomplete"
    result = run_suite(suite, suite["reference"])
    if result["fatal"]:
        return result["fatal"]
    bad = [r for r in result["cases"] if not r.get("ok")]
    if bad:
        first = bad[0]
        return (f"case {first['name']!r}: "
                + (first.get("ref_error") or first.get("error") or "reference disagrees with itself"))
    return None


def suite_prompt(problem, code, entry):
    if problem.get("source") == "own":
        details = "Statement (the candidate's own exercise):\n" + (problem.get("statement") or "")
        if problem.get("starter_code"):
            details += "\nStarter code given with it:\n" + problem["starter_code"]
    else:
        details = ("Write the cases from your own knowledge of this LeetCode problem; do not "
                   "reuse the examples or tests on its page.")
    if problem.get("complexity"):
        details += (f"\nExpected complexity: time {problem['complexity']['time']}, "
                    f"space {problem['complexity']['space']}.")
    label = problem.get("label") or problem.get("title")
    entry_note = (f"the candidate's code defines it: {entry} (same name, same parameter order)"
                  if entry else "the conventional LeetCode signature in Python")
    return SUITE_TEXT.format(entry_note=entry_note, problem=label, details=details,
                             code=numbered(code, 120))


def build_suite(problem, problem_key, code, call):
    """The cached suite for this problem and entry point, or a new one
    written by `call(system, prompt, schema)` and kept only when its
    reference passes it (one retry with the failure quoted)."""
    entry = entry_point(code)
    suite = cached_suite(problem_key, entry)
    if suite:
        return suite
    prompt = suite_prompt(problem, code, entry)
    problem_note = None
    for _attempt in range(2):
        text = prompt if not problem_note else (
            prompt + f"\nYour previous suite was rejected: {problem_note}. Fix it.")
        suite = call(SUITE_SYSTEM, text, SUITE_SCHEMA)
        if entry and suite.get("entry") != entry:
            suite["entry"] = entry
        problem_note = validate(suite)
        if problem_note is None:
            suite["written"] = time.strftime("%Y-%m-%d %H:%M")
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            _cache_path(problem_key, entry).write_text(json.dumps(suite, indent=1), encoding="utf-8")
            return suite
    raise RuntimeError(f"The tutor could not write a working test suite ({problem_note}).")


def forget_suite(problem_key, code):
    """Drop the cached suite (the page's "Write new tests")."""
    path = _cache_path(problem_key, entry_point(code))
    if path.exists():
        path.unlink()
