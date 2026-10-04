"""Roadmap step 7, phase 4: how fast the coding tutor answers.

    .venv\\Scripts\\python experiments\\coding\\tutor_latency.py            dry run: the calls it would make
    .venv\\Scripts\\python experiments\\coding\\tutor_latency.py --confirm  run them, write the results file

The target is the plan's (docs/plan.md step 7, "Speed"), fixed before this
run: hint p95 at most 5 s locally. Each call goes through coach/tutor.py the
way the coding page's does - the prompt, the model, the output guard and any
regeneration - on the engine the local app picks (the signed-in
subscription; quick turns at low effort, the model from LLM_CLI_MODEL).

Workload: five bank problems (LeetCode easy / medium / hard and two of the
author's own exercises), four Hint presses each on an empty editor (levels
0-3), then one typed "where do I start?" each - 25 replies, one at a time.
The results file keeps numbers only (problem id, kind, level, seconds,
guard events, words): the bank is private, so no reply text is committed.
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

from coach import cli_engine, coding, config, tutor  # noqa: E402

RESULTS = Path(__file__).with_name("tutor_latency_results.json")
PROBLEMS = ["code_lc_0001_two_sum", "code_lc_0033_search_in_rotated_sorted_array",
            "code_lc_0042_trapping_rain_water", "code_ml_01_numerically_stable_softmax",
            "code_ml_29_reservoir_sampling"]
HINTS = 4
MESSAGE = "I'm not sure where to start - what should I think about first?"
TARGET_P95_S = 5.0


def setup():
    config.load_env_file()
    provider, note = cli_engine.auto_select("127.0.0.1", os.environ.get("LLM_CLI", "auto"))
    if not provider:
        raise SystemExit(f"No signed-in subscription CLI ({note}).")
    config.MODE, config.CLI_PROVIDER = "cli", provider
    config.CLI_MODEL = os.environ.get("LLM_CLI_MODEL", "").strip()
    config.CLI_EFFORT = os.environ.get("LLM_CLI_EFFORT", "low").strip() or "low"
    config.CLI_QUICK_MODEL = os.environ.get("LLM_CLI_QUICK_MODEL", "").strip()
    return cli_engine.label() + (f"; quick replies on {config.CLI_QUICK_MODEL}"
                                 if config.CLI_QUICK_MODEL else "")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--note", default="", help="what changed since the last run")
    args = parser.parse_args(argv)
    problems = [coding.by_id(pid) for pid in PROBLEMS]
    missing = [pid for pid, p in zip(PROBLEMS, problems) if p is None]
    if missing:
        raise SystemExit(f"not in banks/rag_code: {missing}")
    calls = len(PROBLEMS) * (HINTS + 1)
    label = setup()
    print(f"{calls} tutor replies on {label} (quick turns at low effort), one at a time")
    if not args.confirm:
        print("dry run - add --confirm to run them (counts against your plan)")
        return 0
    rows = []
    for record in problems:
        attempt = tutor.new_attempt(coding.detail(record))
        requests = [("hint", "")] * HINTS + [("message", MESSAGE)]
        for kind, message in requests:
            started = time.perf_counter()
            out = tutor.tutor_turn(attempt, "", message, kind, "cli")
            wall = round(time.perf_counter() - started, 2)
            rows.append({"problem": record["id"], "kind": kind, "level": out["level"],
                         "seconds": wall, "words": len(re.findall(r"\w+", out["reply"])),
                         "regenerated": out["guard"]["regenerated"],
                         "trimmed": out["guard"]["trimmed"]})
            print(f"  {record['id'][:44]:44s} {kind:7s} level {out['level']}  {wall:5.2f} s"
                  + ("  (regenerated)" if out["guard"]["regenerated"] else "")
                  + ("  (trimmed)" if out["guard"]["trimmed"] else ""), flush=True)
    seconds = np.array([r["seconds"] for r in rows])
    hints = np.array([r["seconds"] for r in rows if r["kind"] == "hint"])
    summary = {
        "replies": len(rows),
        "p50_s": round(float(np.percentile(seconds, 50)), 2),
        "p95_s": round(float(np.percentile(seconds, 95)), 2),
        "max_s": round(float(seconds.max()), 2),
        "hint_p50_s": round(float(np.percentile(hints, 50)), 2),
        "hint_p95_s": round(float(np.percentile(hints, 95)), 2),
        "regenerated": sum(r["regenerated"] for r in rows),
        "trimmed": sum(r["trimmed"] for r in rows),
        "levels": [sum(1 for r in rows if r["level"] == lv) for lv in range(5)],
    }
    verdict = "PASS" if summary["hint_p95_s"] <= TARGET_P95_S else "FAIL"
    run = {"measured": datetime.now().isoformat(timespec="seconds"), "note": args.note,
           "engine": label, "quick_effort": "low", "summary": summary, "verdict": verdict,
           "rows": rows}
    # every run stays in the file, the latest last: a failed run is kept,
    # with what changed before the next one
    previous = json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else {}
    runs = previous.get("runs") or []
    runs = runs + [run]
    # one verdict per setting (the latest run of each): a comparison arm must
    # not read as the verdict on the author's own setting
    verdicts = {}
    for item in runs:
        verdicts[item["engine"]] = {"verdict": item["verdict"],
                                    "hint_p95_s": item["summary"]["hint_p95_s"],
                                    "measured": item["measured"]}
    out = {"study": "step 7 phase 4 - tutor reply latency",
           "rule": f"hint p95 <= {TARGET_P95_S} s locally (docs/plan.md step 7, Speed)",
           "verdicts": verdicts, "runs": runs}
    RESULTS.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"hint p50 {summary['hint_p50_s']} s, p95 {summary['hint_p95_s']} s, all p95 "
          f"{summary['p95_s']} s, max {summary['max_s']} s; guard regenerated "
          f"{summary['regenerated']}, trimmed {summary['trimmed']} -> {verdict}")
    print(f"wrote {RESULTS.relative_to(BASE_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
