"""Report-consistency test (plan section 7d): regrade the same five mock
sessions twice with DeepSeek Flash and twice with Claude, and measure how
much each engine's scorecard moves between identical regrades. This is the
number behind the report-engine default: Claude holds it only because its
regrade consistency is measured to be worth the ~$0.15/report; Flash takes
over "only if its scorecard consistency comes within a few points".

Sessions are built deterministically (the offline fake planner + the 20
real Phase 0 answer texts from grader/stt_sentences.jsonl rotated across
5 short sessions), so the test is reproducible and each engine grades each
session as the product would (Claude: adaptive thinking, default sampling;
Flash: temperature 0, thinking spent in completion - the product configs).

  .venv\\Scripts\\python grader\\report_consistency.py            dry run (cost)
  .venv\\Scripts\\python grader\\report_consistency.py --confirm  spend + measure

Results: printed + grader/report_consistency_results.json (committed).
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from coach import config, grading, kb  # noqa: E402
from coach.mock import planning, report, turns  # noqa: E402
from coach.mock.schemas import DIMENSIONS  # noqa: E402

RESULTS_PATH = BASE_DIR / "grader" / "report_consistency_results.json"
SENTENCES_PATH = BASE_DIR / "grader" / "stt_sentences.jsonl"
SESSIONS = 5
REPS = 2
ENGINES = ("deepseek", "claude")

RESUME = ("ML engineer. Built a fraud detection model with LightGBM, evaluated "
          "with PR-AUC and calibration; deployed behind a FastAPI service on "
          "EC2. Built a RAG interview coach with BM25 retrieval and a distilled "
          "sklearn grader evaluated by QWK against a Claude teacher.")


def build_sessions():
    answers = [json.loads(line)["text"] for line in
               SENTENCES_PATH.read_text(encoding="utf-8").splitlines()
               if line.strip() and json.loads(line)["id"].startswith("ans")]
    roles = planning.propose_roles(RESUME, "", "fake")
    role = roles["roles"][0]
    jd_text, _default = planning.resolve_jd({"role": role})
    sessions = []
    for index in range(SESSIONS):
        plan = planning.build_plan(RESUME, jd_text, role, None,
                                   {"style": "neutral", "length": "short"}, "fake")
        state = {"plan": plan, "role": role, "transcript": []}
        offset = index * 4
        while True:
            turn = turns.next_turn(state, "fake")
            if turn["done"]:
                break
            answer = answers[(offset + turn["turn"] - 1) % len(answers)]
            state["transcript"].append({
                "turn": turn["turn"], "phase": turn["phase"],
                "question": turn["question"], "probe_id": turn["probe_id"],
                "answer": answer, "answer_ms": 45000 + 5000 * (turn["turn"] % 3),
            })
        sessions.append(state)
    return sessions


def regrade(sessions, engine):
    """REPS assessments per session; returns per-session list of score dicts."""
    out = []
    for number, state in enumerate(sessions, 1):
        reps = []
        for rep in range(REPS):
            for attempt in range(2):
                try:
                    started = time.perf_counter()
                    result = report.build_report(
                        {"plan": state["plan"], "role": state["role"],
                         "transcript": state["transcript"]}, engine)
                    break
                except Exception as exc:
                    if attempt:
                        raise
                    print(f"  retrying session {number} rep {rep} ({exc})")
            assessment = result["assessment"]
            reps.append({
                "scores": assessment["scores"],
                "overall_call": assessment["overall_call"],
                "per_turn": [t["score"] for t in assessment.get("per_turn", [])],
                "seconds": round(time.perf_counter() - started, 1),
            })
            print(f"  {engine} session {number} rep {rep + 1}: "
                  f"{assessment['overall_call']}, "
                  f"scores {list(assessment['scores'].values())} "
                  f"({reps[-1]['seconds']}s)")
        out.append(reps)
    return out


def consistency(per_session):
    """Regrade agreement between rep pairs of the same session."""
    exact = total = 0
    diffs = []
    call_flips = 0
    per_turn_diffs = []
    for reps in per_session:
        first, second = reps[0], reps[1]
        for name in DIMENSIONS:
            a, b = first["scores"][name], second["scores"][name]
            total += 1
            exact += a == b
            diffs.append(abs(a - b))
        call_flips += first["overall_call"] != second["overall_call"]
        for a, b in zip(first["per_turn"], second["per_turn"]):
            per_turn_diffs.append(abs(a - b))
    return {
        "dimension_scores": total,
        "exact_pct": round(100 * exact / total, 1),
        "mae": round(statistics.mean(diffs), 3),
        "max_diff": max(diffs),
        "call_flips": f"{call_flips}/{len(per_session)}",
        "per_turn_mae": round(statistics.mean(per_turn_diffs), 3)
        if per_turn_diffs else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()
    calls = SESSIONS * REPS
    print(f"{SESSIONS} sessions x {REPS} regrades x {len(ENGINES)} engines = "
          f"{calls * len(ENGINES)} reports")
    print(f"estimated cost: Claude ~${calls * 0.15:.2f} "
          f"({calls} calls x ~$0.15), Flash ~${calls * 0.002:.2f}")
    if not args.confirm:
        print("dry run only - add --confirm to spend")
        return

    config.load_env_file()
    config.MODE = "mock"          # deterministic session construction only;
    kb.load_chunks()              # the report engines are passed explicitly
    grading.load_grader()
    sessions = build_sessions()
    print(f"built {len(sessions)} sessions "
          f"({len(sessions[0]['transcript'])} turns each)\n")

    results = {"sessions": SESSIONS, "reps": REPS,
               "built": time.strftime("%Y-%m-%d"), "engines": {}}
    for engine in ENGINES:
        print(f"== {engine} ==")
        per_session = regrade(sessions, engine)
        stats = consistency(per_session)
        stats["raw"] = per_session
        results["engines"][engine] = stats
        print(f"  -> exact {stats['exact_pct']}%, MAE {stats['mae']}, "
              f"call flips {stats['call_flips']}\n")

    flash = results["engines"]["deepseek"]
    claude = results["engines"]["claude"]
    results["decision"] = (
        "Flash consistency is within 10 exact-percentage points of Claude - "
        "revisit the report default (plan 7d)."
        if flash["exact_pct"] >= claude["exact_pct"] - 10
        else "Claude stays the report default: its regrade consistency "
             "exceeds Flash's by more than the 7d margin.")
    print(results["decision"])
    RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=1),
                            encoding="utf-8")
    print(f"-> {RESULTS_PATH.name}")


if __name__ == "__main__":
    main()
