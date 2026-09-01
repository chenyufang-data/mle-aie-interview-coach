"""Measure the prompt-cache design on real interviewer turns (plan
section 5c, Phase 3).

Two consecutive turns per engine over the same growing session - the
production shape: frozen persona system + append-only history (the
cached prefix) + the per-turn control message (volatile tail). The
second call must read the first call's prefix from cache:

  claude    usage.cache_read_input_tokens > 0 (explicit breakpoints in
            coach/llm._cache_marked; entries under the model's ~1k-token
            minimum silently don't cache, so the fixture session is big)
  deepseek  usage.prompt_cache_hit_tokens > 0 (automatic prefix cache;
            the design's job is only to keep the prefix byte-stable)

  .venv\\Scripts\\python grader\\cache_check.py --confirm   (~$0.05)

Results: printed + grader/cache_check_results.json.
"""

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from coach import config  # noqa: E402

config.load_env_file()

from coach import llm  # noqa: E402
from coach.mock import turns  # noqa: E402

RESULTS_PATH = BASE_DIR / "grader" / "cache_check_results.json"

PLAN = {
    "persona": {"interviewer_intro": "I'm Dana, staff MLE on the risk "
                                     "platform team.",
                "company_type": "payments company", "seniority": "staff"},
    "opening": "Thanks for joining - give me the one-minute version of the "
               "fraud project.",
    "probe_targets": [
        {"id": f"probe_{i}", "topic": topic, "question_hint": hint,
         "source": "project"}
        for i, (topic, hint) in enumerate([
            ("leakage", "How did you catch the leak in the pipeline?"),
            ("imbalance", "Why class weights over resampling here?"),
            ("threshold", "How was the decision threshold chosen?"),
            ("calibration", "How did you verify the scores were calibrated?"),
            ("deployment", "What breaks first when traffic doubles?"),
            ("monitoring", "What drift signal pages you at 3am?"),
        ], start=1)],
    "behavioral_targets": ["a modeling decision you reversed",
                           "a deadline that slipped"],
    "settings": {"style": "neutral", "length": "standard", "level": "Mid-level"},
}
ROLE = {"title": "Machine Learning Engineer", "level": "Mid-level",
        "domain": "fraud detection"}

# Four answered turns of realistic length put the cacheable prefix well
# past Claude's minimum before the first measured call.
HISTORY = [
    ("Give me the one-minute version of the fraud project.",
     "Sure. I built a fraud detection model for card-not-present payments. "
     "The base rate was around 0.3 percent, so everything was shaped by the "
     "imbalance: we optimized PR-AUC rather than ROC-AUC, used LightGBM "
     "with class weights, and tuned the operating threshold against a cost "
     "matrix the risk team signed off on. It served about two thousand "
     "requests a second behind a FastAPI service with a feature store "
     "lookup, and the recall at our fixed precision target roughly doubled "
     "versus the rules engine it replaced."),
    ("Walk me through it end to end - problem, your role, approach, and "
     "how you knew it worked.",
     "The problem was chargeback losses growing faster than payment volume. "
     "I owned the modeling end to end. Data came from the transaction log "
     "joined with device and velocity features; I validated with a "
     "time-based split because random splits leaked future velocity "
     "aggregates into training. The model was gradient boosting after a "
     "logistic baseline, and the key iteration was feature hygiene: "
     "anything computed with post-transaction information got dropped. We "
     "knew it worked from a three-week shadow deployment where the model's "
     "declines were reviewed by analysts before we let it act."),
    ("How did you catch the leak in the pipeline?",
     "The first model looked too good - PR-AUC around 0.92 - so I audited "
     "feature timestamps. A velocity feature was computed over a window "
     "that included the transaction itself, and a settlement flag only "
     "exists after the outcome. I rebuilt the features with strict "
     "point-in-time joins, re-ran the time-based validation, and the "
     "score dropped to a believable 0.71. We added a leakage checklist to "
     "the feature store review process after that."),
    ("Why class weights over resampling here?",
     "We tried SMOTE early and it manufactured borderline fraud examples "
     "that did not look like real attack patterns, which hurt precision "
     "at the operating point. Class weights kept the true data "
     "distribution, trained faster, and made the probability outputs "
     "easier to calibrate afterwards with isotonic regression. "
     "Undersampling the negatives was the fallback, but we had the "
     "compute budget to keep them all."),
    ("How did you verify the scores were actually calibrated?",
     "Reliability diagrams on the holdout window, bucketed by score "
     "decile, before and after isotonic regression - raw gradient "
     "boosting scores were badly overconfident in the top decile. We "
     "also tracked expected calibration error weekly in production, "
     "sliced by merchant category, because calibration drifted for "
     "travel merchants when their volume recovered seasonally. The "
     "cost-matrix thresholding only makes sense if the probabilities "
     "mean what they say, so this gated the launch."),
]
NEXT_ANSWER = (
    "We picked the threshold from the cost matrix: a false positive costs "
    "a manual review plus customer friction, a false negative costs the "
    "chargeback. I swept thresholds on the validation window, plotted "
    "cost against recall, and the risk team chose the knee point. It gets "
    "re-evaluated monthly because the attack mix shifts.")


def build_state():
    transcript = []
    for i, (question, answer) in enumerate(HISTORY, start=1):
        transcript.append({"turn": i, "phase": "warmup" if i == 1 else
                           ("walkthrough" if i == 2 else "deepdive"),
                           "question": question, "probe_id": None,
                           "answer": answer})
    return {"plan": copy.deepcopy(PLAN), "role": dict(ROLE),
            "transcript": transcript}


def measure(engine):
    state = build_state()
    rows = []
    for _ in range(2):
        phase, turn_number, done = turns.position(state)
        assert not done
        system, messages = turns.turn_prompt(state, phase, turn_number)
        started = time.perf_counter()
        text = llm.call_chat(system, messages, engine, thinking=False,
                             max_tokens=150, temperature=0, cache=True)
        spoken, meta = turns.parse_turn_output(text)
        rows.append({"seconds": round(time.perf_counter() - started, 2),
                     "question": spoken[:80], **dict(llm.LAST_USAGE)})
        state["transcript"].append({"turn": turn_number, "phase": phase,
                                    "question": spoken,
                                    "probe_id": meta.get("probe_id"),
                                    "answer": NEXT_ANSWER})
        # Cache entries build asynchronously (DeepSeek especially); real
        # turns are ~30-60 s apart while the candidate answers, so a
        # back-to-back second call would under-report the production case.
        time.sleep(10)
    return rows


def verdict(engine, rows):
    second = rows[1]
    if engine == "claude":
        read = second.get("cache_read") or 0
        return read > 0, f"second call cache_read_input_tokens={read}"
    hit = second.get("cache_hit") or 0
    return hit > 0, f"second call prompt_cache_hit_tokens={hit}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()
    print("cache check: 2 turns x (deepseek, claude), ~150 output tokens "
          "each; estimated cost ~$0.05")
    if not args.confirm:
        print("dry run only - add --confirm to spend")
        return

    results = {}
    for engine, key in (("deepseek", "DEEPSEEK_API_KEY"),
                        ("claude", "ANTHROPIC_API_KEY")):
        if not os.environ.get(key):
            print(f"{engine}: skipped ({key} not set)")
            continue
        rows = measure(engine)
        ok, detail = verdict(engine, rows)
        results[engine] = {"rows": rows, "cache_working": ok,
                           "detail": detail}
        print(f"{engine}: {'CACHED' if ok else 'NOT CACHED'} - {detail}")
        for i, row in enumerate(rows, start=1):
            print(f"  call {i}: {json.dumps(row, ensure_ascii=False)}")

    results["measured_at"] = time.strftime("%Y-%m-%d %H:%M")
    RESULTS_PATH.write_text(json.dumps(results, indent=1, ensure_ascii=False),
                            encoding="utf-8")
    print(f"-> {RESULTS_PATH.name}")


if __name__ == "__main__":
    main()
