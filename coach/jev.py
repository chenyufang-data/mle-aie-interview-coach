"""Optional Jev grade for the local tier (roadmap step 6).

TypeSafe's Jev returns typed answers with probabilities instead of text.
Step 6 (experiments/jev/, docs/plan.md) measured it on the teacher-labelled
rows against the sklearn grader and passed its pre-registered rule; this
module is what the server runs. With TYPESAFE_API_KEY set (read at call
time) and outside --mock mode, the free tier's overall grade and its rubric
hit/partial/miss verdicts come from one Jev call; the grade goes through
the isotonic calibration fitted on seed 42's training-side teacher rows
(coach/assets/jev_calibration.json, tied to the model version it was fitted
for). The cascade and the subscores stay with the sklearn artifact they were
measured against. JEV_DAILY_CAP bounds the calls per day for the whole
server (coach/users.py take_jev); a spent cap, an error or a timeout returns
None and the caller keeps the sklearn grade, and an error also pauses the
route for JEV_BACKOFF_S.

The questions below are the ones the experiment froze before its first
gold call (commit bfcf76b); experiments/jev/questions.py re-exports them so
the experiment and the server always send the same request.
"""

import json
import os
import time
from pathlib import Path
from urllib import error as urlerror
from urllib import request as urlrequest

from coach import config

API_URL = "https://api.typesafe.ai/v1/systemone"

# ------------------------------------------------- the frozen questions

# Ten levels, worst first. TypeSafe's guidance: describe situations, not
# degrees; no numbers in the descriptions; one dimension per question. The
# dimension is the one the teacher's overall score measures: how well the
# answer meets the rubric for this question.
GRADE_LEVELS = [
    "Does not answer the question: off-topic, only restates the question, or empty filler",
    "Stays on the topic but says nothing substantive: generic statements that would fit any question in the area",
    "One correct point, stated minimally; almost all of the rubric is missing",
    "A few relevant points, but with a clear gap or a wrong claim about something central",
    "Covers about half of the rubric, shallowly, with little reasoning",
    "Touches most key points, but some are vague, imprecise or only implied",
    "States the key points correctly with some reasoning; minor gaps remain",
    "Covers nearly all key points accurately, with clear reasoning and practical judgement",
    "Complete, precise and well-reasoned; as good as a strong reference answer",
    "Complete, precise and well-reasoned, and adds insight beyond the reference answer, such as tradeoffs or failure modes",
]

GRADE_INSTRUCTIONS = ("How well does the candidate answer meet the rubric for this "
                      "interview question? Judge the substance of candidate_answer "
                      "against reference_answer, rubric_key_points and common_mistakes; "
                      "wording does not matter.")

# The teacher's own verdict definitions (experiments/distill/label_keypoints.py).
KP_CRITERIA = {
    "hit": "candidate_answer conveys this point's substance (a paraphrase counts fully)",
    "partial": "candidate_answer touches the idea but incompletely, muddled, or only implied",
    "miss": ("the point is absent from candidate_answer, or stated incorrectly: a confident "
             "wrong claim about this point is a miss even if it uses the right vocabulary"),
}
KP_LABELS = ("hit", "partial", "miss")


def build_state(chunk, answer):
    interview = chunk["interview"]
    return {
        "interview_question": interview["question"].strip(),
        "reference_answer": interview.get("model_answer", "").strip(),
        "rubric_key_points": list(interview.get("key_points", [])),
        "common_mistakes": list(interview.get("common_mistakes", [])),
        "candidate_answer": answer.strip(),
    }


def build_questions(chunk, with_keypoints=True):
    questions = {"grade": {"type": "score", "instructions": GRADE_INSTRUCTIONS,
                           "criteria": list(GRADE_LEVELS)}}
    if with_keypoints:
        for i, point in enumerate(chunk["interview"].get("key_points", [])):
            questions[f"kp_{i}"] = {
                "type": "choice",
                "instructions": ("Does candidate_answer cover this rubric key point? "
                                 f"Key point: {point}"),
                "criteria": dict(KP_CRITERIA),
            }
    return questions


def build_request(chunk, answer, model, with_keypoints=True):
    return {"model": model, "state": build_state(chunk, answer),
            "questions": build_questions(chunk, with_keypoints)}


def parse_answers(body, n_points):
    """The grade on the 1..10 scale (1 + the probability-weighted level), its
    confidence and level distribution, and the key-point verdicts."""
    answers = body["answers"]
    grade = answers["grade"]
    probs = grade.get("probabilities") or {}
    out = {
        "grade": 1.0 + float(grade["score"]),
        "grade_confidence": float(grade.get("confidence", 0.0)),
        "grade_probs": [float(probs.get(str(i), 0.0)) for i in range(len(GRADE_LEVELS))],
        "kp": [],
    }
    for i in range(n_points):
        a = answers.get(f"kp_{i}")
        if a is None:
            out["kp"].append(None)
            continue
        p = a.get("probabilities") or {}
        out["kp"].append({"choice": a["choice"],
                          "confidence": float(a.get("confidence", 0.0)),
                          "probs": {k: float(p.get(k, 0.0)) for k in KP_LABELS}})
    return out


# --------------------------------------------------------------- runtime

_unavailable_until = 0.0
_calibration = None


def calibration():
    """{"model", "x", "y", ...} from coach/assets/jev_calibration.json, or None."""
    global _calibration
    if _calibration is None:
        try:
            _calibration = json.loads(Path(config.JEV_CALIBRATION_PATH)
                                      .read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _calibration = {}
    return _calibration or None


def model():
    """The pinned version: JEV_MODEL, else the one the calibration was fitted for."""
    cal = calibration()
    return config.JEV_MODEL or (cal or {}).get("model") or "jev-latest"


def enabled():
    """Configured to serve: a key, not --mock, and a calibration on disk."""
    return (bool(os.environ.get("TYPESAFE_API_KEY")) and config.MODE != "mock"
            and calibration() is not None)


def available():
    return enabled() and time.monotonic() >= _unavailable_until


def label():
    return f"Jev typed decision model ({model()}, calibrated)"


def calibrate(raw_grade):
    """The isotonic map as fitted: linear between its thresholds, clipped at
    the ends (what sklearn's IsotonicRegression.predict does)."""
    cal = calibration()
    xs, ys = cal["x"], cal["y"]
    if raw_grade <= xs[0]:
        return float(ys[0])
    if raw_grade >= xs[-1]:
        return float(ys[-1])
    for i in range(1, len(xs)):
        if raw_grade <= xs[i]:
            x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
            if x1 == x0:
                return float(y1)
            return float(y0 + (y1 - y0) * (raw_grade - x0) / (x1 - x0))
    return float(ys[-1])


def call(chunk, answer, timeout=None):
    """One request; returns the parsed answers (parse_answers) or raises."""
    body = json.dumps(build_request(chunk, answer, model())).encode("utf-8")
    req = urlrequest.Request(API_URL, data=body, method="POST", headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('TYPESAFE_API_KEY', '')}"})
    with urlrequest.urlopen(req, timeout=timeout or config.JEV_TIMEOUT_S) as resp:
        payload = json.load(resp)
    return parse_answers(payload, len(chunk["interview"].get("key_points", [])))


def grade(chunk, answer, user):
    """{"grade": calibrated 1..10, "raw", "confidence", "verdicts"} or None:
    the route is off, the day's cap is spent, or the call failed (the caller
    keeps the sklearn grade)."""
    global _unavailable_until
    if not available():
        return None
    from coach import users
    if not users.take_jev(user):
        return None
    try:
        parsed = call(chunk, answer)
    except (urlerror.URLError, OSError, ValueError, KeyError, TypeError) as exc:
        _unavailable_until = time.monotonic() + config.JEV_BACKOFF_S
        print(f"Jev grader skipped for {config.JEV_BACKOFF_S:.0f}s: {exc}", flush=True)
        return None
    verdicts = [kp["choice"] if kp else None for kp in parsed["kp"]]
    return {"grade": calibrate(parsed["grade"]), "raw": parsed["grade"],
            "confidence": parsed["grade_confidence"],
            "verdicts": verdicts if all(v in KP_LABELS for v in verdicts) else None}
