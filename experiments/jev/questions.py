"""The Jev questions for roadmap step 6, frozen before the first gold call
(docs/plan.md step 6). Pure functions, no network: the harness
(experiments/jev/jev_eval.py) and the runtime route (if the rule passes)
build the same request from here.

One request per answer: the state is the material the teacher graded with,
and the questions are one Score for the overall grade plus one Choice per
rubric key point, all answered in parallel in a single call.
"""

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
