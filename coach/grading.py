"""Engine routing, the smart cascade, and the local distilled grader."""

import random

from retrieval import tokenize

from coach import config, users
from coach.config import CASCADE_FRAC_HIT_MAX, CASCADE_PRED_MAX, GRADER_PATH

# Trained distilled grader for mock mode (grader/train.py artifact); None
# falls back to plain keyword matching. Reassigned by load_grader(); always
# read as grading.GRADER.
GRADER = None


def load_grader():
    global GRADER
    if not GRADER_PATH.exists():
        return
    try:
        import joblib
        from grader import features  # noqa: F401  (unpickling needs FeatureExtractor importable)

        GRADER = joblib.load(GRADER_PATH)
    except Exception as exc:
        print(f"Warning: could not load ML grader ({exc}); mock grading uses keyword matching.")


def grading_route(user, force_llm=False):
    """Decide which grading engine this request gets.

    Returns (engine, reason). engine is "claude", "deepseek", "ollama", or
    "local"; reason says why a request routed local: "mock" (dev mode),
    "free" (no paid key), "quota" (paid but spent today, no DeepSeek key).

    Paid routing: DeepSeek Flash is the quota-free default workhorse
    (measured judge, see grader/judge_agreement.py); Claude serves
    force_llm ("Always Claude") requests and takes quota; an exhausted
    Claude quota degrades to DeepSeek, and only to the local grader when
    no DEEPSEEK_API_KEY is configured. Without users.json (single-user
    setup) everything grades with Claude, as before.
    """
    if config.MODE == "mock":
        return "local", "mock"
    if config.MODE == "ollama":
        return "ollama", None
    if not users.TIERS_ENABLED:
        return "claude", None  # tiers disabled: single-user setup
    if user["tier"] != "paid":
        return "local", "free"
    if not force_llm and config.deepseek_available():
        return "deepseek", None
    if users.quota_take(user):
        return "claude", None
    if config.deepseek_available():
        return "deepseek", None  # Claude quota spent: degrade to Flash
    return "local", "quota"


def clamp_score(value):
    return max(1, min(10, value))


# Why an evaluation was routed to local grading -> (summary label, user hint).
LOCAL_GRADING_LABELS = {
    "mock": ("Mock mode", "Run without --mock for real Claude grading."),
    "free": ("Free tier", "Enter a paid access key for full Claude grading."),
    "quota": ("Daily Claude quota reached", "Quota resets tomorrow; until then grading is local."),
    "cascade": (
        "Smart cascade",
        "This answer clearly misses the rubric, so the local grader handled it "
        "and no Claude quota was spent. Tick 'Always Claude' to force the full "
        "evaluation.",
    ),
    "llm_error": (
        "Grading service unavailable",
        "The DeepSeek grading call failed, so the local grader stepped in. "
        "Submit again, or tick 'Always Claude' to grade with Claude instead.",
    ),
}


def cascade_confident(answer, chunk):
    """True when the student's grade is reliable enough to skip the teacher."""
    if GRADER is None:
        return False
    feats = [GRADER["extractor"].extract(answer, chunk)]
    predicted = float(GRADER["model"].predict(feats)[0])
    frac_hit = feats[0][GRADER["feature_names"].index("kp_frac_hit")]
    return predicted <= CASCADE_PRED_MAX and frac_hit <= CASCADE_FRAC_HIT_MAX


def mock_evaluation(data, chunk, reason="mock"):
    label, hint = LOCAL_GRADING_LABELS[reason]
    answer = data.get("answer", "")
    answer_tokens = set(tokenize(answer))
    word_count = len(answer.split())

    if chunk:
        interview = chunk["interview"]
        if GRADER is not None:
            # Distilled grader: trained model predicts the score. The hit/miss
            # lists come from the semantic key-point classifier when the
            # artifact has one (trained on teacher hit/partial/miss verdicts;
            # judges meaning), falling back to the 0.35 lexical threshold.
            extractor = GRADER["extractor"]
            key_points = interview["key_points"]
            kp_clf = GRADER.get("kp_classifier")
            kp_feats = (extractor.keypoint_features(answer, chunk)
                        if kp_clf is not None else None)
            partials = []
            if kp_feats:
                verdicts = list(kp_clf.predict(kp_feats))
                hits = [p for p, v in zip(key_points, verdicts) if v == "hit"]
                partials = [p for p, v in zip(key_points, verdicts) if v == "partial"]
                misses = [p for p, v in zip(key_points, verdicts) if v == "miss"]
            else:
                coverage_scores = extractor.key_point_coverage(answer, chunk)
                hits = [p for p, c in zip(key_points, coverage_scores) if c >= 0.35]
                misses = [p for p, c in zip(key_points, coverage_scores) if c < 0.35]
            feats = [extractor.extract(answer, chunk)]
            stacked = GRADER.get("stacked_model")
            if stacked is not None and kp_feats:
                # Stacked scorer: base features + the classifier's coverage
                # aggregates. The cascade keeps the plain model - its
                # thresholds were measured against it.
                classes = list(kp_clf.classes_)
                hit_col = classes.index("hit")
                proba = kp_clf.predict_proba(kp_feats)
                agg = [len(hits) / len(key_points),
                       len(partials) / len(key_points),
                       float(proba[:, hit_col].mean()),
                       float(proba[:, hit_col].min())]
                predicted = stacked.predict([feats[0] + agg])[0]
            else:
                predicted = GRADER["model"].predict(feats)[0]
            overall = clamp_score(round(predicted))
            # Subscores from the dedicated multi-output models when the
            # artifact has them (each beats overall-as-proxy on gold rows).
            sub_models = GRADER.get("subscore_models")
            predicted_scores = (
                {name: clamp_score(round(float(m.predict(feats)[0])))
                 for name, m in sub_models.items()}
                if sub_models else None
            )
            summary = (
                f"[{label} - local ML grader ({GRADER['model_name']})] "
                f"Predicted {overall}/10; matched {len(hits)} of "
                f"{len(interview['key_points'])} rubric key points. {hint}"
            )
        else:
            predicted_scores = None
            hits, misses, partials = [], [], []
            for point in interview["key_points"]:
                point_tokens = set(tokenize(point))
                overlap = len(point_tokens & answer_tokens) / max(1, len(point_tokens))
                (hits if overlap >= 0.35 else misses).append(point)
            coverage = len(hits) / max(1, len(interview["key_points"]))
            overall = clamp_score(round(2 + 8 * coverage))
            summary = (
                f"[{label} - keyword matching] Found {len(hits)} of "
                f"{len(interview['key_points'])} rubric key points in your answer. {hint}"
            )
        return {
            "graded_by": (f"local ML grader ({GRADER['model_name']})"
                          if GRADER is not None else "local keyword matching"),
            "overall_score": overall,
            "scores": predicted_scores or {
                "technical_depth": overall,
                "structure": clamp_score(overall + (1 if word_count >= 60 else -1)),
                "practical_judgment": overall,
                "communication": clamp_score(overall + (1 if 40 <= word_count <= 250 else -1)),
            },
            "summary": summary,
            "strengths": (
                [f"Touched on: {point}" for point in hits[:4]]
                or ["You submitted an answer; no rubric key points were detected by keyword matching."]
            ),
            "gaps": (
                ([f"Missed rubric point: {point}" for point in misses]
                 + [f"Only partially covered: {point}" for point in partials])[:4]
                or ["No rubric gaps detected - compare with the model answer to be sure."]
            ),
            "suggested_answer": interview["model_answer"],
            "next_steps": (
                [f"Watch out for: {mistake}" for mistake in interview["common_mistakes"][:3]]
                + ["Compare your answer with the model answer above, then re-record a tighter version."]
            ),
            "follow_up_question": random.choice(interview["followups"]),
        }

    overall = clamp_score(3 + min(4, word_count // 40))
    return {
        "graded_by": "local length heuristic (no rubric)",
        "overall_score": overall,
        "scores": {
            "technical_depth": overall,
            "structure": overall,
            "practical_judgment": overall,
            "communication": overall,
        },
        "summary": (
            f"[{label}] This question has no knowledge-base rubric, so the score "
            f"is a rough length-based placeholder. {hint}"
        ),
        "strengths": ["Answer recorded with local grading."],
        "gaps": ["Local grading cannot judge content for questions without a rubric."],
        "suggested_answer": f"Local grading cannot draft a model answer for this question. {hint}",
        "next_steps": [
            "Practice knowledge-base topics, which grade against a real rubric even locally.",
            hint,
        ],
        "follow_up_question": "How would you measure whether your proposed approach actually works in production?",
    }
