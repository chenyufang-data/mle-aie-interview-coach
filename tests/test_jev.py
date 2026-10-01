"""Offline tests for roadmap step 6 (experiments/jev): the request the
questions build, the response parsing, the calibrated arm and the rule.

Run:  .venv\\Scripts\\python tests\\test_jev.py

No network, no key, no private data: a synthetic chunk and a synthetic
response in the documented shape (docs.typesafe.ai/api).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.jev import jev_eval, questions as jq  # noqa: E402

CHUNK = {"id": "c1", "interview": {
    "question": "When would you prefer Ridge over Lasso?",
    "model_answer": "Ridge when most features matter a little...",
    "key_points": ["L2 shrinks, never zero", "spreads weight across correlated features"],
    "common_mistakes": ["claims Ridge selects features"],
    "followups": ["what about elastic net?"]}}


def test_request_shape():
    req = jq.build_request(CHUNK, "  Ridge keeps all features.  ", "jev-latest")
    assert req["model"] == "jev-latest"
    state = req["state"]
    assert state["candidate_answer"] == "Ridge keeps all features."
    assert state["rubric_key_points"] == CHUNK["interview"]["key_points"]
    assert "followups" not in state  # not part of the rubric the teacher grades with
    q = req["questions"]
    assert set(q) == {"grade", "kp_0", "kp_1"}
    assert q["grade"]["type"] == "score" and len(q["grade"]["criteria"]) == 10
    assert all(not any(ch.isdigit() for ch in level) for level in jq.GRADE_LEVELS), \
        "TypeSafe: no numbers in level descriptions"
    assert q["kp_1"]["type"] == "choice"
    assert set(q["kp_1"]["criteria"]) == {"hit", "partial", "miss"}
    assert "spreads weight" in q["kp_1"]["instructions"]
    assert set(jq.build_questions(CHUNK, with_keypoints=False)) == {"grade"}


def test_parse_answers():
    body = {"model": "jev-1.13.0", "answers": {
        "grade": {"type": "score", "score": 6.4, "confidence": 0.7,
                  "probabilities": {"6": 0.6, "7": 0.4}},
        "kp_0": {"type": "choice", "choice": "hit", "confidence": 0.9,
                 "probabilities": {"hit": 0.9, "partial": 0.08, "miss": 0.02}}},
        "usage": {"input_tokens": 900, "output_tokens": 3}}
    out = jq.parse_answers(body, 2)
    assert abs(out["grade"] - 7.4) < 1e-9          # 1 + level 6.4
    assert out["grade_probs"][6] == 0.6 and len(out["grade_probs"]) == 10
    assert out["kp"][0]["choice"] == "hit" and out["kp"][1] is None


def test_calibrated_is_monotone_and_bounded():
    x_fit = [1, 2, 3, 4, 5, 6, 7, 8]
    y_fit = [1, 1, 3, 4, 4, 6, 8, 9]
    pred = jev_eval.calibrated(x_fit, y_fit, [0, 2.5, 5, 9, 20])
    assert list(pred) == sorted(pred)
    assert pred.min() >= 1 and pred.max() <= 10


def test_rule():
    sk = {s: {"qwk": 0.80, "mae": 1.1} for s in (42, 1)}
    good = {s: {"qwk": 0.90, "mae": 0.6} for s in (42, 1)}
    v = jev_eval.rule_verdict(good, sk, regrade_exact=0.95, p95_ms=400, cost_jev=1e-4,
                              cost_cheaper=6e-4, kp_macro_f1=0.80)
    assert v["pass"] and v["kp_verdicts_replace_classifier"]
    # one seed short of the margin fails quality
    mixed = {**good, 1: {"qwk": 0.84, "mae": 0.6}}
    assert not jev_eval.rule_verdict(mixed, sk, 0.95, 400, 1e-4, 6e-4)["pass"]
    # each other clause fails on its own
    assert not jev_eval.rule_verdict(good, sk, 0.80, 400, 1e-4, 6e-4)["pass"]
    assert not jev_eval.rule_verdict(good, sk, 0.95, 1500, 1e-4, 6e-4)["pass"]
    assert not jev_eval.rule_verdict(good, sk, 0.95, 400, 7e-4, 6e-4)["pass"]
    assert not jev_eval.rule_verdict(good, sk, None, 400, 1e-4, 6e-4)["pass"]
    assert not jev_eval.rule_verdict(good, sk, 0.95, 400, 1e-4, 6e-4,
                                     kp_macro_f1=0.70)["kp_verdicts_replace_classifier"]


def test_kp_eval_matches_train_definition():
    m = jev_eval.kp_eval(["hit", "miss", "partial", "hit"], ["hit", "miss", "hit", "hit"])
    assert m["acc3"] == 0.75 and 0 < m["macro_f1"] <= 1 and 0 < m["hit_f1"] <= 1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all jev tests passed")
