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


# ------------------------------------------------------- the runtime route

def _runtime_setup():
    import os
    import tempfile
    from coach import config, grading, store, users
    store.use(store.FileStore())
    tmp = Path(tempfile.mkdtemp(prefix="coach_jev_"))
    config.USERS_PATH = tmp / "users.json"
    config.USAGE_PATH = tmp / "usage.json"
    config.MODE = "claude"
    config.JEV_DAILY_CAP = 2
    os.environ["TYPESAFE_API_KEY"] = "dummy-never-sent"
    if grading.GRADER is None:
        grading.load_grader()
    return config, grading, users


def _bank_chunk():
    import json
    path = Path(__file__).resolve().parents[1] / "banks" / "rag_ml" / "all_chunks.jsonl"
    return json.loads(path.read_text(encoding="utf-8").splitlines()[0])


def test_calibration_matches_isotonic_interp():
    import numpy as np
    from coach import jev
    cal = jev.calibration()
    assert cal and cal["model"] and len(cal["x"]) == len(cal["y"]) >= 2
    assert cal["y"] == sorted(cal["y"]), "isotonic: non-decreasing"
    for v in np.linspace(0.0, 11.0, 221):
        assert abs(jev.calibrate(float(v)) - float(np.interp(v, cal["x"], cal["y"]))) < 1e-9


def test_route_switches():
    import os
    from coach import jev
    config, _, _ = _runtime_setup()
    assert jev.enabled() and jev.model() == jev.calibration()["model"]
    config.MODE = "mock"
    assert not jev.enabled(), "--mock promises offline: never call Jev"
    config.MODE = "claude"
    key = os.environ.pop("TYPESAFE_API_KEY")
    assert not jev.enabled()
    os.environ["TYPESAFE_API_KEY"] = key


def test_grading_uses_jev_then_falls_back():
    from coach import jev
    config, grading, users = _runtime_setup()
    chunk = _bank_chunk()
    n = len(chunk["interview"]["key_points"])
    verdicts = (["hit", "partial", "miss"] * n)[:n]
    calls = []

    def fake_call(c, answer, timeout=None):
        calls.append(answer)
        return {"grade": 7.0, "grade_confidence": 0.8, "grade_probs": [0.0] * 10,
                "kp": [{"choice": v, "confidence": 0.9, "probs": {}} for v in verdicts]}

    real_call = jev.call
    jev.call = fake_call
    try:
        anon = users.resolve_key(None)
        data = {"answer": "Ridge shrinks all coefficients and keeps every feature."}
        r = grading.mock_evaluation(data, chunk, "free", user=anon)
        assert "Jev" in r["graded_by"], r["graded_by"]
        assert r["overall_score"] == round(jev.calibrate(7.0))
        hits = sum(v == "hit" for v in verdicts)
        assert f"matched {hits} of {n}" in r["summary"]
        assert grading.mock_evaluation(data, chunk, "free", user=anon)["graded_by"].startswith("Jev")
        # third call: the cap of 2 is spent -> the sklearn grade, no call made
        r3 = grading.mock_evaluation(data, chunk, "free", user=anon)
        assert "local ML grader" in r3["graded_by"] and len(calls) == 2
        # no user (the cascade path) never calls Jev
        store_reset = __import__("coach.store", fromlist=["current"]).current().reset_usage
        store_reset()
        assert "local ML grader" in grading.mock_evaluation(data, chunk, "cascade")["graded_by"]
        # a failing call falls back and pauses the route
        def boom(*a, **k):
            raise OSError("network down")
        jev.call = boom
        r4 = grading.mock_evaluation(data, chunk, "free", user=anon)
        assert "local ML grader" in r4["graded_by"] and not jev.available()
    finally:
        jev.call = real_call
        jev._unavailable_until = 0.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all jev tests passed")
