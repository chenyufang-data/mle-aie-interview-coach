"""Offline tests for the mock interview (coach/mock/*), on the fake engine.

Run:  .venv\\Scripts\\python tests\\test_mock.py

Everything here runs without a network or an API key: config.MODE is forced
to "mock", so the deterministic fake engine serves the roles / plan / report
calls and the phase machine drives canned turns — the same path the free
demo uses.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coach import config, grading, kb  # noqa: E402

config.MODE = "mock"
kb.load_chunks()
grading.load_grader()

from coach.mock import engine as engines  # noqa: E402
from coach.mock import metrics, planning, report, turns  # noqa: E402
from coach.mock.schemas import DIMENSIONS, REPORT_SCHEMA, ROLES_SCHEMA  # noqa: E402
from coach.mock.templates import TEMPLATES, catalog, render  # noqa: E402

RESUME = """Chen Yu — Machine Learning Engineer.
Project: fraud detection with XGBoost and LightGBM on imbalanced data; used
SMOTE and class weights, tuned the decision threshold, evaluated with PR-AUC
and recall at fixed precision; features from pandas pipelines with
target encoding, validated with TimeSeriesSplit to avoid leakage.
Project: interview-coach RAG app with BM25 retrieval, prompt engineering,
a distilled sklearn grader (QWK 0.93 vs the teacher) and DeepSeek routing.
Skills: Python, scikit-learn, PCA, k-means clustering, cross-validation, SQL, Docker, EC2."""

ROLE = {"title": "Machine Learning Engineer", "level": "Mid-level",
        "domain": "fraud detection", "template_id": "mle"}


def test_templates():
    assert set(TEMPLATES) == {"mle", "aie", "ds", "platform", "applied_sci"}
    body = render("mle", "Senior", "fraud detection")
    assert "Senior" in body and "fraud detection" in body and "{" not in body
    entries = catalog()
    assert len(entries) == 5 and all(e["body"] for e in entries)


def test_phase_machine():
    for length, expected in (("short", 7), ("standard", 11)):
        total = turns.total_turns(length)
        assert total == expected, (length, total)
        state = {"plan": {"settings": {"length": length}}, "transcript": []}
        seen = []
        for i in range(total):
            phase, turn_no, done = turns.position(state)
            assert not done and turn_no == i + 1
            seen.append(phase)
            state["transcript"].append({"turn": turn_no, "question": "q", "answer": "a"})
        phase, _, done = turns.position(state)
        assert done and phase == "done"
        assert seen[0] == "warmup" and seen[1] == "walkthrough" and seen[-1] == "closing"
        assert "behavioral" in seen
        # Phases never run out of order.
        order = [turns.PHASE_ORDER.index(p) for p in seen]
        assert order == sorted(order), seen


def test_trailer_parsing():
    spoken, meta = turns.parse_turn_output(
        'Why did you pick XGBoost?\n---\n{"probe_id": "probe_2", "rationale": "next target"}')
    assert spoken == "Why did you pick XGBoost?" and meta["probe_id"] == "probe_2"
    spoken, meta = turns.parse_turn_output("Just a question, no trailer.")
    assert spoken == "Just a question, no trailer." and meta == {}
    spoken, meta = turns.parse_turn_output("Question.\n---\n{not json")
    assert spoken == "Question." and meta == {}
    assert turns.parse_turn_output("")[0] == ""


def test_metrics():
    transcript = [
        {"question": "q1", "answer": "Um, so basically we used XGBoost because, "
                                     "you know, it handled the imbalance well.",
         "answer_ms": 30000},
        {"question": "q2", "answer": "I tuned the threshold on the validation set "
                                     "and reported PR-AUC.", "answer_ms": 15000},
        {"question": "q3", "answer": ""},
    ]
    m = metrics.compute(transcript)
    assert m["answers"] == 2
    assert m["filler_total"] >= 3          # um, basically, you know
    assert m["avg_answer_seconds"] == 22.5
    assert m["words_per_minute"] > 0 and m["longest_words"] > m["shortest_words"]
    assert metrics.compute([{"question": "q", "answer": ""}]) == {"answers": 0}


def test_full_offline_flow():
    engine = "fake"
    roles = planning.propose_roles(RESUME, "", engine)
    assert roles["roles"] and roles["profile"]["title"]
    for role in roles["roles"]:
        assert role["template_id"] in TEMPLATES

    jd_text, jd_is_default = planning.resolve_jd({"role": ROLE})
    assert jd_is_default and "Machine Learning Engineer" in jd_text

    plan = planning.build_plan(RESUME, jd_text, ROLE, None,
                               {"style": "tough", "length": "short"}, engine)
    assert 3 <= len(plan["probe_targets"]) <= 8
    assert plan["settings"] == {"style": "tough", "length": "short", "level": "Mid-level"}
    grounded = [t for t in plan["probe_targets"] if t.get("chunk_id")]
    assert grounded, "no probe matched a bank rubric — RUBRIC_MIN_SCORE too high?"
    assert all(t["chunk_id"] in kb.CHUNKS_BY_ID for t in grounded)
    # No chunk is used twice.
    ids = [t["chunk_id"] for t in grounded]
    assert len(ids) == len(set(ids))

    # Play the whole interview against the fake interviewer.
    state = {"plan": plan, "role": ROLE, "transcript": []}
    total = turns.total_turns("short")
    for i in range(total):
        turn = turns.next_turn(state, engine)
        assert not turn["done"] and turn["question"], turn
        answer = ("I chose XGBoost over logistic regression because the interactions "
                  "were non-linear; PR-AUC went from 0.61 to 0.74 and recall at 90% "
                  "precision doubled. The mistake I made was leaking future data "
                  "before switching to TimeSeriesSplit.")
        state["transcript"].append({"turn": turn["turn"], "phase": turn["phase"],
                                    "question": turn["question"],
                                    "probe_id": turn.get("probe_id"),
                                    "answer": answer, "answer_ms": 42000})
    final = turns.next_turn(state, engine)
    assert final["done"]

    result = report.build_report({"plan": plan, "role": ROLE,
                                  "transcript": state["transcript"]}, engine)
    assessment = result["assessment"]
    assert set(assessment["scores"]) == set(DIMENSIONS)
    assert all(1 <= v <= 10 for v in assessment["scores"].values())
    assert assessment["overall_call"] in ("strong hire", "hire", "lean hire", "no hire")
    assert len(assessment["top_actions"]) == 3
    assert result["metrics"]["answers"] == total
    if grading.GRADER is not None and grading.GRADER.get("kp_classifier") is not None:
        asked_grounded = [e for e in state["transcript"]
                          if e.get("probe_id") in {t["id"] for t in grounded}]
        assert len(result["kp_verdicts"]) == len(asked_grounded)
        for verdict in result["kp_verdicts"]:
            assert verdict["hits"] or verdict["partials"] or verdict["misses"]


def test_fake_schema_dispatch():
    assert "roles" in engines.fake_structured("resume with XGBoost", ROLES_SCHEMA)
    assert "per_turn" in engines.fake_structured("Candidate: hi", REPORT_SCHEMA)
    try:
        engines.fake_structured("x", {"properties": {"unknown": {}}})
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_free_tier_refused():
    # With tiers enabled and a free user, the mock is unavailable (needs an LLM).
    from coach import users
    from coach.mock.engine import MockUnavailable, pick_engine
    original_mode, original_tiers = config.MODE, users.TIERS_ENABLED
    try:
        config.MODE = "claude"
        users.TIERS_ENABLED = True
        try:
            pick_engine({"key": None, "name": "anon", "tier": "free", "log": True})
            raise AssertionError("expected MockUnavailable")
        except MockUnavailable:
            pass
    finally:
        config.MODE, users.TIERS_ENABLED = original_mode, original_tiers


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("all mock interview tests passed")
