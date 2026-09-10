"""Practice-session logging: gold pairs (LLM-graded), the unlabeled free
tier, and the opt-in mock log - appended through the state store
(coach/store.py: data/sessions/*.jsonl, or the session_log table)."""

from datetime import datetime

from coach import config, store
from coach.llm import engine_model


def log_real_session(data, result, user=None, engine=None):
    """Persist a real graded exchange. Never breaks a response.

    graded_by matters downstream: grader/evaluate_on_real.py keeps only
    "claude" rows as gold pairs, so DeepSeek-graded sessions never leak into
    the teacher-agreement evaluation.
    """
    if user is not None and not user.get("log", True):
        return
    try:
        engine = engine or config.MODE
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "user": (user or {}).get("name", "anonymous"),
            "graded_by": engine,
            "model": engine_model(engine),
            "role": data.get("role"),
            "level": data.get("level"),
            "topic": data.get("topic"),
            "source": data.get("source"),
            "chunk_id": data.get("chunk_id"),
            "question": data.get("question"),
            "answer": data.get("answer"),
            "timeUsed": data.get("timeUsed"),
            "evaluation": result,
        }
        store.current().append_session("real", record)
    except Exception as exc:
        print(f"Warning: could not log session ({exc})")


def log_mock_session(data, result, user=None, engine=None):
    """Phase 3 mock-interview logging - OPT-IN, unlike the practice logs:
    nothing is written unless the request itself carries log_consent (the
    UI checkbox, default off), because the plan and transcript embed
    resume-derived content. The per-user log flag still vetoes. Returns
    whether a record was written, so the response can say so."""
    if not data.get("log_consent"):
        return False
    if user is not None and not user.get("log", True):
        return False
    try:
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "user": (user or {}).get("name", "anonymous"),
            "report_engine": engine,
            "role": data.get("role"),
            "settings": (data.get("plan") or {}).get("settings"),
            "plan": data.get("plan"),
            "transcript": data.get("transcript"),
            "report": result,
        }
        store.current().append_session("mock", record)
        return True
    except Exception as exc:
        print(f"Warning: could not log mock session ({exc})")
        return False


def log_free_session(data, result, user, reason):
    """Persist a free-tier answer, unlabeled. The stored local_score is the
    student's prediction, recorded for triage only - never a training label."""
    if not user.get("log", True):
        return
    try:
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "user": user["name"],
            "graded_by": "local_ml",
            "reason": reason,
            "role": data.get("role"),
            "level": data.get("level"),
            "topic": data.get("topic"),
            "source": data.get("source"),
            "chunk_id": data.get("chunk_id"),
            "question": data.get("question"),
            "answer": data.get("answer"),
            "timeUsed": data.get("timeUsed"),
            "local_score": result.get("overall_score"),
            "teacher_score": None,
        }
        store.current().append_session("free", record)
    except Exception as exc:
        print(f"Warning: could not log free session ({exc})")
