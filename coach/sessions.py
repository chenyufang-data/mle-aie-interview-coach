"""Practice-session logging: gold pairs (LLM-graded) and unlabeled free tier."""

import json
from datetime import datetime

from coach import config
from coach.config import FREE_SESSIONS_PATH, REAL_SESSIONS_PATH
from coach.llm import engine_model


def log_real_session(data, result, user=None, engine=None):
    """Persist a real graded exchange. Local file only; never breaks a response.

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
        REAL_SESSIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with REAL_SESSIONS_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        print(f"Warning: could not log session ({exc})")


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
        FREE_SESSIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with FREE_SESSIONS_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        print(f"Warning: could not log free session ({exc})")
