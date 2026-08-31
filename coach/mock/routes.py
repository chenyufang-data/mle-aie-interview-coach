"""HTTP dispatch for /api/mock/* (wired from coach/http.py).

Stateless: every request carries what it needs; nothing about a session is
stored server-side (logging arrives with the Phase 3 opt-in). Free tier is
refused with a clear message — the mock needs a real LLM (plan §7d) — while
--mock mode serves the deterministic offline demo engine.
"""

from coach import grading, users
from coach.mock import engine as engines
from coach.mock import planning, report, turns
from coach.mock.templates import catalog
from coach.web import json_response


def handle_get(handler, path):
    if path == "/api/mock/templates":
        json_response(handler, 200, {"templates": catalog(),
                                     "styles": sorted(planning.STYLES),
                                     "lengths": {name: turns.total_turns(name)
                                                 for name in turns.PHASE_QUOTAS}})
        return
    if path == "/api/mock/voice":
        # Voice capabilities: whether the ws loop is up, which audio stack
        # it runs, and which final-transcript engine this deployment offers.
        from coach import config
        from coach.voice import loop as voice_loop
        from coach.voice.final_transcript import available_engine
        json_response(handler, 200, {
            "enabled": config.VOICE_ENABLED,
            "ws_port": voice_loop.VOICE_PORT if config.VOICE_ENABLED else None,
            "audio_backend": voice_loop.audio_backend(),
            "final_stt": available_engine(),
        })
        return
    json_response(handler, 404, {"error": "Unknown endpoint."})


def handle_post(handler, path, data):
    try:
        user = users.resolve_user(handler)
        engine = engines.pick_engine(user, force_llm=bool(data.get("force_llm")))
    except engines.MockUnavailable as exc:
        json_response(handler, 403, {"error": str(exc)})
        return

    if path == "/api/mock/roles":
        resume = (data.get("resume") or "").strip()
        if len(resume) < 80:
            json_response(handler, 400,
                          {"error": "Paste the resume text first (resume_parser.py "
                                    "turns a PDF or .docx into text)."})
            return
        result = planning.propose_roles(resume, (data.get("jd_text") or "").strip(), engine)
        result["engine"] = engine
        json_response(handler, 200, result)
        return

    if path == "/api/mock/start":
        resume = (data.get("resume") or "").strip()
        role = data.get("role") or {}
        if len(resume) < 80 or not role:
            json_response(handler, 400, {"error": "resume and role are required."})
            return
        jd_text, jd_is_default = planning.resolve_jd(data)
        plan = planning.build_plan(resume, jd_text, role, data.get("project"),
                                   data.get("settings") or {}, engine)
        state = {"plan": plan, "role": role, "transcript": []}
        first = turns.next_turn(state, engine)
        json_response(handler, 200, {"plan": plan, "jd_text": jd_text,
                                     "jd_is_default": jd_is_default,
                                     "turn": first, "engine": engine,
                                     "total_turns": turns.total_turns(
                                         plan["settings"]["length"])})
        return

    if path == "/api/mock/turn":
        state = {"plan": data.get("plan") or {}, "role": data.get("role") or {},
                 "transcript": data.get("transcript") or []}
        if not state["plan"].get("probe_targets"):
            json_response(handler, 400, {"error": "plan is required (call /api/mock/start)."})
            return
        json_response(handler, 200, turns.next_turn(state, engine))
        return

    if path == "/api/mock/report":
        if not (data.get("transcript") or []):
            json_response(handler, 400, {"error": "transcript is required."})
            return
        report_engine = data.get("report_engine") or _default_report_engine(engine)
        if grading.GRADER is None:
            grading.load_grader()  # kp verdicts want the artifact even in Claude mode
        result = report.build_report(data, report_engine)
        result["engine"] = report_engine
        json_response(handler, 200, result)
        return

    if path == "/api/mock/keyterms":
        # Deterministic (no LLM): the measured section-7a policy picks the
        # <= 50 realtime keyterms from this session's own material.
        from coach.voice import keyterms as keyterms_module
        chosen = keyterms_module.session_keyterms(
            resume=(data.get("resume") or ""),
            role=data.get("role"), project=data.get("project"))
        json_response(handler, 200, {"keyterms": chosen})
        return

    if path == "/api/mock/transcribe":
        # Final-transcript re-transcription of one answer clip (plan 4a).
        audio_b64 = data.get("audio_base64") or ""
        if not audio_b64:
            json_response(handler, 400, {"error": "audio_base64 is required."})
            return
        if len(audio_b64) > 34_000_000:  # ~25 MB decoded
            json_response(handler, 400, {"error": "audio too large (25 MB cap)."})
            return
        import base64
        from coach.voice import final_transcript
        try:
            audio = base64.b64decode(audio_b64)
            result = final_transcript.transcribe_final(
                audio, data.get("mime") or "audio/webm")
        except Exception as exc:
            json_response(handler, 500, {"error": f"re-transcription failed: {exc}"})
            return
        json_response(handler, 200, result)
        return

    json_response(handler, 404, {"error": "Unknown endpoint."})


def _default_report_engine(turn_engine):
    """Plan §7d: the report defaults to Claude when a key is present (its
    regrade consistency makes scorecards comparable across sessions);
    otherwise it stays on the turn engine."""
    from coach.llm import get_api_key
    if turn_engine == "fake":
        return "fake"
    if get_api_key():
        return "claude"
    return turn_engine
