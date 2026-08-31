"""The mock interview: an experience/project deep-dive round driven by a
job description (pasted or defaulted), the candidate's resume, and the
question banks (docs/mock_interview_plan.md §6–§8a).

    templates   default JD templates (role family × level, domain slot)
    schemas     structured-output schemas for roles / plan / report
    engine      engine selection for mock endpoints + the offline fake engine
    planning    roles proposal and the hidden interview plan (+ rubric retrieval)
    turns       the phase machine and the turn protocol (text + trailer JSON)
    metrics     deterministic communication metrics
    report      the end-of-session scorecard (rubric §8a) + kp verdicts
    routes      HTTP dispatch for /api/mock/*

Phase 1 scope: text mode. The session is stateless server-side — the client
holds {plan, transcript, settings} and sends them per request, matching the
rest of the app. The plan contains the interviewer's hidden probe targets;
a curious user can read it in devtools, which is acceptable for a personal
practice tool and documented in docs/backend.md.
"""
