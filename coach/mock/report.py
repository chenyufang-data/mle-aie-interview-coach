"""The end-of-session report (plan §8/§8a): three visibly separate blocks.

1. Computed communication metrics — deterministic (metrics.py).
2. Per-key-point verdicts on rubric-grounded probes — the distilled kp
   classifier (free, deterministic; the same instrument Phase 0 used).
3. The LLM scorecard — six anchored dimensions, per-turn feedback, red
   flags that quote the candidate, and a hiring call. Default engine is
   Claude when a key is present (95% regrade consistency vs Flash's
   53–57% is what makes session-to-session comparison meaningful);
   the request can override with report_engine.
"""

from coach import grading
from coach.mock import engine as engines
from coach.mock import metrics as metrics_module
from coach.mock import transcription
from coach.mock.schemas import REPORT_SCHEMA

ANCHORS = """Score each dimension 1-10 against these anchors:
- technical_depth: 3 = buzzwords, no mechanism; 6 = correct but stays at "what"; 9 = precise "why", trade-offs, limits.
- ownership_specificity: 3 = "we did", no numbers; 6 = own role clear, some numbers; 9 = "I decided X because Y", concrete figures, honest about others' parts.
- structure: 3 = wanders, answers a different question; 6 = mostly context -> action -> result; 9 = tight STAR, answers exactly what was asked.
- communication_clarity: 3 = undefined jargon, lost thread; 6 = understandable with effort; 9 = a smart non-expert follows it.
- concision_time: 3 = 3-minute answers to 30-second questions; 6 = mostly proportionate; 9 = proportionate (the computed metrics corroborate).
- reflection: 3 = no failures, nothing to change; 6 = names a problem; 9 = what failed, what was learned, what would be done differently.
Every justification and every red flag MUST quote the candidate's own words.
Red flags to check: contradictions between answers, claims that dissolve under
follow-up, inability to explain one's own resume line, blame-shifting. An empty
red_flags list is the correct output when there are none."""


def transcript_text(transcript):
    lines = []
    for entry in transcript:
        lines.append(f"Interviewer (turn {entry.get('turn', '?')}, "
                     f"{entry.get('phase', '?')}): {entry.get('question', '')}")
        answer = entry.get("answer_final") or entry.get("answer") or "(no answer)"
        lines.append(f"Candidate: {answer}")
    return "\n".join(lines)


def voice_note(transcript):
    """A grading caveat when answers arrived by voice."""
    if any(entry.get("answer_final") or entry.get("stt_seconds") is not None
           for entry in transcript):
        return ("Answers are speech transcripts (best available "
                "re-transcription): judge content and structure, never "
                "punctuation, casing or small transcription artifacts.\n\n")
    return ""


def kp_verdicts(plan, transcript):
    """Hit/partial/miss per bank rubric key point, for probes that were
    rubric-grounded and answered. Deterministic; needs the trained artifact."""
    from coach import kb
    grader = grading.GRADER
    if grader is None or grader.get("kp_classifier") is None:
        return []
    extractor = grader["extractor"]
    classifier = grader["kp_classifier"]
    targets = {t["id"]: t for t in plan.get("probe_targets", []) if t.get("chunk_id")}
    results = []
    for entry in transcript:
        target = targets.get(entry.get("probe_id"))
        # Final transcript when present; normalized for voice answers (the
        # Phase 0 surface-form decision), raw for typed ones.
        answer = transcription.grading_answer_text(entry)
        if not target or not answer:
            continue
        chunk = kb.CHUNKS_BY_ID.get(target["chunk_id"])
        if chunk is None:
            continue
        verdicts = list(classifier.predict(extractor.keypoint_features(answer, chunk)))
        points = chunk["interview"]["key_points"]
        results.append({
            "turn": entry.get("turn"),
            "probe_id": target["id"],
            "chunk_id": target["chunk_id"],
            "topic": target.get("topic", ""),
            "hits": [p for p, v in zip(points, verdicts) if v == "hit"],
            "partials": [p for p, v in zip(points, verdicts) if v == "partial"],
            "misses": [p for p, v in zip(points, verdicts) if v == "miss"],
        })
    return results


def _rubric_digest(plan):
    lines = []
    for target in plan.get("probe_targets", []):
        points = target.get("bank_key_points") or target.get("expected_points") or []
        grounded = "bank rubric" if target.get("chunk_id") else "expected specifics"
        lines.append(f"- {target['id']} ({target.get('topic', '')}; {grounded}): "
                     + "; ".join(points))
    return "\n".join(lines)


def build_report(data, engine):
    """data = {plan, role, transcript, resume?}. Returns the three blocks."""
    plan = data.get("plan") or {}
    transcript = data.get("transcript") or []
    computed = metrics_module.compute(transcript)
    verdicts = kp_verdicts(plan, transcript)
    trans = transcription.compare(transcript)
    if trans:
        flips = transcription.kp_flips(plan, transcript)
        if flips:
            trans["kp_flips"] = flips

    prompt = f"""Write the hiring-committee report for this mock interview
(experience/project deep-dive round) for a {data.get('role', {}).get('level', 'Mid-level')}
{data.get('role', {}).get('title', 'Machine Learning Engineer')} opening.

{ANCHORS}

The interviewer's per-probe rubrics (what a strong answer contained):
{_rubric_digest(plan)}

Deterministic communication metrics (use them to corroborate concision_time,
do not restate them all): {computed}

{voice_note(transcript)}Transcript:
{transcript_text(transcript)}

per_turn must cover every interviewer question in order, with every field at
most two sentences. stronger_answer lines are concise and interview-ready, in
the candidate's voice. top_actions has exactly three entries, most impactful
first."""

    assessment = engines.structured(prompt, REPORT_SCHEMA, engine)
    for name, value in list((assessment.get("scores") or {}).items()):
        assessment["scores"][name] = grading.clamp_score(int(value))
    result = {"assessment": assessment, "metrics": computed, "kp_verdicts": verdicts}
    if trans:
        result["transcription"] = trans
    return result
