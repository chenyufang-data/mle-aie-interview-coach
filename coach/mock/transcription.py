"""Two-transcript accounting for the report (plan sections 4a and 8).

A voice session produces two transcripts per answer: the *live* text the
interviewer acted on (Scribe Realtime or local Whisper) and the *final*
text (Scribe batch + the full lexicon, POST /api/mock/transcribe). The
report grades the final text - normalized on both sides for the distilled
kp classifier, the Phase 0 decision that neutralizes its measured
surface-form brittleness (>= 1 point moved on 50% of grades from
lowercasing alone) - shows both, and quantifies the difference:
"transcription cost you N terms and flipped M rubric verdicts", the
per-session instance of the Phase 0 damage number.
"""

from coach import grading


def _norm_tokens(text):
    from grader.stt_text import normalize
    return normalize(text or "")


def grading_answer_text(entry):
    """The text the graders see for one answer: the final transcript when
    present, else the live text; normalized for voice answers, raw for
    typed ones (no STT surface noise to neutralize there)."""
    text = (entry.get("answer_final") or entry.get("answer") or "").strip()
    if not text:
        return ""
    if entry.get("answer_final") or entry.get("stt_seconds") is not None:
        return " ".join(_norm_tokens(text))
    return text


def compare(transcript):
    """Live-vs-final differences; None when no entry carries both."""
    from grader.stt_text import edit_distance

    from coach.voice.keyterms import lexicon
    lex = lexicon()
    rows, errors, ref_len, term_fixes = [], 0, 0, 0
    for entry in transcript:
        live = (entry.get("answer") or "").strip()
        final = (entry.get("answer_final") or "").strip()
        if not live or not final:
            continue
        ref = _norm_tokens(final)
        hyp = _norm_tokens(live)
        if not ref:
            continue
        distance = edit_distance(ref, hyp)
        errors += distance
        ref_len += len(ref)
        fixed = []
        for term, spans in lex.occurrences(ref).items():
            missed = len(spans) - min(len(spans),
                                      lex.lenient_count(hyp, lex.by_term[term]))
            if missed > 0:
                fixed.append(term)
        term_fixes += len(fixed)
        rows.append({"turn": entry.get("turn"),
                     "wer_live_vs_final": round(distance / len(ref), 3),
                     "terms_fixed": sorted(fixed)})
    if not rows:
        return None
    return {
        "answers_compared": len(rows),
        "wer_live_vs_final": round(errors / max(1, ref_len), 3),
        "terms_fixed_total": term_fixes,
        "per_turn": rows,
    }


def kp_flips(plan, transcript):
    """Rubric-verdict flips between live and final text on grounded probes
    (both sides normalized). None without the trained artifact."""
    from coach import kb
    grader = grading.GRADER
    if grader is None or grader.get("kp_classifier") is None:
        return None
    extractor = grader["extractor"]
    classifier = grader["kp_classifier"]
    targets = {t["id"]: t for t in plan.get("probe_targets", []) if t.get("chunk_id")}
    checked, flips, detail = 0, 0, []
    for entry in transcript:
        target = targets.get(entry.get("probe_id"))
        live = (entry.get("answer") or "").strip()
        final = (entry.get("answer_final") or "").strip()
        if not target or not live or not final:
            continue
        chunk = kb.CHUNKS_BY_ID.get(target["chunk_id"])
        if chunk is None:
            continue
        live_text = " ".join(_norm_tokens(live))
        final_text = " ".join(_norm_tokens(final))
        live_verdicts = list(classifier.predict(
            extractor.keypoint_features(live_text, chunk)))
        final_verdicts = list(classifier.predict(
            extractor.keypoint_features(final_text, chunk)))
        points = chunk["interview"]["key_points"]
        checked += len(points)
        for point, live_v, final_v in zip(points, live_verdicts, final_verdicts):
            if live_v != final_v:
                flips += 1
                detail.append({"turn": entry.get("turn"), "point": point,
                               "live": live_v, "final": final_v})
    if not checked:
        return None
    return {"verdicts_checked": checked, "flips": flips, "detail": detail}
