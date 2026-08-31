"""Post-session re-transcription of per-answer recordings (plan section 4a).

The report grades the *final* transcript. The Phase 0 winner produces it:
Scribe v2 batch + the full-lexicon keyterms (3.0% lenient TER on the human
set - nothing met the pre-registered 3% bar; this was the best) whenever an
ElevenLabs key is configured; local faster-whisper with prompt biasing
otherwise (Level 3 - 13.4%, and the report says so via `engine`).

Per-answer clips rather than one session file: the client knows the turn
boundaries, so alignment is free, and each clip costs at most a few of
Scribe's 20-second billing minimums (a whole session lands around
$0.02-0.07 - plan section 5a).
"""

import io
import os
import time

_MIME_EXT = {"audio/webm": ".webm", "audio/ogg": ".ogg",
             "audio/mp4": ".m4a", "audio/wav": ".wav"}


def available_engine():
    """Which final-transcript engine this deployment can offer."""
    if os.environ.get("ELEVENLABS_API_KEY"):
        return "scribe_batch_kt"
    try:
        import faster_whisper  # noqa: F401
        return "whisper_local"
    except ImportError:
        return None


def transcribe_final(audio_bytes, mime="audio/webm"):
    """{"text", "engine", "seconds"} for one answer clip."""
    from coach.voice.keyterms import final_transcript_keyterms
    started = time.perf_counter()
    mime = (mime or "audio/webm").split(";")[0].strip().lower()
    if os.environ.get("ELEVENLABS_API_KEY"):
        from elevenlabs import ElevenLabs
        client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"], timeout=120)
        response = client.speech_to_text.convert(
            model_id="scribe_v2", language_code="en",
            file=("answer" + _MIME_EXT.get(mime, ".webm"), audio_bytes,
                  mime or "application/octet-stream"),
            keyterms=final_transcript_keyterms())
        return {"text": (response.text or "").strip(), "engine": "scribe_batch_kt",
                "seconds": round(time.perf_counter() - started, 2)}
    if available_engine() is None:
        raise RuntimeError(
            "No STT available for the final transcript: set ELEVENLABS_API_KEY "
            "or pip install -r requirements-stt.txt")
    from faster_whisper.audio import decode_audio

    from coach.voice.stt import whisper_singleton
    samples = decode_audio(io.BytesIO(audio_bytes), sampling_rate=16000)
    model, _device = whisper_singleton()
    # initial_prompt is Whisper's weaker keyterm biasing (Phase 0 condition
    # 6): keep it under the ~224-token prompt window.
    prompt = ", ".join(final_transcript_keyterms())[:800]
    segments, _info = model.transcribe(
        samples, language="en", beam_size=5, initial_prompt=prompt,
        condition_on_previous_text=False, vad_filter=False)
    text = " ".join(segment.text.strip() for segment in segments).strip()
    return {"text": text, "engine": "whisper_local",
            "seconds": round(time.perf_counter() - started, 2)}
