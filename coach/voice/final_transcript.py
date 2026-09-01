"""Post-session re-transcription of per-answer recordings (plan section 4a).

The report grades the *final* transcript. Engine order: Scribe v2 batch +
the full-lexicon keyterms is the Phase 0 winner (3.0% lenient TER on the
human set - nothing met the pre-registered 3% bar; this was the best), so
it stays first while an ElevenLabs key is configured; Deepgram Nova-3
batch + the capped keyterm policy is next (the vendor this project is
moving to - UNMEASURED on the human set, and the report says so via
`engine`); local faster-whisper with prompt biasing last (Level 3 -
13.4%). FINAL_STT=scribe|deepgram|whisper forces one.

Per-answer clips rather than one session file: the client knows the turn
boundaries, so alignment is free, and each clip costs at most a few of
the vendor's billing minimums (a whole session lands around $0.02-0.07 -
plan section 5a).
"""

import io
import json
import os
import time
from urllib import request as urlrequest

_MIME_EXT = {"audio/webm": ".webm", "audio/ogg": ".ogg",
             "audio/mp4": ".m4a", "audio/wav": ".wav"}


def available_engine():
    """Which final-transcript engine this deployment can offer."""
    forced = os.environ.get("FINAL_STT")
    if forced:
        return {"scribe": "scribe_batch_kt", "deepgram": "nova3_batch_kt",
                "whisper": "whisper_local"}.get(forced, forced)
    if os.environ.get("ELEVENLABS_API_KEY"):
        return "scribe_batch_kt"
    if os.environ.get("DEEPGRAM_API_KEY"):
        return "nova3_batch_kt"
    try:
        import faster_whisper  # noqa: F401
        return "whisper_local"
    except ImportError:
        return None


def transcribe_final(audio_bytes, mime="audio/webm"):
    """{"text", "engine", "seconds"} for one answer clip."""
    from coach.voice.keyterms import (capped_transcript_keyterms,
                                      final_transcript_keyterms)
    started = time.perf_counter()
    mime = (mime or "audio/webm").split(";")[0].strip().lower()
    engine = available_engine()
    if engine == "scribe_batch_kt":
        from elevenlabs import ElevenLabs
        client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"], timeout=120)
        response = client.speech_to_text.convert(
            model_id="scribe_v2", language_code="en",
            file=("answer" + _MIME_EXT.get(mime, ".webm"), audio_bytes,
                  mime or "application/octet-stream"),
            keyterms=final_transcript_keyterms())
        return {"text": (response.text or "").strip(), "engine": "scribe_batch_kt",
                "seconds": round(time.perf_counter() - started, 2)}
    if engine == "nova3_batch_kt":
        from urllib.parse import urlencode
        from coach.voice.stt import DEEPGRAM_API
        params = [("model", "nova-3"), ("language", "en"),
                  ("smart_format", "true")]
        params += [("keyterm", term) for term in capped_transcript_keyterms()]
        req = urlrequest.Request(
            f"{DEEPGRAM_API}/v1/listen?{urlencode(params)}", data=audio_bytes,
            headers={"Content-Type": mime or "application/octet-stream",
                     "Authorization": f"Token {os.environ['DEEPGRAM_API_KEY']}"},
            method="POST")
        with urlrequest.urlopen(req, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
        alternatives = (((payload.get("results") or {}).get("channels")
                         or [{}])[0].get("alternatives") or [{}])
        return {"text": (alternatives[0].get("transcript") or "").strip(),
                "engine": "nova3_batch_kt",
                "seconds": round(time.perf_counter() - started, 2)}
    if engine != "whisper_local":
        raise RuntimeError(
            "No STT available for the final transcript: set DEEPGRAM_API_KEY "
            "or ELEVENLABS_API_KEY, or pip install -r requirements-stt.txt")
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
