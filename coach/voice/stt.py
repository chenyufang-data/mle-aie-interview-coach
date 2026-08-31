"""Live STT sessions behind one interface (plan section 4a).

  local       in-process faster-whisper (large-v-3-turbo on the GPU; the
              Phase 0 condition-6 stack), keyterms via initial_prompt
  speaches    a Speaches container's OpenAI-style /v1/audio/transcriptions
  elevenlabs  Scribe v2 Realtime over one session-long WebSocket, MANUAL
              commits segmenting the stream per turn; keyterms per session
              (the policy-50 list - measured 9.7% vs 15.4% lenient TER)

The loop calls send() for every incoming frame (only the realtime backend
consumes them; wants_frames says so) and commit(utterance_pcm) at
end-of-turn; buffered backends transcribe the endpointer's utterance,
the realtime backend commits what it already heard - that is what makes
its finalize latency ~0.15 s instead of a full transcription pass.
"""

import asyncio
import io
import json
import os
import struct
import threading
import time
import uuid
from pathlib import Path
from urllib import request as urlrequest

import numpy as np

from coach.config import BASE_DIR

SPEACHES_URL = os.environ.get("SPEACHES_URL", "http://127.0.0.1:8969/v1")
# The CT2 conversion of large-v3-turbo that Speaches' registry actually
# carries (Systran never published a turbo conversion).
SPEACHES_STT_MODEL = os.environ.get(
    "SPEACHES_STT_MODEL", "jootanehorror/faster-whisper-large-v3-turbo-ct2")
WHISPER_MODEL_NAME = os.environ.get("STT_WHISPER_MODEL", "large-v3-turbo")
# Beam 1-2 for the live loop (latency); Phase 0 measured accuracy at beam 5,
# which the offline final-transcript path keeps.
WHISPER_BEAM = int(os.environ.get("STT_BEAM", "2"))

_WHISPER = None
_WHISPER_LOCK = threading.Lock()


def _add_cuda_dlls():
    """GPU DLLs from the pip nvidia-* wheels (no system CUDA install)."""
    try:
        import nvidia.cublas
        import nvidia.cudnn
    except ImportError:
        return
    for pkg in (nvidia.cublas, nvidia.cudnn):
        bin_dir = Path(pkg.__path__[0]) / "bin"
        if bin_dir.exists():
            os.add_dll_directory(str(bin_dir))
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ["PATH"]


def whisper_singleton():
    """(model, device_label), loaded once per process."""
    global _WHISPER
    with _WHISPER_LOCK:
        if _WHISPER is None:
            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
            _add_cuda_dlls()
            from faster_whisper import WhisperModel
            try:
                model = WhisperModel(WHISPER_MODEL_NAME, device="cuda",
                                     compute_type="float16")
                device = "cuda/float16"
            except Exception as exc:
                print(f"voice: GPU unavailable ({str(exc)[:80]}); Whisper on CPU int8")
                model = WhisperModel(WHISPER_MODEL_NAME, device="cpu",
                                     compute_type="int8")
                device = "cpu/int8"
            _WHISPER = (model, device)
    return _WHISPER


def wav_bytes(pcm, rate=16000):
    """A minimal mono 16-bit WAV container around raw PCM."""
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI", b"RIFF", 36 + len(pcm), b"WAVE", b"fmt ",
        16, 1, 1, rate, rate * 2, 2, 16, b"data", len(pcm))
    return header + pcm


class BufferedSTT:
    """Shared shape for backends that transcribe the utterance at commit."""

    wants_frames = False

    def __init__(self, keyterms):
        self.keyterms = list(keyterms or [])

    async def send(self, frame):
        pass

    async def close(self):
        pass


class LocalWhisperSTT(BufferedSTT):
    label = "whisper_local"

    async def commit(self, utterance_pcm):
        started = time.perf_counter()
        text = await asyncio.to_thread(self._transcribe, utterance_pcm)
        return text, time.perf_counter() - started

    def _transcribe(self, pcm):
        model, _device = whisper_singleton()
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        segments, _info = model.transcribe(
            samples, language="en", beam_size=WHISPER_BEAM,
            initial_prompt=", ".join(self.keyterms) or None,
            condition_on_previous_text=False, vad_filter=False)
        return " ".join(segment.text.strip() for segment in segments).strip()


class SpeachesSTT(BufferedSTT):
    label = "speaches"

    async def commit(self, utterance_pcm):
        started = time.perf_counter()
        text = await asyncio.to_thread(self._transcribe, utterance_pcm)
        return text, time.perf_counter() - started

    def _transcribe(self, pcm):
        boundary = uuid.uuid4().hex
        parts = []
        for name, value in (("model", SPEACHES_STT_MODEL), ("language", "en"),
                            ("prompt", ", ".join(self.keyterms))):
            if value:
                parts.append(f"--{boundary}\r\nContent-Disposition: form-data; "
                             f"name=\"{name}\"\r\n\r\n{value}\r\n".encode())
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; "
                     f"name=\"file\"; filename=\"a.wav\"\r\n"
                     f"Content-Type: audio/wav\r\n\r\n".encode())
        body = b"".join(parts) + wav_bytes(pcm) + f"\r\n--{boundary}--\r\n".encode()
        req = urlrequest.Request(
            f"{SPEACHES_URL}/audio/transcriptions", data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST")
        with urlrequest.urlopen(req, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return (payload.get("text") or "").strip()


class ScribeRealtimeSTT:
    """One Scribe v2 Realtime connection for the whole session; MANUAL
    commits segment it into turns (the Phase 0 live condition)."""

    wants_frames = True
    label = "scribe_realtime"

    def __init__(self, keyterms):
        self.keyterms = list(keyterms or [])
        self.connection = None
        self._committed = None       # future for the in-flight commit

    async def _connect(self):
        from elevenlabs import ElevenLabs
        from elevenlabs.realtime.connection import RealtimeEvents
        from elevenlabs.realtime.scribe import AudioFormat, CommitStrategy
        key = os.environ.get("ELEVENLABS_API_KEY")
        if not key:
            raise RuntimeError("ELEVENLABS_API_KEY is not set (.env)")
        client = ElevenLabs(api_key=key, timeout=120)
        options = {"model_id": "scribe_v2_realtime",
                   "audio_format": AudioFormat.PCM_16000, "sample_rate": 16000,
                   "commit_strategy": CommitStrategy.MANUAL,
                   "language_code": "en"}
        if self.keyterms:
            options["keyterms"] = self.keyterms
        self.connection = await client.speech_to_text.realtime.connect(options)

        def on_committed(data):
            future = self._committed
            if future is not None and not future.done():
                future.set_result(data)

        def on_error(data):
            future = self._committed
            if future is not None and not future.done():
                future.set_exception(RuntimeError(json.dumps(data)[:300]))

        self.connection.on(RealtimeEvents.COMMITTED_TRANSCRIPT, on_committed)
        self.connection.on(RealtimeEvents.ERROR, on_error)

    async def send(self, frame):
        import base64
        if self.connection is None:
            await self._connect()
        await self.connection.send(
            {"audio_base_64": base64.b64encode(bytes(frame)).decode()})

    async def commit(self, utterance_pcm):
        import base64
        if self.connection is None:
            # Nothing was streamed (e.g. a reconnect): send the utterance now.
            await self._connect()
            chunk = 3200
            for i in range(0, len(utterance_pcm), chunk):
                await self.connection.send({"audio_base_64": base64.b64encode(
                    utterance_pcm[i:i + chunk]).decode()})
        # Trailing silence so the endpoint sees the last word end.
        await self.connection.send(
            {"audio_base_64": base64.b64encode(b"\x00" * 3200 * 5).decode()})
        self._committed = asyncio.get_running_loop().create_future()
        started = time.perf_counter()
        await self.connection.commit()
        try:
            data = await asyncio.wait_for(self._committed, 20)
        finally:
            self._committed = None
        text = (data.get("text") or data.get("transcript") or "").strip()
        return text, time.perf_counter() - started

    async def close(self):
        if self.connection is not None:
            try:
                await self.connection.close()
            except Exception:
                pass
            self.connection = None


def make_stt(backend, keyterms):
    if backend == "local":
        return LocalWhisperSTT(keyterms)
    if backend == "speaches":
        return SpeachesSTT(keyterms)
    if backend == "elevenlabs":
        return ScribeRealtimeSTT(keyterms)
    raise ValueError(f"unknown AUDIO_BACKEND {backend!r} "
                     "(expected local, speaches, or elevenlabs)")
