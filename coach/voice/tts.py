"""Sentence TTS behind one interface (plan section 4a).

  local       Kokoro-82M via kokoro-onnx, in process (measured on this
              machine: RTF 0.30-0.45 on CPU - faster than playback)
  speaches    the same Kokoro inside a Speaches container, /v1/audio/speech
  deepgram    Aura-2 via /v1/speak, raw linear16 (~$0.030 per 1k chars;
              recommended cloud backend, one vendor with Nova-3 STT)
  elevenlabs  Flash v2.5, pcm_24000 output ($0.05 per 1k chars)

synth() returns one sentence's full audio: at Kokoro/Flash speeds the
first sentence is ready in ~0.2-0.5 s, and sentences stream one by one
from the chunker, so per-chunk streaming inside a sentence is not needed
to meet the section-3 budget. The "browser" audio backend never reaches
this module - the page speaks with speechSynthesis client-side.
"""

import asyncio
import io
import json
import os
import time
from pathlib import Path
from urllib import request as urlrequest

import numpy as np

from coach.config import BASE_DIR

MODELS_DIR = BASE_DIR / "data" / "models"
KOKORO_MODEL = Path(os.environ.get("KOKORO_MODEL", str(MODELS_DIR / "kokoro-v1.0.onnx")))
KOKORO_VOICES = Path(os.environ.get("KOKORO_VOICES", str(MODELS_DIR / "voices-v1.0.bin")))
KOKORO_RELEASE = ("https://github.com/thewh1teagle/kokoro-onnx/releases/"
                  "download/model-files-v1.0/")
DEFAULT_LOCAL_VOICE = os.environ.get("TTS_VOICE", "af_heart")
SPEACHES_URL = os.environ.get("SPEACHES_URL", "http://127.0.0.1:8969/v1")
SPEACHES_TTS_MODEL = os.environ.get(
    "SPEACHES_TTS_MODEL", "speaches-ai/Kokoro-82M-v1.0-ONNX")
# ElevenLabs premade voices, by the account's key (see grader/stt_eval.py).
DEFAULT_ELEVEN_VOICE = os.environ.get("ELEVEN_TTS_VOICE", "XrExE9yKIg1WjnnlVkGX")
DEEPGRAM_API = os.environ.get("DEEPGRAM_URL", "https://api.deepgram.com")
DEFAULT_DEEPGRAM_VOICE = os.environ.get("DEEPGRAM_TTS_VOICE", "aura-2-thalia-en")

_KOKORO = None


class KokoroTTS:
    label = "kokoro_local"
    sample_rate = 24000

    def __init__(self, voice=None):
        self.voice = voice or DEFAULT_LOCAL_VOICE

    def _model(self):
        global _KOKORO
        if _KOKORO is None:
            if not (KOKORO_MODEL.exists() and KOKORO_VOICES.exists()):
                raise RuntimeError(
                    "Kokoro model files are missing. Download "
                    f"{KOKORO_RELEASE}kokoro-v1.0.onnx and voices-v1.0.bin "
                    f"into {MODELS_DIR} (about 340 MB), or run the Speaches "
                    "container and set AUDIO_BACKEND=speaches.")
            from kokoro_onnx import Kokoro
            _KOKORO = Kokoro(str(KOKORO_MODEL), str(KOKORO_VOICES))
        return _KOKORO

    async def synth(self, text):
        return await asyncio.to_thread(self._synth, text)

    def _synth(self, text):
        model = self._model()
        samples, rate = model.create(text, voice=self.voice, speed=1.0)
        pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()
        return pcm, rate


class SpeachesTTS:
    label = "speaches"
    sample_rate = 24000

    def __init__(self, voice=None):
        self.voice = voice or DEFAULT_LOCAL_VOICE

    async def synth(self, text):
        return await asyncio.to_thread(self._synth, text)

    def _synth(self, text):
        payload = {"model": SPEACHES_TTS_MODEL, "voice": self.voice,
                   "input": text, "response_format": "wav"}
        req = urlrequest.Request(
            f"{SPEACHES_URL}/audio/speech",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        with urlrequest.urlopen(req, timeout=120) as response:
            body = response.read()
        import soundfile
        samples, rate = soundfile.read(io.BytesIO(body), dtype="int16")
        if samples.ndim > 1:
            samples = samples[:, 0]
        return samples.astype("<i2").tobytes(), rate


class ElevenLabsTTS:
    label = "eleven_flash_v2_5"
    sample_rate = 24000

    def __init__(self, voice=None):
        self.voice = voice or DEFAULT_ELEVEN_VOICE
        self._client = None

    def _get_client(self):
        if self._client is None:
            from elevenlabs import ElevenLabs
            key = os.environ.get("ELEVENLABS_API_KEY")
            if not key:
                raise RuntimeError("ELEVENLABS_API_KEY is not set (.env)")
            self._client = ElevenLabs(api_key=key, timeout=120)
        return self._client

    async def synth(self, text):
        return await asyncio.to_thread(self._synth, text)

    def _synth(self, text):
        audio = b"".join(self._get_client().text_to_speech.convert(
            self.voice, text=text, model_id="eleven_flash_v2_5",
            output_format="pcm_24000"))
        return audio, 24000


class DeepgramTTS:
    label = "aura_2"
    sample_rate = 24000

    def __init__(self, voice=None):
        self.voice = voice or DEFAULT_DEEPGRAM_VOICE

    async def synth(self, text):
        return await asyncio.to_thread(self._synth, text)

    def _synth(self, text):
        key = os.environ.get("DEEPGRAM_API_KEY")
        if not key:
            raise RuntimeError("DEEPGRAM_API_KEY is not set (.env)")
        from urllib.parse import urlencode
        # container=none: raw linear16 (the default wraps it in a WAV).
        query = urlencode({"model": self.voice, "encoding": "linear16",
                           "sample_rate": "24000", "container": "none"})
        req = urlrequest.Request(
            f"{DEEPGRAM_API}/v1/speak?{query}",
            data=json.dumps({"text": text}).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Token {key}"}, method="POST")
        with urlrequest.urlopen(req, timeout=120) as response:
            return response.read(), 24000


def make_tts(backend, voice=None):
    if backend == "local":
        return KokoroTTS(voice)
    if backend == "speaches":
        return SpeachesTTS(voice)
    if backend == "deepgram":
        return DeepgramTTS(voice)
    if backend == "elevenlabs":
        return ElevenLabsTTS(voice)
    raise ValueError(f"unknown AUDIO_BACKEND {backend!r}")
