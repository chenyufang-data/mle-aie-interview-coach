"""Silero VAD (onnx) and the patient endpointing state machine (plan section 3).

The model: snakers4/silero-vad v5 (MIT, 2.3 MB), run through onnxruntime on
CPU - one 32 ms frame costs well under a millisecond. It lives at
data/models/silero_vad.onnx (gitignored); ensure_model() downloads it once.

Endpointer consumes per-frame speech probabilities and emits events:
  ("speech_start",)          the user began talking (min_speech_ms confirmed)
  ("end_of_turn", pcm)       the user stopped; pcm = the whole utterance
  ("timeout",)               nothing said for turn_timeout_s (once per turn)
Patient by design: candidates pause to think, and cutting a thinking pause
is the worst possible interviewer behaviour (plan section 3) - so the turn
timeout is long, and the live loop sets end_silence_ms to the measured
VOICE_END_SILENCE_MS default of 2000 (1.2 s cut 50% of the real Phase 0
answers mid-thought, 1.8 s cut 15%, 2.0 s cuts 5% - see coach/voice/loop.py
and grader/loop_eval.py). Barge-in is not a separate mechanism: the loop
watches for speech_start while the agent is speaking.
"""

import collections
import os
from pathlib import Path

import numpy as np

from coach.config import BASE_DIR

VAD_MODEL_PATH = Path(os.environ.get(
    "SILERO_VAD_PATH", str(BASE_DIR / "data" / "models" / "silero_vad.onnx")))
VAD_URL = ("https://raw.githubusercontent.com/snakers4/silero-vad/master/"
           "src/silero_vad/data/silero_vad.onnx")
SAMPLE_RATE = 16000
FRAME_SAMPLES = 512            # the v5 model's required frame: 32 ms at 16 kHz
FRAME_BYTES = FRAME_SAMPLES * 2
FRAME_MS = 1000.0 * FRAME_SAMPLES / SAMPLE_RATE


def ensure_model():
    if not VAD_MODEL_PATH.exists():
        VAD_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        from urllib import request
        print(f"downloading Silero VAD -> {VAD_MODEL_PATH}")
        request.urlretrieve(VAD_URL, VAD_MODEL_PATH)
    return VAD_MODEL_PATH


class SileroVAD:
    """Streaming speech probability, one 512-sample 16 kHz frame at a time."""

    def __init__(self, model_path=None):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(model_path or ensure_model()), sess_options=options,
            providers=["CPUExecutionProvider"])
        self.reset()

    def reset(self):
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        # v5 expects 64 samples of left context prepended to every 512-sample
        # frame (the official wrapper does the same); without it the model's
        # STFT window is wrong and speech scores ~0.
        self.context = np.zeros(64, dtype=np.float32)

    def prob(self, frame):
        """frame: 512 int16 samples as bytes, or float32 array in [-1, 1]."""
        if isinstance(frame, (bytes, bytearray)):
            frame = np.frombuffer(frame, dtype="<i2").astype(np.float32) / 32768.0
        frame = frame.astype(np.float32).reshape(-1)
        window = np.concatenate([self.context, frame])
        out, self.state = self.session.run(
            ["output", "stateN"],
            {"input": window.reshape(1, -1),
             "state": self.state,
             "sr": np.array(16000, dtype=np.int64)})
        self.context = frame[-64:]
        return float(out[0][0])


class Endpointer:
    """Turn-taking from a stream of 32 ms frames.

    Testable without audio: pass prob= to feed() and no VAD model is needed
    (tests/test_voice.py drives it with synthetic probability traces).
    """

    def __init__(self, vad=None, start_prob=0.5, end_prob=0.35,
                 min_speech_ms=190, end_silence_ms=1200, turn_timeout_s=30.0,
                 preroll_frames=8):
        self.vad = vad
        self.start_prob = start_prob
        self.end_prob = end_prob
        self.min_speech_ms = min_speech_ms
        self.end_silence_ms = end_silence_ms
        self.turn_timeout_s = turn_timeout_s
        self.preroll = collections.deque(maxlen=preroll_frames)
        self.start_turn()

    def start_turn(self):
        """Reset per-turn state (utterance, timeout); keep the VAD context."""
        self.in_speech = False
        self.pending = []          # frames above start_prob, not yet confirmed
        self.pending_ms = 0.0
        self.silence_ms = 0.0
        self.listened_ms = 0.0
        self.said_anything = False
        self.timeout_fired = False
        self.utterance = bytearray()

    def feed(self, frame, prob=None):
        """One frame in, zero or more events out."""
        if prob is None:
            prob = self.vad.prob(frame)
        frame = bytes(frame)
        events = []
        self.listened_ms += FRAME_MS
        if not self.in_speech:
            if prob >= self.start_prob:
                self.pending.append(frame)
                self.pending_ms += FRAME_MS
                if self.pending_ms >= self.min_speech_ms:
                    self.in_speech = True
                    self.said_anything = True
                    self.silence_ms = 0.0
                    self.utterance = bytearray(b"".join(self.preroll))
                    for pending_frame in self.pending:
                        self.utterance.extend(pending_frame)
                    self.pending = []
                    self.pending_ms = 0.0
                    events.append(("speech_start",))
            else:
                for pending_frame in self.pending:
                    self.preroll.append(pending_frame)
                self.pending = []
                self.pending_ms = 0.0
                self.preroll.append(frame)
                if (not self.said_anything and not self.timeout_fired
                        and self.listened_ms >= self.turn_timeout_s * 1000):
                    self.timeout_fired = True
                    events.append(("timeout",))
        else:
            self.utterance.extend(frame)
            if prob < self.end_prob:
                self.silence_ms += FRAME_MS
                if self.silence_ms >= self.end_silence_ms:
                    pcm = bytes(self.utterance)
                    events.append(("end_of_turn", pcm))
                    self.in_speech = False
                    self.silence_ms = 0.0
                    self.utterance = bytearray()
            else:
                self.silence_ms = 0.0
        return events
