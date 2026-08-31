"""The live voice session server: one WebSocket per interview session.

Started by `python server.py --voice` (a thread beside the HTTP server) or
standalone with `python -m coach.voice.loop`. The REST API stays the
product: the client still does setup (/api/mock/start), keyterm selection
(/api/mock/keyterms) and the report (/api/mock/report) over HTTP; this
socket only runs the audio loop of an interview in progress, holding
{plan, role, transcript} for its own lifetime and mirroring every update
to the client, which keeps the canonical copy (stateless REST unchanged).

Protocol (text frames = JSON control, binary frames = PCM16 mono 16 kHz
from the client; server binary = one sentence of PCM16 at `rate`):

  client -> server
    {"type": "hello", "access_key", "plan", "role", "transcript": [],
     "keyterms": [...], "settings": {"tts_voice": ..., "speak": true}}
    <binary>                       microphone frames, any chunking
    {"type": "played", "seq": n}   sentence n finished playing (latency +
                                   the spoken-prefix truth for barge-in)
    {"type": "end"}                end early; report happens over REST

  server -> client
    {"type": "ready", ...}         auth ok; audio config; total turns
    {"type": "state", "value"}     listening | transcribing | thinking |
                                   speaking | done
    {"type": "speech", "value"}    start | end  (VAD feedback for the UI)
    {"type": "sentence", "seq", "text", "samples", "rate"}  then <binary>
    {"type": "agent_turn", entry}  the question entry as recorded (after
                                   barge-in: the actually-spoken prefix,
                                   cut: true - the DIY loop knows what was
                                   said; Speech Engine cannot, section 10)
    {"type": "user_turn", entry}   the answer entry with live STT text
    {"type": "interrupted", "after_seq": n}
    {"type": "latency", ...}       per-turn instrumentation (section 3)
    {"type": "done", "metrics"}    phase machine finished; p50/p95 e2e
    {"type": "error", "message"}
"""

import asyncio
import json
import os
import statistics
import threading
import time

from coach import config, users
from coach.mock import engine as engines
from coach.mock import turns
from coach.voice import stt as stt_module
from coach.voice import tts as tts_module
from coach.voice.chunker import TurnStream, split_sentences
from coach.voice.vad import FRAME_BYTES, Endpointer, SileroVAD

VOICE_PORT = int(os.environ.get("VOICE_PORT", "8765"))
# End-of-turn silence, measured not guessed: on the 20 real Phase 0 answer
# recordings (each with a 1.0 s thinking pause injected mid-answer), the
# live loop cut 50% of answers mid-thought at 1.2 s, 15% at 1.8 s, and 5%
# at 2.0 s - the equivalence rule's bar (grader/loop_eval.py). The
# afterthought path in on_answer() recovers the answers it still cuts.
END_SILENCE_MS = int(os.environ.get("VOICE_END_SILENCE_MS", "2000"))
NUDGE_TEXT = "No rush - take your time."
GOODBYE_TEXT = "That's all we have time for. Thank you - your report is being written."


def audio_backend():
    return os.environ.get("AUDIO_BACKEND", "local")


class VoiceSession:
    def __init__(self, websocket):
        self.ws = websocket
        self.state = None            # {plan, role, transcript}
        self.engine = None
        self.stt = None
        self.tts = None
        self.endpointer = None
        self.speak = True            # False: browser speaks (text-only frames)
        self.inbuf = bytearray()
        self.agent_task = None       # in-flight agent turn (speaking/thinking)
        self.sent_sentences = []     # texts sent for the current agent turn
        self.played_seq = -1
        self.interrupted = False
        self.pending_entry = None    # question entry awaiting its answer
        self.end_of_speech_at = None # perf_counter at VAD end-of-turn
        self.e2e_latencies = []
        self.turn_latency = {}
        self.nudged = False
        self.closed = False

    # ------------------------------------------------------------ helpers

    async def send_json(self, **payload):
        if not self.closed:
            await self.ws.send(json.dumps(payload))

    async def set_state(self, value):
        await self.send_json(type="state", value=value)

    # -------------------------------------------------------------- setup

    async def run(self):
        try:
            hello = json.loads(await self.ws.recv())
        except Exception:
            return
        if hello.get("type") != "hello":
            await self.send_json(type="error", message="expected a hello message")
            return
        user = users.resolve_key(hello.get("access_key") or "")
        try:
            self.engine = engines.pick_engine(user)
        except engines.MockUnavailable as exc:
            await self.send_json(type="error", message=str(exc))
            return
        plan = hello.get("plan") or {}
        if not plan.get("probe_targets"):
            await self.send_json(type="error",
                                 message="plan is required (call /api/mock/start first)")
            return
        self.state = {"plan": plan, "role": hello.get("role") or {},
                      "transcript": list(hello.get("transcript") or [])}
        settings = hello.get("settings") or {}
        self.speak = settings.get("speak", True)
        backend = audio_backend()
        keyterms = hello.get("keyterms") or []
        try:
            self.stt = stt_module.make_stt(backend, keyterms)
            self.tts = tts_module.make_tts(backend, settings.get("tts_voice")) \
                if self.speak else None
            self.endpointer = Endpointer(vad=SileroVAD(),
                                         end_silence_ms=END_SILENCE_MS)
        except Exception as exc:
            await self.send_json(type="error", message=str(exc))
            return
        if backend == "local":
            # Warm the Whisper model while the interviewer speaks the opening.
            asyncio.create_task(asyncio.to_thread(stt_module.whisper_singleton))
        total = turns.total_turns(plan.get("settings", {}).get("length", "standard"))
        await self.send_json(
            type="ready", engine=self.engine, audio_backend=backend,
            rate=(self.tts.sample_rate if self.tts else None),
            stt=getattr(self.stt, "label", backend), total_turns=total,
            keyterms_used=len(keyterms))
        self.agent_task = asyncio.create_task(self.agent_turn())
        await self.receive_loop()

    # ------------------------------------------------------------ receive

    async def receive_loop(self):
        try:
            async for message in self.ws:
                if isinstance(message, (bytes, bytearray)):
                    await self.on_audio(message)
                    continue
                data = json.loads(message)
                kind = data.get("type")
                if kind == "played":
                    self.played_seq = max(self.played_seq, int(data.get("seq", -1)))
                elif kind == "end":
                    break
        finally:
            self.closed = True
            if self.agent_task:
                self.agent_task.cancel()
            if self.stt:
                await self.stt.close()

    async def on_audio(self, data):
        self.inbuf.extend(data)
        while len(self.inbuf) >= FRAME_BYTES:
            frame = bytes(self.inbuf[:FRAME_BYTES])
            del self.inbuf[:FRAME_BYTES]
            if self.stt.wants_frames:
                try:
                    await self.stt.send(frame)
                except Exception as exc:
                    await self.send_json(type="error", message=f"STT stream: {exc}")
            for event in self.endpointer.feed(frame):
                await self.on_vad_event(event)

    async def on_vad_event(self, event):
        kind = event[0]
        agent_busy = self.agent_task is not None and not self.agent_task.done()
        if kind == "speech_start":
            await self.send_json(type="speech", value="start")
            if agent_busy:
                await self.barge_in()
        elif kind == "end_of_turn":
            await self.send_json(type="speech", value="end")
            if agent_busy:
                return              # backchannel while the agent still talks
            self.end_of_speech_at = time.perf_counter()
            await self.on_answer(event[1])
        elif kind == "timeout" and not agent_busy and not self.nudged:
            self.nudged = True
            await self.speak_text(NUDGE_TEXT)

    # ------------------------------------------------------------ barge-in

    async def barge_in(self):
        """The candidate interrupts the interviewer: stop speaking, record
        only what was actually spoken (sentence-granular)."""
        self.interrupted = True
        self.agent_task.cancel()
        try:
            await self.agent_task
        except (asyncio.CancelledError, Exception):
            pass
        self.agent_task = None
        spoken = " ".join(self.sent_sentences[:self.played_seq + 1]).strip()
        cut = self.played_seq + 1 < len(self.sent_sentences)
        await self.send_json(type="interrupted", after_seq=self.played_seq)
        if self.pending_entry is not None:
            if spoken:
                self.pending_entry["question"] = spoken
            self.pending_entry["cut"] = bool(cut) or not spoken
            await self.send_json(type="agent_turn", entry=self.pending_entry)
        await self.set_state("listening")

    # ------------------------------------------------------------- answers

    async def on_answer(self, pcm):
        await self.set_state("transcribing")
        try:
            text, stt_seconds = await self.stt.commit(pcm)
        except Exception as exc:
            await self.send_json(type="error",
                                 message=f"Transcription failed ({exc}); please answer again.")
            await self.set_state("listening")
            return
        text = text.strip()
        if not text:
            await self.set_state("listening")
            return
        if self.pending_entry is None:
            # Speech with no question pending: the candidate kept talking
            # after the endpointer committed their answer (a thinking pause
            # longer than end_silence_ms - common), or barged in while the
            # interviewer was still composing. Treat it as an afterthought:
            # append to the previous answer and let the interviewer take a
            # fresh turn on the full text. (The equivalence harness found
            # the stall this replaces: a cancelled pre-question turn left
            # the loop listening forever.)
            transcript = self.state["transcript"]
            if not transcript:
                await self.set_state("listening")
                return
            last = transcript[-1]
            last["answer"] = ((last.get("answer") or "") + " " + text).strip()
            await self.send_json(type="user_turn", entry=last)
            if self.agent_task is None or self.agent_task.done():
                self.agent_task = asyncio.create_task(self.agent_turn())
            return
        entry = self.pending_entry
        entry["answer"] = text
        entry["answer_ms"] = int(len(pcm) / 32)      # 32 bytes per ms at 16 kHz
        entry["stt_seconds"] = round(stt_seconds, 3)
        self.pending_entry = None
        self.turn_latency = {"stt_s": round(stt_seconds, 3)}
        await self.send_json(type="user_turn", entry=entry)
        self.agent_task = asyncio.create_task(self.agent_turn())

    # --------------------------------------------------------- agent turns

    async def agent_turn(self):
        try:
            await self._agent_turn()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self.send_json(type="error", message=f"Interviewer turn failed: {exc}")
            await self.set_state("listening")

    async def _agent_turn(self):
        phase, turn_number, done = turns.position(self.state)
        self.sent_sentences = []
        self.played_seq = -1
        self.interrupted = False
        self.nudged = False
        await self.set_state("thinking")
        if done:
            await self.speak_text(GOODBYE_TEXT)
            await self.finish()
            return
        entry = {"turn": turn_number, "phase": phase, "question": "",
                 "probe_id": None, "rationale": None}
        self.pending_entry = entry
        if self.engine == "fake" or phase == "warmup":
            result = turns.next_turn(self.state, self.engine)
            entry["question"] = result["question"]
            entry["probe_id"] = result.get("probe_id")
            entry["rationale"] = result.get("rationale")
            self.state["transcript"].append(entry)
            sentences, _ = split_sentences(result["question"], final=True)
            await self.set_state("speaking")
            for sentence in sentences:
                await self.speak_sentence(sentence)
        else:
            await self.stream_turn(entry, phase, turn_number)
        await self.send_json(type="agent_turn", entry=entry)
        if self.turn_latency:
            first_audio = self.turn_latency.get("first_audio_s")
            if self.end_of_speech_at and first_audio:
                self.e2e_latencies.append(first_audio)
            await self.send_json(type="latency", turn=turn_number,
                                 **self.turn_latency)
            self.turn_latency = {}
        self.endpointer.start_turn()
        await self.set_state("listening")

    async def stream_turn(self, entry, phase, turn_number):
        system, messages = turns.turn_prompt(self.state, phase, turn_number)
        queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def produce():
            from coach.llm import call_chat_stream
            try:
                for delta in call_chat_stream(system, messages, self.engine,
                                              thinking=False, max_tokens=400):
                    loop.call_soon_threadsafe(queue.put_nowait, ("delta", delta))
                loop.call_soon_threadsafe(queue.put_nowait, ("end", None))
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))

        producer = threading.Thread(target=produce, daemon=True)
        llm_started = time.perf_counter()
        producer.start()
        stream = TurnStream()
        first_delta_at = None
        self.state["transcript"].append(entry)
        try:
            while True:
                kind, value = await queue.get()
                if kind == "error":
                    raise RuntimeError(str(value))
                if kind == "end":
                    break
                if first_delta_at is None:
                    first_delta_at = time.perf_counter()
                    self.turn_latency["llm_first_s"] = round(
                        first_delta_at - llm_started, 3)
                for sentence in stream.feed(value):
                    await self.set_state("speaking")
                    await self.speak_sentence(sentence)
            late, spoken, meta = stream.finish()
            for sentence in late:
                await self.set_state("speaking")
                await self.speak_sentence(sentence)
            entry["question"] = spoken
            entry["probe_id"] = meta.get("probe_id")
            entry["rationale"] = meta.get("rationale")
        except asyncio.CancelledError:
            # Barge-in: keep what was actually spoken; the producer thread
            # drains on its own (daemon, bounded output).
            entry["question"] = " ".join(
                self.sent_sentences[:self.played_seq + 1]).strip()
            raise

    async def speak_sentence(self, sentence):
        seq = len(self.sent_sentences)
        self.sent_sentences.append(sentence)
        if self.tts is None:
            await self.send_json(type="sentence", seq=seq, text=sentence,
                                 samples=0, rate=None)
            return
        synth_started = time.perf_counter()
        pcm, rate = await self.tts.synth(sentence)
        if "tts_first_s" not in self.turn_latency:
            self.turn_latency["tts_first_s"] = round(
                time.perf_counter() - synth_started, 3)
        if "first_audio_s" not in self.turn_latency and self.end_of_speech_at:
            self.turn_latency["first_audio_s"] = round(
                time.perf_counter() - self.end_of_speech_at, 3)
        await self.send_json(type="sentence", seq=seq, text=sentence,
                             samples=len(pcm) // 2, rate=rate)
        if not self.closed:
            await self.ws.send(pcm)

    async def speak_text(self, text):
        sentences, _ = split_sentences(text, final=True)
        for sentence in sentences:
            await self.speak_sentence(sentence)

    async def finish(self):
        metrics = {}
        if self.e2e_latencies:
            ordered = sorted(self.e2e_latencies)
            metrics = {
                "turns_measured": len(ordered),
                "e2e_p50_s": round(statistics.median(ordered), 3),
                "e2e_p95_s": round(ordered[max(0, int(len(ordered) * 0.95) - 1)], 3),
            }
        await self.send_json(type="done", metrics=metrics)
        await self.set_state("done")


async def handler(websocket):
    session = VoiceSession(websocket)
    try:
        await session.run()
    except Exception as exc:
        try:
            await websocket.send(json.dumps(
                {"type": "error", "message": str(exc)}))
        except Exception:
            pass


async def serve(host="127.0.0.1", port=VOICE_PORT):
    import websockets
    async with websockets.serve(handler, host, port, max_size=2 ** 23):
        print(f"Voice loop listening on ws://{host}:{port} "
              f"(AUDIO_BACKEND={audio_backend()})")
        await asyncio.Future()


def run_in_thread(host="127.0.0.1", port=VOICE_PORT):
    asyncio.run(serve(host, port))


def main():
    config.load_env_file()
    users.load_users()
    host = os.environ.get("HOST", "127.0.0.1")
    asyncio.run(serve(host, VOICE_PORT))


if __name__ == "__main__":
    main()
