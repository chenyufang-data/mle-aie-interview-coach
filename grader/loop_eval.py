"""Phase 2 equivalence harness (plan section 9): drive the live voice loop
end to end - a scripted interview whose "candidate" is the 20 real human
answer recordings from Phase 0 (references known exactly) - and measure,
per audio backend:

  live TER            lexicon-term loss in what the interviewer heard
  WER                 word error rate vs the reference text
  latency             stt finalize, llm first token, end-of-speech ->
                      first agent audio (p50/p95)
  cut-off rate        answers ended early while a mid-answer 1.0 s
                      thinking pause was still being streamed (the patient
                      endpointer must ride it out; 1.2 s of silence ends
                      a turn)
  barge-in            interrupting the interviewer mid-sentence must stop
                      speech and record the spoken prefix
  downstream damage   distilled-grader score movement, reference vs live
                      text, both normalized (the Phase 0 decision), on the
                      answers that carry a bank chunk

The interviewer is the deterministic fake engine (--mock server: no LLM
spend); the audio path is the real one. Levels measured here: 2/3
(--backend local) and the DIY-cloud "1.5" (--backend deepgram, Nova-3
streaming + Aura-2, or --backend elevenlabs, Scribe Realtime + Flash TTS
- both print the cost and need --confirm). Level 1 (Speech Engine) needs
its sidecar plus a public tunnel and is compared with the same numbers
once it runs.

  .venv\\Scripts\\python grader\\loop_eval.py --backend local
  .venv\\Scripts\\python grader\\loop_eval.py --backend deepgram --confirm
  .venv\\Scripts\\python grader\\loop_eval.py --backend elevenlabs --confirm

Results: printed + merged into grader/loop_eval_results.json.
"""

import argparse
import asyncio
import json
import os
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from grader.stt_text import edit_distance, normalize, term_errors  # noqa: E402

PY = str(BASE_DIR / ".venv" / "Scripts" / "python.exe")
PORT, WS_PORT = 8033, 8790
RESULTS_PATH = BASE_DIR / "grader" / "loop_eval_results.json"
SENTENCES_PATH = BASE_DIR / "grader" / "stt_sentences.jsonl"
AUDIO_DIR = BASE_DIR / "data" / "stt_audio" / "human" / "audio"
PAUSE_S = 1.0            # injected mid-answer thinking pause
# How many times faster than realtime to stream (LOOP_EVAL_SPEED): 4x is
# fine locally; elevenlabs takes 2x (faster streams get backpressured);
# deepgram MUST run 1x - Nova-3 transcribes at the audio clock's own pace,
# so faster-than-realtime input builds a transcript backlog that poisons
# both the text and the stt latency column (measured: the first 2x run
# returned 88% WER of turn-shifted fragments). Defaults set in main().
STREAM_SPEED = float(os.environ.get("LOOP_EVAL_SPEED", "4"))
CLOUD_SPEEDS = {"elevenlabs": 2.0, "deepgram": 1.0}

RESUME = ("ML engineer. Built a fraud detection model with LightGBM, evaluated "
          "with PR-AUC and calibration; deployed behind a FastAPI service on "
          "EC2. Built a RAG interview coach with BM25 retrieval and a distilled "
          "sklearn grader evaluated by QWK against a Claude teacher.")


def post(path, body):
    import urllib.request
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.loads(response.read().decode())


def wait_port(port, timeout=90):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(0.4)
    raise RuntimeError(f"port {port} never opened")


def load_answers():
    items = [json.loads(line) for line in
             SENTENCES_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    answers = [item for item in items if item["id"].startswith("ans")]
    import numpy as np
    from faster_whisper.audio import decode_audio
    out = []
    for item in answers:
        path = AUDIO_DIR / f"{item['id']}.webm"
        if not path.exists():
            continue
        samples = decode_audio(str(path), sampling_rate=16000)
        pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()
        out.append({**item, "pcm": pcm, "seconds": len(samples) / 16000})
    return out


def keyterm_list():
    from coach.voice.keyterms import session_keyterms
    resume_dir = BASE_DIR / "data" / "resume"
    resume = " ".join(p.read_text(encoding="utf-8", errors="ignore")
                      for p in resume_dir.glob("*.txt")) if resume_dir.exists() else RESUME
    return session_keyterms(resume=resume, role={"title": "Machine Learning Engineer"})


class SessionDriver:
    """Streams scripted answers into one ws session; collects measurements."""

    def __init__(self, ws, answers, barge_on_turn=None):
        self.ws = ws
        self.answers = answers
        self.barge_on_turn = barge_on_turn
        self.rows = []
        self.latencies = []
        self.barge = None
        self.queue = asyncio.Queue()
        self.audio_seen = asyncio.Event()

    async def receiver(self):
        debug = os.environ.get("LOOP_EVAL_DEBUG")
        async for message in self.ws:
            if isinstance(message, (bytes, bytearray)):
                self.audio_seen.set()
                continue
            msg = json.loads(message)
            if debug:
                print(f"  <{msg['type']}>", flush=True)
            if msg["type"] == "error":
                print("  server error:", msg["message"])
            await self.queue.put(msg)

    async def next_msg(self, kinds, timeout=180):
        while True:
            msg = await asyncio.wait_for(self.queue.get(), timeout)
            if msg["type"] == "latency":
                self.latencies.append(msg)
            if msg["type"] in kinds:
                return msg

    async def stream_pcm(self, pcm, realtime_factor=STREAM_SPEED):
        chunk = 3200                                     # 100 ms
        for i in range(0, len(pcm), chunk):
            await self.ws.send(pcm[i:i + chunk])
            await asyncio.sleep(0.1 / realtime_factor)

    async def run(self, keyterms, plan, role):
        await self.ws.send(json.dumps({
            "type": "hello", "access_key": "", "plan": plan, "role": role,
            "transcript": [], "keyterms": keyterms, "settings": {"speak": True}}))
        receiver = asyncio.create_task(self.receiver())
        try:
            await self._run_turns()
        finally:
            receiver.cancel()

    async def _run_turns(self):
        for index, answer in enumerate(self.answers):
            turn_no = index + 1
            if self.barge_on_turn == turn_no:
                await self._barge_then_answer(answer)
            else:
                await self.next_msg({"agent_turn"})
            # Stream the answer concurrently and wait for the turn to end.
            # A user_turn that lands while the actual audio is still being
            # streamed means the endpointer ended the turn inside the
            # injected (or an adjacent natural) pause: a measured cut-off.
            # Either way, stop streaming immediately - a real candidate
            # stops talking when the interviewer starts.
            msg = None
            for attempt in range(2):
                self.audio_done = False
                stream_task = asyncio.create_task(self._stream_answer(answer))
                try:
                    msg = await self.next_msg({"user_turn", "error"})
                finally:
                    stream_task.cancel()
                    try:
                        await stream_task
                    except (asyncio.CancelledError, Exception):
                        pass
                if msg["type"] == "user_turn":
                    break
                print(f"  turn error ({msg.get('message', '')[:90]}); "
                      f"{'retrying' if attempt == 0 else 'giving up'}")
                msg = None
            if msg is None:
                break                # session unrecoverable; keep what we have
            self._record(answer, msg, cut=not self.audio_done,
                         barge=self.barge_on_turn == turn_no)
            # A real microphone never stops sending frames; the cancel
            # above does. Close any utterance the server may have started
            # from answer frames sent while its STT was running, so the
            # loop's afterthought path can commit it and move on.
            for _ in range(50):
                await self.ws.send(b"\x00" * 3200)
                await asyncio.sleep(0.1 / STREAM_SPEED)
        await self.ws.send(json.dumps({"type": "end"}))

    async def _stream_answer(self, answer):
        print(f"  streaming {answer['id']} ({answer['seconds']:.0f}s)", flush=True)
        silence = b"\x00" * 3200
        half = (len(answer["pcm"]) // 2) & ~1
        await self.stream_pcm(answer["pcm"][:half])
        for _ in range(int(PAUSE_S * 10)):   # mid-answer thinking pause
            await self.ws.send(silence)
            await asyncio.sleep(0.1 / STREAM_SPEED)
        await self.stream_pcm(answer["pcm"][half:])
        self.audio_done = True
        for _ in range(70):                  # > end_silence_ms of trailer
            await self.ws.send(silence)
            await asyncio.sleep(0.1 / STREAM_SPEED)

    async def _barge_then_answer(self, answer):
        """Interrupt the interviewer at their first sentence, then answer.
        Keyed off the sentence control message, not the binary audio - an
        Event raced the previous turn's trailing flush and hung forever."""
        await self.next_msg({"sentence"}, timeout=60)
        started = time.perf_counter()
        await self.stream_pcm(answer["pcm"][:3200 * 20], realtime_factor=8.0)
        try:
            msg = await self.next_msg({"interrupted", "agent_turn"}, timeout=15)
            self.barge = {"ok": msg["type"] == "interrupted",
                          "stop_s": round(time.perf_counter() - started, 2),
                          "via": msg["type"]}
            if msg["type"] == "interrupted":
                await self.next_msg({"agent_turn"}, timeout=15)
        except asyncio.TimeoutError:
            self.barge = {"ok": False}

    def _record(self, answer, msg, cut, barge=False):
        entry = msg["entry"]
        ref = normalize(answer["text"])
        hyp = normalize(entry.get("answer") or "")
        rows = term_errors(_lexicon(), ref, hyp)
        self.rows.append({
            "id": answer["id"], "chunk_id": answer.get("chunk_id"),
            "ref_text": answer["text"], "hyp_text": entry.get("answer") or "",
            "wer": round(edit_distance(ref, hyp) / max(1, len(ref)), 4),
            "terms": rows, "stt_s": entry.get("stt_seconds"),
            "cut_during_pause": cut,
            # The barge turn's utterance starts with the injected interrupt
            # speech (a duplicated answer prefix), so it is excluded from
            # the text-accuracy aggregates.
            "barge": barge,
        })


_LEX = None


def _lexicon():
    global _LEX
    if _LEX is None:
        from coach.voice.keyterms import lexicon
        _LEX = lexicon()
    return _LEX


def grade_damage(rows):
    """Distilled-grader movement ref vs live text, both normalized."""
    import server
    server.load_chunks()
    server.load_grader()
    if server.GRADER is None:
        return None
    grader = server.GRADER
    extractor = grader["extractor"]
    moved, graded = 0, 0
    for row in rows:
        chunk = server.CHUNKS_BY_ID.get(row.get("chunk_id") or "")
        if chunk is None or row["cut_during_pause"] or row.get("barge"):
            continue
        ref_text = " ".join(normalize(row["ref_text"]))
        hyp_text = " ".join(normalize(row["hyp_text"]))
        scores = []
        for text in (ref_text, hyp_text):
            raw = float(grader["model"].predict([extractor.extract(text, chunk)])[0])
            scores.append(server.clamp_score(round(raw)))
        graded += 1
        if abs(scores[0] - scores[1]) >= 1:
            moved += 1
    return {"graded": graded, "moved_ge1": moved,
            "rate": round(moved / graded, 3) if graded else None}


def summarize(backend, drivers):
    rows = [row for driver in drivers for row in driver.rows]
    latencies = [l for driver in drivers for l in driver.latencies]
    occurrences = strict = lenient = 0
    clean = [row for row in rows
             if not row["cut_during_pause"] and not row.get("barge")]
    for row in clean:
        for term_row in row["terms"]:
            occurrences += term_row["count"]
            strict += term_row["strict_hits"]
            lenient += term_row["lenient_hits"]
    wers = [row["wer"] for row in clean]
    stt = sorted(row["stt_s"] for row in rows if row.get("stt_s"))
    first_audio = sorted(l["first_audio_s"] for l in latencies
                         if l.get("first_audio_s"))
    cut = sum(1 for row in rows if row["cut_during_pause"])
    barge = next((driver.barge for driver in drivers if driver.barge), None)

    def pct(values, q):
        if not values:
            return None
        return round(values[min(len(values) - 1, max(0, int(len(values) * q) - (0 if q < 1 else 1)))], 3)

    summary = {
        "backend": backend,
        "answers": len(rows),
        "wer": round(sum(wers) / len(wers), 4) if wers else None,
        "occurrences": occurrences,
        "ter_strict": round(1 - strict / occurrences, 4) if occurrences else None,
        "ter_lenient": round(1 - lenient / occurrences, 4) if occurrences else None,
        "stt_p50_s": round(statistics.median(stt), 3) if stt else None,
        "stt_p95_s": pct(stt, 0.95),
        "first_audio_p50_s": round(statistics.median(first_audio), 3) if first_audio else None,
        "first_audio_p95_s": pct(first_audio, 0.95),
        "cutoff_rate": round(cut / len(rows), 3) if rows else None,
        "barge_in": barge,
        "damage": grade_damage(rows),
        "measured_at": time.strftime("%Y-%m-%d %H:%M"),
    }
    return summary, rows


async def drive(backend, answers, keyterms):
    import websockets
    drivers = []
    # sessions of at most 11 turns (standard length), barge-in test on the
    # first session's turn 3
    groups = [answers[i:i + 11] for i in range(0, len(answers), 11)]
    for index, group in enumerate(groups):
        roles = post("/api/mock/roles", {"resume": RESUME})
        role = roles["roles"][0]
        start = post("/api/mock/start", {
            "resume": RESUME, "role": role, "project": None,
            "settings": {"style": "neutral",
                         "length": "standard" if len(group) > 7 else "short"}})
        # ping_interval=None: the scripted driver must not enforce protocol
        # ping timeouts - a server busy with a slow STT commit would get
        # disconnected mid-measurement (browsers, the real client, send no
        # protocol pings at all).
        async with websockets.connect(f"ws://127.0.0.1:{WS_PORT}",
                                      max_size=2 ** 23,
                                      ping_interval=None) as ws:
            # Barge on turn 2: the walkthrough question is the longest the
            # fake engine speaks, so the interrupt lands mid-sentence.
            driver = SessionDriver(ws, group,
                                   barge_on_turn=2 if index == 0 else None)
            await driver.run(keyterms, start["plan"], role)
            drivers.append(driver)
        print(f"  session {index + 1}/{len(groups)}: {len(driver.rows)} answers")
    return drivers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="local",
                        choices=["local", "speaches", "deepgram", "elevenlabs"])
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()

    if args.backend in CLOUD_SPEEDS and "LOOP_EVAL_SPEED" not in os.environ:
        globals()["STREAM_SPEED"] = CLOUD_SPEEDS[args.backend]

    answers = load_answers()
    total_seconds = sum(answer["seconds"] for answer in answers)
    print(f"{len(answers)} scripted answers, {total_seconds / 60:.1f} min of audio, "
          f"backend {args.backend}")
    if args.backend == "elevenlabs":
        est = total_seconds / 3600 * 0.44 + 0.05   # realtime STT + Flash TTS
        print(f"estimated ElevenLabs cost: ~${est:.2f}")
        if not args.confirm:
            print("dry run only - add --confirm to spend")
            return
    if args.backend == "deepgram":
        # Nova-3 streaming ~$0.46/h + Aura-2 ~$0.030/1k chars of questions.
        est = total_seconds / 3600 * 0.46 + 0.10
        print(f"estimated Deepgram cost: ~${est:.2f} (new accounts carry "
              "~$200 signup credit)")
        if not args.confirm:
            print("dry run only - add --confirm to spend")
            return

    keyterms = keyterm_list()
    env = dict(os.environ, PORT=str(PORT), VOICE_PORT=str(WS_PORT),
               AUDIO_BACKEND=args.backend, PYTHONUNBUFFERED="1")
    # Server output goes to a FILE: an undrained PIPE fills up on the audio
    # stack's per-sentence warnings and freezes the server mid-print.
    log_path = BASE_DIR / "data" / "loop_eval_server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        server_proc = subprocess.Popen(
            [PY, "server.py", "--mock", "--voice"], cwd=str(BASE_DIR), env=env,
            stdout=log, stderr=subprocess.STDOUT)
        try:
            wait_port(PORT)
            wait_port(WS_PORT)
            drivers = asyncio.run(drive(args.backend, answers, keyterms))
        finally:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_proc.kill()

    summary, rows = summarize(args.backend, drivers)
    print(json.dumps(summary, indent=1))
    existing = json.loads(RESULTS_PATH.read_text(encoding="utf-8")) \
        if RESULTS_PATH.exists() else {}
    existing[args.backend] = {"summary": summary,
                              "per_answer": [{k: v for k, v in row.items()
                                              if k not in ("terms",)}
                                             for row in rows]}
    RESULTS_PATH.write_text(json.dumps(existing, ensure_ascii=False, indent=1),
                            encoding="utf-8")
    print(f"-> {RESULTS_PATH.name}")


if __name__ == "__main__":
    main()
