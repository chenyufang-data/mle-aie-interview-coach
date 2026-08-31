"""Level 1 sidecar: ElevenLabs Speech Engine, bring-your-own-agent.

The hosted loop does STT (Scribe Realtime + keywords), turn-taking,
barge-in, TTS and playback; ElevenLabs connects to THIS server's public
WebSocket (`ws_url`) and exchanges text only:

  they send   init {conversation_id} / user_transcript {history} / ping
  we send     agent_response {content, event_id, is_final} / pong

(Wire shapes verified against elevenlabs SDK 2.65 types. Two consequences
measured nowhere else: the transcript is the only observability - on a
barge-in the sidecar never learns the spoken prefix, which the DIY loop
does know - and `ws_url` must be publicly reachable, so a local run needs
a tunnel. Both recorded in plan section 10.)

Usage (single-tenant, a personal practice tool):

  1. python -m coach.voice.sidecar --setup wss://<public-host>/ws
       creates/updates the Speech Engine via the API: Scribe Realtime ASR
       with this session's keyterms, Flash v2.5 TTS, patient turn-taking.
       Prints SPEECH_ENGINE_ID for .env.
  2. python -m coach.voice.sidecar --serve [--port 3001]
       serves the upstream protocol; the NEXT conversation adopts the plan
       registered via register_plan() (the mock page calls
       /api/mock/start first, then hands the plan over).

The interview logic is untouched: the same turn_prompt / call_chat_stream
/ TurnStream path as coach/voice/loop.py, so Level 1 and Level 2 run the
same interviewer - which is what makes the equivalence test (plan
section 9) a comparison of audio stacks, not of interviewers.
"""

import argparse
import asyncio
import json
import os
import threading

from coach import config, users
from coach.mock import engine as engines
from coach.mock import turns
from coach.voice.chunker import TurnStream

SIDECAR_PORT = int(os.environ.get("SIDECAR_PORT", "3001"))
_PENDING = {"plan": None, "role": None, "engine": None}
# The most recent conversation's state, so the mock page can mirror the
# transcript (GET /api/mock/level1/transcript) and build the report.
LAST = {"state": None, "conversation_id": None}


def register_plan(plan, role, engine):
    """Stash the next conversation's interview (single-tenant by design)."""
    _PENDING.update({"plan": plan, "role": role, "engine": engine})


class SidecarSession:
    def __init__(self, websocket):
        self.ws = websocket
        self.state = None
        self.engine = None
        self.event_id = 0

    async def run(self):
        async for message in self.ws:
            data = json.loads(message)
            kind = data.get("type")
            if kind == "ping":
                await self.ws.send(json.dumps({"type": "pong"}))
            elif kind == "init":
                if _PENDING["plan"] is None:
                    await self._say("No interview is set up yet - open the mock "
                                    "page and start a session first.", final=True)
                    continue
                self.state = {"plan": _PENDING["plan"],
                              "role": _PENDING["role"], "transcript": []}
                self.engine = _PENDING["engine"]
                LAST["state"] = self.state
                LAST["conversation_id"] = data.get("conversation_id")
                await self.agent_turn()
            elif kind == "user_transcript":
                if self.state is None:
                    continue
                answer = ""
                for row in data.get("user_transcript") or []:
                    if row.get("role") == "user":
                        answer = row.get("content") or answer
                if self.state["transcript"]:
                    self.state["transcript"][-1]["answer"] = answer
                await self.agent_turn()
            elif kind == "close":
                break
            elif kind == "error":
                print("speech engine error:", data.get("message"))

    async def _say(self, content, final):
        self.event_id += 1
        await self.ws.send(json.dumps({
            "type": "agent_response", "content": content,
            "event_id": self.event_id, "is_final": final}))

    async def agent_turn(self):
        phase, turn_number, done = turns.position(self.state)
        if done:
            await self._say("That's all we have time for. Thank you - your "
                            "report is on the mock page.", final=True)
            return
        entry = {"turn": turn_number, "phase": phase, "question": "",
                 "probe_id": None, "rationale": None, "answer": None}
        if self.engine == "fake" or phase == "warmup":
            result = turns.next_turn(self.state, self.engine)
            entry.update(question=result["question"],
                         probe_id=result.get("probe_id"),
                         rationale=result.get("rationale"))
            self.state["transcript"].append(entry)
            await self._say(result["question"], final=True)
            return
        system, messages = turns.turn_prompt(self.state, phase, turn_number)
        self.state["transcript"].append(entry)
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

        threading.Thread(target=produce, daemon=True).start()
        stream = TurnStream()
        while True:
            kind, value = await queue.get()
            if kind == "error":
                raise RuntimeError(str(value))
            if kind == "end":
                break
            for sentence in stream.feed(value):
                await self._say(sentence, final=False)
        late, spoken, meta = stream.finish()
        for sentence in late:
            await self._say(sentence, final=False)
        await self._say("", final=True)
        entry.update(question=spoken, probe_id=meta.get("probe_id"),
                     rationale=meta.get("rationale"))


async def serve(port):
    import websockets

    async def handler(websocket):
        await SidecarSession(websocket).run()

    async with websockets.serve(handler, "0.0.0.0", port):
        print(f"Speech Engine sidecar on ws://0.0.0.0:{port} "
              "(expose it publicly and set that URL as ws_url)")
        await asyncio.Future()


def setup(ws_url, keyterms=(), voice_id="XrExE9yKIg1WjnnlVkGX"):
    """Create or update the Speech Engine agent config via the API.

    Verified live 2026-08-31: `asr.keywords` and the patient turn config
    are accepted; English agents must use eleven_turbo_v2 / eleven_flash_v2
    for TTS (flash v2.5 is rejected with a 400 - it serves non-English
    agents), and creating with an invalid tts block 500s, so configure
    stepwise-compatible fields only.
    """
    from elevenlabs import ElevenLabs
    client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"], timeout=60)
    kwargs = {
        "asr": {"provider": "scribe_realtime",
                "keywords": list(keyterms)[:50]},
        "tts": {"model_id": "eleven_flash_v2",
                **({"voice_id": voice_id} if voice_id else {})},
        # Patient turn-taking: candidates pause to think (plan section 3).
        "turn": {"turn_eagerness": "patient", "turn_timeout": 30},
    }
    engine_id = os.environ.get("SPEECH_ENGINE_ID")
    if engine_id:
        result = client.speech_engine.update(
            engine_id, speech_engine={"ws_url": ws_url}, **kwargs)
    else:
        result = client.speech_engine.create(
            speech_engine={"ws_url": ws_url}, name="mock-interviewer", **kwargs)
        engine_id = getattr(result, "speech_engine_id", None)
        if not engine_id:  # the create response hides the id; list() has it
            listing = client.speech_engine.list()
            for item in getattr(listing, "speech_engines", []) or []:
                if item.name == "mock-interviewer":
                    engine_id = item.speech_engine_id
    print("SPEECH_ENGINE_ID =", engine_id)
    return result if not engine_id else type(
        "Result", (), {"speech_engine_id": engine_id})()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--setup", metavar="WS_URL")
    parser.add_argument("--port", type=int, default=SIDECAR_PORT)
    args = parser.parse_args()
    config.load_env_file()
    users.load_users()
    if args.setup:
        from coach.voice.keyterms import session_keyterms
        setup(args.setup, keyterms=session_keyterms())
        return
    if args.serve:
        asyncio.run(serve(args.port))
        return
    parser.print_help()


if __name__ == "__main__":
    main()
