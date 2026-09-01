"""Offline unit tests for the voice loop's deterministic parts: the
endpointer state machine (synthetic probability traces - no VAD model
needed), the trailer-safe sentence chunker, the keyterm policy endpoint
logic, SSE parsing, the shared turn prompt, and the two-transcript report
block. No network, no audio models; CI-safe."""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from coach.llm import _sse_events                                  # noqa: E402
from coach.mock import transcription                               # noqa: E402
from coach.mock.turns import turn_prompt                           # noqa: E402
from coach.voice.chunker import TurnStream, split_sentences        # noqa: E402
from coach.voice.final_transcript import available_engine          # noqa: E402
from coach.voice.keyterms import (final_transcript_keyterms,       # noqa: E402
                                  session_keyterms)
from coach.voice.vad import FRAME_BYTES, Endpointer                # noqa: E402


def test_endpointer():
    frame = b"x" * FRAME_BYTES
    # speech then silence -> one turn with the utterance attached
    ep = Endpointer(end_silence_ms=1200)
    events = []
    for _ in range(20):
        events += ep.feed(frame, prob=0.9)
    for _ in range(45):
        events += ep.feed(frame, prob=0.1)
    assert [e[0] for e in events] == ["speech_start", "end_of_turn"], events
    pcm = events[-1][1]
    assert len(pcm) >= 20 * FRAME_BYTES

    # a thinking pause shorter than end_silence_ms must NOT end the turn
    ep.start_turn()
    events = []
    for _ in range(20):
        events += ep.feed(frame, prob=0.9)
    for _ in range(30):                      # ~960 ms pause
        events += ep.feed(frame, prob=0.1)
    for _ in range(20):
        events += ep.feed(frame, prob=0.9)
    for _ in range(45):
        events += ep.feed(frame, prob=0.1)
    assert [e[0] for e in events] == ["speech_start", "end_of_turn"], events

    # a blip shorter than min_speech_ms never becomes a turn
    ep.start_turn()
    events = []
    for _ in range(3):
        events += ep.feed(frame, prob=0.9)
    for _ in range(50):
        events += ep.feed(frame, prob=0.0)
    assert events == []

    # timeout fires once when nothing is said, then stays quiet
    ep2 = Endpointer(turn_timeout_s=1.0)
    events = []
    for _ in range(60):
        events += ep2.feed(frame, prob=0.0)
    assert [e[0] for e in events] == ["timeout"], events
    print("endpointer state machine ok")


def test_chunker():
    stream = TurnStream()
    out = []
    for delta in ["Thanks. Tell me about", " the QWK metric. Why",
                  " did you pick it?\n---\n"
                  "{\"probe_id\": \"probe_2\", \"rationale\": \"metric\"}"]:
        out += stream.feed(delta)
    late, spoken, meta = stream.finish()
    sentences = out + late
    assert sentences[0] == "Thanks."
    assert meta["probe_id"] == "probe_2"
    assert "---" not in spoken and "probe_2" not in spoken

    # malformed trailer: stripped, meta degrades to {}
    stream = TurnStream()
    stream.feed("Good. What broke?\n---\n{not json")
    _late, spoken, meta = stream.finish()
    assert meta == {} and spoken == "Good. What broke?"

    # no trailer at all
    stream = TurnStream()
    out = stream.feed("One question only")
    late, spoken, meta = stream.finish()
    assert spoken == "One question only" and (out + late) == ["One question only"]

    # dash line split across deltas is still held back
    stream = TurnStream()
    out = stream.feed("Done here.\n--")
    out += stream.feed("-\n{\"probe_id\": null}")
    _late, spoken, meta = stream.finish()
    assert spoken == "Done here." and meta == {"probe_id": None}

    # trailing dash line with no trailer body
    stream = TurnStream()
    stream.feed("Last question then.\n---")
    _late, spoken, meta = stream.finish()
    assert spoken == "Last question then." and meta == {}

    # decimals and abbreviations do not split; boundary at buffer end waits
    sentences, rest = split_sentences(
        "We used v2.5 e.g. for speed. Then shipped. tail", final=False)
    assert sentences == ["We used v2.5 e.g. for speed.", "Then shipped."]
    assert rest == "tail"
    sentences, rest = split_sentences("It was v2.", final=False)
    assert sentences == [] and rest == "It was v2."
    print("sentence chunker ok")


def test_keyterms():
    chosen = session_keyterms(
        resume="I used LightGBM and QWK with PyTorch on fraud detection",
        role={"title": "fraud MLE", "probe_themes": ["heteroscedasticity"]})
    assert len(chosen) == 50
    assert all(len(term) <= 20 for term in chosen)
    # measured 100%-failure terms take slots
    assert "heteroscedasticity" in chosen
    batch = final_transcript_keyterms()
    assert len(batch) >= 300
    assert all(len(term) < 50 for term in batch)
    print("keyterm policy ok")


def test_sse_events():
    lines = [
        b"event: something",
        b"data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}",
        b"",
        b"data: not json",
        b"data: {\"choices\":[{\"delta\":{\"content\":\" there\"}}]}",
        b"data: [DONE]",
        b"data: {\"choices\":[{\"delta\":{\"content\":\"never\"}}]}",
    ]
    events = list(_sse_events(lines))
    texts = [event["choices"][0]["delta"]["content"] for event in events]
    assert texts == ["Hi", " there"], texts
    print("sse parsing ok")


def test_turn_prompt():
    state = {
        "plan": {"probe_targets": [
            {"id": "probe_1", "topic": "leakage", "question_hint": "how caught"}],
            "behavioral_targets": ["a failure"],
            "settings": {"length": "short", "style": "neutral"},
            "persona": {"company_type": "bank", "seniority": "senior"}},
        "role": {"title": "MLE", "level": "Mid-level"},
        "transcript": [
            {"turn": 1, "phase": "warmup", "question": "Hi - one minute version?",
             "probe_id": None, "answer": "Sure, I built a fraud model."}],
    }
    system, messages = turn_prompt(state, "deepdive", 3)
    assert "interviewer" in system.lower()
    assert messages[-1]["role"] == "user"
    assert "DEEPDIVE" in messages[-1]["content"]
    assert "probe_1" in messages[-1]["content"]
    # the transcript rides as chat history before the control message
    assert messages[0]["role"] == "assistant"
    assert messages[1]["content"] == "Sure, I built a fraud model."
    print("turn prompt ok")


def test_transcription_block():
    entries = [
        # voice answer where the live STT mangled two lexicon terms
        {"turn": 2, "answer": "we tuned light jbm using quadratic weighted copper",
         "answer_final": "We tuned LightGBM using quadratic weighted kappa.",
         "stt_seconds": 0.4},
        # typed answer: no final transcript, not compared
        {"turn": 3, "answer": "Plain typed answer."},
    ]
    block = transcription.compare(entries)
    assert block["answers_compared"] == 1
    assert block["wer_live_vs_final"] > 0
    fixed = block["per_turn"][0]["terms_fixed"]
    assert "LightGBM" in fixed, fixed
    assert block["terms_fixed_total"] >= 1

    # perfect live transcript: zero WER, nothing fixed
    clean = transcription.compare([
        {"turn": 1, "answer": "We used LightGBM.",
         "answer_final": "We used LightGBM.", "stt_seconds": 0.2}])
    assert clean["wer_live_vs_final"] == 0.0
    assert clean["terms_fixed_total"] == 0

    # text-only session: no block at all
    assert transcription.compare([{"turn": 1, "answer": "typed"}]) is None

    # grading text: voice answers normalized, typed answers untouched
    voice_text = transcription.grading_answer_text(entries[0])
    assert voice_text == voice_text.lower() and "." not in voice_text
    assert transcription.grading_answer_text(entries[1]) == "Plain typed answer."
    # kp_flips degrades to None without the trained artifact loaded
    from coach import grading
    if grading.GRADER is None:
        assert transcription.kp_flips({"probe_targets": []}, entries) is None
    print("transcription block ok")


def test_final_transcript_meta():
    assert available_engine() in (None, "scribe_batch_kt", "whisper_local")
    print("final transcript meta ok")


def test_sidecar_protocol():
    """The Speech Engine upstream contract: every agent_response chunk
    echoes the user_transcript's event_id (ElevenLabs discards mismatched
    ids - the first live run was silent for exactly this) and every
    response terminates with an empty is_final=true chunk."""
    import asyncio
    import json as jsonlib

    from coach.voice import sidecar

    plan = {"opening": "Welcome - give me the one minute version of you.",
            "probe_targets": [{"id": "probe_1", "topic": "leakage",
                               "question_hint": "How did you catch it?"}],
            "behavioral_targets": ["a failure"],
            "settings": {"length": "short", "style": "neutral"},
            "persona": {"company_type": "bank", "seniority": "senior"}}
    sidecar.register_plan(plan, {"title": "MLE", "level": "Mid-level"}, "fake")

    class FakeWS:
        def __init__(self, incoming):
            self.incoming = list(incoming)
            self.sent = []

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self.incoming:
                raise StopAsyncIteration
            return jsonlib.dumps(self.incoming.pop(0))

        async def send(self, message):
            self.sent.append(jsonlib.loads(message))

    ws = FakeWS([
        {"type": "init", "conversation_id": "conv_1"},
        {"type": "ping"},
        {"type": "user_transcript", "event_id": 7, "user_transcript": [
            {"role": "agent", "content": plan["opening"]},
            {"role": "user", "content": "I build fraud models."}]},
        {"type": "close"},
    ])
    asyncio.run(sidecar.SidecarSession(ws).run())

    assert any(f.get("type") == "pong" for f in ws.sent), ws.sent
    responses = [f for f in ws.sent if f.get("type") == "agent_response"]
    for frame in responses:
        if frame["is_final"]:
            assert frame["content"] == "", frame
        else:
            assert frame["content"], frame
    assert responses[-1]["is_final"], responses
    # the opener (no user turn yet) carries no event_id; everything after
    # the user_transcript echoes ITS id, never a counter of ours
    assert "event_id" not in responses[0], responses[0]
    echoed = [f for f in responses if "event_id" in f]
    assert echoed and all(f["event_id"] == 7 for f in echoed), responses
    state = sidecar.LAST["state"]
    assert state["transcript"][0]["question"] == plan["opening"]
    assert state["transcript"][0]["answer"] == "I build fraud models."

    # a reconnect with the SAME conversation_id resumes silently instead
    # of restarting the interview from the top
    ws2 = FakeWS([{"type": "init", "conversation_id": "conv_1"},
                  {"type": "close"}])
    asyncio.run(sidecar.SidecarSession(ws2).run())
    assert not ws2.sent, ws2.sent
    assert sidecar.LAST["state"] is state
    print("sidecar protocol ok")


if __name__ == "__main__":
    test_endpointer()
    test_chunker()
    test_keyterms()
    test_sse_events()
    test_turn_prompt()
    test_transcription_block()
    test_final_transcript_meta()
    test_sidecar_protocol()
    print("all voice tests passed")
