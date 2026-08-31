"""The phase machine and the turn protocol (plan §3, §7d).

The phase sequence is enforced in code — the model chooses content *within*
a phase (follow up on a weakness vs move to the next unprobed target), never
the phase itself. The turn protocol keeps the spoken text first so the voice
path can stream it to TTS later, with a `\\n---\\n{json}` trailer the server
strips: {"probe_id": ..., "rationale": ...}.
"""

import json
import re

from coach.mock import engine as engines
from coach.mock.planning import persona_system

# phase -> number of interviewer turns, by session length.
PHASE_QUOTAS = {
    "short": {"warmup": 1, "walkthrough": 1, "deepdive": 3, "behavioral": 1, "closing": 1},
    "standard": {"warmup": 1, "walkthrough": 1, "deepdive": 6, "behavioral": 2, "closing": 1},
}
PHASE_ORDER = ("warmup", "walkthrough", "deepdive", "behavioral", "closing")

_TRAILER_MARKER = re.compile(r"\n-{3,}\s*\n")

PHASE_INSTRUCTIONS = {
    "warmup": "Warm-up: greet briefly and ask for the one-minute version of the "
              "chosen project. Do not probe yet.",
    "walkthrough": "Walkthrough: ask the candidate to take you through the project "
                   "end to end — problem, their role, approach, how they knew it worked.",
    "deepdive": "Deep dive: EITHER follow up on the weakest or vaguest part of the "
                "last answer (missing number, unnamed alternative, unclear ownership) "
                "OR move to the next unprobed target from the plan. Prefer the "
                "follow-up when the last answer left an obvious gap.",
    "behavioral": "Behavioral: ask about one of the plan's behavioral targets, "
                  "anchored in the project just discussed.",
    "closing": "Closing: one final reflective question (what they would do "
               "differently, or what they learned), then you will end.",
}


def quotas(length):
    return PHASE_QUOTAS.get(length, PHASE_QUOTAS["standard"])


def total_turns(length):
    return sum(quotas(length).values())


def position(state):
    """(phase, turn_number, done) for the NEXT interviewer turn, derived
    deterministically from how many turns the transcript already holds."""
    length = (state.get("plan", {}).get("settings", {}) or {}).get("length", "standard")
    asked = len(state.get("transcript", []))
    turn_number = asked + 1
    remaining = quotas(length)
    cumulative = 0
    for phase in PHASE_ORDER:
        cumulative += remaining[phase]
        if asked < cumulative:
            return phase, turn_number, False
    return "done", turn_number, True


def parse_turn_output(text):
    """Split the model's output into (spoken, meta).

    Everything after the last --- line is the trailer: stripped from the
    spoken text even when it is not valid JSON (the candidate must never see
    it), in which case meta degrades to {} — never a failed turn."""
    matches = list(_TRAILER_MARKER.finditer(text or ""))
    if not matches:
        return (text or "").strip(), {}
    last = matches[-1]
    spoken = text[:last.start()].strip()
    try:
        meta = json.loads(text[last.end():].strip())
        if not isinstance(meta, dict):
            meta = {}
    except ValueError:
        meta = {}
    return spoken, meta


def _plan_digest(plan, asked_probe_ids):
    lines = []
    for target in plan.get("probe_targets", []):
        status = "ASKED" if target["id"] in asked_probe_ids else "unprobed"
        rubric = " [bank rubric attached]" if target.get("chunk_id") else ""
        lines.append(f"- {target['id']} ({status}{rubric}) {target.get('topic', '')}: "
                     f"{target.get('question_hint', '')}")
    behavioral = ", ".join(plan.get("behavioral_targets", []))
    return "\n".join(lines) + (f"\nBehavioral targets: {behavioral}" if behavioral else "")


def _transcript_messages(state):
    messages = []
    for entry in state.get("transcript", []):
        messages.append({"role": "assistant", "content": entry.get("question", "")})
        if entry.get("answer") is not None:
            messages.append({"role": "user", "content": entry.get("answer", "")})
    return messages


def turn_prompt(state, phase, turn_number):
    """(system, messages) for an LLM interviewer turn — shared by next_turn
    (blocking, /api/mock/turn) and the voice loop's streamed variant
    (coach/voice/loop.py), so both paths run the same interview."""
    asked_ids = {entry.get("probe_id") for entry in state.get("transcript", [])
                 if entry.get("probe_id")}
    system = persona_system(state["plan"], state.get("role", {}))
    instruction = (
        f"[Interview control — invisible to the candidate]\n"
        f"You are now on interviewer turn {turn_number} of "
        f"{total_turns(state['plan'].get('settings', {}).get('length', 'standard'))}, "
        f"phase: {phase.upper()}.\n{PHASE_INSTRUCTIONS[phase]}\n\n"
        f"The hidden plan:\n{_plan_digest(state['plan'], asked_ids)}\n\n"
        f"Reply with your next spoken question, then the --- trailer.")
    messages = _transcript_messages(state) + [{"role": "user", "content": instruction}]
    return system, messages


def next_turn(state, engine):
    """The next interviewer message. state = {plan, role, transcript:[{question,
    answer, probe_id, ...}]} — stateless server, the client sends it back
    each time."""
    phase, turn_number, done = position(state)
    if done:
        return {"done": True, "turn": turn_number, "phase": "done",
                "question": "", "probe_id": None}
    if engine == "fake":
        result = engines.fake_turn(state)
        spoken, meta = result["spoken"], result["meta"]
    elif phase == "warmup":
        # Turn 1 is the plan's opening, written at start time - no LLM call.
        spoken, meta = state["plan"].get("opening", ""), {}
    else:
        system, messages = turn_prompt(state, phase, turn_number)
        # Thinking off: measured ~0.8 s to first token vs ~1.4 s+ with it on.
        text = engines.chat(system, messages, engine, thinking=False, max_tokens=400)
        spoken, meta = parse_turn_output(text)
    return {
        "done": False,
        "turn": turn_number,
        "phase": phase,
        "question": spoken,
        "probe_id": meta.get("probe_id"),
        "rationale": meta.get("rationale"),
    }
