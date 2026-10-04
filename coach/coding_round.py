"""The mock coding round (roadmap step 7, phase 5): an interviewer, not a tutor.

One attempt from coach/tutor.py with mode "mock". The phases are enforced
here; the model only chooses what to say within them:

  discuss   the interviewer presents the problem (number and title for a
            LeetCode problem - the statement stays on leetcode.com - or the
            author's own statement), answers clarifying questions, and asks
            for the approach and its complexity before any code
  coding    from the first real code; the interviewer answers when asked
  review    after "I'm done": a walk through an example, then follow-ups
            (the bank's, or the model's for a problem outside the bank)
  closed    the closing line; the page asks for the reports

Help requests climb the same server-enforced ladder and pass the same output
guard as the tutor's, and count in the coding report. *Interview conditions*
(`strict`): Run and Check are refused until "I'm done", the whitespace notes
are hidden (the page), and the interviewer never checks in unasked; without
them, Run and Check work throughout and a long quiet stretch while coding
gets a check-in.

The communication report: the deterministic metrics of coach/mock/metrics.py
over the candidate's turns, time to first code and the longest quiet stretch
from the timeline, then a judged verdict per criterion with its evidence and
two or three suggestions.
"""

import re
import time

from coach import code_tests, tutor
from coach.mock import metrics

FOLLOWUPS = 2
CRITERIA = {
    "clarifying_questions": "asked clarifying questions before solving",
    "approach_before_code": "stated an approach before writing code",
    "complexity_stated": "stated time and space complexity",
    "tests_walked": "walked through a test case and edge cases",
    "thinking_aloud": "thought aloud while coding",
}
CHECK_IN = "How is it going? Talk me through where you are."
WALKTHROUGH = ("Okay. Before we look at any results, walk me through your code on a small "
               "example, and tell me one edge case you thought about.")
CLOSING = "Thanks, that's time. Your reports are below."

INTERVIEWER_SYSTEM = """You are the interviewer in a live coding round for a machine learning or AI
engineering role. Speak like a real interviewer: natural, neutral and brief - one to three
sentences, at most 60 words, plain text. Never write code, never give the solution, the
algorithm or the formula.

Phase "discuss": answer clarifying questions briefly and accurately - inputs, outputs,
constraints, edge cases - the way an interviewer would. If the candidate has not described
an approach yet, ask them to, with its time and space complexity, before they code. If a
complexity claim is vague or wrong, ask once how they know - do not correct it.
Phase "coding": the candidate is coding. Answer what they ask, briefly; volunteer nothing.

Hints only when the candidate asks for help getting unstuck, at no more than the level the
server allows:
{levels}

Never restate a LeetCode problem statement, its examples or its constraints word for word.
Return JSON: "reply", "level_used" (the hint level the reply uses; 0 when it is not a
hint), "help_request" (true only when the candidate asked for help getting unstuck)."""

REVIEW_SYSTEM = """You are the interviewer in a live coding round; the candidate has finished
coding. React to their last answer in one short, neutral sentence - no grading, no praise
inflation, never the correct answer - then ask the next question you are given, lightly
rephrased if you like. At most 60 words, plain text, no code.
Return JSON: "reply", "level_used" (0), "help_request" (false)."""

COMM_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "verdicts": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "properties": {"criterion": {"type": "string", "enum": list(CRITERIA)},
                           "verdict": {"type": "string", "enum": ["yes", "partly", "no"]},
                           "evidence": {"type": "string"}},
            "required": ["criterion", "verdict", "evidence"]}},
        "suggestions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["verdicts", "suggestions"],
}
COMM_SYSTEM = """You assess how a candidate communicated in a live coding round, from its
timeline (what was said, when code appeared, runs and checks). For each criterion give
"yes", "partly" or "no" and the evidence - a short quote or a timestamped fact from the
timeline. Judge communication, not correctness. Then two or three concrete suggestions
for the next round, each one sentence."""


# --------------------------------------------------------------- the round

def start(problem, user, strict):
    attempt = tutor.new_attempt(problem, user)
    followups = list(((problem.get("rubric") or {}).get("followups") or [])[:FOLLOWUPS])
    followups += [None] * (FOLLOWUPS - len(followups))  # the model picks the missing ones
    attempt.update(mode="mock", strict=bool(strict), phase="discuss", followups=followups,
                   followups_asked=0, coding_started=None, done_at=None, checked_in=0)
    opening = intro_text(problem)
    attempt["messages"].append({"role": "interviewer", "text": opening, "t": 0.0,
                                "kind": "intro", "level": None, "help": False})
    return attempt, opening


def intro_text(problem):
    if problem.get("source") == "own":
        return (f"Here's today's problem: {problem.get('title')}. The statement is on the left. "
                "Take a minute with it, then tell me in your own words what it asks, and ask "
                "me anything that is unclear before you start.")
    return (f"Let's work on {problem.get('label') or problem.get('title')}. Open it on LeetCode "
            "and take a minute to read it. Then tell me in your own words what it asks, and "
            "ask me anything that is unclear before you start.")


def real_code(code):
    """Lines that are the candidate's own work - not blanks, comments,
    imports, def/class headers or `pass` from a starter skeleton."""
    count = 0
    for line in (code or "").splitlines():
        text = line.strip()
        if (not text or text.startswith(("#", "import ", "from ", "def ", "class ", "@"))
                or text in ("pass", "...") or text.startswith(("print(", "\"\"\"", "'''"))):
            continue
        count += 1
    return count


def advance(attempt, code):
    """discuss -> coding at the first real code (two lines of it)."""
    if attempt.get("phase") == "discuss" and real_code(code) >= 2:
        attempt["phase"] = "coding"
        attempt["coding_started"] = round(time.time() - attempt["started"], 1)


def locked(attempt):
    """Run and Check are refused under interview conditions until done."""
    return (attempt.get("mode") == "mock" and attempt.get("strict")
            and attempt.get("phase") in ("discuss", "coding"))


def _say(attempt, text, kind, started):
    attempt["messages"].append({"role": "interviewer", "text": text, "level": None,
                                "help": False, "kind": kind,
                                "t": round(time.time() - attempt["started"], 1)})
    return {"reply": text, "level": None, "help": False, "phase": attempt["phase"],
            "next_level": tutor.ladder_caps(attempt)[1], "guard": dict(tutor.NO_GUARD),
            "seconds": round(time.perf_counter() - started, 2)}


def _log_candidate(attempt, message, ms, voice):
    attempt["messages"].append({"role": "user", "text": message, "ms": ms, "voice": voice,
                                "t": round(time.time() - attempt["started"], 1)})


def interviewer_turn(attempt, code, message, kind, engine, ms=None, voice=False):
    """kind: "message" (typed or spoken), "hint" (Ask for a hint), "done"
    (I'm done coding), "checkin" (the page saw a long quiet stretch)."""
    if attempt.get("mode") != "mock":
        raise tutor.TutorError("This practice session is not an interview round.")
    if attempt["phase"] == "closed":
        raise tutor.TutorError("The round is over - the reports are below.")
    if kind not in ("message", "hint", "done", "checkin"):
        raise tutor.TutorError("Unknown request.")
    if kind == "message" and not (message or "").strip():
        raise tutor.TutorError("Type or say something first.")
    started = time.perf_counter()
    tutor.snapshot(attempt, code, kind)
    advance(attempt, code)

    if kind == "checkin":
        if attempt["strict"] or attempt["phase"] != "coding":
            return {"skipped": True, "phase": attempt["phase"]}
        attempt["checked_in"] += 1
        return _say(attempt, CHECK_IN, "checkin", started)

    if kind == "done":
        if attempt["phase"] in ("discuss", "coding"):
            attempt["phase"] = "review"
            attempt["done_at"] = round(time.time() - attempt["started"], 1)
            out = _say(attempt, WALKTHROUGH, "walkthrough", started)
            out["unlocked"] = bool(attempt["strict"])
            return out
        raise tutor.TutorError("You have already finished coding.")

    if attempt["phase"] == "review":
        _log_candidate(attempt, message or "(asked for a hint)", ms, voice)
        if attempt["followups_asked"] >= FOLLOWUPS:
            attempt["phase"] = "closed"
            out = _say(attempt, CLOSING, "closing", started)
            out["finished"] = True
            return out
        question = attempt["followups"][attempt["followups_asked"]]
        attempt["followups_asked"] += 1
        if engine == "fake":
            text = question or "What is the time and space complexity of your solution, and why?"
            return _say(attempt, text, "followup", started)
        ask = (f"Next question: {question}" if question else
               "Next question: ask ONE follow-up of your own about their solution - its "
               "complexity, an edge case, or a variant of the problem.")
        prompt = "\n\n".join([tutor.problem_text(attempt["problem"]),
                              "Their final code (line numbers added):\n"
                              + code_tests.numbered(code),
                              "Conversation so far (latest last):\n"
                              + tutor.conversation_text(attempt, limit=8), ask])
        out, guard = tutor.guarded_reply(attempt, engine, REVIEW_SYSTEM, prompt, 0)
        reply = _say(attempt, out["reply"], "followup", started)
        reply["guard"] = guard
        return reply

    # discuss / coding: a question, an approach, or a request for help
    caps = tutor.ladder_caps(attempt)
    _reached, next_cap, now_cap = caps
    if kind == "hint":
        _log_candidate(attempt, "(asked for a hint)", ms, voice)
    if engine == "fake":
        # offline: the bank's hint when asked, else a plain interviewer prompt
        out = (tutor.fake_reply(attempt, "hint", caps, message) if kind == "hint" else
               {"reply": "Go on - and what time and space complexity do you expect?",
                "level_used": 0, "help_request": False})
        guard = dict(tutor.NO_GUARD)
    else:
        if kind == "hint":
            request = (f"The candidate asked for a hint. Give one at exactly level {next_cap} "
                       f"({tutor.LEVEL_TEXT[next_cap]}).")
        else:
            request = (f"The candidate said: «{message}»\nIf they are asking for help "
                       f"getting unstuck you may hint up to level {next_cap}; otherwise do not "
                       "hint at all - answer as the interviewer.")
        conditions = ("Interview conditions are ON: the candidate cannot run or test code until "
                      "they say they are done - never suggest running it; ask them to trace it "
                      "by hand instead." if attempt["strict"] else
                      "Interview conditions are off: the candidate may run their code.")
        prompt = "\n\n".join([
            f"Phase: {attempt['phase']}. {conditions}",
            tutor.problem_text(attempt["problem"]),
            "Candidate's code (line numbers added):\n" + code_tests.numbered(code),
            tutor.last_run_text(attempt),
            "Conversation so far (latest last):\n" + tutor.conversation_text(attempt),
            request])
        system = INTERVIEWER_SYSTEM.format(levels="\n".join(
            f"  level {k}: {v}" for k, v in tutor.LEVEL_TEXT.items() if k < 4))
        out, guard = tutor.guarded_reply(attempt, engine, system, prompt, next_cap,
                                         next_cap if kind == "hint" else None)
    reply = tutor.settle_turn(attempt, out, guard, kind, message, next_cap, started,
                              role="interviewer", ms=ms, voice=voice)
    reply["phase"] = attempt["phase"]
    reply.pop("offer_solution", None)
    return reply


# ------------------------------------------------------ communication report

def timeline_text(attempt):
    events = []
    for m in attempt["messages"]:
        who = "Candidate" if m["role"] == "user" else "Interviewer"
        spoken = " (spoken)" if m.get("voice") else ""
        events.append((m["t"], f"{who}{spoken}: {m['text']}"))
    for s in attempt["snapshots"]:
        events.append((s["t"], f"[code: {real_code(s['code'])} lines of their own, at {s['reason']}]"))
    for r in attempt["runs"]:
        if r["kind"] == "check":
            events.append((r["t"], f"[checked: {r.get('passed')} of {r.get('total')} tests passed]"))
        else:
            events.append((r["t"], "[ran: " + ("timed out" if r["timed_out"] else
                                              f"exit {r.get('exit_code')}") + "]"))
    if attempt.get("coding_started") is not None:
        events.append((attempt["coding_started"], "[coding started]"))
    if attempt.get("done_at") is not None:
        events.append((attempt["done_at"], "[said they were done coding]"))
    events.sort(key=lambda e: e[0])
    return "\n".join(f"[{int(t) // 60:02d}:{int(t) % 60:02d}] {text}" for t, text in events)


def quiet_stretch(attempt, until):
    """The longest gap between the candidate's own actions (messages, code
    snapshots, runs) - the page snapshots after about 20 s without typing,
    so a gap here means neither talking nor typing."""
    times = sorted([0.0] + [m["t"] for m in attempt["messages"] if m["role"] == "user"]
                   + [s["t"] for s in attempt["snapshots"]] + [r["t"] for r in attempt["runs"]]
                   + [until])
    return round(max((b - a for a, b in zip(times, times[1:])), default=0.0))


def communication(attempt, engine):
    # pace and timing come from spoken answers only: a typed answer's time
    # measures typing, not speaking
    turns = [{"answer": m["text"], "answer_ms": m.get("ms") if m.get("voice") else None}
             for m in attempt["messages"]
             if m["role"] == "user" and not m["text"].startswith("(")]
    until = round(time.time() - attempt["started"], 1)
    numbers = metrics.compute(turns)
    numbers.update(
        spoken_turns=sum(1 for m in attempt["messages"] if m["role"] == "user" and m.get("voice")),
        time_to_first_code_s=attempt.get("coding_started"),
        coding_seconds=(round(attempt["done_at"] - attempt["coding_started"])
                        if attempt.get("done_at") is not None and attempt.get("coding_started") is not None
                        else None),
        longest_quiet_s=quiet_stretch(attempt, until),
        check_ins=attempt.get("checked_in", 0),
    )
    report = {"metrics": numbers, "verdicts": [], "suggestions": []}
    if engine != "fake" and turns:
        out = tutor.make_caller(engine)(
            COMM_SYSTEM,
            "Criteria:\n" + "\n".join(f"- {k}: {v}" for k, v in CRITERIA.items())
            + "\n\nTimeline:\n" + timeline_text(attempt),
            COMM_SCHEMA)
        seen = set()
        for v in out.get("verdicts") or []:
            if v.get("criterion") in CRITERIA and v["criterion"] not in seen:
                seen.add(v["criterion"])
                report["verdicts"].append({**v, "label": CRITERIA[v["criterion"]]})
        report["suggestions"] = [s for s in (out.get("suggestions") or []) if s][:3]
    return report


def communication_markdown(report):
    m = report["metrics"]
    lines = ["", "## Communication", ""]
    if m.get("answers"):
        lines.append(f"- Your turns: {m['answers']} ({m.get('spoken_turns', 0)} spoken), "
                     f"{m['total_words']} words, {m['filler_per_100_words']} fillers per 100 words")
    if m.get("words_per_minute"):
        lines.append(f"- Pace while answering: {m['words_per_minute']:.0f} words per minute")
    if m.get("time_to_first_code_s") is not None:
        lines.append(f"- Time to first code: {int(m['time_to_first_code_s']) // 60} min "
                     f"{int(m['time_to_first_code_s']) % 60:02d} s")
    lines.append(f"- Longest quiet stretch: {m['longest_quiet_s']} s")
    for v in report["verdicts"]:
        lines.append(f"- {v['label'].capitalize()}: **{v['verdict']}** - {v['evidence']}")
    if report["suggestions"]:
        lines += ["", "Suggestions:", ""] + [f"- {s}" for s in report["suggestions"]]
    return "\n".join(lines)
