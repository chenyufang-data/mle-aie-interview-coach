"""Engine selection for the mock endpoints, and the offline fake engine.

The mock reuses grading_route(): paid keys get DeepSeek Flash turns
(quota-free) with "Always Claude" available; the free tier cannot run a
mock — it needs a real LLM (plan §7d) — and gets a clear 403. In --mock
dev mode every LLM call is served by the deterministic fake below, which
is also what the offline unit tests and CI exercise.
"""

import hashlib
import json

from coach import grading


class MockUnavailable(Exception):
    """Raised when this user cannot run a mock interview (HTTP 403)."""


def pick_engine(user, force_llm=False):
    engine, reason = grading.grading_route(user, force_llm=force_llm)
    if engine != "local":
        return engine
    if reason == "mock":
        return "fake"
    raise MockUnavailable(
        "The mock interview needs a real LLM interviewer. Enter a paid access "
        "key, or run the server in --mock mode for the offline demo interviewer."
    )


# ------------------------------------------------------------- fake engine

def _pick(seed, options):
    digest = int(hashlib.sha1(seed.encode("utf-8")).hexdigest(), 16)
    return options[digest % len(options)]


def fake_structured(prompt, schema):
    """Deterministic stand-in for call_model in --mock mode.

    Recognizes the mock's three schema calls by their top-level properties
    and fabricates plausible, stable output from the prompt text alone —
    enough for the full flow (roles → plan → turns → report) to run
    offline, for tests, CI and the free demo.
    """
    properties = schema.get("properties", {})
    if "roles" in properties:
        return _fake_roles(prompt)
    if "probe_targets" in properties:
        return _fake_plan(prompt)
    if "per_turn" in properties:
        return _fake_report(prompt)
    raise ValueError("fake_structured: unrecognized schema")


def _keywords(prompt, limit=8):
    import re
    # Fabricate from the resume section only, not the instructions around it
    # (otherwise "Mid-level" and "deep-dive" become probe topics).
    for marker in ("Candidate resume:", "Resume:"):
        if marker in prompt:
            prompt = prompt.split(marker, 1)[1]
            break
    seen = []
    for token in re.findall(r"\b[A-Z][A-Za-z0-9]*[A-Z0-9][A-Za-z0-9]*\b|\b[A-Za-z]+-[A-Za-z0-9]+\b", prompt):
        if token.lower() not in (s.lower() for s in seen):
            seen.append(token)
        if len(seen) == limit:
            break
    return seen or ["machine learning"]


def _fake_roles(prompt):
    keywords = _keywords(prompt)
    project = {"name": "Main project", "summary": "The project described in the resume.",
               "keywords": keywords[:5]}
    return {
        "profile": {
            "title": "Machine Learning Engineer", "level": "Mid-level",
            "domain": ", ".join(keywords[:2]),
            "must_have_skills": keywords[:4],
            "responsibilities": ["Own models end to end"],
            "stack": keywords[:3],
            "what_they_evaluate": ["Measured decisions", "Production judgment"],
        },
        "roles": [
            {"title": "Machine Learning Engineer", "level": "Mid-level",
             "domain": ", ".join(keywords[:2]), "template_id": "mle",
             "why": "Closest match to the resume's modelling work (offline demo).",
             "projects": [project], "probe_themes": keywords[:5]},
            {"title": "AI / LLM Application Engineer", "level": "Mid-level",
             "domain": "LLM applications", "template_id": "aie",
             "why": "Fits the LLM-flavoured parts of the resume (offline demo).",
             "projects": [project], "probe_themes": ["RAG", "evaluation", "cost and latency"]},
        ],
    }


def _fake_plan(prompt):
    keywords = _keywords(prompt, limit=10)
    targets = []
    for index, keyword in enumerate(keywords[:6]):
        targets.append({
            "id": f"probe_{index + 1}",
            "topic": keyword,
            "question_hint": f"Walk me through how you used {keyword} and why.",
            "source": "project" if index % 2 == 0 else "role_theme",
            "expected_points": [
                f"what decision was made around {keyword}",
                "which alternative was considered and rejected",
                "the metric that justified it, with a number",
            ],
        })
    return {
        "persona": {"interviewer_intro": "I'm the hiring manager for this role.",
                    "company_type": "product company", "seniority": "senior"},
        "opening": "Thanks for joining. To warm up: give me the one-minute "
                   "version of the project you're most proud of.",
        "probe_targets": targets,
        "behavioral_targets": ["a time something you shipped went wrong"],
    }


def _fake_report(prompt):
    # Turn count from the transcript embedded in the prompt.
    turns = prompt.count("Candidate:")
    per_turn = [{
        "turn": index + 1, "question": f"(question {index + 1})",
        "answer_summary": "Offline demo summary.",
        "what_was_good": "Answered the question asked.",
        "what_was_missing": "Concrete numbers and a named alternative.",
        "stronger_answer": "State the decision, the alternative, the metric with its value, the outcome.",
        "score": _pick(f"turn{index}{prompt[:64]}", [5, 6, 7]),
    } for index in range(max(1, turns))]
    return {
        "overall_impression": "Offline demo report: the interview flow works; "
                              "scores here are placeholders, not judgments.",
        "overall_call": "lean hire",
        "scores": {"technical_depth": 6, "ownership_specificity": 6, "structure": 6,
                   "communication_clarity": 6, "concision_time": 6, "reflection": 6},
        "justifications": {name: "Offline demo — run with a real LLM for graded feedback."
                           for name in ("technical_depth", "ownership_specificity", "structure",
                                        "communication_clarity", "concision_time", "reflection")},
        "per_turn": per_turn,
        "red_flags": [],
        "keep_doing": ["Practicing full sessions end to end."],
        "top_actions": ["Add a number to every claim.",
                        "Name the alternative you rejected.",
                        "Close each answer with the outcome."],
    }


def fake_turn(state):
    """Deterministic interviewer turn for --mock mode: serve the next unasked
    probe's question hint (or the phase's canned line)."""
    from coach.mock import turns as turns_module
    phase, turn_no, done = turns_module.position(state)
    if done:
        return {"spoken": "That's all we have time for — thank you.", "meta": {}}
    if phase == "warmup":
        return {"spoken": state["plan"]["opening"], "meta": {}}
    if phase == "walkthrough":
        return {"spoken": "Pick the project most relevant to this role and walk me "
                          "through it end to end: the problem, your role, the approach, "
                          "and how you knew it worked.", "meta": {}}
    if phase == "behavioral":
        target = (state["plan"].get("behavioral_targets") or ["a difficult moment"])[0]
        return {"spoken": f"Tell me about {target}, and what you did about it.",
                "meta": {}}
    if phase == "closing":
        return {"spoken": "Last one: knowing everything you know now, what would you "
                          "do differently on that project?", "meta": {}}
    asked = {t.get("probe_id") for t in state.get("transcript", [])}
    for target in state["plan"]["probe_targets"]:
        if target["id"] not in asked:
            return {"spoken": target["question_hint"],
                    "meta": {"probe_id": target["id"], "rationale": "next unprobed target"}}
    return {"spoken": "Dig one level deeper on your last answer: what was the "
                      "hardest tradeoff, and how did you decide?", "meta": {}}


def structured(prompt, schema, engine):
    if engine == "fake":
        return fake_structured(prompt, schema)
    from coach.llm import call_model
    return call_model(prompt, schema, engine)


def chat(system, messages, engine, thinking=False, max_tokens=700):
    if engine == "fake":
        raise RuntimeError("fake engine has no free-form chat; use fake_turn")
    from coach.llm import call_chat
    # cache=True: interviewer turns share the frozen persona + append-only
    # history prefix, the shape prompt caching pays for (plan section 5c).
    return call_chat(system, messages, engine, thinking=thinking,
                     max_tokens=max_tokens, cache=True)
