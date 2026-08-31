"""System prompt, prompt builders, and structured-output schemas."""

QUESTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "question": {"type": "string"},
        "what_interviewer_is_testing": {"type": "string"},
        "answer_guidance": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["question", "what_interviewer_is_testing", "answer_guidance"],
}

EVALUATION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "overall_score": {"type": "integer"},
        "scores": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "technical_depth": {"type": "integer"},
                "structure": {"type": "integer"},
                "practical_judgment": {"type": "integer"},
                "communication": {"type": "integer"},
            },
            "required": [
                "technical_depth",
                "structure",
                "practical_judgment",
                "communication",
            ],
        },
        "summary": {"type": "string"},
        "strengths": {
            "type": "array",
            "items": {"type": "string"},
        },
        "gaps": {
            "type": "array",
            "items": {"type": "string"},
        },
        "suggested_answer": {"type": "string"},
        "next_steps": {
            "type": "array",
            "items": {"type": "string"},
        },
        "follow_up_question": {"type": "string"},
    },
    "required": [
        "overall_score",
        "scores",
        "summary",
        "strengths",
        "gaps",
        "suggested_answer",
        "next_steps",
        "follow_up_question",
    ],
}


SYSTEM_PROMPT = """You are a rigorous interview coach for Machine Learning Engineer (MLE)
and AI Engineer (AIE) candidates. Act like a realistic interviewer: ask one focused
question at a time, evaluate answers against senior hiring expectations, and give
specific, actionable feedback. Prefer practical tradeoffs, production constraints,
metrics, failure modes, and clear communication over generic textbook recitation.
All scores are integers from 1 (poor) to 10 (outstanding)."""


def build_question_prompt(data):
    role = data.get("role", "MLE")
    topic = data.get("topic", "ML system design")
    level = data.get("level", "Mid-level")
    focus = data.get("focus", "").strip()

    return f"""Create one interview question.

Role track: {role}
Candidate level: {level}
Topic: {topic}
Optional focus from user: {focus or "none"}

Make it realistic for a technical screen or onsite interview. Ask only one question.
Give 3 to 5 short answer_guidance bullets describing what a strong answer covers.
Do not include the ideal answer, because the candidate has not answered yet."""


def build_evaluation_prompt(data, chunk=None):
    is_followup = data.get("source") == "followup"

    prompt = f"""Evaluate this candidate answer.

Role track: {data.get("role", "MLE")}
Candidate level: {data.get("level", "Mid-level")}
Topic: {data.get("topic", "ML system design")}
Interviewer question: {data.get("question", "")}
Time used: {data.get("timeUsed", "unknown")}
Candidate answer: {data.get("answer", "")}
"""

    if is_followup:
        prompt += f"""
This is a follow-up question probing deeper into the candidate's previous answer.
Judge the answer in the context of that exchange: reward building on or correcting
their earlier reasoning, and flag contradictions with it.

Original question: {data.get("parent_question") or "(not recorded)"}
Candidate's answer to the original question: {data.get("parent_answer") or "(not recorded)"}
"""

    if chunk:
        interview = chunk["interview"]
        key_points = "\n".join(f"- {point}" for point in interview["key_points"])
        mistakes = "\n".join(f"- {mistake}" for mistake in interview["common_mistakes"])
        followups = "\n".join(f"- {item}" for item in interview["followups"])
        if is_followup:
            prompt += f"""
Course knowledge-base background for the ORIGINAL question. Use it to judge
technical accuracy, but do NOT treat the key points as this answer's rubric -
they belong to the original question, so hold the candidate only to the parts
this follow-up actually asks about.

Reference model answer to the original question:
{interview["model_answer"]}

Key points of the original question:
{key_points}

Common mistakes in this area:
{mistakes}
"""
        else:
            prompt += f"""
Grade against this rubric from the course knowledge base.

Reference model answer:
{interview["model_answer"]}

Key points a strong answer must hit (use these as the rubric):
{key_points}

Common mistakes to check the answer for:
{mistakes}

Candidate follow-up probes (choose the most fitting one as your follow_up_question,
adapting the wording to build on what the candidate actually said):
{followups}

In "gaps", explicitly call out rubric key points the candidate missed or got wrong.
"""

    prompt += """
Return candid but constructive feedback: 2 to 4 strengths, 2 to 4 gaps, and 3 to 5
concrete next_steps. The suggested_answer should be concise, interview-ready, and
stronger than the candidate answer."""
    return prompt
