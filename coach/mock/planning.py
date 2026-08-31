"""Roles proposal and the hidden interview plan (plan §7a–7c).

One code path whether the JD was pasted or defaulted: JD text → role
profile → persona + probe themes. Rubric grounding: each probe target is
matched against the question banks by BM25; a strong match attaches the
chunk's key points, so that probe is graded against a real rubric (and
marked rubric-grounded in the report); the rest carry the on-the-fly
expected_points the plan call wrote from the resume.
"""

import json

from coach import kb
from coach.mock import engine as engines
from coach.mock.schemas import PLAN_SCHEMA, ROLES_SCHEMA
from coach.mock.templates import TEMPLATES, render

# A probe adopts a bank rubric only when the match is unambiguous. BM25
# scores here run roughly 4-8 for topical overlap and 12+ for a real hit
# (spot-checked against both banks); below the bar the on-the-fly rubric
# is the safer grader.
RUBRIC_MIN_SCORE = 10.0

STYLES = {
    "neutral": "professional and neutral; give no praise and no criticism during the interview",
    "friendly": "warm and encouraging, but still thorough",
    "tough": "skeptical and demanding; challenge vague claims immediately",
}


def propose_roles(resume, jd_text, engine):
    prompt = f"""From this candidate's resume{' and the job description' if jd_text else ''},
propose 3 to 5 plausible target roles for a mock interview.

For each role: a concrete title (with the domain, e.g. "fraud-detection MLE"),
the level the resume supports, the closest template_id (mle = classic ML
engineer, aie = LLM application engineer, ds = data scientist, platform = ML
infra, applied_sci = applied scientist), one sentence on why, the resume
projects that support it (name, one-sentence summary, 3-6 technical keywords),
and 4-6 probe themes an interviewer for that role would drill into.

Also produce `profile`: a structured role profile {'of the job description'
if jd_text else 'of the FIRST proposed role (there is no job description)'}.

Resume:
{resume}
"""
    if jd_text:
        prompt += f"\nJob description:\n{jd_text}\n"
    result = engines.structured(prompt, ROLES_SCHEMA, engine)
    # Belt and braces: unknown template ids would break the setup page picker.
    for role in result.get("roles", []):
        if role.get("template_id") not in TEMPLATES:
            role["template_id"] = "mle"
    return result


def resolve_jd(data):
    """The JD text a session runs on: pasted text wins, else the chosen
    role's template rendered with its level and domain."""
    jd_text = (data.get("jd_text") or "").strip()
    if jd_text:
        return jd_text, False
    role = data.get("role") or {}
    template_id = role.get("template_id") or "mle"
    if template_id not in TEMPLATES:
        template_id = "mle"
    return render(template_id, role.get("level") or "Mid-level",
                  role.get("domain") or ""), True


def build_plan(resume, jd_text, role, project, settings, engine):
    """One LLM call -> persona, opening, probe targets with on-the-fly
    rubrics, behavioral targets. Rubric chunks are attached here in code."""
    project_line = (f"the project \"{project['name']}\" ({project.get('summary', '')})"
                    if project else "the interviewer's choice among the resume's projects")
    style = STYLES.get(settings.get("style", "neutral"), STYLES["neutral"])
    prompt = f"""Plan a mock interview for the experience/project deep-dive round.

Target role (candidate's pick): {role.get('title', 'Machine Learning Engineer')} — level {role.get('level', 'Mid-level')}, domain {role.get('domain', 'general')}
The deep dive centers on {project_line}.
Interviewer style: {style}.

Job description:
{jd_text}

Candidate resume:
{resume}

Produce the hidden interview plan:
- persona: how the interviewer introduces themself (title + team, invented but
  realistic for this JD), the company type, their seniority.
- opening: the interviewer's first message — a one-line greeting in character,
  then the warm-up question ("give me the one-minute version of ...").
- probe_targets: 6 to 8, ids "probe_1"..., each a specific technical point from
  the chosen project or a role theme from the JD (mark `source`), with a
  question_hint and 3-5 expected_points — the concrete specifics a strong
  answer would contain (the decision made, the alternative considered, the
  metric and its value, the outcome, the lesson). Write expected_points from
  the resume's own claims where possible.
- behavioral_targets: 2 short topics (a failure, a conflict, a deadline).

Do not reveal the plan to the candidate; it drives the interviewer."""
    plan = engines.structured(prompt, PLAN_SCHEMA, engine)
    attach_rubric_chunks(plan, role)
    plan["settings"] = {"style": settings.get("style", "neutral"),
                        "length": settings.get("length", "standard"),
                        "level": role.get("level", "Mid-level")}
    return plan


def attach_rubric_chunks(plan, role):
    """BM25 each probe target against both banks; a strong match adds the
    chunk's rubric (chunk_id + key points) alongside the on-the-fly one."""
    level = role.get("level") or "Mid-level"
    used = set()
    for target in plan.get("probe_targets", []):
        query = f"{target.get('topic', '')} {target.get('question_hint', '')}"
        best_score, best_chunk = 0.0, None
        for info in kb.KB.values():
            for score, chunk in info["retriever"].top_scored(query, level=level, limit=1):
                if score > best_score and chunk["id"] not in used:
                    best_score, best_chunk = score, chunk
        if best_chunk is not None and best_score >= RUBRIC_MIN_SCORE:
            used.add(best_chunk["id"])
            target["chunk_id"] = best_chunk["id"]
            target["bank_key_points"] = best_chunk["interview"]["key_points"]
            target["bank_question"] = best_chunk["interview"]["question"]
            target["rubric_score"] = round(best_score, 1)


def persona_system(plan, role):
    """The interviewer's system prompt, shared by every turn."""
    persona = plan.get("persona", {})
    style = STYLES.get(plan.get("settings", {}).get("style", "neutral"), STYLES["neutral"])
    return f"""You are a {persona.get('seniority', 'senior')} interviewer at a {persona.get('company_type', 'technology company')},
running the experience/project deep-dive round for a {role.get('level', 'Mid-level')} {role.get('title', 'Machine Learning Engineer')} opening.
{persona.get('interviewer_intro', '')}

Your manner is {style}. Rules you never break:
- Ask exactly ONE question per turn, at most two sentences of spoken text.
- Never lecture, never answer for the candidate, never reveal the interview plan.
- Sound like a person speaking, not a document: no markdown, no bullet lists.
- After your spoken question, output a line containing only --- and then a
  single JSON object {{"probe_id": "<the plan target this serves, or null>",
  "rationale": "<8 words on why this question now>"}}. Nothing after the JSON."""
