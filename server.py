from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib import error as urlerror
from urllib import request as urlrequest
import argparse
import json
import mimetypes
import os
import random
import threading

import anthropic

from retrieval import Retriever, tokenize


BASE_DIR = Path(__file__).resolve().parent
PUBLIC_DIR = BASE_DIR / "public"
ENV_PATH = BASE_DIR / ".env"
# One question-bank corpus per track; both share the id/interview/metadata schema.
CORPUS_PATHS = {
    "MLE": BASE_DIR / "rag_ml" / "all_chunks.jsonl",
    "AIE": BASE_DIR / "rag_ai" / "all_chunks.jsonl",
}
DEFAULT_MODEL = "claude-opus-4-8"


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


# role -> {"chunks": [...], "retriever": Retriever, "modules": [...]}
KB = {}
# Chunk ids are unique across both corpora, so evaluation can look them up globally.
CHUNKS_BY_ID = {}

# Trained distilled grader for mock mode (grader/train.py artifact); None
# falls back to plain keyword matching.
GRADER_PATH = BASE_DIR / "grader" / "model.joblib"
GRADER = None

# Every real (non-mock) graded exchange is appended here: each row is a gold
# (answer, teacher score) pair for evaluating and later retraining the
# distilled grader, and the practice history for progress features.
# Overridable so Docker can point it at a persistent volume.
REAL_SESSIONS_PATH = Path(
    os.environ.get("REAL_SESSIONS_PATH", BASE_DIR / "grader" / "real_sessions.jsonl")
)

# Freemium tiers (Claude mode only): a request's X-Access-Key header is looked
# up in USERS_PATH; keys with tier "paid" get Claude grading, capped at
# PAID_DAILY_QUOTA Claude calls per day (question generation + evaluation
# combined). Everyone else gets the distilled local grader. When USERS_PATH
# does not exist, tiers are disabled and every request grades with Claude —
# the original single-user behaviour.
USERS_PATH = Path(os.environ.get("USERS_PATH", BASE_DIR / "users.json"))
USAGE_PATH = Path(os.environ.get("USAGE_PATH", BASE_DIR / "grader" / "usage.json"))
PAID_DAILY_QUOTA = int(os.environ.get("PAID_DAILY_QUOTA", "30"))
USERS = {}
# True whenever USERS_PATH exists — even if it fails to parse. A present-but-
# broken users file must fail CLOSED (tiers on, no paid keys -> everyone free),
# never open (everyone gets Claude), or a corrupt file becomes a cost leak.
TIERS_ENABLED = False
_usage_lock = threading.Lock()

# "claude" (default, needs ANTHROPIC_API_KEY), "mock" (free, offline), or
# "ollama" (free local model, needs Ollama running at localhost:11434).
MODE = "claude"
OLLAMA_MODEL = "llama3.2"
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"

_client = None


def load_env_file():
    if not ENV_PATH.exists():
        return

    for raw_line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            # Tolerate a bare Anthropic key pasted without a variable name.
            if line.startswith("sk-ant-"):
                os.environ.setdefault("ANTHROPIC_API_KEY", line)
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def load_chunks():
    for role, path in CORPUS_PATHS.items():
        if not path.exists():
            print(f"Warning: {path} not found; {role} course knowledge base disabled.")
            continue
        with path.open(encoding="utf-8") as handle:
            chunks = [json.loads(line) for line in handle if line.strip()]
        KB[role] = {
            "chunks": chunks,
            "retriever": Retriever(chunks),
            # Module names in corpus (class) order, for the topic dropdown.
            "modules": list(dict.fromkeys(chunk["metadata"]["module"] for chunk in chunks)),
        }
        CHUNKS_BY_ID.update({chunk["id"]: chunk for chunk in chunks})


def log_real_session(data, result, user=None):
    """Persist a real graded exchange. Local file only; never breaks a response."""
    try:
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "user": (user or {}).get("name", "anonymous"),
            "graded_by": MODE,
            "model": (os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL)
                      if MODE == "claude" else OLLAMA_MODEL),
            "role": data.get("role"),
            "level": data.get("level"),
            "topic": data.get("topic"),
            "source": data.get("source"),
            "chunk_id": data.get("chunk_id"),
            "question": data.get("question"),
            "answer": data.get("answer"),
            "timeUsed": data.get("timeUsed"),
            "evaluation": result,
        }
        with REAL_SESSIONS_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        print(f"Warning: could not log session ({exc})")


def load_grader():
    global GRADER
    if not GRADER_PATH.exists():
        return
    try:
        import joblib
        from grader import features  # noqa: F401  (unpickling needs FeatureExtractor importable)

        GRADER = joblib.load(GRADER_PATH)
    except Exception as exc:
        print(f"Warning: could not load ML grader ({exc}); mock grading uses keyword matching.")


def load_users():
    global USERS, TIERS_ENABLED
    if not USERS_PATH.exists():
        return
    TIERS_ENABLED = True
    try:
        # utf-8-sig: tolerate the BOM that Windows editors and PowerShell
        # (Set-Content -Encoding utf8) prepend.
        USERS = json.loads(USERS_PATH.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        print(
            f"Warning: could not read {USERS_PATH.name} ({exc}); "
            "tiers stay ON with no paid keys - every request is free tier."
        )


def resolve_user(handler):
    key = (handler.headers.get("X-Access-Key") or "").strip()
    entry = USERS.get(key)
    if entry and entry.get("tier") == "paid":
        return {"key": key, "name": entry.get("name", "paid user"), "tier": "paid"}
    return {"key": None, "name": "anonymous", "tier": "free"}


def _read_usage():
    if not USAGE_PATH.exists():
        return {}
    try:
        return json.loads(USAGE_PATH.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


def quota_left(user):
    if user["tier"] != "paid":
        return 0
    today = datetime.now().date().isoformat()
    with _usage_lock:
        row = _read_usage().get(user["key"])
    used = row["used"] if row and row.get("date") == today else 0
    return max(0, PAID_DAILY_QUOTA - used)


def quota_take(user):
    """Reserve one paid Claude call; False once today's quota is spent."""
    today = datetime.now().date().isoformat()
    with _usage_lock:
        usage = _read_usage()
        row = usage.get(user["key"])
        used = row["used"] if row and row.get("date") == today else 0
        if used >= PAID_DAILY_QUOTA:
            return False
        usage[user["key"]] = {"date": today, "used": used + 1}
        USAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
        USAGE_PATH.write_text(json.dumps(usage), encoding="utf-8")
    return True


def grading_route(user):
    """Decide whether this request may spend an LLM call.

    Returns (use_llm, reason); reason says why a request routed local:
    "mock" (dev mode), "free" (no paid key), "quota" (paid but spent today).
    """
    if MODE == "mock":
        return False, "mock"
    if MODE == "ollama":
        return True, None
    if not TIERS_ENABLED:
        return True, None  # tiers disabled: single-user setup grades with Claude
    if user["tier"] != "paid":
        return False, "free"
    if not quota_take(user):
        return False, "quota"
    return True, None


def get_api_key():
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_KEY", "API_KEY"):
        value = os.environ.get(name)
        if value:
            return value
    return None


def get_client():
    global _client
    if _client is None:
        api_key = get_api_key()
        if not api_key:
            raise RuntimeError("No Anthropic API key found. Add ANTHROPIC_API_KEY to .env.")
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


def call_claude(user_prompt, schema):
    response = get_client().messages.create(
        model=os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL),
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": user_prompt}],
        output_config={"format": {"type": "json_schema", "schema": schema}},
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("The model declined this request. Try a different topic or focus.")
    text = next((block.text for block in response.content if block.type == "text"), "")
    if not text:
        raise RuntimeError("Model response did not include output text")
    return json.loads(text)


def call_ollama(user_prompt, schema):
    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "format": schema,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    }
    req = urlrequest.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=300) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urlerror.URLError as exc:
        raise RuntimeError(
            "Could not reach Ollama at localhost:11434. Install it from https://ollama.com, "
            f"run 'ollama pull {OLLAMA_MODEL}', and make sure the Ollama app is running."
        ) from exc
    content = body.get("message", {}).get("content", "")
    if not content:
        raise RuntimeError("Ollama response did not include output text")
    return json.loads(content)


def call_model(user_prompt, schema):
    if MODE == "ollama":
        return call_ollama(user_prompt, schema)
    return call_claude(user_prompt, schema)


def clamp_score(value):
    return max(1, min(10, value))


# Why an evaluation was routed to local grading -> (summary label, user hint).
LOCAL_GRADING_LABELS = {
    "mock": ("Mock mode", "Run without --mock for real Claude grading."),
    "free": ("Free tier", "Enter a paid access key for full Claude grading."),
    "quota": ("Daily Claude quota reached", "Quota resets tomorrow; until then grading is local."),
}


def mock_evaluation(data, chunk, reason="mock"):
    label, hint = LOCAL_GRADING_LABELS[reason]
    answer = data.get("answer", "")
    answer_tokens = set(tokenize(answer))
    word_count = len(answer.split())

    if chunk:
        interview = chunk["interview"]
        if GRADER is not None:
            # Distilled grader: trained model predicts the score; per-key-point
            # coverage from its feature extractor drives the hit/miss lists.
            extractor = GRADER["extractor"]
            coverage_scores = extractor.key_point_coverage(answer, chunk)
            hits = [p for p, c in zip(interview["key_points"], coverage_scores) if c >= 0.35]
            misses = [p for p, c in zip(interview["key_points"], coverage_scores) if c < 0.35]
            predicted = GRADER["model"].predict([extractor.extract(answer, chunk)])[0]
            overall = clamp_score(round(predicted))
            summary = (
                f"[{label} - local ML grader ({GRADER['model_name']})] "
                f"Predicted {overall}/10; matched {len(hits)} of "
                f"{len(interview['key_points'])} rubric key points. {hint}"
            )
        else:
            hits, misses = [], []
            for point in interview["key_points"]:
                point_tokens = set(tokenize(point))
                overlap = len(point_tokens & answer_tokens) / max(1, len(point_tokens))
                (hits if overlap >= 0.35 else misses).append(point)
            coverage = len(hits) / max(1, len(interview["key_points"]))
            overall = clamp_score(round(2 + 8 * coverage))
            summary = (
                f"[{label} - keyword matching] Found {len(hits)} of "
                f"{len(interview['key_points'])} rubric key points in your answer. {hint}"
            )
        return {
            "overall_score": overall,
            "scores": {
                "technical_depth": overall,
                "structure": clamp_score(overall + (1 if word_count >= 60 else -1)),
                "practical_judgment": overall,
                "communication": clamp_score(overall + (1 if 40 <= word_count <= 250 else -1)),
            },
            "summary": summary,
            "strengths": (
                [f"Touched on: {point}" for point in hits[:4]]
                or ["You submitted an answer; no rubric key points were detected by keyword matching."]
            ),
            "gaps": (
                [f"Missed rubric point: {point}" for point in misses[:4]]
                or ["No rubric gaps detected by keyword matching - compare with the model answer to be sure."]
            ),
            "suggested_answer": interview["model_answer"],
            "next_steps": (
                [f"Watch out for: {mistake}" for mistake in interview["common_mistakes"][:3]]
                + ["Compare your answer with the model answer above, then re-record a tighter version."]
            ),
            "follow_up_question": random.choice(interview["followups"]),
        }

    overall = clamp_score(3 + min(4, word_count // 40))
    return {
        "overall_score": overall,
        "scores": {
            "technical_depth": overall,
            "structure": overall,
            "practical_judgment": overall,
            "communication": overall,
        },
        "summary": (
            f"[{label}] This question has no knowledge-base rubric, so the score "
            f"is a rough length-based placeholder. {hint}"
        ),
        "strengths": ["Answer recorded with local grading."],
        "gaps": ["Local grading cannot judge content for questions without a rubric."],
        "suggested_answer": f"Local grading cannot draft a model answer for this question. {hint}",
        "next_steps": [
            "Practice knowledge-base topics, which grade against a real rubric even locally.",
            hint,
        ],
        "follow_up_question": "How would you measure whether your proposed approach actually works in production?",
    }


def select_chunk(role, module, level, focus, exclude_ids):
    kb = KB.get(role)
    if kb is None:
        path = CORPUS_PATHS.get(role)
        detail = f"{path.parent.name}/{path.name} missing" if path else f"unknown track {role!r}"
        raise RuntimeError(f"Course knowledge base for the {role} track is not loaded ({detail}).")

    candidates = kb["retriever"].search(
        query=focus,
        module=module,
        level=level,
        exclude_ids=exclude_ids,
        limit=5,
    )
    # Any of the top matches is a good serve; sampling keeps repeat sessions
    # from always landing on the same question.
    return random.choice(candidates)


def kb_question_payload(chunk):
    meta = chunk["metadata"]
    return {
        "question": chunk["interview"]["question"],
        "what_interviewer_is_testing": f"{meta['module']} - {meta['topic']} ({meta['difficulty']})",
        "answer_guidance": chunk["interview"]["key_points"],
        "chunk_id": chunk["id"],
        "source": "kb",
    }


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


def json_response(handler, status, payload):
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def read_json(handler):
    length = int(handler.headers.get("Content-Length", "0"))
    if length == 0:
        return {}
    return json.loads(handler.rfile.read(length).decode("utf-8"))


class InterviewCoachHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/meta":
            # The frontend builds the knowledge-base topic groups from this,
            # so the module lists live only in the corpora, not in app.js.
            user = resolve_user(self)
            json_response(self, 200, {
                "kb": {
                    role: {"modules": info["modules"], "chunks": len(info["chunks"])}
                    for role, info in KB.items()
                },
                "user": {
                    "name": user["name"],
                    "tier": user["tier"],
                    "tiers_enabled": TIERS_ENABLED and MODE == "claude",
                    "paid_quota": PAID_DAILY_QUOTA,
                    "paid_left_today": quota_left(user),
                },
            })
            return
        if path == "/":
            path = "/index.html"

        file_path = (PUBLIC_DIR / path.lstrip("/")).resolve()
        if not str(file_path).startswith(str(PUBLIC_DIR.resolve())):
            self.send_error(403)
            return
        if not file_path.exists() or not file_path.is_file():
            self.send_error(404)
            return

        content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        body = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        try:
            data = read_json(self)
            if self.path == "/api/question":
                if data.get("source") == "kb":
                    chunk = select_chunk(
                        data.get("role", "MLE"),
                        data.get("topic", ""),
                        data.get("level", ""),
                        data.get("focus", "").strip(),
                        set(data.get("exclude") or []),
                    )
                    json_response(self, 200, kb_question_payload(chunk))
                    return
                use_llm, _ = grading_route(resolve_user(self))
                if not use_llm:
                    # Local route (mock mode, free tier, or spent quota): serve a
                    # knowledge-base question relevant to the chosen general
                    # topic instead of spending an LLM call.
                    chunk = select_chunk(
                        data.get("role", "MLE"),
                        "Any course topic",
                        data.get("level", ""),
                        f"{data.get('topic', '')} {data.get('focus', '')}".strip(),
                        set(data.get("exclude") or []),
                    )
                    json_response(self, 200, kb_question_payload(chunk))
                    return
                result = call_model(build_question_prompt(data), QUESTION_SCHEMA)
                result["source"] = "ai"
                json_response(self, 200, result)
                return

            if self.path == "/api/evaluate":
                if not data.get("answer", "").strip():
                    json_response(self, 400, {"error": "Answer is required."})
                    return
                user = resolve_user(self)
                chunk = CHUNKS_BY_ID.get(data.get("chunk_id") or "")
                use_llm, reason = grading_route(user)
                if not use_llm:
                    # A follow-up has no rubric of its own; keyword-matching the
                    # parent chunk's key points against it would mis-grade.
                    local_chunk = None if data.get("source") == "followup" else chunk
                    json_response(self, 200, mock_evaluation(data, local_chunk, reason))
                    return
                result = call_model(
                    build_evaluation_prompt(data, chunk),
                    EVALUATION_SCHEMA,
                )
                log_real_session(data, result, user)
                json_response(self, 200, result)
                return

            json_response(self, 404, {"error": "Unknown endpoint."})
        except anthropic.AuthenticationError:
            json_response(self, 500, {"error": "Anthropic API key was rejected. Check ANTHROPIC_API_KEY in .env."})
        except anthropic.RateLimitError:
            json_response(self, 500, {"error": "Rate limited by the Anthropic API. Wait a moment and try again."})
        except anthropic.APIStatusError as exc:
            json_response(self, 500, {"error": f"Anthropic API error {exc.status_code}: {exc.message}"})
        except anthropic.APIConnectionError:
            json_response(self, 500, {"error": "Could not reach the Anthropic API. Check your network connection."})
        except Exception as exc:
            json_response(self, 500, {"error": str(exc)})

    def log_message(self, format, *args):
        print("%s - %s" % (self.address_string(), format % args))


def parse_args():
    parser = argparse.ArgumentParser(description="MLE/AIE Interview Coach server")
    backend = parser.add_mutually_exclusive_group()
    backend.add_argument(
        "--mock",
        action="store_true",
        help="Free testing mode: no API key, no network, no cost. Questions come from the "
        "course knowledge base and grading uses the trained local ML grader "
        "(grader/model.joblib), falling back to rubric keyword matching.",
    )
    backend.add_argument(
        "--ollama",
        nargs="?",
        const="llama3.2",
        metavar="MODEL",
        help="Use a free local Ollama model instead of the Anthropic API (default: llama3.2). "
        "Requires Ollama running at localhost:11434.",
    )
    return parser.parse_args()


def main():
    global MODE, OLLAMA_MODEL

    args = parse_args()
    if args.mock:
        MODE = "mock"
    elif args.ollama:
        MODE = "ollama"
        OLLAMA_MODEL = args.ollama

    load_env_file()
    load_chunks()
    load_users()
    # Claude mode needs the local grader too when tiers are on: it serves
    # free-tier and over-quota requests.
    if MODE == "mock" or (MODE == "claude" and TIERS_ENABLED):
        load_grader()
    # HOST=0.0.0.0 is required inside a container; the localhost default keeps
    # a bare `python server.py` private to this machine.
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    server = ThreadingHTTPServer((host, port), InterviewCoachHandler)
    display_host = "127.0.0.1" if host == "0.0.0.0" else host
    print(f"MLE/AIE Interview Coach running at http://{display_host}:{port}")
    for role, info in KB.items():
        print(
            f"Knowledge base [{role}]: {len(info['chunks'])} interview chunks "
            f"from {CORPUS_PATHS[role].parent.name}/."
        )
    if MODE == "mock":
        brain = (
            f"trained ML grader ({GRADER['model_name']})"
            if GRADER is not None
            else "keyword matching (train one with grader\\train.py)"
        )
        print(f"Backend: MOCK mode (free) - KB questions, grading via {brain}.")
    elif MODE == "ollama":
        print(f"Backend: Ollama local model '{OLLAMA_MODEL}' (free) at localhost:11434.")
    else:
        print(f"Backend: Anthropic API, model {os.environ.get('ANTHROPIC_MODEL', DEFAULT_MODEL)}.")
        if TIERS_ENABLED:
            paid = sum(1 for entry in USERS.values() if entry.get("tier") == "paid")
            brain = GRADER["model_name"] if GRADER is not None else "keyword matching"
            print(
                f"Tiers: ON - {paid} paid key(s), {PAID_DAILY_QUOTA} Claude calls/day each; "
                f"free tier grades with the local model ({brain})."
            )
        else:
            print("Tiers: off (no users.json) - every request grades with Claude.")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
