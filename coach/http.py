"""The HTTP handler: API routes plus static serving of public/."""

import mimetypes
from http.server import BaseHTTPRequestHandler

import anthropic

from coach import config, grading, kb, sessions, stt_dev, users
from coach.mock import routes as mock_routes
from coach.config import PAID_CASCADE, PAID_DAILY_QUOTA, PUBLIC_DIR
from coach.llm import call_model, engine_model
from coach.prompts import (EVALUATION_SCHEMA, QUESTION_SCHEMA,
                           build_evaluation_prompt, build_question_prompt)
from coach.web import json_response, read_json


class InterviewCoachHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/meta":
            # The frontend builds the knowledge-base topic groups from this,
            # so the module lists live only in the corpora, not in app.js.
            user = users.resolve_user(self)
            json_response(self, 200, {
                "kb": {
                    role: {"modules": info["modules"], "chunks": len(info["chunks"])}
                    for role, info in kb.KB.items()
                },
                "user": {
                    "name": user["name"],
                    "tier": user["tier"],
                    "tiers_enabled": users.TIERS_ENABLED and config.MODE == "claude",
                    "paid_quota": PAID_DAILY_QUOTA,
                    "paid_left_today": users.quota_left(user),
                    # Paid tier's default (quota-free) judge; "claude" means
                    # no DeepSeek key is configured and quota meters everything.
                    "paid_grader": (config.deepseek_model() if config.deepseek_available()
                                    else "claude"),
                },
            })
            return
        if path.startswith("/api/stt/"):
            stt_dev.stt_get(self, path)
            return
        if path.startswith("/api/mock/"):
            mock_routes.handle_get(self, path)
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
        # Without this, browsers heuristically cache the HTML/JS and serve
        # stale pages after an update (a shipped feature "not showing up").
        # The files are tiny and local, so always revalidating is free.
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        try:
            data = read_json(self)
            if self.path == "/api/stt/record":
                stt_dev.stt_record(self, data)
                return
            if self.path.startswith("/api/mock/"):
                mock_routes.handle_post(self, self.path, data)
                return
            if self.path == "/api/question":
                if data.get("chunk_id"):
                    # Exact-question request: the mock report's rubric
                    # tie-in links a missed probe straight to the bank
                    # question it was graded against (plan section 9 P3).
                    chunk = kb.CHUNKS_BY_ID.get(data["chunk_id"])
                    if chunk is None:
                        json_response(self, 404, {
                            "error": f"unknown chunk_id {data['chunk_id']!r}"})
                        return
                    json_response(self, 200, kb.kb_question_payload(chunk))
                    return
                if data.get("source") == "kb":
                    chunk = kb.select_chunk(
                        data.get("role", "MLE"),
                        data.get("topic", ""),
                        data.get("level", ""),
                        data.get("focus", "").strip(),
                        set(data.get("exclude") or []),
                    )
                    json_response(self, 200, kb.kb_question_payload(chunk))
                    return
                engine, _ = grading.grading_route(users.resolve_user(self))
                if engine != "local":
                    try:
                        result = call_model(build_question_prompt(data),
                                            QUESTION_SCHEMA, engine)
                        result["source"] = "ai"
                        json_response(self, 200, result)
                        return
                    except Exception:
                        if engine != "deepseek":
                            raise
                        # DeepSeek unreachable: fall through to the KB serve
                        # rather than surprise-spending Claude credits.
                # Local route (mock mode, free tier, or spent quota): serve a
                # knowledge-base question relevant to the chosen general
                # topic instead of spending an LLM call.
                chunk = kb.select_chunk(
                    data.get("role", "MLE"),
                    "Any course topic",
                    data.get("level", ""),
                    f"{data.get('topic', '')} {data.get('focus', '')}".strip(),
                    set(data.get("exclude") or []),
                )
                json_response(self, 200, kb.kb_question_payload(chunk))
                return

            if self.path == "/api/evaluate":
                if not data.get("answer", "").strip():
                    json_response(self, 400, {"error": "Answer is required."})
                    return
                user = users.resolve_user(self)
                chunk = kb.CHUNKS_BY_ID.get(data.get("chunk_id") or "")
                # Smart cascade — checked BEFORE grading_route so no quota is
                # taken for answers the student grades reliably. Only for paid
                # users on rubric questions; "force_llm": true opts out.
                if (config.MODE == "claude" and users.TIERS_ENABLED and PAID_CASCADE
                        and user["tier"] == "paid"
                        and chunk is not None
                        and data.get("source") != "followup"
                        and not data.get("force_llm")
                        and grading.cascade_confident(data.get("answer", ""), chunk)):
                    json_response(self, 200, grading.mock_evaluation(data, chunk, "cascade"))
                    return
                engine, reason = grading.grading_route(
                    user, force_llm=bool(data.get("force_llm")))
                if engine != "local":
                    try:
                        result = call_model(
                            build_evaluation_prompt(data, chunk),
                            EVALUATION_SCHEMA, engine,
                        )
                    except Exception:
                        if engine != "deepseek":
                            raise
                        # DeepSeek down or malformed twice: degrade to the
                        # local grader rather than surprise-spending Claude.
                        engine, reason = "local", "llm_error"
                    else:
                        result["graded_by"] = engine_model(engine)
                        sessions.log_real_session(data, result, user, engine)
                        json_response(self, 200, result)
                        return
                # A follow-up has no rubric of its own; keyword-matching the
                # parent chunk's key points against it would mis-grade.
                local_chunk = None if data.get("source") == "followup" else chunk
                result = grading.mock_evaluation(data, local_chunk, reason)
                if config.MODE == "claude":
                    # Tiered deployment (not --mock dev mode): collect the
                    # free-tier answer unlabeled for future teacher labeling.
                    sessions.log_free_session(data, result, user, reason)
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
