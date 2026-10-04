"""Coding drills (roadmap step 7): the coding bank, LeetCode references, the
local code runner, and the author's notes while reviewing the bank.

The bank is banks/rag_code (ingest/ingest_code.py). For a LeetCode problem
it holds number, title and link only, plus our own rubric and hints; the app
never shows, fetches or stores LeetCode's statements (docs/plan.md step 7) -
the user opens the problem on leetcode.com in a separate window.

Every /api/code/ route is local-only: config.CODE_LOCAL is True only when
the server binds loopback (the demo box binds 0.0.0.0 in its container), and
each request must come from loopback, name a loopback Host (DNS rebinding),
and carry the X-Coach-Local header, which a page on another site cannot send
without a CORS preflight that this server never grants. Run executes the
user's own code with this Python on their own machine, in a temporary
folder, with a time limit and none of the server's API keys in its
environment.
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from urllib import parse as urlparse

from coach import config
from coach.web import json_response

LOOPBACK_CLIENTS = ("127.0.0.1", "::1")
LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")
SLUG = re.compile(r"leetcode\.com/problems/([a-z0-9-]+)", re.I)
NUMBER = re.compile(r"(?:lc|leetcode)?\s*#?\s*(\d{1,4})", re.I)
REVIEW_STATUSES = ("keep", "fix", "retire")
CODE_LIMIT = 200_000
OUTPUT_LIMIT = 64_000
# what a child Python needs to start (Windows needs SYSTEMROOT); nothing else
# from the server's environment, which holds the API keys from .env
CHILD_ENV_KEYS = ("PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "HOME", "USERPROFILE", "LANG")

_lock = threading.Lock()
_runs = threading.BoundedSemaphore(2)
_cache = {"mtime": None, "records": []}


# --------------------------------------------------------------------- bank

def records():
    """The bank without retired records, re-read when the file changes."""
    try:
        mtime = config.CODE_BANK_PATH.stat().st_mtime
    except OSError:
        return []
    with _lock:
        if _cache["mtime"] != mtime:
            rows = [json.loads(line) for line in
                    config.CODE_BANK_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
            _cache.update(mtime=mtime, records=[
                r for r in rows if r["metadata"].get("review", {}).get("status") != "retire"])
        return _cache["records"]


def by_id(record_id):
    return next((r for r in records() if r["id"] == record_id), None)


def summary(record):
    c = record["code"]
    return {"id": record["id"], "label": record["interview"]["question"],
            "source": c["source"], "family": c["family"], "number": c["number"],
            "title": c["title"], "difficulty": c["difficulty"], "role": c["role"],
            "group": record["metadata"].get("group", ""), "link": c.get("link")}


def detail(record):
    """What the coding page shows: the summary, the author's own statement
    and starter code when there is one, and the coaching (hints behind a
    button, the rubric behind the self-check)."""
    c, i = record["code"], record["interview"]
    out = summary(record)
    out.update(statement=c.get("statement"), starter_code=c.get("starter_code"),
               approaches=c["approaches"], complexity=c["complexity"], hints=c["hints"],
               rubric={"key_points": i["key_points"], "edge_cases": c["edge_cases"],
                       "common_mistakes": i["common_mistakes"], "code_quality": c["code_quality"],
                       "communication": c["communication"], "followups": i["followups"]},
               review=review_decisions().get(record["id"]))
    return out


# ------------------------------------------------------- LeetCode references

def slugify(title):
    """LeetCode's slug rule: drop punctuation, then spaces and hyphens to one
    hyphen ("Sqrt(x)" -> sqrtx, "Pow(x, n)" -> powx-n)."""
    kept = re.sub(r"[^a-z0-9\s-]", "", title.lower())
    return re.sub(r"[\s-]+", "-", kept).strip("-")


def title_from_slug(slug):
    return " ".join(part.capitalize() for part in slug.split("-"))


def leetcode_link(slug):
    return f"https://leetcode.com/problems/{slug}/"


def resolve(query):
    """The LeetCode problem the user names - a bank entry by number, link or
    title, or any other problem by link or title. Nothing is looked up
    online, so a number outside the bank needs the link or the title."""
    q = (query or "").strip()
    if not q:
        raise ValueError("Type a LeetCode number, title or link.")
    leetcode = [r for r in records() if r["code"]["source"] == "leetcode"]
    match = SLUG.search(q)
    if match:
        slug = match.group(1).lower()
        title = None
    elif NUMBER.fullmatch(q):
        number = int(NUMBER.fullmatch(q).group(1))
        hit = next((r for r in leetcode if r["code"]["number"] == number), None)
        if hit:
            return detail(hit)
        raise ValueError(f"LeetCode {number} is not in your bank, and the app does not look "
                         "problems up online. Paste its link or type its title.")
    else:
        slug, title = slugify(q), re.sub(r"\s+", " ", q)
        if not slug:
            raise ValueError("Type a LeetCode number, title or link.")
    hit = next((r for r in leetcode if r["code"]["slug"] == slug), None)
    if hit:
        return detail(hit)
    return {"id": None, "label": title or title_from_slug(slug), "source": "leetcode",
            "family": "dsa", "number": None, "title": title or title_from_slug(slug),
            "link": leetcode_link(slug), "difficulty": "", "hints": [], "rubric": None,
            "from_title": title is not None}


# --------------------------------------------------------------------- run

def _text(value):
    if value is None:
        return ""
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else value


def _clip(text):
    return text if len(text) <= OUTPUT_LIMIT else text[:OUTPUT_LIMIT] + "\n... (output cut)"


def child_env():
    """The environment for code this app runs: the OS basics, no API keys."""
    env = {k: os.environ[k] for k in CHILD_ENV_KEYS if os.environ.get(k)}
    env.update(PYTHONIOENCODING="utf-8", PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1")
    return env


def run_code(code, stdin=""):
    """Run the user's Python file once; output, exit code, time."""
    with _runs, tempfile.TemporaryDirectory(prefix="coach-run-") as tmp:
        path = Path(tmp) / "solution.py"
        path.write_text(code, encoding="utf-8")
        env = child_env()
        started = time.perf_counter()
        try:
            proc = subprocess.run([sys.executable, "-I", str(path)], input=stdin or "",
                                  capture_output=True, text=True, encoding="utf-8",
                                  errors="replace", cwd=tmp, env=env,
                                  timeout=config.CODE_RUN_TIMEOUT_S)
            stdout, stderr, exit_code, timed_out = proc.stdout, proc.stderr, proc.returncode, False
        except subprocess.TimeoutExpired as exc:
            stdout, stderr, exit_code, timed_out = _text(exc.stdout), _text(exc.stderr), None, True
        seconds = time.perf_counter() - started
    stderr = stderr.replace(str(path), "solution.py")
    return {"stdout": _clip(stdout), "stderr": _clip(stderr), "exit_code": exit_code,
            "timed_out": timed_out, "seconds": round(seconds, 3),
            "time_limit": config.CODE_RUN_TIMEOUT_S}


# ------------------------------------------------------------ bank review

def review_decisions():
    try:
        saved = json.loads(config.CODE_REVIEW_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {d["id"]: d for d in saved.get("decided", [])}


def save_review(record_id, status, note):
    """Keep / fix / retire + note, in the file tools/review_bank.py --apply
    reads; an empty status clears the decision."""
    if by_id(record_id) is None:
        raise ValueError("Unknown problem id.")
    if status and status not in REVIEW_STATUSES:
        raise ValueError("Status must be keep, fix or retire.")
    with _lock:
        decided = review_decisions()
        if status:
            decided[record_id] = {"id": record_id, "status": status, "note": (note or "").strip()}
        else:
            decided.pop(record_id, None)
        config.CODE_REVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
        config.CODE_REVIEW_PATH.write_text(json.dumps(
            {"bank": "rag_code", "decided": sorted(decided.values(), key=lambda d: d["id"])},
            ensure_ascii=False, indent=1), encoding="utf-8")
    return decided.get(record_id)


# ------------------------------------------------------------------ routes

def refusal(handler):
    """Why this request may not use the coding routes, or None."""
    if not config.CODE_LOCAL:
        return "Coding drills run in the local app only."
    if handler.client_address[0] not in LOOPBACK_CLIENTS:
        return "Coding drills only serve this computer."
    host = urlparse.urlsplit("//" + (handler.headers.get("Host") or "")).hostname
    if host not in LOOPBACK_HOSTS:
        return "Coding drills only answer requests addressed to localhost."
    origin = handler.headers.get("Origin")
    if origin and urlparse.urlsplit(origin).hostname not in LOOPBACK_HOSTS:
        return "Coding drills only answer this app's own pages."
    if handler.headers.get("X-Coach-Local") != "1":
        return "Missing the X-Coach-Local header the coding page sends."
    return None


def handle_get(handler, path):
    reason = refusal(handler)
    if reason:
        json_response(handler, 403, {"error": reason})
        return
    if path == "/api/code/bank":
        json_response(handler, 200, {"problems": [summary(r) for r in records()],
                                     "time_limit": config.CODE_RUN_TIMEOUT_S})
        return
    if path in ("/api/code/history", "/api/code/record"):
        # imported under another name: a plain `records` here would shadow
        # this module's records() for the whole function (UnboundLocalError)
        from coach import records as history_records, users
        user = users.resolve_user(handler)
        if path == "/api/code/history":
            json_response(handler, 200, history_records.history(user))
            return
        query = urlparse.parse_qs(urlparse.urlsplit(handler.path).query)
        detail_row = history_records.attempt_detail(user, (query.get("id") or [""])[0])
        if detail_row is None:
            json_response(handler, 404, {"error": "No such attempt in your history."})
            return
        json_response(handler, 200, {"attempt": detail_row})
        return
    if path == "/api/code/problem":
        query = urlparse.parse_qs(urlparse.urlsplit(handler.path).query)
        record = by_id((query.get("id") or [""])[0])
        if record is None:
            json_response(handler, 404, {"error": "No such problem in the bank."})
            return
        json_response(handler, 200, {"problem": detail(record)})
        return
    json_response(handler, 404, {"error": "Not found."})


def _code(data):
    code = data.get("code") or ""
    if not isinstance(code, str):
        raise ValueError("Code must be text.")
    if len(code) > CODE_LIMIT:
        raise ValueError("That file is too long to run here.")
    return code


def _attempt(data):
    from coach import tutor
    attempt = tutor.get_attempt(str(data.get("attempt_id") or ""))
    if attempt is None:
        raise LookupError("This practice session has ended (the server restarted). "
                          "Pick the problem again to start a new one.")
    return attempt


def _engine(handler):
    """The tutor's engine: the subscription locally, the fake in --mock."""
    from coach import users
    from coach.mock.engine import pick_engine
    return pick_engine(users.resolve_user(handler))


def code_changed_since_start():
    """True when the app's Python files changed on disk after this server
    started: it still runs the old code, and a page served now (always the
    new one) may call routes the old code lacks. Restart to load it."""
    if not config.STARTED_AT:
        return False
    files = [config.BASE_DIR / "server.py", *(config.BASE_DIR / "coach").rglob("*.py")]
    return any(f.stat().st_mtime > config.STARTED_AT + 1 for f in files if f.exists())


def engine_label(engine):
    if engine == "fake":
        return "offline mode - the bank's hints, no model"
    from coach.llm import engine_model
    if engine == "cli" and config.CLI_QUICK_MODEL:
        return f"{engine_model(engine)}; quick replies on {config.CLI_QUICK_MODEL}"
    return engine_model(engine)


def stt_engine():
    """Push-to-talk follows the live loop's STT backend (local Whisper by
    default) rather than the mock report's cloud-first order."""
    from coach.voice import final_transcript
    backend = os.environ.get("STT_BACKEND", os.environ.get("AUDIO_BACKEND", "local"))
    engine = {"local": "whisper_local", "deepgram": "nova3_batch_kt",
              "elevenlabs": "scribe_batch_kt"}.get(backend)
    if engine == "whisper_local":
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            engine = None
    return engine or final_transcript.available_engine()


def speak(text):
    """One sentence of the tutor's reply as WAV, with the live loop's TTS."""
    import asyncio
    import base64
    import io
    import wave
    from coach.voice import tts as tts_module
    backend = os.environ.get("TTS_BACKEND", os.environ.get("AUDIO_BACKEND", "local"))
    engine = tts_module.make_tts(backend)
    pcm, rate = asyncio.run(engine.synth(text))
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm)
    return {"audio_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
            "mime": "audio/wav", "engine": engine.label}


def problem_for(data):
    """A bank problem by id, or any LeetCode problem by its link or title."""
    if data.get("problem_id"):
        record = by_id(str(data["problem_id"]))
        if record is None:
            raise LookupError("No such problem in the bank.")
        return detail(record)
    named = data.get("problem") or {}
    return resolve(named.get("link") or named.get("title") or "")


def handle_post(handler, path, data):
    reason = refusal(handler)
    if reason:
        json_response(handler, 403, {"error": reason})
        return
    from coach import code_tests, coding_round, tutor, users
    from coach.mock.engine import MockUnavailable
    try:
        if path == "/api/code/resolve":
            json_response(handler, 200, {"problem": resolve(data.get("query"))})
        elif path == "/api/code/run":
            code = _code(data)
            if not code.strip():
                raise ValueError("Write some code first.")
            attempt = _attempt(data) if data.get("attempt_id") else None
            if attempt is not None:
                coding_round.advance(attempt, code)
                if coding_round.locked(attempt):
                    raise PermissionError("Run is locked until you say you're done "
                                          "(interview conditions).")
            result = run_code(code, str(data.get("stdin") or ""))
            if attempt is not None:
                entry = tutor.record_run(attempt, code, result)
                result.update(offer_hint=entry["failed"], next_level=tutor.ladder_caps(attempt)[1],
                              phase=attempt.get("phase"))
            json_response(handler, 200, result)
        elif path == "/api/code/review":
            saved = save_review(str(data.get("id") or ""), str(data.get("status") or ""),
                                str(data.get("note") or ""))
            json_response(handler, 200, {"review": saved})
        elif path == "/api/code/attempt":
            problem = problem_for(data)
            engine = _engine(handler)
            user = users.resolve_user(handler)["name"]
            if data.get("mode") == "mock":
                attempt, opening = coding_round.start(problem, user, bool(data.get("strict")))
                json_response(handler, 200, {"attempt_id": attempt["id"], "mode": "mock",
                                             "strict": attempt["strict"], "phase": "discuss",
                                             "opening": opening, "tutor": engine_label(engine),
                                             "checks": engine != "fake"})
                return
            attempt = tutor.new_attempt(problem, user)
            json_response(handler, 200, {"attempt_id": attempt["id"],
                                         "tutor": engine_label(engine),
                                         "checks": engine != "fake"})
        elif path == "/api/code/snapshot":
            attempt = _attempt(data)
            tutor.snapshot(attempt, _code(data), str(data.get("reason") or "pause")[:20])
            coding_round.advance(attempt, _code(data))
            json_response(handler, 200, {"ok": True, "phase": attempt.get("phase")})
        elif path == "/api/code/check":
            attempt = _attempt(data)
            code = _code(data)
            if not code.strip():
                raise ValueError("Write some code first.")
            coding_round.advance(attempt, code)
            if coding_round.locked(attempt):
                raise PermissionError("Check is locked until you say you're done "
                                      "(interview conditions).")
            engine = _engine(handler)
            if engine == "fake":
                raise MockUnavailable("Checking needs a model to write the tests: run the "
                                      "server with your subscription or an API key.")
            if data.get("fresh"):
                code_tests.forget_suite(attempt["problem_key"], code)
            started = time.perf_counter()
            caller = tutor.make_caller(engine)
            suite = code_tests.build_suite(attempt["problem"], attempt["problem_key"], code,
                                           lambda system, prompt, schema: caller(system, prompt, schema))
            written = round(time.perf_counter() - started, 1)
            result = code_tests.run_suite(suite, code)
            entry = tutor.record_run(attempt, code, result, kind="check")
            result.update(offer_hint=entry["failed"], entry=suite["entry"],
                          next_level=tutor.ladder_caps(attempt)[1], phase=attempt.get("phase"),
                          suite_written=suite.get("written"), seconds_to_tests=written)
            json_response(handler, 200, result)
        elif path == "/api/code/interviewer":
            attempt = _attempt(data)
            ms = data.get("ms")
            reply = coding_round.interviewer_turn(
                attempt, _code(data), str(data.get("message") or "")[:2000],
                str(data.get("kind") or "message"), _engine(handler),
                ms=int(ms) if isinstance(ms, (int, float)) and 0 < ms < 3_600_000 else None,
                voice=bool(data.get("voice")))
            json_response(handler, 200, reply)
        elif path == "/api/code/tutor":
            attempt = _attempt(data)
            if attempt.get("mode") == "mock":
                raise tutor.TutorError("This is an interview round - talk to the interviewer.")
            reply = tutor.tutor_turn(attempt, _code(data), str(data.get("message") or "")[:2000],
                                     str(data.get("kind") or "message"), _engine(handler),
                                     confirmed=bool(data.get("confirmed")))
            json_response(handler, 200, reply)
        elif path == "/api/code/observe":
            attempt = _attempt(data)
            items = tutor.observe(attempt, _code(data), _engine(handler))
            # the tutor stays quiet while the user works: only a count here,
            # the watch-outs themselves are in the report
            json_response(handler, 200, {"logged": len(items)})
        elif path == "/api/code/finish":
            attempt = _attempt(data)
            outcome = data.get("leetcode")
            if outcome not in (None, "accepted", "rejected", "not-submitted"):
                raise ValueError("Unknown LeetCode outcome.")
            engine = _engine(handler)
            report = tutor.finish(attempt, _code(data), engine, outcome)
            markdown = tutor.report_markdown(report)
            if attempt.get("mode") == "mock":
                report.update(mode="mock", strict=attempt["strict"],
                              communication=coding_round.communication(attempt, engine))
                markdown += coding_round.communication_markdown(report["communication"]) + "\n"
            from coach import records as history_records
            saved = history_records.save(attempt, report, users.resolve_user(handler))
            json_response(handler, 200, {"report": report, "markdown": markdown,
                                         "saved": saved is not None})
        elif path == "/api/code/star":
            from coach import records as history_records
            ok = history_records.star(users.resolve_user(handler), str(data.get("key") or ""),
                              bool(data.get("starred")))
            if not ok:
                raise LookupError("That problem is not in your history.")
            json_response(handler, 200, {"starred": bool(data.get("starred"))})
        elif path == "/api/code/history/clear":
            from coach import records as history_records
            if data.get("confirm") is not True:
                raise ValueError("Confirm to delete your coding history.")
            json_response(handler, 200, {"deleted": history_records.clear(users.resolve_user(handler))})
        elif path == "/api/code/transcribe":
            import base64
            from coach.voice import final_transcript
            audio_b64 = data.get("audio_base64") or ""
            if not audio_b64 or len(audio_b64) > 12_000_000:
                raise ValueError("Send one clip of up to about two minutes.")
            terms = []
            attempt = tutor.get_attempt(str(data.get("attempt_id") or ""))
            if attempt:
                terms = list(attempt["problem"].get("approaches") or [])
                terms.append(attempt["problem"].get("title") or "")
            result = final_transcript.transcribe_final(
                base64.b64decode(audio_b64), data.get("mime") or "audio/webm",
                engine=stt_engine(), terms=[t for t in terms if t])
            json_response(handler, 200, result)
        elif path == "/api/code/speak":
            text = str(data.get("text") or "").strip()
            if not text or len(text) > 1200:
                raise ValueError("Send one sentence to speak.")
            json_response(handler, 200, speak(text))
        else:
            json_response(handler, 404, {"error": "Not found."})
    except (ValueError, tutor.TutorError) as exc:
        json_response(handler, 400, {"error": str(exc)})
    except LookupError as exc:
        json_response(handler, 404, {"error": str(exc)})
    except (MockUnavailable, PermissionError) as exc:
        json_response(handler, 403, {"error": str(exc)})
    except RuntimeError as exc:
        json_response(handler, 503, {"error": str(exc)})
