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


def run_code(code, stdin=""):
    """Run the user's Python file once; output, exit code, time."""
    with _runs, tempfile.TemporaryDirectory(prefix="coach-run-") as tmp:
        path = Path(tmp) / "solution.py"
        path.write_text(code, encoding="utf-8")
        env = {k: os.environ[k] for k in CHILD_ENV_KEYS if os.environ.get(k)}
        env.update(PYTHONIOENCODING="utf-8", PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1")
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
    if path == "/api/code/problem":
        query = urlparse.parse_qs(urlparse.urlsplit(handler.path).query)
        record = by_id((query.get("id") or [""])[0])
        if record is None:
            json_response(handler, 404, {"error": "No such problem in the bank."})
            return
        json_response(handler, 200, {"problem": detail(record)})
        return
    json_response(handler, 404, {"error": "Not found."})


def handle_post(handler, path, data):
    reason = refusal(handler)
    if reason:
        json_response(handler, 403, {"error": reason})
        return
    try:
        if path == "/api/code/resolve":
            json_response(handler, 200, {"problem": resolve(data.get("query"))})
        elif path == "/api/code/run":
            code = data.get("code") or ""
            if not isinstance(code, str) or not code.strip():
                raise ValueError("Write some code first.")
            if len(code) > CODE_LIMIT:
                raise ValueError("That file is too long to run here.")
            json_response(handler, 200, run_code(code, str(data.get("stdin") or "")))
        elif path == "/api/code/review":
            saved = save_review(str(data.get("id") or ""), str(data.get("status") or ""),
                                str(data.get("note") or ""))
            json_response(handler, 200, {"review": saved})
        else:
            json_response(handler, 404, {"error": "Not found."})
    except ValueError as exc:
        json_response(handler, 400, {"error": str(exc)})
