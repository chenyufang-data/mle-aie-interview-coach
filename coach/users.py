"""Freemium tiers: access keys, and the paid tier's daily Claude quota.

A request's X-Access-Key header is looked up in USERS_PATH; keys with tier
"paid" get Claude grading, capped at PAID_DAILY_QUOTA Claude calls per day
(question generation + evaluation combined). Everyone else gets the distilled
local grader. When USERS_PATH does not exist, tiers are disabled and every
request grades with Claude — the original single-user behaviour.
"""

import json
import threading
from datetime import datetime

from coach.config import PAID_DAILY_QUOTA, USAGE_PATH, USERS_PATH

USERS = {}
# True whenever USERS_PATH exists — even if it fails to parse. A present-but-
# broken users file must fail CLOSED (tiers on, no paid keys -> everyone free),
# never open (everyone gets Claude), or a corrupt file becomes a cost leak.
TIERS_ENABLED = False
_usage_lock = threading.Lock()


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
    if entry:
        return {
            "key": key,
            "name": entry.get("name", "user"),
            "tier": "paid" if entry.get("tier") == "paid" else "free",
            # Per-user answer-collection opt-out: {"log": false} in users.json.
            "log": entry.get("log", True),
        }
    return {"key": None, "name": "anonymous", "tier": "free", "log": True}


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
