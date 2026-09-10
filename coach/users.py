"""Freemium tiers: access keys, the paid tier's daily Claude quota, and the
daily LLM-call and live-voice budgets that make a shared or demo key safe
to hand out.

A request's X-Access-Key header is looked up in the access keys (users.json,
or the access_keys table when DATABASE_URL selects the Postgres store -
coach/store.py); keys with tier "paid" get LLM grading (DeepSeek Flash by
default, Claude on "Always Claude" under PAID_DAILY_QUOTA Claude calls per
day, question generation and evaluation combined). Everyone else gets the
distilled local grader. When there are no keys at all (no users.json, no
database), tiers are disabled and every request grades with Claude - the
original single-user behaviour.

Budgets (2026-09-07, for the public demo): every LLM call - any engine,
practice grading or a mock route - counts against the key's own
"daily_llm_calls" (0 or absent = unlimited) and against the server-wide
config.LLM_DAILY_CAP (0 = unlimited). A refused call grades locally with the
reason "budget"; the mock refuses with a clear message. Live voice is
metered the same way in seconds against "daily_voice_minutes" and
VOICE_DAILY_MINUTES. The counters live in the store keyed by a digest of
the key, plus one "_server" row for the whole instance; a reservation is
one locked read-modify-write there (a file rewrite under a process lock,
or a row-locked transaction in Postgres).
"""

import hashlib
from datetime import datetime

from coach import config, store

USERS = {}
# True whenever access keys exist (users.json present - even if it fails to
# parse - or the Postgres store is on). A present-but-broken users file
# must fail CLOSED (tiers on, no paid keys -> everyone free), never open
# (everyone gets Claude), or a corrupt file becomes a cost leak.
TIERS_ENABLED = False
SERVER_ROW = "_server"
_users_stamp = None


def load_users():
    global USERS, TIERS_ENABLED, _users_stamp
    st = store.current()
    if not st.users_present():
        return
    TIERS_ENABLED = True
    try:
        _users_stamp = st.users_stamp()
        USERS = st.load_users()
    except Exception as exc:
        print(
            f"Warning: could not read the access keys ({exc}); "
            "tiers stay ON with no paid keys - every request is free tier."
        )


def _maybe_reload():
    """Pick up key changes without a restart - a users.json edit (revoking a
    leaked key: delete its line) on the next request, a table change within
    USERS_REFRESH_S seconds on the Postgres store."""
    if store.current().users_stamp() != _users_stamp:
        load_users()


def _usage_id(key):
    """Usage rows are keyed by a digest of the access key, never the raw
    key: a shared or backed-up usage file must not leak every paid key. The
    raw keys rest only in users.json (gitignored) or the access_keys table."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def _cap(value):
    """A budget field -> int, 0 meaning unlimited; junk is 0."""
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def resolve_key(key):
    """User record for a bare access key (the voice WebSocket has no
    request headers; everything else goes through resolve_user)."""
    _maybe_reload()
    key = (key or "").strip()
    entry = USERS.get(key)
    if entry:
        return {
            "key": key,
            "name": entry.get("name", "user"),
            "tier": "paid" if entry.get("tier") == "paid" else "free",
            # Per-user answer-collection opt-out: {"log": false} in users.json.
            "log": entry.get("log", True),
            # Per-key daily LLM-call budget, any engine (0 = unlimited).
            "llm_cap": _cap(entry.get("daily_llm_calls")),
            # Per-key daily live-voice budget in minutes (0 = unlimited).
            "voice_cap": _cap(entry.get("daily_voice_minutes")),
        }
    return {"key": None, "name": "anonymous", "tier": "free", "log": True,
            "llm_cap": 0, "voice_cap": 0}


def resolve_user(handler):
    return resolve_key(handler.headers.get("X-Access-Key") or "")


def _today():
    return datetime.now().date().isoformat()


def _row(usage, row_id, today):
    """Today's counters for a usage row; a stale or missing row starts at 0.
    "used" counts Claude calls (the quota), "llm" counts every LLM call,
    "voice" counts seconds of live voice (metered in ticks, so a float)."""
    row = usage.get(row_id)
    if not row or row.get("date") != today:
        return {"date": today, "used": 0, "llm": 0, "voice": 0.0}
    return {"date": today, "used": int(row.get("used", 0)),
            "llm": int(row.get("llm", 0)),
            "voice": float(row.get("voice", 0) or 0)}


def _ids(user):
    """The usage rows a user touches: the server row, and the key's own row
    when it has a key (anonymous callers only count against the server)."""
    key_id = _usage_id(user["key"]) if user.get("key") else None
    return key_id, ([SERVER_ROW, key_id] if key_id else [SERVER_ROW])


def _read_rows(user):
    today = _today()
    key_id, ids = _ids(user)
    raw = store.current().usage_read(ids, today)
    key_row = _row(raw, key_id, today) if key_id else None
    return key_row, _row(raw, SERVER_ROW, today)


def quota_left(user):
    if user["tier"] != "paid":
        return 0
    key_row, _ = _read_rows(user)
    return max(0, config.PAID_DAILY_QUOTA - key_row["used"])


def budget_left(user):
    """LLM calls left today as {"key": n | None, "server": n | None};
    None means that budget is unlimited."""
    key_row, server_row = _read_rows(user)
    key_left = None
    if key_row is not None and user.get("llm_cap"):
        key_left = max(0, user["llm_cap"] - key_row["llm"])
    server_left = None
    if config.LLM_DAILY_CAP:
        server_left = max(0, config.LLM_DAILY_CAP - server_row["llm"])
    return {"key": key_left, "server": server_left}


def voice_left(user):
    """Live-voice seconds left today as {"key": n | None, "server": n | None};
    None means that budget is unlimited. Caps are minutes in users.json and
    the environment; the counters are seconds, because a session is metered
    in ticks (coach/voice/loop.py) and a clip by its length."""
    key_row, server_row = _read_rows(user)
    key_left = None
    if key_row is not None and user.get("voice_cap"):
        key_left = max(0.0, user["voice_cap"] * 60 - key_row["voice"])
    server_left = None
    if config.VOICE_DAILY_MINUTES:
        server_left = max(0.0, config.VOICE_DAILY_MINUTES * 60 - server_row["voice"])
    return {"key": key_left, "server": server_left}


def minutes_left(seconds):
    """Seconds left -> whole minutes for display (None stays None). A
    started minute counts as one, since a session gets a whole tick."""
    if seconds is None:
        return None
    return int(-(-seconds // 60))


def take_voice(user, seconds):
    """Charge `seconds` of live voice to the key (when it has one) and to
    the server.

    Returns None when the session may go on, or "voice" when the key's or
    the server's daily voice allowance is already spent - nothing is
    written then. A tick is charged whole once it starts, so a session
    ends within one tick of the allowance running out: the overrun is
    bounded by the tick length, never by the session length."""
    today = _today()
    seconds = max(0.0, float(seconds))
    key_id, ids = _ids(user)
    with store.current().usage_transaction(ids, today) as txn:
        key_row = _row(txn.rows, key_id, today) if key_id else None
        server_row = _row(txn.rows, SERVER_ROW, today)
        if (key_row is not None and user.get("voice_cap")
                and key_row["voice"] >= user["voice_cap"] * 60):
            return "voice"
        if (config.VOICE_DAILY_MINUTES
                and server_row["voice"] >= config.VOICE_DAILY_MINUTES * 60):
            return "voice"
        if key_row is not None:
            key_row["voice"] = round(key_row["voice"] + seconds, 3)
            txn.write(key_id, key_row)
        server_row["voice"] = round(server_row["voice"] + seconds, 3)
        txn.write(SERVER_ROW, server_row)
    return None


def take_call(user, engine):
    """Reserve one LLM call for a paid user on `engine`.

    Returns None when the call may proceed, "quota" when engine is Claude
    and the key's daily Claude quota is spent, or "budget" when the key's
    or the server's daily LLM budget is spent. Nothing is written on a
    refusal, so the caller may retry with another engine (a spent Claude
    quota degrades to DeepSeek; a spent budget never does)."""
    today = _today()
    key_id = _usage_id(user["key"])
    with store.current().usage_transaction([SERVER_ROW, key_id], today) as txn:
        key_row = _row(txn.rows, key_id, today)
        server_row = _row(txn.rows, SERVER_ROW, today)
        if engine == "claude" and key_row["used"] >= config.PAID_DAILY_QUOTA:
            return "quota"
        if user.get("llm_cap") and key_row["llm"] >= user["llm_cap"]:
            return "budget"
        if config.LLM_DAILY_CAP and server_row["llm"] >= config.LLM_DAILY_CAP:
            return "budget"
        if engine == "claude":
            key_row["used"] += 1
        key_row["llm"] += 1
        server_row["llm"] += 1
        txn.write(key_id, key_row)
        txn.write(SERVER_ROW, server_row)
    return None


def quota_take(user):
    """Reserve one paid Claude call (quota and budgets); False when refused."""
    return take_call(user, "claude") is None
