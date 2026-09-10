"""State store: one interface over the app's mutable state, two backends
(roadmap step 4, 2026-09-10).

The state is the access keys and tiers (users.json), the daily counters
(usage.json), the three session logs (data/sessions/*.jsonl) and the mock's
plan cache (data/mock_cache/). The FILE backend is the default and keeps
every path and format exactly as before, so a clone runs with zero
services. The POSTGRES backend is selected by DATABASE_URL (psycopg 3 with
a small connection pool - the stdlib server is threaded) and turns the
quota reservation, a read-modify-write of usage.json under a process lock,
into a row-locked transaction: the one thing the file backend cannot offer
to a second process or host.

Policy stays in coach/users.py (what a cap means, when a call is refused);
the store only offers primitives: a snapshot of the keys, a locked
read-modify-write of usage rows, appends to a log, a small key-value
cache. Both backends are exercised by the same tests (tests/test_store.py
re-runs tests/test_users.py against Postgres in CI).
"""

import hashlib
import json
import os
import threading
import time
from contextlib import contextmanager
from datetime import date, datetime

from coach import config

SESSION_KINDS = ("real", "free", "mock")


class StoreError(RuntimeError):
    """The configured store cannot serve (bad DATABASE_URL, no server, no
    driver). Raised at startup so an operator who set DATABASE_URL never
    gets the file backend by accident."""


class UsageTxn:
    """What usage_transaction() hands to policy code: `rows` is the raw
    usage rows found (row id -> dict; a missing id is simply absent) and
    write(row_id, row) stages a row to persist when the block ends. A
    refusal stages nothing, and nothing is written."""

    def __init__(self, rows):
        self.rows = rows
        self.writes = {}

    def write(self, row_id, row):
        self.writes[row_id] = dict(row)


def session_path(kind):
    return {"real": config.REAL_SESSIONS_PATH,
            "free": config.FREE_SESSIONS_PATH,
            "mock": config.MOCK_SESSIONS_PATH}[kind]


def session_fingerprint(kind, record):
    """Identity of a log row: the same record imported twice (the migration
    tool re-run on the same folder) lands once."""
    payload = json.dumps(record, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(f"{kind}\x00{payload}".encode("utf-8")).hexdigest()


def _cap(value):
    """A budget field -> int, 0 meaning unlimited; junk is 0 (as users.py)."""
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _record_time(record):
    """The record's own timestamp when it carries one (imports keep their
    order), else None so the database stamps now()."""
    stamp = record.get("timestamp") if isinstance(record, dict) else None
    if not stamp:
        return None
    try:
        return datetime.fromisoformat(str(stamp))
    except ValueError:
        return None


class FileStore:
    """The original files under data/ (and users.json), read through the
    config paths at call time so tests and compose can point them
    elsewhere; one process lock serialises the counter updates."""

    backend = "file"

    def __init__(self):
        self._lock = threading.Lock()

    # --- access keys -------------------------------------------------
    def users_present(self):
        return config.USERS_PATH.exists()

    def users_stamp(self):
        """Changes whenever the keys may have changed (file mtime)."""
        try:
            return config.USERS_PATH.stat().st_mtime
        except OSError:
            return None

    def load_users(self):
        # utf-8-sig: tolerate the BOM that Windows editors and PowerShell
        # (Set-Content -Encoding utf8) prepend.
        return json.loads(config.USERS_PATH.read_text(encoding="utf-8-sig"))

    def replace_users(self, users):
        config.USERS_PATH.parent.mkdir(parents=True, exist_ok=True)
        config.USERS_PATH.write_text(json.dumps(users, ensure_ascii=False, indent=2),
                                     encoding="utf-8")

    # --- daily counters ----------------------------------------------
    def _read_usage(self):
        if not config.USAGE_PATH.exists():
            return {}
        try:
            return json.loads(config.USAGE_PATH.read_text(encoding="utf-8-sig"))
        except Exception:
            return {}

    def usage_read(self, ids, today):
        with self._lock:
            usage = self._read_usage()
        return {row_id: usage[row_id] for row_id in ids if row_id in usage}

    @contextmanager
    def usage_transaction(self, ids, today):
        with self._lock:
            usage = self._read_usage()
            txn = UsageTxn({row_id: usage[row_id] for row_id in ids if row_id in usage})
            yield txn
            if txn.writes:
                usage.update(txn.writes)
                config.USAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
                config.USAGE_PATH.write_text(json.dumps(usage), encoding="utf-8")

    def usage_dump(self):
        with self._lock:
            return self._read_usage()

    def reset_usage(self):
        with self._lock:
            try:
                config.USAGE_PATH.unlink()
            except FileNotFoundError:
                pass

    # --- session logs ------------------------------------------------
    def append_session(self, kind, record):
        path = session_path(kind)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def read_sessions(self, kind):
        path = session_path(kind)
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]

    # --- plan cache --------------------------------------------------
    def _cache_path(self, digest):
        return config.mock_cache_dir() / f"{digest}.json"

    def cache_get(self, digest):
        try:
            return json.loads(self._cache_path(digest).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def cache_put(self, digest, payload):
        path = self._cache_path(digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def cache_trim(self, limit):
        try:
            files = sorted(config.mock_cache_dir().glob("*.json"),
                           key=lambda p: p.stat().st_mtime)
        except OSError:
            return
        for path in files[:-limit] if limit else files:
            try:
                path.unlink()
            except OSError:
                pass

    def cache_count(self):
        try:
            return sum(1 for _ in config.mock_cache_dir().glob("*.json"))
        except OSError:
            return 0

    # --- about -------------------------------------------------------
    def info(self):
        return {"backend": "file"}

    def counts(self):
        keys = 0
        if self.users_present():
            try:
                keys = sum(1 for v in self.load_users().values() if isinstance(v, dict))
            except Exception:
                keys = 0
        return {"keys": keys,
                "sessions": {kind: len(self.read_sessions(kind)) for kind in SESSION_KINDS},
                "cached_plans": self.cache_count()}

    def describe(self):
        return (f"file backend - {config.USERS_PATH.name}, {config.USAGE_PATH.name}, "
                f"{config.REAL_SESSIONS_PATH.parent.name}/, {config.mock_cache_dir().name}/")

    def close(self):
        pass


SCHEMA = (
    """CREATE TABLE IF NOT EXISTS access_keys (
        key text PRIMARY KEY,
        name text NOT NULL DEFAULT 'user',
        tier text NOT NULL DEFAULT 'free',
        log boolean NOT NULL DEFAULT true,
        daily_llm_calls integer NOT NULL DEFAULT 0,
        daily_voice_minutes integer NOT NULL DEFAULT 0,
        created_at timestamptz NOT NULL DEFAULT now(),
        revoked_at timestamptz)""",
    """CREATE TABLE IF NOT EXISTS usage_counters (
        row_id text NOT NULL,
        day date NOT NULL,
        used integer NOT NULL DEFAULT 0,
        llm integer NOT NULL DEFAULT 0,
        voice double precision NOT NULL DEFAULT 0,
        PRIMARY KEY (row_id, day))""",
    """CREATE TABLE IF NOT EXISTS session_log (
        id bigserial PRIMARY KEY,
        kind text NOT NULL,
        ts timestamptz NOT NULL DEFAULT now(),
        fingerprint text NOT NULL UNIQUE,
        record jsonb NOT NULL)""",
    "CREATE INDEX IF NOT EXISTS session_log_kind_id ON session_log (kind, id)",
    """CREATE TABLE IF NOT EXISTS plan_cache (
        digest text PRIMARY KEY,
        payload jsonb NOT NULL,
        updated_at timestamptz NOT NULL DEFAULT now())""",
)


class PostgresStore:
    """Four tables (SCHEMA above, created if missing at startup). The usage
    counters keep one row per (id, day) - the file kept only today's - so
    the history stays; reads and reservations look at today's rows only,
    exactly as users.py's policy did with the file. A reservation is one
    transaction: make today's rows exist, lock them (SELECT ... FOR UPDATE),
    let the policy decide, write the changed rows, commit - two processes
    can never both spend the last call."""

    backend = "postgres"

    def __init__(self, url, min_size=1, max_size=8, connect_timeout=10.0):
        try:
            import psycopg
            from psycopg.types.json import Jsonb
            from psycopg_pool import ConnectionPool
        except ImportError as exc:
            raise StoreError("DATABASE_URL is set but the psycopg driver is not "
                             "installed (pip install -r requirements-db.txt)") from exc
        self._psycopg = psycopg
        self._jsonb = Jsonb
        self.url = url
        try:
            self.pool = ConnectionPool(url, min_size=min_size, max_size=max_size,
                                       open=True, timeout=connect_timeout,
                                       kwargs={"autocommit": True})
            self.pool.wait(timeout=connect_timeout)
            with self.pool.connection() as conn:
                for statement in SCHEMA:
                    conn.execute(statement)
        except StoreError:
            raise
        except Exception as exc:
            # Stop the pool's reconnect workers before giving up, or they
            # keep retrying in the background of a process that is exiting.
            pool = getattr(self, "pool", None)
            if pool is not None:
                try:
                    pool.close(timeout=1.0)
                except Exception:
                    pass
            raise StoreError(f"cannot use DATABASE_URL ({self.location()}): {exc}") from exc

    def location(self):
        """host:port/dbname, never the password."""
        try:
            info = self._psycopg.conninfo.conninfo_to_dict(self.url)
        except Exception:
            return "?"
        return f"{info.get('host', '?')}:{info.get('port', 5432)}/{info.get('dbname', '?')}"

    # --- access keys -------------------------------------------------
    def users_present(self):
        return True

    def users_stamp(self):
        """Changes every USERS_REFRESH_S seconds, so users.py re-reads the
        table that often and a revoked key stops working within that."""
        return int(time.monotonic() // max(0.5, config.USERS_REFRESH_S))

    def load_users(self):
        with self.pool.connection() as conn:
            rows = conn.execute(
                "SELECT key, name, tier, log, daily_llm_calls, daily_voice_minutes "
                "FROM access_keys WHERE revoked_at IS NULL").fetchall()
        return {key: {"name": name, "tier": tier, "log": log,
                      "daily_llm_calls": llm, "daily_voice_minutes": voice}
                for key, name, tier, log, llm, voice in rows}

    def upsert_users(self, users):
        """A users.json-shaped dict -> rows: insert or update the fields,
        keep an existing revocation. Returns the number of rows written."""
        written = 0
        with self.pool.connection() as conn, conn.transaction():
            for key, entry in users.items():
                if not isinstance(entry, dict) or not str(key).strip():
                    continue
                conn.execute(
                    "INSERT INTO access_keys (key, name, tier, log, daily_llm_calls, "
                    "daily_voice_minutes) VALUES (%s, %s, %s, %s, %s, %s) "
                    "ON CONFLICT (key) DO UPDATE SET name = EXCLUDED.name, "
                    "tier = EXCLUDED.tier, log = EXCLUDED.log, "
                    "daily_llm_calls = EXCLUDED.daily_llm_calls, "
                    "daily_voice_minutes = EXCLUDED.daily_voice_minutes",
                    (str(key), str(entry.get("name", "user")),
                     "paid" if entry.get("tier") == "paid" else "free",
                     bool(entry.get("log", True)),
                     _cap(entry.get("daily_llm_calls")),
                     _cap(entry.get("daily_voice_minutes"))))
                written += 1
        return written

    def replace_users(self, users):
        with self.pool.connection() as conn:
            conn.execute("TRUNCATE access_keys")
        self.upsert_users(users)

    def revoke_key(self, key):
        """Revoke one key (kept for the record); rows affected."""
        with self.pool.connection() as conn:
            return conn.execute(
                "UPDATE access_keys SET revoked_at = now() "
                "WHERE key = %s AND revoked_at IS NULL", (key,)).rowcount

    def list_keys(self):
        """Every key row, revoked ones included, with the key shown only by
        its first four characters - for tools and the runbook."""
        with self.pool.connection() as conn:
            rows = conn.execute(
                "SELECT key, name, tier, log, daily_llm_calls, daily_voice_minutes, "
                "created_at, revoked_at FROM access_keys ORDER BY created_at").fetchall()
        return [{"key_prefix": key[:4] + "…", "name": name, "tier": tier, "log": log,
                 "daily_llm_calls": llm, "daily_voice_minutes": voice,
                 "created_at": created.isoformat(timespec="seconds"),
                 "revoked_at": revoked.isoformat(timespec="seconds") if revoked else None}
                for key, name, tier, log, llm, voice, created, revoked in rows]

    # --- daily counters ----------------------------------------------
    @staticmethod
    def _rows_dict(rows, today):
        return {row_id: {"date": today, "used": used, "llm": llm, "voice": voice}
                for row_id, used, llm, voice in rows}

    def usage_read(self, ids, today):
        day = date.fromisoformat(today)
        with self.pool.connection() as conn:
            rows = conn.execute(
                "SELECT row_id, used, llm, voice FROM usage_counters "
                "WHERE row_id = ANY(%s) AND day = %s", (list(ids), day)).fetchall()
        return self._rows_dict(rows, today)

    @contextmanager
    def usage_transaction(self, ids, today):
        day = date.fromisoformat(today)
        ids = list(ids)
        with self.pool.connection() as conn, conn.transaction():
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO usage_counters (row_id, day) VALUES (%s, %s) "
                    "ON CONFLICT DO NOTHING", [(row_id, day) for row_id in ids])
            rows = conn.execute(
                "SELECT row_id, used, llm, voice FROM usage_counters "
                "WHERE row_id = ANY(%s) AND day = %s FOR UPDATE", (ids, day)).fetchall()
            txn = UsageTxn(self._rows_dict(rows, today))
            yield txn
            for row_id, row in txn.writes.items():
                conn.execute(
                    "UPDATE usage_counters SET used = %s, llm = %s, voice = %s "
                    "WHERE row_id = %s AND day = %s",
                    (int(row.get("used", 0)), int(row.get("llm", 0)),
                     float(row.get("voice", 0) or 0), row_id, day))

    def usage_dump(self):
        """The newest day of every row, in the file's shape (tests, tools)."""
        with self.pool.connection() as conn:
            rows = conn.execute(
                "SELECT DISTINCT ON (row_id) row_id, day, used, llm, voice "
                "FROM usage_counters ORDER BY row_id, day DESC").fetchall()
        return {row_id: {"date": day.isoformat(), "used": used, "llm": llm, "voice": voice}
                for row_id, day, used, llm, voice in rows}

    def upsert_usage(self, usage):
        """usage.json rows -> today's-state rows (the file is the newest)."""
        written = 0
        with self.pool.connection() as conn, conn.transaction():
            for row_id, row in usage.items():
                if not isinstance(row, dict) or not row.get("date"):
                    continue
                try:
                    day = date.fromisoformat(str(row["date"]))
                except ValueError:
                    continue
                conn.execute(
                    "INSERT INTO usage_counters (row_id, day, used, llm, voice) "
                    "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (row_id, day) DO UPDATE SET "
                    "used = EXCLUDED.used, llm = EXCLUDED.llm, voice = EXCLUDED.voice",
                    (str(row_id), day, _cap(row.get("used")), _cap(row.get("llm")),
                     float(row.get("voice", 0) or 0)))
                written += 1
        return written

    def reset_usage(self):
        with self.pool.connection() as conn:
            conn.execute("TRUNCATE usage_counters")

    # --- session logs ------------------------------------------------
    def append_session(self, kind, record):
        """Returns True when the row was new (False: same record already
        stored - the migration tool re-run)."""
        with self.pool.connection() as conn:
            return conn.execute(
                "INSERT INTO session_log (kind, ts, fingerprint, record) "
                "VALUES (%s, COALESCE(%s, now()), %s, %s) "
                "ON CONFLICT (fingerprint) DO NOTHING",
                (kind, _record_time(record), session_fingerprint(kind, record),
                 self._jsonb(record))).rowcount == 1

    def read_sessions(self, kind):
        with self.pool.connection() as conn:
            return [record for (record,) in conn.execute(
                "SELECT record FROM session_log WHERE kind = %s ORDER BY id",
                (kind,)).fetchall()]

    # --- plan cache --------------------------------------------------
    def cache_get(self, digest):
        with self.pool.connection() as conn:
            row = conn.execute("SELECT payload FROM plan_cache WHERE digest = %s",
                               (digest,)).fetchone()
        return row[0] if row else None

    def cache_put(self, digest, payload):
        with self.pool.connection() as conn:
            conn.execute(
                "INSERT INTO plan_cache (digest, payload) VALUES (%s, %s) "
                "ON CONFLICT (digest) DO UPDATE SET payload = EXCLUDED.payload, "
                "updated_at = now()", (digest, self._jsonb(payload)))

    def cache_trim(self, limit):
        with self.pool.connection() as conn:
            if not limit:
                conn.execute("DELETE FROM plan_cache")
            else:
                conn.execute(
                    "DELETE FROM plan_cache WHERE digest IN (SELECT digest FROM plan_cache "
                    "ORDER BY updated_at DESC, digest OFFSET %s)", (int(limit),))

    def cache_count(self):
        with self.pool.connection() as conn:
            return conn.execute("SELECT count(*) FROM plan_cache").fetchone()[0]

    # --- about -------------------------------------------------------
    def info(self):
        return {"backend": "postgres"}

    def counts(self):
        with self.pool.connection() as conn:
            keys = conn.execute(
                "SELECT count(*) FROM access_keys WHERE revoked_at IS NULL").fetchone()[0]
            sessions = dict(conn.execute(
                "SELECT kind, count(*) FROM session_log GROUP BY kind").fetchall())
            cached = conn.execute("SELECT count(*) FROM plan_cache").fetchone()[0]
            usage_rows = conn.execute("SELECT count(*) FROM usage_counters").fetchone()[0]
        return {"keys": keys,
                "sessions": {kind: int(sessions.get(kind, 0)) for kind in SESSION_KINDS},
                "cached_plans": cached, "usage_rows": usage_rows}

    def describe(self):
        c = self.counts()
        return (f"postgres at {self.location()} - {c['keys']} active key(s), "
                f"{sum(c['sessions'].values())} session row(s), "
                f"{c['cached_plans']} cached plan(s), {c['usage_rows']} usage day-row(s)")

    def close(self):
        try:
            self.pool.close()
        except Exception:
            pass


_current = None


def init():
    """Select the backend from DATABASE_URL - read from the environment
    here, after config.load_env_file(), so a .env line works - and connect.
    Raises StoreError when a configured database cannot serve."""
    global _current
    url = os.environ.get("DATABASE_URL", "").strip() or config.DATABASE_URL
    config.DATABASE_URL = url
    if _current is not None:
        _current.close()
    _current = PostgresStore(url) if url else FileStore()
    return _current


def current():
    global _current
    if _current is None:
        _current = init()
    return _current


def use(store_obj):
    """Swap the active store (tests, the migration tool). The store swapped
    out is left open: the caller may still hold it (the store tests hand
    one Postgres store back and forth with the tier suite) and closes it."""
    global _current
    _current = store_obj
    return store_obj
