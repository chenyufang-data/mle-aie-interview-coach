"""Import a data/ folder and users.json into the Postgres state store
(roadmap step 4, coach/store.py).

  python tools/migrate_to_postgres.py --dry-run
  python tools/migrate_to_postgres.py --data data --users users.json
  python tools/migrate_to_postgres.py --data /data --users /app/users.json   # inside the container

The database comes from --database-url, else DATABASE_URL in the
environment or the .env. What moves: users.json -> access_keys,
data/usage.json -> usage_counters, data/sessions/{real,free,mock}_sessions.jsonl
-> session_log, data/mock_cache/*.json -> plan_cache. Idempotent: keys, usage
rows and cache entries upsert, session rows dedupe by fingerprint - a
re-run on the same folder changes nothing, which is also what the test
checks. --dry-run prints what the folder holds and, when a database is
reachable, what its tables hold now; it writes nothing.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coach import config, store  # noqa: E402

SESSION_FILES = {"real": "real_sessions.jsonl", "free": "free_sessions.jsonl",
                 "mock": "mock_sessions.jsonl"}


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _read_jsonl(path):
    rows, bad = [], 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            bad += 1
            continue
        if isinstance(row, dict):
            rows.append(row)
        else:
            bad += 1
    return rows, bad


def collect(data_dir, users_path):
    """Everything the folder and the users file hold, with notes on what
    could not be read."""
    data_dir = Path(data_dir)
    found = {"keys": {}, "usage": {}, "sessions": {k: [] for k in store.SESSION_KINDS},
             "cache": {}, "notes": []}
    users_path = Path(users_path) if users_path else None
    if users_path and users_path.exists():
        try:
            raw = _read_json(users_path)
            found["keys"] = {k: v for k, v in raw.items() if isinstance(v, dict)}
        except Exception as exc:
            found["notes"].append(f"{users_path}: unreadable ({exc})")
    else:
        found["notes"].append(f"no users file at {users_path}")
    usage_path = data_dir / "usage.json"
    if usage_path.exists():
        try:
            raw = _read_json(usage_path)
            found["usage"] = {k: v for k, v in raw.items() if isinstance(v, dict)}
        except Exception as exc:
            found["notes"].append(f"{usage_path}: unreadable ({exc})")
    for kind, name in SESSION_FILES.items():
        path = data_dir / "sessions" / name
        if path.exists():
            rows, bad = _read_jsonl(path)
            found["sessions"][kind] = rows
            if bad:
                found["notes"].append(f"{path}: {bad} unreadable line(s) skipped")
    cache_dir = data_dir / "mock_cache"
    if cache_dir.is_dir():
        for path in sorted(cache_dir.glob("*.json")):
            try:
                found["cache"][path.stem] = _read_json(path)
            except Exception as exc:
                found["notes"].append(f"{path}: unreadable ({exc})")
    return found


def _print_found(found, data_dir, users_path):
    print(f"folder {data_dir}; users file {users_path}")
    print(f"  access keys        {len(found['keys'])}")
    print(f"  usage rows         {len(found['usage'])}")
    for kind in store.SESSION_KINDS:
        print(f"  {kind + ' sessions':<18} {len(found['sessions'][kind])}")
    print(f"  cached plans       {len(found['cache'])}")
    for note in found["notes"]:
        print(f"  note: {note}")


def _print_counts(label, counts):
    sessions = ", ".join(f"{k} {v}" for k, v in counts["sessions"].items())
    print(f"{label}: {counts['keys']} active key(s), usage day-rows "
          f"{counts.get('usage_rows', '?')}, sessions {sessions}, "
          f"{counts['cached_plans']} cached plan(s)")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Import data/ and users.json into Postgres")
    parser.add_argument("--data", default=str(config.BASE_DIR / "data"),
                        help="the data/ folder (default: the repo's)")
    parser.add_argument("--users", default=str(config.USERS_PATH),
                        help="users.json (default: the repo's)")
    parser.add_argument("--database-url", default=None,
                        help="postgresql://... (default: DATABASE_URL from the environment or .env)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print what would be imported and the current table counts; write nothing")
    args = parser.parse_args(argv)

    config.load_env_file()
    url = args.database_url or os.environ.get("DATABASE_URL", "").strip() or config.DATABASE_URL
    found = collect(args.data, args.users)
    _print_found(found, args.data, args.users)

    if args.dry_run:
        if url:
            try:
                st = store.PostgresStore(url)
                _print_counts(f"tables now at {st.location()}", st.counts())
                st.close()
            except store.StoreError as exc:
                print(f"  database not checked: {exc}")
        else:
            print("  no DATABASE_URL: table counts not shown")
        print("dry run: nothing written")
        return 0

    if not url:
        print("No database: pass --database-url or set DATABASE_URL (environment or .env).")
        return 2
    try:
        st = store.PostgresStore(url)
    except store.StoreError as exc:
        print(f"Cannot import: {exc}")
        return 2
    try:
        _print_counts("before", st.counts())
        keys = st.upsert_users(found["keys"])
        usage = st.upsert_usage(found["usage"])
        new_sessions = {kind: sum(1 for row in rows if st.append_session(kind, row))
                        for kind, rows in found["sessions"].items()}
        for digest, payload in found["cache"].items():
            st.cache_put(digest, payload)
        print(f"written: {keys} key(s) upserted, {usage} usage row(s) upserted, "
              + ", ".join(f"{n} new {kind} session(s)" for kind, n in new_sessions.items())
              + f", {len(found['cache'])} cached plan(s) upserted")
        _print_counts("after", st.counts())
    finally:
        st.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
