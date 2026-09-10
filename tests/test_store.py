"""Tests for the state store (coach/store.py, roadmap step 4).

Always: the file backend's contract (keys, a locked usage transaction that
writes only what the policy stages, session appends, the plan cache) and a
20-thread race for a 5-call budget that exactly 5 win. With
TEST_DATABASE_URL naming a reachable Postgres (CI's service container, or
a local `docker run pgvector/pgvector:pg16`): the same contract and race
on the Postgres backend, key revocation, duplicate-append dedupe, the
whole tier/budget suite (tests/test_users.py) re-run against Postgres, and
the migration tool's round trip and idempotency.

Run:  .venv\\Scripts\\python tests\\test_store.py
      TEST_DATABASE_URL=postgresql://coach:<password>@127.0.0.1:5433/coach python tests/test_store.py
"""

import importlib.util
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from coach import config, store, users  # noqa: E402

os.environ.setdefault("DEEPSEEK_API_KEY", "dummy-routing-only-never-called")
os.environ.setdefault("ANTHROPIC_API_KEY", "dummy-routing-only-never-called")
config.MODE = "claude"
TODAY = "2026-09-10"


def temp_paths():
    tmp = Path(tempfile.mkdtemp(prefix="coach_store_"))
    config.USERS_PATH = tmp / "users.json"
    config.USAGE_PATH = tmp / "usage.json"
    config.REAL_SESSIONS_PATH = tmp / "sessions" / "real_sessions.jsonl"
    config.FREE_SESSIONS_PATH = tmp / "sessions" / "free_sessions.jsonl"
    config.MOCK_SESSIONS_PATH = tmp / "sessions" / "mock_sessions.jsonl"
    os.environ["MOCK_CACHE_DIR"] = str(tmp / "mock_cache")
    return tmp


def wipe(st):
    """Empty every table of a Postgres store (the file store is fresh per
    temp folder)."""
    if st.backend == "postgres":
        with st.pool.connection() as conn:
            conn.execute("TRUNCATE access_keys, usage_counters, session_log, plan_cache")


def check_contract(st):
    """The store contract, identical for both backends."""
    st.reset_usage()
    # keys: users.json's shape in, the same shape out
    st.replace_users({"k1": {"name": "a", "tier": "paid", "daily_llm_calls": 3},
                      "k2": {"tier": "free", "log": False}})
    assert st.users_present()
    loaded = st.load_users()
    assert loaded["k1"]["name"] == "a" and loaded["k1"]["tier"] == "paid"
    assert loaded["k1"]["daily_llm_calls"] == 3 and loaded["k2"]["tier"] == "free"
    assert loaded["k2"]["log"] is False
    # usage: a transaction that stages nothing writes nothing
    with st.usage_transaction(["_server", "x"], TODAY) as txn:
        row = txn.rows.get("x")
        assert row is None or (row["llm"] == 0 and row["used"] == 0 and row["voice"] == 0)
    assert users._row(st.usage_read(["x"], TODAY), "x", TODAY)["llm"] == 0
    assert all(r.get("llm", 0) == 0 for r in st.usage_dump().values())
    # a staged write persists, in the file's row shape
    with st.usage_transaction(["_server", "x"], TODAY) as txn:
        txn.write("x", {"date": TODAY, "used": 1, "llm": 2, "voice": 3.5})
        txn.write("_server", {"date": TODAY, "used": 0, "llm": 2, "voice": 3.5})
    rows = st.usage_read(["x", "_server"], TODAY)
    assert rows["x"]["used"] == 1 and rows["x"]["llm"] == 2 and rows["x"]["voice"] == 3.5
    assert rows["_server"]["llm"] == 2 and rows["x"]["date"] == TODAY
    # another day starts from zero (the file keeps a stale row, the table
    # another day-row; users._row hides the difference)
    assert users._row(st.usage_read(["x"], "2026-09-11"), "x", "2026-09-11")["llm"] == 0
    dump = st.usage_dump()
    assert dump["x"]["llm"] == 2 and dump["_server"]["voice"] == 3.5
    # an exception inside the block writes nothing
    try:
        with st.usage_transaction(["x"], TODAY) as txn:
            txn.write("x", {"date": TODAY, "used": 9, "llm": 9, "voice": 9})
            raise RuntimeError("policy blew up")
    except RuntimeError:
        pass
    assert st.usage_read(["x"], TODAY)["x"]["llm"] == 2
    # sessions: three kinds, appended and read back in order
    for kind in store.SESSION_KINDS:
        assert st.read_sessions(kind) == []
    st.append_session("free", {"timestamp": "2026-09-10T10:00:00", "user": "u", "answer": "a"})
    st.append_session("free", {"timestamp": "2026-09-10T10:00:05", "user": "v", "answer": "b"})
    st.append_session("real", {"timestamp": "2026-09-10T10:00:01", "user": "u",
                               "evaluation": {"overall_score": 7, "strengths": ["x"]}})
    assert [r["user"] for r in st.read_sessions("free")] == ["u", "v"]
    assert st.read_sessions("real")[0]["evaluation"]["overall_score"] == 7
    assert st.read_sessions("mock") == []
    # plan cache: get/put/overwrite/trim keeps the newest
    assert st.cache_get("d1") is None
    st.cache_put("d1", {"roles": [1]})
    time.sleep(0.02)
    st.cache_put("d2", {"roles": [2]})
    assert st.cache_get("d1") == {"roles": [1]} and st.cache_count() == 2
    time.sleep(0.02)
    st.cache_put("d1", {"roles": [3]})
    assert st.cache_get("d1") == {"roles": [3]}
    st.cache_trim(1)
    assert st.cache_count() == 1 and st.cache_get("d1") == {"roles": [3]}
    assert st.cache_get("d2") is None
    st.cache_trim(0)
    assert st.cache_count() == 0
    # about
    assert st.info() == {"backend": st.backend}
    counts = st.counts()
    assert counts["keys"] == 2 and counts["sessions"] == {"real": 1, "free": 2, "mock": 0}
    assert counts["cached_plans"] == 0 and isinstance(st.describe(), str)


def check_race(st):
    """20 threads race for a key capped at 5 calls a day: exactly 5 win and
    the counters say 5 - the file store by its process lock, the Postgres
    store by the row lock (two processes would get the same answer there)."""
    st.replace_users({"race": {"name": "r", "tier": "paid", "daily_llm_calls": 5}})
    users._users_stamp = None
    users.load_users()
    st.reset_usage()
    config.LLM_DAILY_CAP = 0
    config.PAID_DAILY_QUOTA = 100
    user = users.resolve_key("race")
    results, lock, start = [], threading.Lock(), threading.Event()

    def worker():
        start.wait()
        outcome = users.take_call(user, "deepseek")
        with lock:
            results.append(outcome)

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    start.set()
    for t in threads:
        t.join(30)
    assert results.count(None) == 5 and results.count("budget") == 15, results
    dump = st.usage_dump()
    assert dump[users._usage_id("race")]["llm"] == 5 and dump["_server"]["llm"] == 5


def test_file_store():
    temp_paths()
    st = store.use(store.FileStore())
    check_contract(st)
    check_race(st)
    # the file backend wrote the same files as before the store existed
    assert config.USAGE_PATH.exists() and config.FREE_SESSIONS_PATH.exists()
    assert json.loads(config.USERS_PATH.read_text(encoding="utf-8"))["race"]["tier"] == "paid"


def test_init_selects_the_backend():
    saved = os.environ.pop("DATABASE_URL", None)
    try:
        config.DATABASE_URL = ""
        st = store.init()
        assert st.backend == "file" and store.current() is st
    finally:
        if saved is not None:
            os.environ["DATABASE_URL"] = saved


def test_bad_database_url_fails_loudly():
    if importlib.util.find_spec("psycopg") is None:
        print("SKIP: psycopg not installed")
        return
    try:
        store.PostgresStore("postgresql://nobody:nothing@127.0.0.1:1/none", connect_timeout=2)
        raise AssertionError("StoreError expected")
    except store.StoreError as exc:
        assert "127.0.0.1:1/none" in str(exc) and "nothing" not in str(exc)


def _run_tier_suite(st):
    """tests/test_users.py, every test, against this store."""
    import test_users  # noqa: E402  (tests/ is on sys.path)
    store.use(st)
    names = [n for n, fn in vars(test_users).items() if n.startswith("test_") and callable(fn)]
    for name in names:
        wipe(st)
        getattr(test_users, name)()
        print(f"ok {st.backend} {name}")
    return len(names)


def _migration_fixture():
    """A data/ folder as the app writes it: two keys, two usage rows, three
    session files, one cached plan, one unreadable log line."""
    data = Path(tempfile.mkdtemp(prefix="coach_migrate_"))
    (data / "sessions").mkdir()
    (data / "mock_cache").mkdir()
    users_path = data / "users.json"
    users_path.write_text(json.dumps({
        "paid-key-1": {"name": "me", "tier": "paid", "daily_llm_calls": 60,
                       "daily_voice_minutes": 30},
        "free-key-2": {"name": "friend", "tier": "free", "log": False},
        "junk": "not a record"}), encoding="utf-8")
    (data / "usage.json").write_text(json.dumps({
        users._usage_id("paid-key-1"): {"date": TODAY, "used": 2, "llm": 22, "voice": 466.0},
        "_server": {"date": TODAY, "used": 2, "llm": 22, "voice": 466.0}}), encoding="utf-8")
    free = [{"timestamp": "2026-09-08T09:02:08", "user": "anonymous", "graded_by": "local_ml",
             "answer": "a", "local_score": 4},
            {"timestamp": "2026-09-08T09:03:35", "user": "anonymous", "graded_by": "local_ml",
             "answer": "b", "local_score": 6}]
    (data / "sessions" / "free_sessions.jsonl").write_text(
        "\n".join(json.dumps(r) for r in free) + "\nnot json\n", encoding="utf-8")
    (data / "sessions" / "real_sessions.jsonl").write_text(json.dumps(
        {"timestamp": "2026-09-02T01:57:00", "user": "me", "graded_by": "claude",
         "evaluation": {"overall_score": 8}}) + "\n", encoding="utf-8")
    (data / "mock_cache" / ("a" * 64 + ".json")).write_text(
        json.dumps({"roles": [{"title": "MLE"}]}), encoding="utf-8")
    return data, users_path


def _load_migrate():
    spec = importlib.util.spec_from_file_location(
        "migrate_to_postgres", ROOT / "tools" / "migrate_to_postgres.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_migration(st, url):
    migrate = _load_migrate()
    data, users_path = _migration_fixture()
    wipe(st)
    found = migrate.collect(data, users_path)
    assert len(found["keys"]) == 2 and len(found["usage"]) == 2
    assert len(found["sessions"]["free"]) == 2 and len(found["sessions"]["real"]) == 1
    assert len(found["cache"]) == 1 and any("unreadable line" in n for n in found["notes"])
    # dry run: writes nothing
    assert migrate.main(["--data", str(data), "--users", str(users_path),
                         "--database-url", url, "--dry-run"]) == 0
    assert st.counts()["keys"] == 0
    # import, then the app reads the imported state through the store
    assert migrate.main(["--data", str(data), "--users", str(users_path),
                         "--database-url", url]) == 0
    counts = st.counts()
    assert counts["keys"] == 2 and counts["sessions"] == {"real": 1, "free": 2, "mock": 0}
    assert counts["cached_plans"] == 1 and counts["usage_rows"] == 2
    users._users_stamp = None
    users.load_users()
    me = users.resolve_key("paid-key-1")
    assert me["tier"] == "paid" and me["llm_cap"] == 60 and me["voice_cap"] == 30
    assert users.resolve_key("free-key-2")["log"] is False
    assert users.resolve_key("junk")["key"] is None
    assert st.usage_dump()[users._usage_id("paid-key-1")]["voice"] == 466.0
    assert st.cache_get("a" * 64) == {"roles": [{"title": "MLE"}]}
    assert st.read_sessions("free")[1]["answer"] == "b"
    # a second run changes nothing
    assert migrate.main(["--data", str(data), "--users", str(users_path),
                         "--database-url", url]) == 0
    assert st.counts() == counts
    # no database named: a clear exit code, nothing else
    saved = os.environ.pop("DATABASE_URL", None)
    try:
        config.DATABASE_URL = ""
        assert migrate.main(["--data", str(data), "--users", str(users_path)]) == 2
    finally:
        if saved is not None:
            os.environ["DATABASE_URL"] = saved


def check_pgvector(st):
    """PgVectorRetriever returns exactly what DenseRetriever returns for the
    same vectors (the fake embedder of the dense tests), reuses a stored
    corpus by fingerprint and reloads it when the bank changes."""
    import test_dense_retrieval as dense_tests  # noqa: E402  (tests/ is on sys.path)
    from retrieval_dense import DenseRetriever, PgVectorRetriever, vector_literal

    assert vector_literal([0.5, -1, 2]) == "[0.5,-1.0,2.0]"
    embedder = dense_tests.FakeEmbedder()
    chunks = list(dense_tests.CHUNKS)
    vectors = embedder.embed_docs([dense_tests.retrieval_text(c) for c in chunks])
    with st.pool.connection() as conn:
        conn.execute("DROP TABLE IF EXISTS chunk_vectors_test")
    numpy_arm = DenseRetriever(chunks, embedder, vectors)
    pg = PgVectorRetriever(chunks, embedder, vectors, name="t", pool=st.pool,
                           table="chunk_vectors_test")
    assert pg.build_seconds is not None and pg.size_mb() > 0
    queries = ["overfitting regularization", "test information leaking into training",
               "gradient boosting trees", "what is precision", ""]
    for query in queries:
        for kwargs in ({}, {"module": "Regression"}, {"level": "Senior"},
                       {"exclude_ids": [chunks[0]["id"]]}):
            a = [c["id"] for c in numpy_arm.search(query=query, limit=3, **kwargs)]
            b = [c["id"] for c in pg.search(query=query, limit=3, **kwargs)]
            assert a == b, (query, kwargs, a, b)
        a = [(round(s, 5), c["id"]) for s, c in numpy_arm.top_scored(query, limit=2)]
        b = [(round(s, 5), c["id"]) for s, c in pg.top_scored(query, limit=2)]
        assert a == b, (query, a, b)
    # the second construction reuses the stored rows; a changed bank reloads
    again = PgVectorRetriever(chunks, embedder, vectors, name="t", pool=st.pool,
                              table="chunk_vectors_test")
    assert again.build_seconds is None
    changed = [dict(c, interview=dict(c["interview"], question=c["interview"]["question"] + " (v2)"))
               for c in chunks]
    changed_vectors = embedder.embed_docs([dense_tests.retrieval_text(c) for c in changed])
    reloaded = PgVectorRetriever(changed, embedder, changed_vectors, name="t", pool=st.pool,
                                 table="chunk_vectors_test")
    assert reloaded.build_seconds is not None and reloaded.fingerprint != pg.fingerprint
    with st.pool.connection() as conn:
        rows = conn.execute("SELECT count(*), count(DISTINCT fingerprint) FROM chunk_vectors_test "
                            "WHERE corpus = 't'").fetchone()
        assert rows == (len(chunks), 1)
        conn.execute("DROP TABLE chunk_vectors_test")


def test_postgres_store():
    url = os.environ.get("TEST_DATABASE_URL", "").strip()
    if not url:
        print("SKIP: TEST_DATABASE_URL not set - the Postgres backend was not tested here")
        return
    temp_paths()
    st = store.use(store.PostgresStore(url))
    wipe(st)
    check_contract(st)
    check_race(st)
    check_pgvector(st)
    # revocation: the key stops resolving on the next reload
    st.replace_users({"gone": {"name": "g", "tier": "paid"}})
    users._users_stamp = None
    users.load_users()
    assert users.resolve_key("gone")["tier"] == "paid"
    assert st.revoke_key("gone") == 1 and st.revoke_key("gone") == 0
    users._users_stamp = None
    assert users.resolve_key("gone")["tier"] == "free"
    listed = st.list_keys()
    assert listed[0]["revoked_at"] and listed[0]["key_prefix"] == "gone…"
    # the same record appended twice lands once
    assert st.append_session("mock", {"timestamp": "2026-09-10T11:00:00", "x": 1}) is True
    assert st.append_session("mock", {"timestamp": "2026-09-10T11:00:00", "x": 1}) is False
    assert len(st.read_sessions("mock")) == 1
    n = _run_tier_suite(st)
    print(f"tier/budget suite: {n} tests passed on postgres")
    check_migration(st, url)
    wipe(st)
    store.use(store.FileStore())


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("all store tests passed")
