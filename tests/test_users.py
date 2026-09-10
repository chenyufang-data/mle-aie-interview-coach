"""Offline tests for the tier budgets (coach/users.py), the grading route,
the mock engine gate, the request-body cap and the anonymous-LLM refusal.

Run:  .venv\\Scripts\\python tests\\test_users.py

No network, no API key: the DeepSeek key in the environment is a dummy that
only makes routing prefer DeepSeek; nothing is ever called.
"""

import io
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coach import config, grading, store, users, web  # noqa: E402
from coach.mock import engine as engines  # noqa: E402
import server  # noqa: E402

# The store under test: the file backend in a temp folder by default -
# pinned here so a DATABASE_URL in the developer's environment or .env can
# never point these tests at a real database; tests/test_store.py swaps in
# the Postgres backend it was given and re-runs these tests.
store.use(store.FileStore())
TMP = Path(tempfile.mkdtemp(prefix="coach_users_"))
config.USERS_PATH = TMP / "users.json"
config.USAGE_PATH = TMP / "usage.json"
config.MODE = "claude"
os.environ["DEEPSEEK_API_KEY"] = "dummy-routing-only-never-called"
os.environ["ANTHROPIC_API_KEY"] = "dummy-routing-only-never-called"

KEYS = {
    "demo-key": {"name": "demo", "tier": "paid", "daily_llm_calls": 2},
    "owner-key": {"name": "me", "tier": "paid"},
    "guest-key": {"name": "guest", "tier": "free"},
}


def setup(keys=KEYS, server_cap=0, quota=30):
    st = store.current()
    st.replace_users(keys)
    users._users_stamp = None
    users.load_users()
    st.reset_usage()
    config.LLM_DAILY_CAP = server_cap
    config.PAID_DAILY_QUOTA = quota


def usage():
    return store.current().usage_dump()


def test_per_key_budget():
    setup()
    demo = users.resolve_key("demo-key")
    assert demo["tier"] == "paid" and demo["llm_cap"] == 2
    assert grading.grading_route(demo) == ("deepseek", None)
    assert grading.grading_route(demo) == ("deepseek", None)
    assert grading.grading_route(demo) == ("local", "budget")
    assert users.budget_left(demo) == {"key": 0, "server": None}
    # The owner's uncapped key is untouched by the demo key's budget.
    owner = users.resolve_key("owner-key")
    assert owner["llm_cap"] == 0
    for _ in range(5):
        assert grading.grading_route(owner) == ("deepseek", None)
    assert users.budget_left(owner) == {"key": None, "server": None}
    # Usage rows are keyed by digest, never the raw key; the server row exists.
    rows = usage()
    assert "demo-key" not in rows and "owner-key" not in rows
    assert rows["_server"]["llm"] == 7


def test_server_cap():
    setup(server_cap=3)
    owner = users.resolve_key("owner-key")
    for _ in range(3):
        assert grading.grading_route(owner) == ("deepseek", None)
    assert grading.grading_route(owner) == ("local", "budget")
    # Server-wide: the demo key is refused too, before its own cap.
    demo = users.resolve_key("demo-key")
    assert grading.grading_route(demo) == ("local", "budget")
    assert users.budget_left(demo) == {"key": 2, "server": 0}
    assert usage()["_server"]["llm"] == 3  # refusals write nothing


def test_claude_quota_degrades_then_budget():
    setup(quota=1)
    demo = users.resolve_key("demo-key")
    # "Always Claude" takes the Claude quota AND one budget unit.
    assert grading.grading_route(demo, force_llm=True) == ("claude", None)
    assert users.quota_left(demo) == 0
    # Quota spent: degrades to DeepSeek, which still costs a budget unit.
    assert grading.grading_route(demo, force_llm=True) == ("deepseek", None)
    # Budget spent (2 of 2): local, even with force_llm.
    assert grading.grading_route(demo, force_llm=True) == ("local", "budget")
    row = [r for k, r in usage().items() if k != "_server"][0]
    assert row["used"] == 1 and row["llm"] == 2


def test_quota_without_deepseek():
    setup(quota=1)
    saved = os.environ.pop("DEEPSEEK_API_KEY")
    try:
        owner = users.resolve_key("owner-key")
        assert grading.grading_route(owner) == ("claude", None)
        assert grading.grading_route(owner) == ("local", "quota")
    finally:
        os.environ["DEEPSEEK_API_KEY"] = saved


def test_force_llm_without_claude_key():
    setup()
    saved = os.environ.pop("ANTHROPIC_API_KEY")
    try:
        demo = users.resolve_key("demo-key")
        # The demo box has no Claude key: "Always Claude" routes to DeepSeek
        # and takes no Claude quota, only a budget unit.
        assert grading.grading_route(demo, force_llm=True) == ("deepseek", None)
        assert users.quota_left(demo) == config.PAID_DAILY_QUOTA
        assert users.budget_left(demo)["key"] == 1
    finally:
        os.environ["ANTHROPIC_API_KEY"] = saved


def test_free_and_anonymous():
    setup(server_cap=1)
    guest = users.resolve_key("guest-key")
    anon = users.resolve_key("no-such-key")
    assert guest["tier"] == "free" and anon["tier"] == "free" and anon["key"] is None
    assert grading.grading_route(guest) == ("local", "free")
    assert grading.grading_route(anon) == ("local", "free")
    assert usage() == {}  # free routing never touches usage
    assert users.budget_left(anon) == {"key": None, "server": 1}
    assert users.quota_left(anon) == 0


def test_mock_gate_messages():
    setup(quota=1)
    demo = users.resolve_key("demo-key")
    assert engines.pick_engine(demo) == "deepseek"
    assert engines.pick_engine(demo) == "deepseek"
    try:
        engines.pick_engine(demo)
        raise AssertionError("budget refusal expected")
    except engines.MockUnavailable as exc:
        assert "allowance" in str(exc)
    try:
        engines.pick_engine(users.resolve_key("guest-key"))
        raise AssertionError("free tier refusal expected")
    except engines.MockUnavailable as exc:
        assert "paid access" in str(exc)
    config.MODE = "mock"
    try:
        assert engines.pick_engine(users.resolve_key("guest-key")) == "fake"
    finally:
        config.MODE = "claude"


def test_cap_field_tolerates_junk():
    setup({"k1": {"tier": "paid", "daily_llm_calls": "12"},
           "k2": {"tier": "paid", "daily_llm_calls": "lots"},
           "k3": {"tier": "paid", "daily_llm_calls": -5}})
    assert users.resolve_key("k1")["llm_cap"] == 12
    assert users.resolve_key("k2")["llm_cap"] == 0
    assert users.resolve_key("k3")["llm_cap"] == 0


class FakeHandler:
    def __init__(self, body, length=None):
        self.headers = {"Content-Length": str(len(body) if length is None else length)}
        self.rfile = io.BytesIO(body)


def test_read_json_cap():
    assert web.read_json(FakeHandler(b"")) == {}
    assert web.read_json(FakeHandler(b'{"a": 1}')) == {"a": 1}
    big = b'{"answer": "' + b"x" * 200 + b'"}'
    try:
        web.read_json(FakeHandler(big), limit=100)
        raise AssertionError("413 expected")
    except web.PayloadTooLarge as exc:
        assert "limit" in str(exc)
    for bad in (b"[1, 2]", b"{not json", b"\xff\xfe"):
        try:
            web.read_json(FakeHandler(bad))
            raise AssertionError("400 expected")
        except web.PayloadTooLarge:
            raise AssertionError("bad JSON must not read as too large")
        except ValueError:
            pass
    try:
        web.read_json(FakeHandler(b"{}", length="abc"))
        raise AssertionError("400 expected")
    except ValueError:
        pass
    # The file routes keep their larger caps; the default is 1 MB.
    from coach.http import BODY_LIMITS
    assert web.DEFAULT_BODY_LIMIT == 1_000_000
    assert BODY_LIMITS["/api/mock/parse_file"] > 14_000_000
    assert BODY_LIMITS["/api/mock/transcribe"] > 34_000_000


def test_anonymous_llm_refusal():
    saved = users.TIERS_ENABLED
    try:
        users.TIERS_ENABLED = False
        config.MODE = "claude"
        config.ALLOW_ANONYMOUS_LLM = False
        assert server.anonymous_llm_refused("0.0.0.0", False)
        assert not server.anonymous_llm_refused("127.0.0.1", False)
        assert not server.anonymous_llm_refused("0.0.0.0", True)
        config.ALLOW_ANONYMOUS_LLM = True
        assert not server.anonymous_llm_refused("0.0.0.0", False)
        config.ALLOW_ANONYMOUS_LLM = False
        users.TIERS_ENABLED = True
        assert not server.anonymous_llm_refused("0.0.0.0", False)
        users.TIERS_ENABLED = False
        config.MODE = "mock"
        assert not server.anonymous_llm_refused("0.0.0.0", False)
    finally:
        users.TIERS_ENABLED = saved
        config.MODE = "claude"


def test_voice_budget():
    """Live-voice minutes: per-key "daily_voice_minutes" and the server-wide
    VOICE_DAILY_MINUTES, charged in seconds. The tick that reaches a cap is
    charged whole and the next one is refused; LLM counters are untouched."""
    setup({"demo-key": {"name": "demo", "tier": "paid", "daily_voice_minutes": 2},
           "owner-key": {"name": "me", "tier": "paid"}})
    config.VOICE_DAILY_MINUTES = 0
    try:
        demo, owner = users.resolve_key("demo-key"), users.resolve_key("owner-key")
        assert demo["voice_cap"] == 2 and owner["voice_cap"] == 0
        assert users.voice_left(demo) == {"key": 120, "server": None}
        assert users.take_voice(demo, 60) is None
        assert users.take_voice(demo, 60) is None
        assert users.voice_left(demo)["key"] == 0
        assert users.take_voice(demo, 60) == "voice"
        assert usage()[users._usage_id("demo-key")]["voice"] == 120
        assert users.minutes_left(users.voice_left(demo)["key"]) == 0
        # the owner's key is unlimited; a server-wide cap still applies to it
        assert users.take_voice(owner, 45.5) is None
        assert users.voice_left(owner) == {"key": None, "server": None}
        config.VOICE_DAILY_MINUTES = 3
        assert users.voice_left(owner)["server"] == 14.5
        assert users.minutes_left(14.5) == 1 and users.minutes_left(None) is None
        assert users.take_voice(owner, 60) is None       # 165.5 s used: the tick still starts
        assert users.take_voice(owner, 60) == "voice"    # 225.5 s used: over the 180 s cap
        config.VOICE_DAILY_MINUTES = 0
        # anonymous callers have no key row; the server row still counts them
        anon = users.resolve_key("")
        assert anon["voice_cap"] == 0
        assert users.take_voice(anon, 30) is None
        assert usage()[users.SERVER_ROW]["voice"] == 255.5
        assert usage()[users._usage_id("demo-key")]["llm"] == 0
        assert users.budget_left(demo) == {"key": None, "server": None}
    finally:
        config.VOICE_DAILY_MINUTES = 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("all tier/budget tests passed")
