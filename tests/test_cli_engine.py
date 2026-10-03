"""Offline tests for the local subscription engine (coach/cli_engine.py,
server.py --cli): the commands it builds, how it reads each CLI's output,
streaming, the loopback rule, and the routing around it.

Run:  .venv\\Scripts\\python tests\\test_cli_engine.py

No CLI is run: subprocess is replaced by fakes that answer the way
`claude -p --output-format json|stream-json` and `codex exec -o` do.
"""

import io
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coach import cli_engine, config, grading, llm, store  # noqa: E402
from coach.mock import routes as mock_routes  # noqa: E402

SCHEMA = {"type": "object", "additionalProperties": False,
          "properties": {"overall_score": {"type": "integer"}}, "required": ["overall_score"]}


class FakeProc:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout, self.stderr, self.returncode = stdout, stderr, returncode


def use(provider, model=""):
    config.CLI_PROVIDER = provider
    config.CLI_MODEL = model
    config.CLI_EFFORT = "low"
    config.CLI_TIMEOUT_S = 30


def with_fake_run(fake):
    real = cli_engine.subprocess.run
    cli_engine.subprocess.run = fake
    return real


def test_claude_command():
    use("claude")
    args = cli_engine.claude_args("SYS", SCHEMA, "low")
    assert args[1:3] == ["-p", "--output-format"] and args[3] == "json"
    i = args.index("--tools")
    assert args[i + 1] == "", "no file or shell tools"
    for flag in ("--safe-mode", "--no-session-persistence"):
        assert flag in args
    assert "--bare" not in args, "bare mode ignores the subscription login"
    assert args[args.index("--model") + 1] == "sonnet"
    assert args[args.index("--system-prompt") + 1] == "SYS"
    assert json.loads(args[args.index("--json-schema") + 1]) == SCHEMA
    assert "--json-schema" not in cli_engine.claude_args("SYS")
    streaming = cli_engine.claude_args("SYS", stream=True)
    assert streaming[3] == "stream-json" and "--include-partial-messages" in streaming
    use("claude", "opus")
    assert cli_engine.claude_args("S")[cli_engine.claude_args("S").index("--model") + 1] == "opus"


def test_codex_command():
    use("codex")
    args = cli_engine.codex_args("WD", "OUT", "SCHEMA.json", "low")
    assert args[1] == "exec" and args[-1] == "-", "prompt from stdin"
    assert args[args.index("--sandbox") + 1] == "read-only"
    assert args[args.index("-C") + 1] == "WD" and args[args.index("-o") + 1] == "OUT"
    assert "--ephemeral" in args and "--skip-git-repo-check" in args
    assert args[args.index("--output-schema") + 1] == "SCHEMA.json"
    assert "-m" not in args, "no model named: the user's Codex default"
    assert 'model_reasoning_effort="low"' in args
    use("codex", "gpt-x")
    assert cli_engine.codex_args("WD", "OUT")[cli_engine.codex_args("WD", "OUT").index("-m") + 1] == "gpt-x"


def test_claude_complete_reads_structured_output():
    use("claude")
    seen = {}

    def fake(args, input=None, cwd=None, **kw):
        seen.update(args=args, input=input, cwd=cwd)
        return FakeProc(json.dumps({"is_error": False, "structured_output": {"overall_score": 7},
                                    "result": "", "num_turns": 2}))
    real = with_fake_run(fake)
    try:
        assert cli_engine.complete("SYS", "grade this", SCHEMA) == {"overall_score": 7}
        assert seen["input"] == "grade this"
        assert not os.path.exists(seen["cwd"]), "the empty working folder is removed"
        cli_engine.subprocess.run = lambda *a, **k: FakeProc(json.dumps({"is_error": False, "result": " hello "}))
        assert cli_engine.complete("SYS", "hi") == "hello"
        cli_engine.subprocess.run = lambda *a, **k: FakeProc(
            json.dumps({"is_error": True, "result": "You've hit your usage limit"}), returncode=1)
        try:
            cli_engine.complete("SYS", "hi", SCHEMA)
            raise AssertionError("expected RuntimeError")
        except RuntimeError as exc:
            assert "usage limit" in str(exc)
    finally:
        cli_engine.subprocess.run = real


def test_codex_complete_reads_last_message():
    use("codex")

    def fake(args, input=None, cwd=None, **kw):
        out = Path(args[args.index("-o") + 1])
        schema_file = Path(args[args.index("--output-schema") + 1])
        assert json.loads(schema_file.read_text(encoding="utf-8")) == SCHEMA
        assert input.startswith("SYS\n\ngrade this") and input.endswith("JSON object only.")
        out.write_text('{"overall_score": 5}', encoding="utf-8")
        return FakeProc(stderr="progress...")
    real = with_fake_run(fake)
    try:
        assert cli_engine.complete("SYS", "grade this", SCHEMA) == {"overall_score": 5}
        cli_engine.subprocess.run = lambda *a, **k: FakeProc(
            stderr="ERROR: model requires a newer version of Codex", returncode=1)
        try:
            cli_engine.complete("SYS", "x", SCHEMA)
            raise AssertionError("expected RuntimeError")
        except RuntimeError as exc:
            assert "newer version of Codex" in str(exc)
    finally:
        cli_engine.subprocess.run = real


def test_claude_stream_yields_deltas():
    use("claude")
    lines = [json.dumps({"type": "system", "subtype": "init"}),
             json.dumps({"type": "stream_event", "event": {"delta": {"type": "text_delta", "text": "Hi "}}}),
             json.dumps({"type": "stream_event", "event": {"delta": {"type": "text_delta", "text": "there."}}}),
             json.dumps({"type": "result", "is_error": False, "result": "Hi there."})]

    class FakePopen:
        def __init__(self, args, **kw):
            assert "stream-json" in args
            self.stdin = io.StringIO()
            self.stdout = iter(line + "\n" for line in lines)
        def wait(self, timeout=None):
            return 0
        def poll(self):
            return 0
        def kill(self):
            pass
    real = cli_engine.subprocess.Popen
    cli_engine.subprocess.Popen = FakePopen
    try:
        assert list(cli_engine.stream("SYS", "prompt")) == ["Hi ", "there."]
    finally:
        cli_engine.subprocess.Popen = real


def test_transcript_and_loopback():
    text = cli_engine.transcript([{"role": "user", "content": "Tell me about X."},
                                  {"role": "assistant", "content": [{"type": "text", "text": "Sure."}]},
                                  {"role": "user", "content": "And Y?"}])
    assert text.index("USER:\nTell me about X.") < text.index("ASSISTANT:\nSure.") < text.index("USER:\nAnd Y?")
    assert text.rstrip().endswith("no commentary.")
    assert not cli_engine.bind_refused("127.0.0.1") and not cli_engine.bind_refused("localhost")
    assert cli_engine.bind_refused("0.0.0.0") and cli_engine.bind_refused("192.168.1.5")


def test_routing_in_cli_mode():
    store.use(store.FileStore())
    tmp = Path(tempfile.mkdtemp(prefix="coach_cli_"))
    config.USERS_PATH, config.USAGE_PATH = tmp / "users.json", tmp / "usage.json"
    old_mode = config.MODE
    os.environ["ANTHROPIC_API_KEY"] = "dummy-never-used"
    try:
        config.MODE = "cli"
        use("claude")
        anon = {"key": None, "name": "anonymous", "tier": "free"}
        assert grading.grading_route(anon) == ("cli", None), "no tiers on a personal machine"
        assert mock_routes._default_report_engine("cli") == "cli", "never the API key behind the user's back"
        assert "on your subscription" in llm.engine_model("cli")
        calls = []
        real = cli_engine.complete
        cli_engine.complete = lambda system, prompt, schema=None, effort=None: (
            calls.append((schema is not None, effort)) or {"overall_score": 6})
        try:
            assert llm.call_model("grade", SCHEMA, "cli") == {"overall_score": 6}
            assert calls[-1] == (True, "low")
        finally:
            cli_engine.complete = real
        if grading.GRADER is None:
            grading.load_grader()
        chunk = json.loads((Path(__file__).resolve().parents[1] / "banks" / "rag_ml" /
                            "all_chunks.jsonl").read_text(encoding="utf-8").splitlines()[0])
        result = grading.mock_evaluation({"answer": "some answer"}, chunk, "cli_error")
        assert "Subscription call failed" in result["summary"]
    finally:
        config.MODE = old_mode


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all cli engine tests passed")
