"""Local-only LLM engine on the user's own subscription (`server.py --cli`).

Instead of an API key, every LLM call goes through a coding-agent CLI the
user installed and signed in to themselves: Claude Code (`claude -p`, a
Claude Pro/Max plan) or OpenAI Codex (`codex exec`, a ChatGPT plan). The
app never sees the credentials; it runs the CLI as a subprocess and reads
its JSON. Each call costs plan allowance, not API money.

Local only, by design and by the vendors' terms: Anthropic does not permit
routing requests through Free/Pro/Max plan credentials on behalf of other
users, and a ChatGPT plan is personal. server.py therefore refuses to start
this engine on anything but a loopback address, so one person's plan can
never serve someone else.

How each CLI is run:
- Claude Code: `-p --output-format json`, the app's system prompt replacing
  Claude Code's, `--tools ""` (no file or shell access), `--safe-mode` (the
  user's hooks, skills, plugins, MCP servers and CLAUDE.md stay out), no
  session saved, an explicit model and effort (otherwise the user's
  interactive defaults apply), `--json-schema` for structured calls (the
  answer arrives in `structured_output`). Not `--bare`: bare mode ignores
  the subscription login and reads only API keys.
- Codex: `exec` in a read-only sandbox rooted at an empty temporary folder,
  ephemeral, `--output-schema` for structured calls, the final message read
  from `-o`. Codex has no system-prompt flag, so the system text leads the
  prompt.

Measured 2026-10-02 on the app's grading prompt (Claude Code, Sonnet, low
effort): 10-11 s per evaluation, of which about 0.7 s is starting the CLI;
the rest is generating ~1,400 tokens of feedback, the same work an API call
does.
"""

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from coach import config

PROVIDERS = ("claude", "codex")
LOOPBACK = ("127.0.0.1", "localhost", "::1")

# Variables that make each CLI bill an API account instead of the user's
# signed-in plan. The server loads .env, where the app's own API keys live,
# so a child process would inherit them - and Claude Code prefers an
# ANTHROPIC_API_KEY in its environment over the subscription login. They are
# removed from every CLI call (found 2026-10-03: a server run with the key in
# .env reported "api_key" and billed the API, not the plan).
API_KEY_VARS = {
    "claude": ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
    "codex": ("OPENAI_API_KEY", "CODEX_API_KEY"),
}
SUBSCRIPTION_AUTH = ("claude.ai", "oauth_token")   # Claude Code authMethod values


def provider():
    return config.CLI_PROVIDER


def model():
    """The model the CLI is asked for: LLM_CLI_MODEL, else Sonnet for Claude
    Code and the user's own Codex default for Codex."""
    if config.CLI_MODEL:
        return config.CLI_MODEL
    return "sonnet" if provider() == "claude" else None


def label():
    name = {"claude": "Claude Code", "codex": "Codex"}.get(provider(), "CLI")
    return f"{name} ({model() or 'its default model'}) on your subscription"


def binary(name=None):
    return shutil.which(name or provider())


def child_env(name=None):
    """os.environ without the API-key variables of this CLI, so it runs on
    the user's subscription login."""
    env = dict(os.environ)
    for var in API_KEY_VARS.get(name or provider(), ()):
        env.pop(var, None)
    return env


def bind_refused(host):
    """True when this engine must not serve: anything but loopback."""
    return host not in LOOPBACK


def preference_order(setting):
    """LLM_CLI -> the CLIs to try, in order; () means the subscription is off.
    `auto` (the default) and `claude` try Claude Code first; `codex` tries
    Codex first; `off` keeps the API keys."""
    value = (setting or "auto").strip().lower()
    if value in ("off", "none", "api", "0", "false", "no"):
        return ()
    if value == "codex":
        return ("codex", "claude")
    return ("claude", "codex")


def auto_select(host, setting):
    """The subscription engine as the local default (roadmap step 7, phase 1).

    Returns (provider, message): the CLI to use and its sign-in status, or
    (None, why-not). Only on a loopback bind - a server bound to anything
    else (the demo box, a container) never picks a personal plan, and says
    nothing about it. A CLI that is not installed is skipped silently; one
    that is installed but not on a subscription is skipped with a reason."""
    order = preference_order(setting)
    if not order:
        return None, "LLM_CLI=off: the API keys serve"
    if bind_refused(host):
        return None, None
    reasons = []
    for name in order:
        if not binary(name):
            continue
        ok, message = check_ready(name)
        if ok:
            return name, message
        reasons.append(message)
    return None, ("; ".join(reasons) if reasons else None)


def check_ready(name):
    """(ok, message) for startup: the CLI is installed and signed in."""
    path = binary(name)
    if not path:
        hint = ("install Claude Code (https://code.claude.com) and run `claude` once to sign in"
                if name == "claude" else
                "install the Codex CLI and run `codex login` with your ChatGPT account")
        return False, f"`{name}` is not on PATH: {hint}."
    args = [path, "auth", "status"] if name == "claude" else [path, "login", "status"]
    try:
        proc = subprocess.run(args, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=60, env=child_env(name))
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"could not run `{name}` ({exc})."
    if proc.returncode != 0:
        fix = "run `claude` and sign in" if name == "claude" else "run `codex login`"
        return False, f"`{name}` is not signed in: {fix}."
    if name == "claude":
        try:
            status = json.loads(proc.stdout)
        except ValueError:
            status = {}
        method = status.get("authMethod")
        if method not in SUBSCRIPTION_AUTH:
            return False, (f"`claude` is signed in with {method or 'an unknown method'}, not a "
                           "Claude subscription; run `claude auth login` with your Claude account "
                           "(or use the app's API-key mode instead of --cli).")
        return True, f"`claude` signed in ({method}, {status.get('subscriptionType') or 'plan unknown'})"
    said = (proc.stdout.strip() or proc.stderr.strip())
    if "chatgpt" not in said.lower():
        return False, (f"`codex` reports \"{said}\", not a ChatGPT sign-in; run `codex login` "
                       "with your ChatGPT account (or use the app's API-key mode).")
    return True, f"`codex` signed in ({said})"


def transcript(messages):
    """A chat history as one prompt: the CLIs take a single prompt, so the
    turns are written out and the model is asked for the next reply."""
    lines = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, list):
            content = "\n".join(part.get("text", "") for part in content if isinstance(part, dict))
        who = "ASSISTANT" if message.get("role") == "assistant" else "USER"
        lines.append(f"{who}:\n{content}")
    lines.append("Write the next ASSISTANT message only, exactly as it should be sent - "
                 "no label, no commentary.")
    return "\n\n".join(lines)


# ----------------------------------------------------------- the commands

def claude_args(system, schema=None, effort=None, stream=False):
    args = [binary("claude") or "claude", "-p",
            "--output-format", "stream-json" if stream else "json",
            "--system-prompt", system,
            "--tools", "", "--safe-mode", "--no-session-persistence",
            "--model", model() or "sonnet"]
    if effort:
        args += ["--effort", effort]
    if schema is not None:
        args += ["--json-schema", json.dumps(schema)]
    if stream:
        args += ["--verbose", "--include-partial-messages"]
    return args


def codex_args(workdir, out_path, schema_path=None, effort=None):
    args = [binary("codex") or "codex", "exec", "--skip-git-repo-check", "--ephemeral",
            "--sandbox", "read-only", "-C", str(workdir), "--color", "never",
            "-o", str(out_path)]
    if model():
        args += ["-m", model()]
    if effort:
        args += ["-c", f'model_reasoning_effort="{effort}"']
    if schema_path is not None:
        args += ["--output-schema", str(schema_path)]
    return args + ["-"]


def _run(args, stdin, cwd):
    try:
        proc = subprocess.run(args, input=stdin, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", cwd=cwd,
                              timeout=config.CLI_TIMEOUT_S, env=child_env())
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"{label()} did not answer within {config.CLI_TIMEOUT_S:.0f} s") from None
    except OSError as exc:
        raise RuntimeError(f"could not start {label()}: {exc}") from None
    return proc


def _last_line(text):
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    return lines[-1][:300] if lines else "no output"


def complete(system, prompt, schema=None, effort=None):
    """One call: the parsed object when `schema` is given, else the text."""
    workdir = Path(tempfile.mkdtemp(prefix="coach_cli_"))
    try:
        if provider() == "claude":
            proc = _run(claude_args(system, schema, effort), prompt, workdir)
            try:
                body = json.loads(proc.stdout)
            except ValueError:
                raise RuntimeError(f"{label()} failed: {_last_line(proc.stderr or proc.stdout)}") from None
            if body.get("is_error") or proc.returncode != 0:
                raise RuntimeError(f"{label()} failed: {str(body.get('result') or '')[:300]}")
            if schema is not None:
                result = body.get("structured_output")
                if not isinstance(result, dict):
                    raise RuntimeError(f"{label()} returned no structured output")
                return result
            return (body.get("result") or "").strip()

        out_path = workdir / "last_message.txt"
        schema_path = None
        if schema is not None:
            schema_path = workdir / "schema.json"
            schema_path.write_text(json.dumps(schema), encoding="utf-8")
        text = f"{system}\n\n{prompt}"
        if schema is not None:
            text += "\n\nReply with the JSON object only."
        proc = _run(codex_args(workdir, out_path, schema_path, effort), text, workdir)
        if proc.returncode != 0 or not out_path.exists():
            raise RuntimeError(f"{label()} failed: {_last_line(proc.stderr)}")
        answer = out_path.read_text(encoding="utf-8").strip()
        if schema is None:
            return answer
        try:
            return json.loads(answer)
        except ValueError:
            raise RuntimeError(f"{label()} did not return valid JSON") from None
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def stream(system, prompt, effort=None):
    """Text deltas as they arrive. Claude Code streams them
    (`stream-json` with partial messages); Codex prints only its final
    message, so it arrives as one piece."""
    if provider() != "claude":
        yield complete(system, prompt, None, effort)
        return
    workdir = Path(tempfile.mkdtemp(prefix="coach_cli_"))
    proc = None
    try:
        try:
            proc = subprocess.Popen(claude_args(system, None, effort, stream=True),
                                    stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, text=True, encoding="utf-8",
                                    errors="replace", cwd=workdir, env=child_env())
        except OSError as exc:
            raise RuntimeError(f"could not start {label()}: {exc}") from None
        proc.stdin.write(prompt)
        proc.stdin.close()
        yielded = False
        for line in proc.stdout:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("type") == "stream_event":
                delta = (event.get("event") or {}).get("delta") or {}
                if delta.get("type") == "text_delta" and delta.get("text"):
                    yielded = True
                    yield delta["text"]
            elif event.get("type") == "result":
                if event.get("is_error"):
                    raise RuntimeError(f"{label()} failed: {str(event.get('result') or '')[:300]}")
                if not yielded and event.get("result"):
                    yield event["result"]
        proc.wait(timeout=config.CLI_TIMEOUT_S)
    finally:
        if proc is not None and proc.poll() is None:
            proc.kill()
        shutil.rmtree(workdir, ignore_errors=True)


def effort_for(thinking):
    """The configured effort for reasoning calls; low for quick turns."""
    return config.CLI_EFFORT if thinking else "low"


def environment_note():
    """What the startup banner prints about the subscription engine."""
    return (f"Backend: {label()} - local only; calls count against your plan, "
            f"not an API key (effort {config.CLI_EFFORT}).")

