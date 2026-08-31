"""The three grading engines: Claude (teacher), DeepSeek (paid workhorse),
Ollama (free local)."""

import json
import os
from urllib import error as urlerror
from urllib import request as urlrequest

import anthropic

from coach import config
from coach.config import DEEPSEEK_URL, DEFAULT_MODEL, OLLAMA_URL
from coach.prompts import SYSTEM_PROMPT

_client = None


def get_api_key():
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_KEY", "API_KEY"):
        value = os.environ.get(name)
        if value:
            return value
    return None


def get_client():
    global _client
    if _client is None:
        api_key = get_api_key()
        if not api_key:
            raise RuntimeError("No Anthropic API key found. Add ANTHROPIC_API_KEY to .env.")
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


def call_claude(user_prompt, schema):
    response = get_client().messages.create(
        model=os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL),
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": user_prompt}],
        output_config={"format": {"type": "json_schema", "schema": schema}},
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("The model declined this request. Try a different topic or focus.")
    text = next((block.text for block in response.content if block.type == "text"), "")
    if not text:
        raise RuntimeError("Model response did not include output text")
    return json.loads(text)


def call_ollama(user_prompt, schema):
    payload = {
        "model": config.OLLAMA_MODEL,
        "stream": False,
        "format": schema,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    }
    req = urlrequest.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=300) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urlerror.URLError as exc:
        raise RuntimeError(
            "Could not reach Ollama at localhost:11434. Install it from https://ollama.com, "
            f"run 'ollama pull {config.OLLAMA_MODEL}', and make sure the Ollama app is running."
        ) from exc
    content = body.get("message", {}).get("content", "")
    if not content:
        raise RuntimeError("Ollama response did not include output text")
    return json.loads(content)


def call_deepseek(user_prompt, schema):
    """Grade with DeepSeek via its OpenAI-compatible API.

    Two quirks verified by grader/judge_agreement.py: DeepSeek's
    Anthropic-compat endpoint silently ignores output_config, so the schema
    rides in the prompt instead; and ~3% of json_object responses arrive as
    malformed JSON, so one parse-retry before giving up.
    """
    payload = {
        "model": config.deepseek_model(),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt
             + "\n\nRespond with ONLY a JSON object matching this JSON schema:\n"
             + json.dumps(schema)},
        ],
        "response_format": {"type": "json_object"},
        # V4's hybrid thinking spends completion tokens on reasoning before
        # the JSON: the mock's 7-turn report measured ~6.2k completion tokens
        # with run-to-run variance even at temperature 0, so 8k truncated
        # occasionally. 12k is accepted by the API and leaves headroom.
        "max_tokens": 12000,
        "temperature": 0,
    }
    last_error = None
    for _attempt in range(2):
        req = urlrequest.Request(
            DEEPSEEK_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}"},
            method="POST",
        )
        try:
            with urlrequest.urlopen(req, timeout=300) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urlerror.URLError as exc:
            raise RuntimeError(f"Could not reach the DeepSeek API ({exc})") from exc
        try:
            choice = body["choices"][0]
            if choice.get("finish_reason") == "length":
                raise ValueError("output truncated at max_tokens")
            return json.loads(choice["message"]["content"])
        except (KeyError, IndexError, ValueError) as exc:
            last_error = exc
    raise RuntimeError(f"DeepSeek returned malformed JSON twice ({last_error})")


def call_chat(system, messages, engine, thinking=False, max_tokens=700, temperature=0.7):
    """Plain-text chat, for the mock interviewer's conversational turns.

    Unlike call_model there is no output schema: the turn protocol wants
    spoken text first (streamable to TTS later) with a `\\n---\\n{json}`
    trailer the caller strips. `thinking` maps to each vendor's control —
    measured on DeepSeek V4 Flash: first content token ~0.8 s with thinking
    disabled vs ~1.4 s enabled on a short prompt, and the gap grows with
    prompt size, so interviewer turns run with thinking off while the
    report keeps it on.
    """
    if engine == "ollama":
        payload = {
            "model": config.OLLAMA_MODEL,
            "stream": False,
            "think": bool(thinking),
            "messages": [{"role": "system", "content": system}] + messages,
        }
        req = urlrequest.Request(
            OLLAMA_URL, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urlrequest.urlopen(req, timeout=300) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urlerror.URLError as exc:
            raise RuntimeError("Could not reach Ollama at localhost:11434.") from exc
        return body.get("message", {}).get("content", "").strip()
    if engine == "deepseek":
        payload = {
            "model": config.deepseek_model(),
            "messages": [{"role": "system", "content": system}] + messages,
            "thinking": {"type": "enabled" if thinking else "disabled"},
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        req = urlrequest.Request(
            DEEPSEEK_URL, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}"},
            method="POST")
        try:
            with urlrequest.urlopen(req, timeout=300) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urlerror.URLError as exc:
            raise RuntimeError(f"Could not reach the DeepSeek API ({exc})") from exc
        return (body["choices"][0]["message"].get("content") or "").strip()
    response = get_client().messages.create(
        model=os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL),
        max_tokens=max_tokens if not thinking else max(max_tokens, 4000),
        system=system,
        thinking={"type": "adaptive"} if thinking else {"type": "disabled"},
        temperature=temperature,
        messages=messages,
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("The model declined this request.")
    return next((block.text for block in response.content if block.type == "text"), "").strip()


def call_model(user_prompt, schema, engine):
    if engine == "ollama":
        return call_ollama(user_prompt, schema)
    if engine == "deepseek":
        return call_deepseek(user_prompt, schema)
    return call_claude(user_prompt, schema)


def engine_model(engine):
    """Human-readable model name behind a grading engine, for logs and UI."""
    return {
        "claude": os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL),
        "ollama": config.OLLAMA_MODEL,
        "deepseek": config.deepseek_model(),
    }.get(engine, "local")
