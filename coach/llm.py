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
        "max_tokens": 8000,
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
            return json.loads(body["choices"][0]["message"]["content"])
        except (KeyError, IndexError, ValueError) as exc:
            last_error = exc
    raise RuntimeError(f"DeepSeek returned malformed JSON twice ({last_error})")


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
