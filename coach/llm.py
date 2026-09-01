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

# Usage of the most recent chat call, engine-labeled, incl. the cache
# counters (Anthropic: cache_read/creation_input_tokens; DeepSeek:
# prompt_cache_hit/miss_tokens). grader/cache_check.py measures the
# prompt-cache design against these; they are the only ground truth
# that caching is actually working.
LAST_USAGE = {}


def _record_usage(engine, **fields):
    LAST_USAGE.clear()
    LAST_USAGE["engine"] = engine
    LAST_USAGE.update({k: v for k, v in fields.items() if v is not None})


def _cache_marked(system, messages):
    """Claude prompt-cache breakpoints for the interviewer turn shape
    (plan section 5c; placement per the prefix-match contract):

    system (the frozen persona) and the transcript history are the stable,
    append-only prefix; the [Interview control] instruction in the LAST
    message changes every turn and never recurs, so a breakpoint there
    would be a pure write surcharge. Mark the system block and the last
    HISTORY block instead - each turn then reads the previous turn's
    entry and writes only the newly appended Q&A. (Entries below the
    model's ~1k-token minimum silently don't cache; with the persona plus
    a few real answers the prefix crosses it early in a session.)"""
    system_blocks = [{"type": "text", "text": system,
                     "cache_control": {"type": "ephemeral"}}]
    marked = list(messages)
    if len(marked) >= 2:
        last_history = dict(marked[-2])
        content = last_history.get("content")
        if isinstance(content, str):
            last_history["content"] = [{"type": "text", "text": content,
                                        "cache_control": {"type": "ephemeral"}}]
            marked[-2] = last_history
    return system_blocks, marked


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


def call_chat(system, messages, engine, thinking=False, max_tokens=700,
              temperature=0.7, cache=False):
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
        usage = body.get("usage") or {}
        _record_usage("deepseek",
                      prompt_tokens=usage.get("prompt_tokens"),
                      cache_hit=usage.get("prompt_cache_hit_tokens"),
                      cache_miss=usage.get("prompt_cache_miss_tokens"))
        return (body["choices"][0]["message"].get("content") or "").strip()
    if cache:
        system, messages = _cache_marked(system, messages)
    # No temperature: sampling params are rejected (400) on current Claude
    # models - found live by grader/cache_check.py.
    response = get_client().messages.create(
        model=os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL),
        max_tokens=max_tokens if not thinking else max(max_tokens, 4000),
        system=system,
        thinking={"type": "adaptive"} if thinking else {"type": "disabled"},
        messages=messages,
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("The model declined this request.")
    _record_usage("claude",
                  input_tokens=response.usage.input_tokens,
                  cache_read=response.usage.cache_read_input_tokens,
                  cache_write=response.usage.cache_creation_input_tokens)
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


def _sse_events(lines):
    """Parse an iterable of raw SSE lines ("data: {...}") into JSON events.
    Stops at [DONE]; skips keep-alives and malformed lines. Factored out so
    tests can feed canned bytes."""
    for raw in lines:
        line = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            return
        try:
            yield json.loads(payload)
        except ValueError:
            continue


def call_chat_stream(system, messages, engine, thinking=False, max_tokens=700,
                     temperature=0.7, cache=False):
    """call_chat, streamed: yields text deltas as they arrive.

    The voice loop pipes these through the sentence chunker to TTS so the
    first audio never waits for the full turn (plan section 3). Deltas are
    raw - the `---` trailer is stripped downstream (coach/voice/chunker.py).
    DeepSeek: SSE with stream=true (verified 2026-08-30; reasoning_content
    deltas are skipped - with thinking disabled there are none anyway).
    """
    if engine == "deepseek":
        payload = {
            "model": config.deepseek_model(),
            "messages": [{"role": "system", "content": system}] + messages,
            "thinking": {"type": "enabled" if thinking else "disabled"},
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            # The final SSE chunk then carries usage incl. the prompt-cache
            # hit/miss counters (DeepSeek caches stable prefixes on its own).
            "stream_options": {"include_usage": True},
        }
        req = urlrequest.Request(
            DEEPSEEK_URL, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}"},
            method="POST")
        try:
            with urlrequest.urlopen(req, timeout=300) as response:
                for event in _sse_events(response):
                    usage = event.get("usage")
                    if usage:
                        _record_usage("deepseek",
                                      prompt_tokens=usage.get("prompt_tokens"),
                                      cache_hit=usage.get("prompt_cache_hit_tokens"),
                                      cache_miss=usage.get("prompt_cache_miss_tokens"))
                    for choice in event.get("choices") or []:
                        piece = (choice.get("delta") or {}).get("content")
                        if piece:
                            yield piece
        except urlerror.URLError as exc:
            raise RuntimeError(f"Could not reach the DeepSeek API ({exc})") from exc
        return
    if engine == "ollama":
        payload = {"model": config.OLLAMA_MODEL, "stream": True, "think": bool(thinking),
                   "messages": [{"role": "system", "content": system}] + messages}
        req = urlrequest.Request(OLLAMA_URL, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urlrequest.urlopen(req, timeout=300) as response:
                for raw in response:
                    if not raw.strip():
                        continue
                    body = json.loads(raw.decode("utf-8"))
                    piece = body.get("message", {}).get("content")
                    if piece:
                        yield piece
                    if body.get("done"):
                        return
        except urlerror.URLError as exc:
            raise RuntimeError("Could not reach Ollama at localhost:11434.") from exc
        return
    if cache:
        system, messages = _cache_marked(system, messages)
    # No temperature here either (400 on current Claude models).
    with get_client().messages.stream(
        model=os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL),
        max_tokens=max_tokens if not thinking else max(max_tokens, 4000),
        system=system,
        thinking={"type": "adaptive"} if thinking else {"type": "disabled"},
        messages=messages,
    ) as stream:
        for piece in stream.text_stream:
            if piece:
                yield piece
        final = stream.get_final_message()
        _record_usage("claude",
                      input_tokens=final.usage.input_tokens,
                      cache_read=final.usage.cache_read_input_tokens,
                      cache_write=final.usage.cache_creation_input_tokens)
