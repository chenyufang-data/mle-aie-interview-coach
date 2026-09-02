"""MLE/AIE Interview Coach — entrypoint.

The backend lives in the coach/ package (one module per concern; see
coach/__init__.py for the map). This file parses arguments, wires the
modules together, and runs the HTTP server.

It is also a backwards-compatible facade: the grader scripts and tests do
`import server` and use names like server.build_evaluation_prompt,
server.GRADER, server.KB — the module-level __getattr__ below resolves any
such name against the coach modules at access time, so even globals that
are reassigned at startup (GRADER, MODE, USERS, TIERS_ENABLED) always read
current.
"""

import argparse
import os
from http.server import ThreadingHTTPServer

from coach import (config, grading, http, kb, llm, prompts, sessions,  # noqa: F401
                   stt_dev, users, web)

_FACADE = (config, kb, users, llm, prompts, grading, sessions, web, stt_dev, http)


def __getattr__(name):
    for module in _FACADE:
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(f"module 'server' has no attribute {name!r}")


def parse_args():
    parser = argparse.ArgumentParser(description="MLE/AIE Interview Coach server")
    backend = parser.add_mutually_exclusive_group()
    backend.add_argument(
        "--mock",
        action="store_true",
        help="Free testing mode: no API key, no network, no cost. Questions come from the "
        "course knowledge base and grading uses the trained local ML grader "
        "(grader/model.joblib), falling back to rubric keyword matching.",
    )
    backend.add_argument(
        "--ollama",
        nargs="?",
        const="llama3.2",
        metavar="MODEL",
        help="Use a free local Ollama model instead of the Anthropic API (default: llama3.2). "
        "Requires Ollama running at localhost:11434.",
    )
    voice = parser.add_mutually_exclusive_group()
    voice.add_argument(
        "--voice",
        action="store_true",
        help="Force the live voice-loop WebSocket server (coach/voice) on "
        "VOICE_PORT (default 8765) and fail loudly if it cannot start. "
        "Without either voice flag the loop starts automatically whenever "
        "its optional dependencies (requirements-stt.txt) are installed. "
        "AUDIO_BACKEND picks the audio stack: local (faster-whisper + "
        "Kokoro, default), speaches, deepgram, or elevenlabs; "
        "STT_BACKEND/TTS_BACKEND override each side.",
    )
    voice.add_argument(
        "--no-voice",
        action="store_true",
        help="Skip the voice loop even when its dependencies are available "
        "(fast text-only server).",
    )
    return parser.parse_args()


def voice_availability(host):
    """None when the voice loop can run here, else a short reason.

    The voice stack's heavy dependencies are optional (requirements-stt.txt),
    so a plain install serves HTTP-only and says why; models load lazily per
    session, so availability is only imports plus a free WebSocket port."""
    try:
        import numpy  # noqa: F401
        import onnxruntime  # noqa: F401
        import websockets  # noqa: F401
    except ImportError as exc:
        return (f"missing dependency {getattr(exc, 'name', exc)!r} "
                "(pip install -r requirements-stt.txt)")
    import socket

    from coach.voice import loop as voice_loop
    with socket.socket() as probe:
        try:
            probe.bind((host, voice_loop.VOICE_PORT))
        except OSError:
            return (f"ws port {voice_loop.VOICE_PORT} is already in use "
                    "(another server with voice running?)")
    return None


def main():
    args = parse_args()
    if args.mock:
        config.MODE = "mock"
    elif args.ollama:
        config.MODE = "ollama"
        config.OLLAMA_MODEL = args.ollama

    config.load_env_file()
    kb.load_chunks()
    users.load_users()
    # Claude mode needs the local grader too when tiers are on: it serves
    # free-tier and over-quota requests.
    if config.MODE == "mock" or (config.MODE == "claude" and users.TIERS_ENABLED):
        grading.load_grader()
    # HOST=0.0.0.0 is required inside a container; the localhost default keeps
    # a bare `python server.py` private to this machine.
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    server = ThreadingHTTPServer((host, port), http.InterviewCoachHandler)
    display_host = "127.0.0.1" if host == "0.0.0.0" else host
    print(f"MLE/AIE Interview Coach running at http://{display_host}:{port}")
    for role, info in kb.KB.items():
        print(
            f"Knowledge base [{role}]: {len(info['chunks'])} interview chunks "
            f"from {config.CORPUS_PATHS[role].parent.name}/."
        )
    if config.MODE == "mock":
        brain = (
            f"trained ML grader ({grading.GRADER['model_name']})"
            if grading.GRADER is not None
            else "keyword matching (train one with grader\\train.py)"
        )
        print(f"Backend: MOCK mode (free) - KB questions, grading via {brain}.")
    elif config.MODE == "ollama":
        print(f"Backend: Ollama local model '{config.OLLAMA_MODEL}' (free) at localhost:11434.")
    else:
        print(f"Backend: Anthropic API, model {os.environ.get('ANTHROPIC_MODEL', config.DEFAULT_MODEL)}.")
        if users.TIERS_ENABLED:
            paid = sum(1 for entry in users.USERS.values() if entry.get("tier") == "paid")
            brain = grading.GRADER["model_name"] if grading.GRADER is not None else "keyword matching"
            print(
                f"Tiers: ON - {paid} paid key(s), {config.PAID_DAILY_QUOTA} Claude calls/day each; "
                f"free tier grades with the local model ({brain})."
            )
            print(
                "Paid workhorse: "
                + (f"{config.deepseek_model()} (quota-free; Claude on 'Always Claude')."
                   if config.deepseek_available()
                   else "Claude only (set DEEPSEEK_API_KEY in .env to add the "
                        "measured DeepSeek judge).")
            )
            print(
                "Cascade: "
                + ("ON - clearly-weak paid answers grade locally, no quota spent."
                   if config.PAID_CASCADE and grading.GRADER is not None else "off.")
            )
        else:
            print("Tiers: off (no users.json) - every request grades with Claude.")
    # Voice defaults to ON when the optional stack can run (user decision
    # 2026-09-02): --voice forces it, --no-voice skips it, and a plain
    # public install (no requirements-stt.txt) degrades to HTTP-only with
    # the reason stated here and on the mock page.
    if args.no_voice:
        voice_reason = "disabled with --no-voice"
    else:
        voice_reason = voice_availability(host)
        if voice_reason and args.voice:
            raise SystemExit(f"--voice: cannot start the voice loop - {voice_reason}")
    if voice_reason:
        config.VOICE_DISABLED_REASON = voice_reason
        print(f"Voice loop: off - {voice_reason}.")
    else:
        import threading

        from coach.voice import loop as voice_loop

        config.VOICE_ENABLED = True
        threading.Thread(target=voice_loop.run_in_thread,
                         kwargs={"host": host}, daemon=True).start()
        print(
            f"Voice loop: ws://{display_host}:{voice_loop.VOICE_PORT} "
            f"(stt={voice_loop.stt_backend()}, tts={voice_loop.tts_backend()})"
        )
        if os.environ.get("ELEVENLABS_API_KEY"):
            # Level 1 sidecar in the same process, so /api/mock/level1/start
            # can register the plan with it (module state is shared).
            import asyncio

            from coach.voice import sidecar

            threading.Thread(
                target=lambda: asyncio.run(sidecar.serve(sidecar.SIDECAR_PORT)),
                daemon=True,
            ).start()
            print(
                f"Speech Engine sidecar: ws://0.0.0.0:{sidecar.SIDECAR_PORT} "
                "(needs a public tunnel as ws_url; see coach/voice/sidecar.py)"
            )
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
