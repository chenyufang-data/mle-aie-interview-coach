"""The layered backend of the MLE/AIE Interview Coach.

Each module owns one concern; `server.py` at the repo root is the thin
entrypoint and a backwards-compatible facade over all of them (the grader
scripts and tests do `import server` and use its names).

    config    env loading, paths, constants, runtime mode flags
    kb        question-bank corpora, chunk selection (BM25 via retrieval.py)
    users     freemium tiers, access keys, daily Claude quota (fail-closed)
    llm       the three grading engines: Claude, DeepSeek, Ollama
    prompts   system prompt, question/evaluation prompts, output schemas
    grading   engine routing, smart cascade, the local distilled grader
    sessions  practice-session logging (gold pairs vs unlabeled free tier)
    web       tiny JSON request/response helpers
    stt_dev   Phase 0 recording routes (localhost only)
    http      the HTTP handler wiring it all together

A rule the split must keep: `config.MODE`, `config.OLLAMA_MODEL`,
`users.USERS`, `users.TIERS_ENABLED` and `grading.GRADER` are REASSIGNED at
startup, so cross-module readers access them as module attributes
(`config.MODE`), never `from ... import MODE` — that would freeze the
import-time value.
"""
