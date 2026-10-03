"""The layered backend of the MLE/AIE Interview Coach.

Each module owns one concern; `server.py` at the repo root is the thin
entrypoint and a backwards-compatible facade over all of them (the experiment
scripts and tests do `import server` and use its names). Everything the
server imports lives here; experiments/ and ingest/ import from it, never the
other way round.

    config    env loading, paths, constants, runtime mode flags
    kb        question-bank corpora (banks/), chunk selection
    retrieval BM25 over the banks (the CI gate and the fallback)
    retrieval_dense  bge-small embeddings + BM25 hybrid; numpy or pgvector vectors
    resume_parser    PDF/.docx/.txt to plain text (/api/mock/parse_file)
    users     freemium tiers, access keys, daily Claude quota (fail-closed)
    store     the state store behind users/sessions/plan cache: file backend
              (default, data/) or Postgres when DATABASE_URL is set
    llm       the three grading engines: Claude, DeepSeek, Ollama
    prompts   system prompt, question/evaluation prompts, output schemas
    grading   engine routing, smart cascade, the local distilled grader
    features  the distilled grader's lexical features (the artifact unpickles them)
    slm       optional fine-tuned small-model grade for the local tier (SLM_URL)
    jev       optional Jev grade + rubric verdicts for the free tier (TYPESAFE_API_KEY)
    cli_engine  local-only LLM engine on the user's own Claude Code / Codex subscription (--cli)
    stt_text  WER, term error rate, lexicon and the keyterm policy (voice, reports)
    sessions  practice-session logging (gold pairs vs unlabeled free tier)
    web       tiny JSON request/response helpers
    stt_dev   Phase 0 recording routes (localhost only)
    http      the HTTP handler wiring it all together
    assets/   runtime inputs: grader_model.joblib, stt_lexicon.json,
              stt_failure_rates.json

A rule the split must keep: `config.MODE`, `config.OLLAMA_MODEL`,
`users.USERS`, `users.TIERS_ENABLED` and `grading.GRADER` are REASSIGNED at
startup, so cross-module readers access them as module attributes
(`config.MODE`), never `from ... import MODE` — that would freeze the
import-time value.
"""
