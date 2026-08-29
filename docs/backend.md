# Backend requirements

Scope: `server.py`, `retrieval.py`, the `grader/` package, and the two question
banks (`rag_ml/`, `rag_ai/`). The backend owns all data, retrieval, AI calls, and
grading. It knows nothing about presentation.

## Runtime requirements

- Python 3.10+ with the packages in `requirements.txt`
  (`anthropic`, `scikit-learn`). The standard-library `http.server` is the web
  server — no web framework.
- Per grading mode:
  - **Default (Claude)**: `.env` in the project root with `ANTHROPIC_API_KEY`.
    Optional: `ANTHROPIC_MODEL` (default `claude-opus-4-8`), `PORT` (default 8000).
  - **`--ollama [model]`**: a local [Ollama](https://ollama.com) install with the
    chosen model pulled. No API key.
  - **`--mock`**: nothing — fully offline. Uses `grader/model.joblib` (the trained
    local grader) if present, keyword rubric matching otherwise.
- Environment overrides: `PORT` (default `8000`), `HOST` (default `127.0.0.1`;
  set to `0.0.0.0` inside a container), `REAL_SESSIONS_PATH` (default
  `data/sessions/real_sessions.jsonl`; point it at a mounted volume in Docker so
  practice history survives rebuilds), `USERS_PATH` / `USAGE_PATH` /
  `PAID_DAILY_QUOTA` (freemium tiers, below).
- Freemium tiers (Claude mode only): if `users.json` exists (copy
  `users.sample.json`; gitignored — it holds real access keys), requests are
  routed per user. A request whose `X-Access-Key` header matches a `"paid"`
  key gets real AI grading; all other requests get the distilled local
  grader and knowledge-base questions — the free tier. Without `users.json`,
  tiers are off and every request grades with Claude (single-user setup).
  `--mock`/`--ollama` ignore tiers.
- Paid-tier judge routing: with `DEEPSEEK_API_KEY` in `.env` (optional
  `DEEPSEEK_MODEL`, default `deepseek-v4-flash`), paid evaluations and
  question generation default to DeepSeek Flash, **quota-free** — the judge
  is measured, not assumed (`grader/judge_agreement.py`: 94% within-±1,
  QWK 0.93 vs the Claude teacher on the 121 held-out gold rows). Claude
  serves `"force_llm": true` ("Always Claude") requests, capped at
  `PAID_DAILY_QUOTA` calls per day (state in `data/usage.json`); an
  exhausted quota degrades to DeepSeek, and a failed DeepSeek call (two
  malformed responses or network error) degrades to the local grader —
  never silently to Claude, so Anthropic spend stays strictly quota-bound.
  Without `DEEPSEEK_API_KEY`, paid routing is all-Claude under quota with
  local-grader degradation, exactly as before.
- Smart cascade (`PAID_CASCADE`, default on): paid evaluations of rubric
  questions first ask the distilled model; when it is confident by the
  measured rule (predicted <= 2.5 AND `kp_frac_hit` <= 0.25 — see
  `grader/cascade_analysis.py`, 100% within-±1 on gold rows), the local
  result is served and **no quota is taken**. `"force_llm": true` in the
  request body opts out. Follow-ups and non-KB questions never cascade.
- Docker: `docker/backend.Dockerfile` packages exactly this scope — server,
  retrieval, grader artifact, and both banks. Training scripts, datasets, gold
  labels, and `.env` never enter the image; the key is injected at runtime via
  compose `env_file`. (`--ollama` is not wired for Docker: it expects Ollama on
  localhost.)

## Responsibilities (must)

- Serve the static frontend from `public/` (path-traversal-safe).
- Own the corpora: load both banks at startup, validate ids are unique across
  them, and expose module lists via `GET /api/meta` so the frontend stays
  corpus-agnostic.
- Question selection: metadata filtering (module + difficulty-by-level) plus BM25
  ranking (`retrieval.py`) over the requested track's bank, honouring the
  session's `exclude` list; or Claude-generated questions for general topics.
- Evaluation: grade against the chunk's rubric (model answer, key points, common
  mistakes) with structured outputs; grade follow-ups with the parent question,
  previous answer, and course material as context (rubric as background, not as a
  checklist). In `--mock` mode, predict with the distilled local grader; its
  hit/miss rubric feedback uses a per-key-point classifier distilled from
  teacher hit/partial/miss verdicts when the artifact carries one
  (`grader/label_keypoints.py` + `train.py`), with the 0.35 lexical threshold
  as fallback.
- Log every real (non-mock) graded session to `data/sessions/real_sessions.jsonl` —
  these Claude-vs-local gold pairs feed `grader/evaluate_on_real.py`. Logging
  must never break the response (best-effort, wrapped in try/except).
- Answer collection is disclosed and opt-out-able: the answer page shows a
  notice, and any key in `users.json` can set `"log": false` to be excluded
  from both logs. Free-tier answers (tiered Claude mode only, never `--mock`)
  go to `data/sessions/free_sessions.jsonl` **unlabeled** — the local model's score
  is stored for triage but must never be used as a training label
  (self-distillation would amplify the student's own errors); they become
  training data only after teacher labeling.
- Map failures (bad key, rate limit, network, Ollama down) to JSON errors the
  frontend can display.

## Must not

- Serve corpus-specific knowledge through any channel other than the API.
- Let the API key reach the frontend, logs, or version control (`.env` is
  gitignored; `load_env_file` reads it at startup).

## API contract

All endpoints accept an optional `X-Access-Key` header identifying the user
when tiers are enabled.

### `GET /api/meta`

```json
{ "kb": { "MLE": { "modules": ["..."], "chunks": 191 },
          "AIE": { "modules": ["..."], "chunks": 91 } },
  "user": { "name": "anonymous", "tier": "free", "tiers_enabled": true,
            "paid_quota": 30, "paid_left_today": 0,
            "paid_grader": "deepseek-v4-flash" } }
```

`paid_grader` is the paid tier's quota-free default judge; `"claude"` means no
DeepSeek key is configured and the quota meters every paid LLM call.

`tiers_enabled` is false when `users.json` is absent or the server is not in
Claude mode; the frontend hides its tier badge then.

### `POST /api/question`

Request: `role` (`"MLE"` | `"AIE"`), `level`, `topic`, `focus`,
`source` (`"kb"` to draw from the course bank, otherwise AI-generated),
`exclude` (list of already-served chunk ids).

Response:

```json
{ "question": "...",
  "what_interviewer_is_testing": "...",
  "answer_guidance": ["..."],
  "source": "kb",
  "chunk_id": "06_linear_regression_model__03_..." }
```

`chunk_id` is present only for knowledge-base questions; `source` is `"ai"` for
generated ones. In `--mock` mode, general-topic requests are also answered from
the knowledge base (focus-matched) instead of calling a model.

### `POST /api/evaluate`

Request: `role`, `level`, `topic`, `focus`, `question`, `answer` (required,
non-empty), `timeUsed`, and — when applicable — `chunk_id`, `source`
(`"followup"` for follow-up answers), `parent_question`, `parent_answer`,
`force_llm` (boolean; skip the smart cascade for this request).

Response (structured-output schema, enforced server-side; `graded_by` names
the engine's model — e.g. `deepseek-v4-flash`, the Claude model id, or a
`local ...` label — and drives the "Graded by" chip in the frontend; it also
lands in the session log, where `grader/evaluate_on_real.py` keeps only
Claude-graded rows as gold pairs):

```json
{ "graded_by": "deepseek-v4-flash",
  "overall_score": 7,
  "scores": { "technical_depth": 7, "structure": 6,
              "practical_judgment": 7, "communication": 8 },
  "summary": "...",
  "strengths": ["..."],
  "gaps": ["..."],
  "suggested_answer": "...",
  "next_steps": ["..."],
  "follow_up_question": "..." }
```

Errors are returned as `{ "error": "..." }` with status 400 (missing answer) or
500 (upstream/API failures).
