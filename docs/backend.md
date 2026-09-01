# Backend requirements

Scope: `server.py` + the `coach/` package, `retrieval.py`, the `grader/`
package, and the two question
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

## Code layout

`server.py` is the entrypoint (arg parsing, wiring, `main()`) and a
backwards-compatible facade — scripts may `import server` and use any
backend name. The implementation lives in `coach/`, one module per concern:
`config` (env, paths, mode flags), `kb` (corpora + chunk selection),
`users` (tiers/quota, fail-closed), `llm` (Claude/DeepSeek/Ollama calls,
blocking and streamed), `prompts` (system prompt, builders, schemas),
`grading` (routing, cascade, local distilled grader), `sessions` (logging),
`web` (JSON helpers), `stt_dev` (localhost-only recording routes), `http`
(the handler). `coach/mock/` is the mock interview; `coach/voice/` is the
live voice loop (below).
Convention: startup-reassigned globals (`config.MODE`, `users.TIERS_ENABLED`,
`grading.GRADER`, …) are read as module attributes, never `from`-imported.

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

### Mock interview: `/api/mock/*` (coach/mock/)

Stateless like the rest of the API: the client holds `{plan, transcript,
settings}` and sends them per request. Engine selection reuses
`grading_route()` — paid keys get DeepSeek Flash turns (quota-free, "Always
Claude" via `force_llm`), the free tier gets 403 (a mock needs a real LLM),
and `--mock` mode serves a deterministic offline demo engine (also what
`tests/test_mock.py` and CI run).

- `GET /api/mock/templates` → default JD templates (§7c), styles, lengths.
- `POST /api/mock/roles` `{resume, jd_text?}` → `{profile, roles[]}` (§7a).
- `POST /api/mock/start` `{resume, jd_text?, role, project?, settings}` →
  `{plan, jd_text, jd_is_default, turn, total_turns}`. The hidden plan
  carries probe targets with on-the-fly rubrics; targets that BM25-match a
  bank chunk (score ≥ 10) also carry `chunk_id` + `bank_key_points` —
  rubric-grounded probes. Note: the plan travels to the client, so a curious
  user can read it in devtools; acceptable for a personal practice tool.
- `POST /api/mock/turn` `{plan, role, transcript}` → the next interviewer
  message. The phase machine (warm-up → walkthrough → deep dive →
  behavioral → closing) is enforced in code; the model only chooses content
  within the phase, with thinking disabled (measured ~0.8 s vs ~1.4 s first
  token). Turn protocol: spoken text, then a `---` line and a JSON trailer
  `{probe_id, rationale}` that is stripped server-side.
- `POST /api/mock/report` `{plan, role, transcript, report_engine?}` →
  `{assessment, metrics, kp_verdicts, engine}`: the §8a scorecard (six
  anchored dimensions, per-turn feedback, red flags that quote, hiring
  call), deterministic communication metrics, and distilled-classifier
  hit/partial/miss verdicts for rubric-grounded probes. Report engine
  defaults to Claude when a key is present (regrade consistency), else the
  turn engine.

- `GET /api/mock/voice` → `{enabled, ws_port, audio_backend, final_stt}`:
  whether the live loop WebSocket is up (`--voice`), which audio stack it
  runs, and which final-transcript engine this deployment offers
  (`scribe_batch_kt` with an ElevenLabs key, `nova3_batch_kt` with a
  Deepgram key, `whisper_local` with faster-whisper installed, else null;
  `FINAL_STT=scribe|deepgram|whisper` forces one).
- `POST /api/mock/keyterms` `{resume?, role?, project?}` → `{keyterms}`:
  the ≤ 50 realtime STT keyterms for this session, chosen deterministically
  by the measured §7a policy (failure rates from
  `grader/stt_failure_rates.json` + presence in the session's own material
  + rarity; the policy measured 9.7% vs naive-50's 11.7% lenient TER).
- `POST /api/mock/transcribe` `{audio_base64, mime}` → `{text, engine,
  seconds}`: final-transcript re-transcription of one recorded answer clip
  (≤ 25 MB) — Scribe v2 batch + the full lexicon when `ELEVENLABS_API_KEY`
  is set (the Phase 0 winner: 3.0% lenient TER), Deepgram Nova-3 batch +
  the top-100 keyterm policy with a Deepgram key (unmeasured on the human
  set, and the report's `engine` field says which ran), local Whisper
  otherwise.
  The client attaches the result as `answer_final` on the transcript entry;
  the report grades it (normalized both sides for the kp classifier) and
  reports live-vs-final drift as the `transcription` block.

### Live voice loop: `coach/voice/` (`python server.py --voice`)

One WebSocket per interview session on `VOICE_PORT` (default 8765), beside
the HTTP server. REST still owns setup, keyterms and the report; the
socket runs only the audio loop, holding `{plan, role, transcript}` for
its own lifetime and mirroring every entry to the client. Modules: `vad`
(Silero VAD on onnxruntime — `data/models/silero_vad.onnx`, auto-downloaded
— plus the patient endpointer: 1.2 s end-of-turn silence, 30 s turn
timeout, barge-in via speech-during-speaking), `chunker` (trailer-safe
sentence chunking of the streamed LLM turn), `stt` / `tts` (backends),
`keyterms` (the §7a policy), `final_transcript`, `loop` (the session
server; the protocol is documented in its docstring).

`AUDIO_BACKEND` selects the audio stack — all four drive the same loop;
`STT_BACKEND` / `TTS_BACKEND` override each side for mix-and-match (e.g.
local Whisper with a cloud voice):

| Backend | STT | TTS | Needs |
| --- | --- | --- | --- |
| `local` (default) | faster-whisper `large-v3-turbo` in-process (GPU), keyterms via `initial_prompt` | Kokoro-82M via kokoro-onnx (CPU, RTF ≈ 0.3–0.45) | `requirements-stt.txt` + model files in `data/models/` |
| `speaches` | the same models inside a [Speaches](https://speaches.ai) container | ↑ | Docker; `SPEACHES_URL` (default `http://127.0.0.1:8969/v1`) |
| `deepgram` | Nova-3 streaming, one session-long connection, finals + open-interim assembly per turn, policy-50 via keyterm prompting | Aura-2 (`linear16`, `DEEPGRAM_TTS_VOICE`) | `DEEPGRAM_API_KEY` — measured: TER 14.8% lenient / WER 12%, STT finalize ~0.00 s, first-audio p50 1.35 / p95 4.48 s, damage 6/18 — worse than the Scribe row on this lexicon (ablated: 16.7% TER without keyterms, so prompting helps and stays on); the cloud option when no ElevenLabs key is present |
| `elevenlabs` | Scribe v2 Realtime, one session-long connection, MANUAL commits, policy-50 keyterms | Flash v2.5 (`pcm_24000`) | `ELEVENLABS_API_KEY` — the measured cloud row (TER 3.7%, first-audio p95 0.63 s) |

The turn LLM streams (`coach/llm.py call_chat_stream`, thinking off) →
sentence chunker → per-sentence TTS, so first audio does not wait for the
full turn. Interruption truncates the recorded question to the sentences
actually played (`cut: true`) — observability the hosted Speech Engine
loop cannot offer (its upstream protocol is text-only; verified against
SDK 2.65 types, plan §10).

### Developer routes: `/api/stt/*` (localhost only)

Used by `public/stt_record.html` to record the Phase 0 STT test set
(`grader/stt_sentences.jsonl`). They answer only loopback clients — behind
nginx in Docker every request arrives from the proxy and gets 403 — because
they write under `data/stt_audio/human/` (override the root with
`STT_AUDIO_DIR`).

- `GET /api/stt/items` → `{ items, recorded, dir }`: the sentence set plus the
  manifest of takes already saved.
- `POST /api/stt/record` with `{ id, mime, audio_base64, duration_ms,
  web_speech: { text, supported, error }, user_agent }` → saves
  `audio/<id>.<ext>` and updates `manifest.json`; re-recording replaces the
  take and bumps its `take` counter.
- `GET /api/stt/audio/<id>` → the saved take, for playback.
