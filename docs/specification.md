# Project Specification — MLE/AIE Interview Coach

A complete map of the codebase: what every file does, how the pieces connect,
and where each measured number comes from. Companion documents:
[backend.md](backend.md) and [frontend.md](frontend.md) hold the strict
frontend/backend contracts; [mock_interview_plan.md](mock_interview_plan.md)
is the lab notebook behind the mock-interview subsystem;
[stt_evaluation.md](stt_evaluation.md) is the generated Phase 0 STT report;
the top-level [README](../README.md) is the user-facing guide.

## 1. What the system is

A web app for practicing Machine Learning Engineer (MLE) and AI Engineer (AIE)
interviews, with two surfaces:

- **Question practice** — asks realistic questions from curated banks, times
  the answer, grades it against a per-question rubric, and returns structured
  feedback (score, four subscores, strengths, gaps, a stronger sample answer,
  next steps, and a follow-up question). Questions can be bookmarked and
  replayed later.
- **AI mock interview** — a full multi-phase interview run by an LLM
  interviewer persona built from the user's resume and a job description
  (pasted, uploaded, or defaulted). Typed or **live voice**: streaming STT,
  patient endpointing, barge-in, streamed TTS. Ends with a three-block report
  (computed communication metrics, per-key-point rubric verdicts, LLM
  scorecard).

Three grading engines serve different users and cost points:

| Engine | Model | Marginal cost | Who gets it |
| --- | --- | --- | --- |
| Claude (teacher) | `claude-opus-4-8` | cents-to-dimes/answer | "Always Claude" requests, metered by daily quota |
| DeepSeek workhorse | `deepseek-v4-flash` | ~$0.0005/answer | paid tier default, quota-free |
| Distilled local grader | sklearn, `grader/model.joblib` | zero, offline, ms | free tier, cascade, all degradation paths |

The project doubles as an end-to-end ML case study: every component that
ships carries a held-out metric, and designs that failed their metrics are
documented as deliberate negative results — a "route confident-good answers
locally" cascade rule, a stacked scorer, and a full STT/TTS vendor swap
(Deepgram) that was built, harnessed, measured, and rejected.

## 2. Repository layout (110 tracked files)

```
mle-aie-interview-coach/
├── server.py               (205 lines)   entrypoint + backwards-compat facade
├── coach/                  (31 files)    the backend package (see §3, §6)
├── retrieval.py            (127)         BM25 retrieval over question banks
├── resume_parser.py        (173)         PDF/.docx/.txt → plain text (CLI + library)
├── requirements.txt                      anthropic + scikit-learn (core install)
├── requirements-stt.txt                  optional voice stack (whisper, kokoro, VAD…)
├── example_resume.txt                    fictional resume so anyone can try the mock
├── README.md                             user guide + measured results
├── users.sample.json                     access-key template for freemium tiers
├── .env.sample                           documented environment variables
├── docker-compose.yml                    two-service deployment
├── docker/                 (3 files)     backend.Dockerfile, frontend.Dockerfile, nginx.conf
├── docs/                   (7 files)     contracts, this spec, mock plan, STT report, screenshots
├── data/                   (gitignored)  personal + runtime data; only data/README.md tracked
├── public/                 (10 files)    dependency-free frontend
├── grader/                 (28 files)    distillation subsystem + experiment harnesses (§5)
├── tests/                  (8 files)     regression suites + Playwright e2e (§8)
├── tools/                  (3 files)     level1_up, strip_chunks, backup_private
├── rag_ml/                               MLE bank: 191 chunks over 15 modules
├── rag_ai/                               AIE bank: 91 chunks over 6 modules
└── rag_exp/                              "Real Qs" bank: 57 chunks from real interview reports
```

The public banks are **stripped**: each chunk carries only `id`, the
`interview` object (question, model answer, key points, common mistakes,
followups) and retrieval `metadata` (module, topic, tags, difficulty).
The complete banks — with course-derived lesson text and source references —
live in a private repository; `tools/strip_chunks.py` produces the public
versions. `rag_exp/all_chunks.jsonl` itself is generated locally by
`grader/ingest_questions.py` and only its README is tracked here.

Runtime-only files (gitignored, never committed): `.env` (API keys),
`users.json` (real access keys), and everything under `data/` — resume text,
gathered interview experiences, `data/sessions/*.jsonl` and
`data/mock_sessions/` (collected answers), `data/usage.json` (quota state),
`data/plan_cache/`, recordings, downloaded voice models. `data/README.md`
documents the layout; in Docker the same tree is the `coach-data` volume at
`/data`.

## 3. Backend — the `coach/` package

Python 3.10+ standard library only for serving (`http.server.
ThreadingHTTPServer`, thread per request — no Flask/FastAPI, a deliberate
zero-dependency choice), plus the `anthropic` SDK for Claude and plain
`urllib` for DeepSeek/Ollama. The voice stack (`requirements-stt.txt`) is
optional; without it the server runs HTTP-only and says why.

One module per concern:

| Module | Owns |
| --- | --- |
| `coach/config.py` | `.env` loading, paths, constants, runtime mode flags |
| `coach/kb.py` | question-bank corpora, one `Retriever` per track, chunk-id validation |
| `coach/users.py` | freemium tiers, access keys, daily Claude quota (fail-closed) |
| `coach/llm.py` | the three grading engines: Claude, DeepSeek, Ollama |
| `coach/prompts.py` | system prompt, question/evaluation prompts, output schemas |
| `coach/grading.py` | engine routing, smart cascade, the local distilled grader |
| `coach/sessions.py` | practice-session logging (gold pairs vs unlabeled free tier) |
| `coach/web.py` | tiny JSON request/response helpers |
| `coach/stt_dev.py` | Phase 0 recording routes (localhost only) |
| `coach/http.py` | the HTTP handler wiring it all together |
| `coach/mock/` | the mock-interview subsystem (§6) |
| `coach/voice/` | the live voice loop (§6) |

`server.py` parses arguments, wires the modules, and runs the server. It is
also a backwards-compatible facade: grader scripts and tests still do
`import server` and read names like `server.GRADER` — a module-level
`__getattr__` resolves them against the coach modules **at access time**, so
globals reassigned at startup always read current. The same rule holds
inside the package: cross-module readers access mutable state as module
attributes (`config.MODE`), never `from ... import MODE`.

Key behaviors:

- **Users and tiers** — `load_users()` is fail-closed: a corrupt `users.json`
  keeps tiers ON with zero paid keys, never silently off. The file is
  hot-reloaded on mtime change, so revoking a key needs no restart. Quota
  state in `usage.json` is keyed by a SHA-256 digest of the access key —
  no raw keys at rest.
- **Routing** — `grading_route(user, force_llm)` returns `(engine, reason)`,
  the single decision point for the whole freemium design:
  1. `--mock` → local; `--ollama` → ollama; no `users.json` → claude
     (single-user mode, unchanged behavior).
  2. free user → local distilled grader.
  3. paid + default → **deepseek** (quota-free).
  4. paid + `force_llm` ("Always Claude") → **claude**, takes quota;
     exhausted quota → deepseek; no DeepSeek key → local.
  5. Any DeepSeek failure → local grader (never silently Claude, so
     Anthropic spend is strictly quota-bound).
- **Smart cascade** — before any LLM call, paid rubric-question answers the
  student grades reliably are answered locally: rule `predicted <= 2.5 AND
  kp_frac_hit <= 0.25`, chosen by replaying gold rows
  (`grader/cascade_analysis.py`): 12% coverage at 100% within-±1 agreement.
  The intuitive "route confident-good answers" rule measured 47% and was
  rejected.
- **Engines** — `call_claude` (structured outputs via `output_config`
  json_schema + adaptive thinking), `call_deepseek` (OpenAI-compatible
  endpoint; schema embedded in the prompt because DeepSeek's Anthropic-compat
  endpoint silently ignores `output_config`; one parse-retry because ~3% of
  `json_object` responses arrive malformed), `call_ollama` (local model),
  `mock_evaluation` (the distilled grader, §5).
- **Handlers** — `GET /api/meta`, `POST /api/question`, `POST /api/evaluate`,
  the `/api/mock/*` family (§6), plus path-traversal-safe static serving of
  `public/` with `Cache-Control: no-cache` (stdlib servers send no cache
  headers, and browsers heuristically cache stale HTML/JS without it).
- **Session logging** — `log_real_session` (LLM-graded rows; `graded_by`
  field lets `evaluate_on_real.py` keep only Claude rows as gold pairs) and
  `log_free_session` (free-tier answers stored **unlabeled** — the student's
  own score is never a training label, avoiding self-distillation). Both
  honor the per-key `"log": false` opt-out; the UI discloses collection.
  Mock sessions log only with an explicit opt-in checkbox.

## 4. Retrieval — `retrieval.py`

Pure-Python BM25 (k1=1.5, b=0.75) over short per-question documents
(question + key points + tags), not raw lesson text, across three tracks:
MLE (`rag_ml`, 191 chunks), AIE (`rag_ai`, 91), and the optional "Real Qs"
track (`rag_exp`, 57 chunks distilled from real interview reports — absent
in a fresh clone; `coach/kb.py` warn-skips it). Query-time filters: module
and difficulty-by-level metadata, plus a session `exclude` list so questions
do not repeat. One of the top-5 hits is sampled at random for variety.
Deliberately not embedding-based — measured at 100% Recall@5 / 0.91 MRR on
23 curated cases (`tests/test_retrieval.py`), which does not leave room for
an embedding index to pay its complexity.

## 5. The grader subsystem — `grader/`

The LLM-distillation pipeline, in dependency order:

| File | Role |
| --- | --- |
| `generate_answers.py` | Manufactures ~3,900 synthetic answers from the corpora across quality tiers (reference answers, paraphrases, key-point recalls, fragments, vague filler, question echoes, confidently-wrong "mistake" answers), with style corruption at every level so messy writing isn't mistaken for low quality. Seed-deterministic: regenerating preserves `row_id`s so gold labels re-join. |
| `features.py` | `FeatureExtractor` → 16 answer-level features (`FEATURE_NAMES`): idf-weighted stemmed-token overlap and char n-gram cosine per key point, mistake similarity, question echo, length/structure stats. Deliberately **no** whole-answer similarity to the model answer — that feature taught "quoting the reference = good" and tanked real paraphrases. Also 10 per-key-point features (`KP_FEATURE_NAMES`) for the verdict classifier. |
| `label_teacher.py` | Gold labels: grades sampled synthetic answers through the app's exact evaluation prompt with Claude. 598 rows labeled (~$21 total, across three authorized runs). Dry-run cost estimate; nothing sent without `--confirm`; resumable. |
| `label_keypoints.py` | Dense supervision: one teacher call returns a hit/partial/miss verdict for every rubric key point of a row (~3,000 verdicts for ~$8). `--consistency N` regrades rows to measure label stability: 95% exact, zero hard hit↔miss flips. |
| `train.py` | Trains everything and writes `model.joblib`: (1) overall scorer — grouped split **by chunk** so evaluation is always on unseen questions; teacher labels override construction labels at 3× weight. (2) Four subscore regressors. (3) Per-key-point verdict classifier. (4) A stacked scorer — gated: ships only if gold metrics improve (they didn't: QWK 0.770 vs 0.784, so it doesn't). |
| `model.joblib` | The artifact: extractor + overall model + subscore models + kp classifier + all metrics. Loaded for every local grading path, including the mock report's verdict block. |
| `cascade_analysis.py` | Replays the training split and measures candidate cascade rules on gold rows — the evidence behind the shipped thresholds. |
| `judge_agreement.py` | Re-grades the 121 held-out gold rows with candidate judge models: DeepSeek Flash 0.59 MAE / 94% within-±1 / QWK 0.93 vs the teacher; regrade consistency 57% exact (vs Claude's 95%) — hence "runtime judge yes, teacher no". |
| `evaluate_on_real.py` | The real-distribution check: compares local predictions against Claude scores on actual logged practice answers as they accumulate. |
| `ingest_questions.py` | Builds `rag_exp/` from hand-collected interview experiences (gitignored spreadsheets/pastes under `data/interview_exp/`): parse → normalize → dedupe (lexical containment) → intent-merge HR-screen phrasings → classify by round → free dry-run preview with cost estimate → `--generate --confirm` teacher run writing rubric chunks. Idempotent (existing ids skip); the teacher prompt strips person/employer names. 57 chunks for ≈$1.77. |
| `stt_testset.py`, `stt_text.py`, `stt_lexicon.json`, `stt_sentences.jsonl` | Phase 0 STT experiment: test-set builder, pure-text metrics layer (normalization, WER, term error rate over a 339-term lexicon, keyterm-selection policy), the committed lexicon and 88-item test set. |
| `stt_eval.py`, `stt_eval_results.json`, `make_failure_rates.py`, `stt_failure_rates.json` | Runs STT conditions over the recordings, measures WER/TER **and downstream grade damage**, renders `docs/stt_evaluation.md`; per-term failure rates feed the runtime keyterm policy. |
| `loop_eval.py`, `loop_eval_results.json` | Phase 2 harness: drives the live voice loop end to end with the 20 real human answer recordings as the scripted "candidate" and measures live TER, first-audio latency p50/p95, cut-offs, and downstream damage per audio backend. The evidence behind the Deepgram rejection and the barge-in number. |
| `report_consistency.py`, `report_consistency_results.json` | Regrades the same five mock sessions twice per engine: Claude 73.3% exact dimension scores / MAE 0.3 vs Flash 43.3% / MAE 0.8 — why Claude writes the mock report. |
| `cache_check.py`, `cache_check_results.json` | Measures the prompt-cache design on real interviewer turns: Claude reads ~1,123 tokens from cache and pays for only the ~374-token fresh tail (≈ −75% turn input cost); DeepSeek's cache builds asynchronously. |
| `dataset.jsonl`, `labels_teacher.jsonl`, `labels_keypoints.jsonl`, `judge_agreement_results.jsonl` | Committed data: synthetic rows, 598 gold overall+subscore labels, 623 per-key-point verdict records, judge experiment results. |

Headline gold-set numbers (121 held-out Claude-labeled rows, questions never
seen in training): distilled student **MAE 1.11 / 70% within-±1 / QWK 0.784**
vs keyword baseline 2.05 / 45% / 0.574. Per-key-point classifier: **78%
3-class accuracy / hit-F1 0.87** vs 70% / 0.76 for the lexical threshold.

## 6. The mock interview — `coach/mock/` and `coach/voice/`

Design history and every measurement live in
[mock_interview_plan.md](mock_interview_plan.md); this is the shipped shape.

**Session flow** (`coach/mock/routes.py`, all under `/api/mock/`):
`parse_file` (resume/JD upload → `resume_parser.py`) → `roles` (LLM proposes
role profiles from resume + JD; defaults from `templates.py` when no JD) →
`start` (builds the hidden interview plan) → `turn` (the interview) →
`report`. Voice sessions add `keyterms`, `transcribe`, and the Level 1
endpoints. The free tier gets a clear 403 — a mock needs a real LLM; in
`--mock` mode an offline fake engine (`engine.py`) runs the whole flow for
tests.

- `planning.py` — one code path whether the JD was pasted or defaulted:
  JD → role profile → persona + probe themes. Each probe target is matched
  against the question banks by BM25 and a strong match attaches that
  chunk's rubric, so the report can score against curated key points.
  `rag_exp` chunks are round-filtered (`MOCK_ROUNDS`: technical, experience,
  system_design) — behavioral/coding questions never ground a tech-round
  probe.
- `plan_cache.py` — the two setup LLM calls are deterministic functions of
  their inputs, so results are cached under a content hash; repeat practice
  with the same resume costs no setup calls.
- `turns.py` — the phase machine: the sequence is enforced in code, the
  model chooses content *within* a phase, never the phase itself. The turn
  protocol puts spoken text first, then a JSON trailer, so the voice loop
  can stream speech before the structured part arrives.
- `report.py` — three visibly separate blocks: computed communication
  metrics (`metrics.py`, deterministic, never modelled), per-key-point
  verdicts from the distilled classifier on rubric-grounded probes, and the
  LLM scorecard. Claude writes the scorecard because consistency was
  measured, not assumed (73.3% vs Flash's 43.3% exact on identical
  regrades).
- `transcription.py` — two-transcript accounting: the *live* text the
  interviewer acted on, and the *final* text (Scribe batch + full-lexicon
  keyterms — the best Phase 0 condition) that the report grades.
- Prompt shape — frozen persona system + append-only history, so provider
  prompt caching applies; measured in `grader/cache_check.py` at ≈ −75%
  Claude turn input cost.

**Live voice** (`coach/voice/`, WebSocket on `VOICE_PORT` 8765): `loop.py`
runs one session per socket beside the HTTP server. `stt.py` and `tts.py`
put four backends each behind one interface — `AUDIO_BACKEND` picks
local (faster-whisper + Kokoro, default), speaches, deepgram, or
elevenlabs; `STT_BACKEND`/`TTS_BACKEND` override each side. `vad.py` is
Silero VAD plus a patient endpointing state machine; `chunker.py` streams
speakable sentences out of LLM deltas ahead of the JSON trailer;
`keyterms.py` selects ≤50 per-session STT keyterms scored by measured
per-term failure rates. `sidecar.py` + `tools/level1_up.py` are the
optional Level 1 stack: ElevenLabs Speech Engine runs the hosted audio loop
and exchanges text with this server over a tunnel.

Voice starts **by default** whenever its optional dependencies import and
the port is free (`server.py voice_availability`); `--voice` forces it and
fails loudly, `--no-voice` skips it, and a plain install degrades to
HTTP-only with the reason surfaced on the mock page.

Measured (harness: `grader/loop_eval.py`, 20 real recordings as the
candidate): shipped local config live TER 5.6–9.3% lenient, first-audio
p50 0.77–0.83 s / p95 ≤ 1.74 s (bar: 2.0 s); best cloud STT (Scribe
Realtime) live TER 3.7% lenient, first-audio p50 0.48 / p95 0.63 s;
barge-in interrupts mid-question in 0.32 s. The Deepgram stack
(Nova-3 + Aura-2) was fully built and measured through the same harness:
TER 14.8% lenient, first-audio p95 4.48 s — fails both bars, kept only as
the keyless-cloud fallback.

## 7. Frontend — `public/` (zero dependencies, no build step)

- `index.html` — setup page: track (MLE / AIE / Real Qs), level, topic
  (module lists fetched from `/api/meta`, never hard-coded), optional focus
  text, a collapsed "Saved questions" section (bookmarks), and a banner to
  the mock interview.
- `interview.html` — question card, elapsed timer with pause, word count,
  answer box, "Always Claude" checkbox (`force_llm`), evaluation rendering
  with a "Graded by <model>" chip, a bookmark star, follow-up / new-question
  flow, data-collection disclosure note.
- `mock.html` / `mock.js` — mock setup (resume/JD paste or upload, round,
  voice mode with the disabled-reason surfaced) and the interview itself:
  chat transcript, mic capture via `mock-audio-worklet.js`, the voice
  WebSocket, and the final report.
- `account.js` — the login-lite account chip: the access key lives in
  `localStorage`, is attached per-request via `Account.headers()`, and is
  validated server-side; the chip hides entirely when tiers are off.
- `app.js` — page init, sessionStorage state, bookmarks (localStorage,
  capped), rendering. The frontend never grades, never ranks, never stores
  a key server-side.
- `styles.css` — the entire visual design (design tokens, light theme,
  Google Fonts links with system fallbacks — the only external resource).
- `stt_record.html` / `stt_record.js` — Phase 0 recording tool, localhost
  only.

## 8. Tests and evaluation harnesses

CI (`.github/workflows/tests.yml`) runs six suites on every push:

- `tests/test_retrieval.py` + `retrieval_cases.json` — 23 curated queries,
  fails below 90% Recall@5 (currently 100%, MRR 0.91).
- `tests/test_grader.py` — artifact sanity: ordering invariants
  (reference ≫ vague ≫ off-topic) on probe answers per corpus.
- `tests/test_stt_text.py` — the STT text layer: normalization, WER/TER,
  keyterm policy (pure Python, no audio).
- `tests/test_mock.py` — phase machine, turn protocol, metrics, rubric
  round-filter, and the full offline flow on the fake engine
  (roles → plan → turns → report).
- `tests/test_ingest.py` — rag_exp parsing/dedupe/classify on synthetic
  fixtures (no private data needed).
- `tests/test_voice.py` — deterministic voice parts: endpointer state
  machine, sentence chunker, keyterm policy, two-transcript report block.

Local-only: `tests/e2e_smoke.py` drives a real Chromium via Playwright
(installed in `.venv`, deliberately not in requirements) through
start → answer → grade → bookmark → replay; it self-skips when Playwright
is absent. It exists because two frontend bugs survived code reading and
fell immediately to a real browser. The `grader/` analysis scripts (§5) are
the deeper evaluation layer.

## 9. Deployment

- **Local dev**: `python server.py` (frontend + API on one port; voice
  auto-starts when the optional stack is installed); `--mock` (fully
  offline) and `--ollama` (local LLM) modes cost nothing.
- **Docker**: two services mirroring the documented contract split —
  nginx serving `public/` and proxying `/api/` (300s timeouts,
  `Cache-Control: no-cache`), and a Python backend image containing exactly
  the runtime scope (no training scripts, no datasets, no `.env`; the key is
  injected at runtime via compose `env_file`). Named volume `coach-data`
  persists session logs and quota state; `users.json` is bind-mounted
  read-only.
- **EC2**: ran on a t3.micro (backend peaks ~300–400 MB); currently parked.
  Operational notes learned the hard way: add swap before `docker build` on
  1 GB RAM; AL2023 needs compose/buildx plugins installed manually; a silent
  security-group drop looks like a 21-second client timeout.
- **Public/private split**: this repository is public. Personal data never
  enters it — `data/` is gitignored, the public banks are stripped of course
  lesson text (`tools/strip_chunks.py`), and the generated `rag_exp` bank
  plus its raw paste sources are mirrored to a private sibling repository by
  `tools/backup_private.py` (raw interview spreadsheets stay local-only,
  they name real people).

## 10. Design decisions worth knowing (with their evidence)

1. **Fail-closed tiers**: a parse failure in `users.json` once silently
   disabled tiers and let test requests hit paid Claude. The fix inverted the
   default: file exists → tiers ON no matter what. Unit-tested with BOM and
   corrupt-file fixtures.
2. **Measured cascade over intuitive cascade**: the "obvious" rule (route
   high-confidence good answers locally) was refuted by replay (47%
   within-±1); only the clearly-weak side shipped (100%).
3. **DeepSeek as workhorse, Claude as teacher**: agreement qualified Flash
   for runtime grading (94% within-±1 at ~1/200th cost), but 53–57% exact
   regrade consistency disqualified it from writing gold labels.
4. **Dense supervision improves explanations, not the score**: per-key-point
   verdicts lifted hit-F1 0.76 → 0.87, but the stacked scorer failed its gate
   — the score keeps the plain model. Both outcomes are recorded.
5. **No self-distillation**: free-tier answers are logged unlabeled; the
   student's own predictions are never training labels.
6. **Vendor swaps are experiments**: the Deepgram voice stack was built to
   completion, run through the same harness as the incumbents, and rejected
   on its numbers (TER 14.8% vs 3.7%, first-audio p95 4.48 s vs 0.63 s). It
   survives as the keyless-cloud fallback, not the default.
7. **The report engine was chosen by a consistency experiment**, not model
   reputation: regrading identical mock sessions, Claude reproduced 73.3% of
   its own dimension scores exactly (MAE 0.3) vs Flash's 43.3% (MAE 0.8) —
   so Claude writes the scorecard while Flash runs the turns.
8. **Prompt caching was measured, not assumed**: `grader/cache_check.py`
   confirmed the frozen-system + append-only-history shape actually hits the
   cache (≈ −75% Claude turn input cost) and caught a latent
   temperature-mismatch bug that had been silently breaking cache reuse.
