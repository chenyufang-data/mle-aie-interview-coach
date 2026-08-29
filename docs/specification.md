# Project Specification — MLE/AIE Interview Coach

A complete map of the codebase: what every file does, how the pieces connect,
and where each measured number comes from. Companion documents:
[backend.md](backend.md) and [frontend.md](frontend.md) hold the strict
frontend/backend contracts; [interview_prep.md](interview_prep.md) frames the
project for job interviews; the top-level [README](../README.md) is the
user-facing guide.

## 1. What the system is

A web app for practicing Machine Learning Engineer (MLE) and AI Engineer (AIE)
interviews. It asks realistic questions, times the answer, grades it against a
curated rubric, and returns structured feedback (score, four subscores,
strengths, gaps, a stronger sample answer, next steps, and a follow-up
question).

Three grading engines serve different users and cost points:

| Engine | Model | Marginal cost | Who gets it |
| --- | --- | --- | --- |
| Claude (teacher) | `claude-opus-4-8` | cents-to-dimes/answer | "Always Claude" requests, metered by daily quota |
| DeepSeek workhorse | `deepseek-v4-flash` | ~$0.0005/answer | paid tier default, quota-free |
| Distilled local grader | sklearn, `grader/model.joblib` | zero, offline, ms | free tier, cascade, all degradation paths |

The project doubles as an end-to-end ML case study: every component that ships
carries a held-out metric, and two designs that failed their metrics
(a "route confident-good answers locally" cascade rule; a stacked scorer)
are documented as deliberate negative results.

## 2. Repository layout (95 tracked files)

```
mle-aie-interview-coach/
├── server.py               (1039 lines)  backend: HTTP server, routing, grading engines
├── retrieval.py            (110)         BM25 retrieval over question banks
├── requirements.txt                      anthropic + scikit-learn (that's all)
├── README.md                             user guide + measured results
├── users.sample.json                     access-key template for freemium tiers
├── .env.sample                           documented environment variables
├── docker-compose.yml                    two-service deployment
├── docker/                 (3 files)     backend.Dockerfile, frontend.Dockerfile, nginx.conf
├── docs/                   (6 files)     contracts, specification, interview prep (en/zh), mock-interview plan
├── data/                   (gitignored)  personal + runtime data; only data/README.md is tracked
├── public/                 (4 files)     dependency-free frontend
├── grader/                 (14 files)    the distillation subsystem (see §5)
├── tests/                  (3 files)     retrieval + grader regression suites
├── rag_ml/                 (44 files)    MLE question bank: 191 chunks over 20 modules
└── rag_ai/                 (16 files)    AIE question bank: 91 chunks over 6 modules
```

Runtime-only files (gitignored, never committed): `.env` (API keys),
`users.json` (real access keys), and everything under `data/` — resume text,
gathered interview questions, `data/sessions/*.jsonl` (collected answers),
`data/usage.json` (quota state), recordings. `data/README.md` documents the
layout; in Docker the same tree is the `coach-data` volume at `/data`.

## 3. Backend — `server.py`

Python 3.10+ standard library only for serving (`http.server.
ThreadingHTTPServer`, thread per request — no Flask/FastAPI, a deliberate
zero-dependency choice), plus the `anthropic` SDK for Claude and plain
`urllib` for DeepSeek/Ollama.

Main sections, top to bottom:

- **Config and env** — `load_env_file()` parses `.env`; env vars: `PORT`,
  `HOST`, `ANTHROPIC_MODEL`, `DEEPSEEK_API_KEY`/`DEEPSEEK_MODEL`,
  `USERS_PATH`/`USAGE_PATH`/`PAID_DAILY_QUOTA`, `PAID_CASCADE`,
  `REAL_SESSIONS_PATH`/`FREE_SESSIONS_PATH`.
- **Corpora** — `load_chunks()` loads both banks, builds one `Retriever` per
  track, validates chunk-id uniqueness, exposes module lists via `/api/meta`.
- **Users and tiers** — `load_users()` (fail-closed: a corrupt `users.json`
  keeps tiers ON with zero paid keys, never silently off), `resolve_user()`
  (bearer-style `X-Access-Key` header), `quota_take()`/`quota_left()`
  (per-key daily counter under a lock, state in `usage.json`).
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
- **Prompts and schemas** — `build_question_prompt`,
  `build_evaluation_prompt` (rubric as checklist for first answers; rubric as
  background for follow-ups), `QUESTION_SCHEMA`, `EVALUATION_SCHEMA`.
- **Handlers** — `GET /api/meta`, `POST /api/question`, `POST /api/evaluate`,
  plus path-traversal-safe static file serving of `public/`.
- **Session logging** — `log_real_session` (LLM-graded rows; `graded_by`
  field lets `evaluate_on_real.py` keep only Claude rows as gold pairs) and
  `log_free_session` (free-tier answers stored **unlabeled** — the student's
  own score is never a training label, avoiding self-distillation). Both
  honor the per-key `"log": false` opt-out; the UI discloses collection.

## 4. Retrieval — `retrieval.py`

Pure-Python BM25 (k1=1.5, b=0.75) over short per-question documents
(question + key points + tags), not raw lesson text. Query-time filters:
module and difficulty-by-level metadata, plus a session `exclude` list so
questions do not repeat. One of the top-5 hits is sampled at random for
variety. Deliberately not embedding-based — measured at 100% Recall@5 / 0.91
MRR on 23 curated cases (`tests/test_retrieval.py`), which does not leave
room for an embedding index to pay its complexity.

## 5. The grader subsystem — `grader/`

The LLM-distillation pipeline, in dependency order:

| File | Role |
| --- | --- |
| `generate_answers.py` | Manufactures ~3,900 synthetic answers from the corpora across quality tiers (reference answers, paraphrases, key-point recalls, fragments, vague filler, question echoes, confidently-wrong "mistake" answers), with style corruption at every level so messy writing isn't mistaken for low quality. Seed-deterministic: regenerating preserves `row_id`s so gold labels re-join. |
| `features.py` | `FeatureExtractor` → 16 answer-level features (`FEATURE_NAMES`): idf-weighted stemmed-token overlap and char n-gram cosine per key point, mistake similarity, question echo, length/structure stats. Deliberately **no** whole-answer similarity to the model answer — that feature taught "quoting the reference = good" and tanked real paraphrases. Also 10 per-key-point features (`KP_FEATURE_NAMES`, both lexical channels exposed separately) for the verdict classifier. |
| `label_teacher.py` | Gold labels: grades sampled synthetic answers through the app's exact evaluation prompt with Claude. 598 rows labeled (~$21 total, across three authorized runs). Dry-run cost estimate; nothing sent without `--confirm`; resumable. |
| `label_keypoints.py` | Dense supervision: one teacher call returns a hit/partial/miss verdict for every rubric key point of a row (~3,000 verdicts for ~$8). `--consistency N` regrades rows to measure label stability: 95% exact, zero hard hit↔miss flips. |
| `train.py` | Trains everything and writes `model.joblib`: (1) overall scorer — Ridge vs HistGradientBoosting, grouped split **by chunk** so evaluation is always on unseen questions; teacher labels override construction labels at 3× weight. (2) Four subscore regressors (each must beat overall-as-proxy). (3) Per-key-point verdict classifier (must beat the 0.35 lexical threshold). (4) A stacked scorer fed the classifier's aggregates — gated: ships only if gold metrics improve (they didn't: QWK 0.770 vs 0.784, so it doesn't). |
| `model.joblib` | The artifact: extractor + overall model + subscore models + kp classifier + all metrics. Loaded by `server.py` for every local grading path. |
| `cascade_analysis.py` | Replays the training split and measures candidate cascade rules on gold rows — the evidence behind the shipped thresholds. |
| `judge_agreement.py` | Re-grades the 121 held-out gold rows with candidate judge models (DeepSeek V4) through the same prompt: Flash 0.59 MAE / 94% within-±1 / QWK 0.93 vs the teacher; regrade consistency 57% exact (vs Claude's 95%) — hence "runtime judge yes, teacher no". |
| `evaluate_on_real.py` | The real-distribution check: compares local predictions against Claude scores on actual logged practice answers as they accumulate. |
| `dataset.jsonl`, `labels_teacher.jsonl`, `labels_keypoints.jsonl`, `judge_agreement_results.jsonl` | Committed data: synthetic rows, 598 gold overall+subscore labels, 623 per-key-point verdict records, judge experiment results. |

Headline gold-set numbers (121 held-out Claude-labeled rows, questions never
seen in training): distilled student **MAE 1.11 / 70% within-±1 / QWK 0.784**
vs keyword baseline 2.05 / 45% / 0.574. Per-key-point classifier: **78%
3-class accuracy / hit-F1 0.87** vs 70% / 0.76 for the lexical threshold.

## 6. Frontend — `public/` (zero dependencies, no build step)

- `index.html` — setup page: track (MLE/AIE), level, topic (module lists
  fetched from `/api/meta`, never hard-coded), optional focus text, optional
  access key with live validity badge (green check / red cross, decided by
  the server) and tier/quota badge.
- `interview.html` — question card, elapsed timer with pause, word count,
  answer box with 🎤 voice input (browser Web Speech API, client-side only,
  hidden outside secure contexts) and the "Always Claude" checkbox
  (`force_llm`), evaluation rendering with a "Graded by <model>" chip,
  follow-up / new-question flow, data-collection disclosure note.
- `app.js` (638 lines) — page init, sessionStorage state, the three API
  calls, rendering. The frontend never grades, never ranks, never sees a key.
- `styles.css` (633 lines) — the entire visual design.

## 7. Tests and evaluation harnesses

- `tests/test_retrieval.py` + `retrieval_cases.json` — 23 curated queries,
  fails below 90% Recall@5 (currently 100%, MRR 0.91).
- `tests/test_grader.py` — artifact sanity: ordering invariants
  (reference ≫ vague ≫ off-topic) on probe answers per corpus.
- The three `grader/` analysis scripts above are the deeper evaluation layer;
  routing/cascade/tier unit checks live in session scratchpads and re-run on
  demand.

## 8. Deployment

- **Local dev**: `python server.py` (serves frontend + API on one port);
  `--mock` (fully offline) and `--ollama` (local LLM) modes cost nothing.
- **Docker**: two services mirroring the documented contract split —
  nginx serving `public/` and proxying `/api/` (300s timeouts,
  `Cache-Control: no-cache`), and a Python backend image containing exactly
  the runtime scope (no training scripts, no datasets, no `.env`; the key is
  injected at runtime via compose `env_file`). Named volume `coach-data`
  persists session logs and quota state; `users.json` is bind-mounted
  read-only.
- **EC2**: runs on a t3.micro (backend peaks ~300–400 MB). Operational notes
  that were learned the hard way: add swap before `docker build` on 1 GB RAM;
  AL2023 needs compose/buildx plugins installed manually; a silent security-
  group drop looks like a 21-second client timeout.

## 9. Design decisions worth knowing (with their evidence)

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
