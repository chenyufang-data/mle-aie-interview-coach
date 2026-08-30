# MLE/AIE Interview Coach

A local interview practice app for Machine Learning Engineer and AI Engineer interviews,
powered by Claude and two curated, rubric-grounded question banks with real questions,
model answers, and grading rubrics:

- `rag_ml/` (191 chunks over 20 modules of classical ML and data analysis) -
  serves the **MLE** track.
- `rag_ai/` (91 chunks over 6 modules of LLM and agent engineering) - serves
  the **AIE** track.

Both banks are derived from course material I studied; the public repository
ships the interview questions, model answers, and rubrics only. The complete
edition — with each chunk's lesson text and the builder scripts — lives in a
private repository, and `tools/strip_chunks.py` regenerates the public banks
from it (see [Data and privacy](#data-and-privacy)).

Both banks share the same chunk schema (`id` / `interview` / `metadata`), so the
server treats them uniformly and routes by track.

The project doubles as an end-to-end LLM + classical-ML case study: rubric-grounded
Claude grading with structured outputs, evaluated BM25 retrieval, and a distilled
scikit-learn grader that approximates the Claude judge offline — with a measured
agreement number for every component (see
[Local ML grader](#local-ml-grader-llm-distillation)).

## Architecture

Frontend and backend have strictly separated requirements — see
[`docs/frontend.md`](docs/frontend.md) and [`docs/backend.md`](docs/backend.md)
for the full contracts, including the three-endpoint JSON API between them.
In short: the frontend (`public/`, dependency-free vanilla JS, no build step)
owns presentation and session flow; the backend (`server.py`, Python stdlib HTTP
server) owns the corpora, retrieval, AI calls, grading, and session logging.
Nothing corpus-specific is hard-coded in the frontend — module lists come from
`GET /api/meta`.

## What it does

- Choose an interview track, level, and topic on the setup page.
- Two question sources in the topic dropdown:
  - **General topics** - Claude generates a fresh, realistic interview question.
    These cover ground the course banks do not (e.g. fine-tuning, cost/serving,
    and safety on the AIE track).
  - **Course knowledge base** - the app picks a real question from your track's
    course bank (MLE -> `rag_ml`, AIE -> `rag_ai`), filtered by module and your
    level, then ranked against your optional focus text with BM25 keyword search
    (`retrieval.py`; one of the top matches is chosen at random so sessions stay
    varied). The module lists in the dropdown come from the server (`/api/meta`),
    so the frontend never hard-codes corpus contents.
- Answer with the elapsed timer running — typed, or spoken via the 🎤 button
  (browser speech recognition, client-side only; needs HTTPS or localhost) —
  then submit.
- The coach returns a scored evaluation, strengths, gaps, a stronger sample answer,
  concrete next steps, and a follow-up question. Knowledge-base questions are graded
  against the chunk's model answer, key points, and common mistakes.
- After the evaluation, continue with **Practice the follow-up** or **New question**
  (knowledge-base mode avoids repeating questions within a session). Follow-up answers
  are graded with the original question, your previous answer, and the course material
  as context, so the coach can tell whether you built on your earlier reasoning.

## Setup

One-time: create the virtual environment and install dependencies.

```powershell
cd mle-aie-interview-coach
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

Copy `.env.sample` to `.env` and fill in your Anthropic API key (skip this if
you only want the free `--mock` or `--ollama` modes):

```powershell
copy .env.sample .env
```

`.env` also accepts the optional `ANTHROPIC_MODEL` (default `claude-opus-4-8`)
and `PORT` (default `8000`). It is gitignored, so your key never reaches the repo.

## Run

```powershell
cd mle-aie-interview-coach
.venv\Scripts\python server.py
```

(Or activate first with `.venv\Scripts\Activate.ps1`, then `python server.py`.)

## Run with Docker

The compose setup deploys the two documented halves as separate services:
a dependency-free nginx image serving `public/` (frontend, the only published
port) and a Python image with the API, question banks, and grader artifact
(backend, reachable only through nginx's `/api/` proxy).

```bash
docker compose up --build -d
```

Then open http://127.0.0.1:8080. The Anthropic key is read from the same `.env`
at runtime by compose — it is dockerignored and never baked into an image. For
the free offline ML grader instead, uncomment the `--mock` command line in
`docker-compose.yml` (no `.env` needed at all). Practice history persists in
the `coach-data` volume. `--ollama` mode is not wired for Docker (it expects
Ollama on localhost).

This fits comfortably on a small host such as an EC2 `t3.micro` (the backend
peaks around 300-400 MB RAM). Two tips there: build the images off-box or add
swap (1 GB RAM is tight for `docker build`), and either restrict the security
group to your own IP or run `--mock` if the instance is publicly reachable —
the app has no authentication, and in Claude mode every evaluation spends your
API credits.

## Free testing modes (no API cost)

Test the app without spending Anthropic API credits:

```powershell
# Fully free and offline - no API key needed at all.
# Questions come from the course knowledge base (even for "general" topics),
# and grading uses the trained local ML grader (see below), falling back to
# keyword rubric matching if no model has been trained.
.venv\Scripts\python server.py --mock

# Free local AI model via Ollama (install from https://ollama.com first,
# then e.g. `ollama pull llama3.2`). Real AI grading, zero API cost.
.venv\Scripts\python server.py --ollama            # uses llama3.2
.venv\Scripts\python server.py --ollama qwen3:8b   # or pick your model
```

Local evaluations are clearly labeled in the summary (e.g. `[Mock mode - local
ML grader ...]`). Run without flags for the real Claude-graded experience.

## Free and paid tiers (freemium demo)

The two graders can also serve different users at once — the classic freemium
cost structure, with the distilled model as the zero-marginal-cost free tier
and Claude as the metered paid tier:

```powershell
copy users.sample.json users.json   # then change the key; users.json is gitignored
.venv\Scripts\python server.py      # Claude mode with tiers ON
```

With `users.json` present, requests are routed per user instead of per server:

- **Free (no key, or unknown key)**: course-bank questions and instant grading
  by the distilled local model. Zero API cost, works offline.
- **Paid (a key listed with `"tier": "paid"`, entered in the setup page's
  access-key field)**: AI-generated questions and real LLM grading. With
  `DEEPSEEK_API_KEY` in `.env`, the workhorse judge is DeepSeek V4 Flash —
  quota-free at ~$0.0005 per evaluation, and measured before it shipped
  (94% within-±1 agreement with the Claude teacher; see the judge table
  below). Claude handles "Always Claude" requests, capped at
  `PAID_DAILY_QUOTA` calls per day (default 30); a spent quota degrades to
  Flash, and a failed Flash call degrades to the local model — labeled as
  such — rather than failing. Without a DeepSeek key, the paid tier is
  all-Claude under quota, as before. Every evaluation carries a "Graded by"
  chip naming the model that scored it.

**Smart cascade** (paid tier, `PAID_CASCADE=0` to disable): answers the
distilled model grades reliably are served locally *without* spending quota.
The routing rule is measured, not guessed — `grader/cascade_analysis.py`
replays the 121 held-out gold rows and shows that routing only
clearly-below-rubric answers (predicted <= 2.5, rubric coverage <= 0.25) keeps
~12% of evaluations local at **100% within-±1 agreement** with Claude, while
the intuitive "route confident-good answers" rule measured at only 47% — so it
does not ship. The "Always Claude" checkbox on the answer page opts out per
request.

Delete `users.json` (or never create it) and the app behaves exactly as before:
single-user, every request graded by Claude. In Docker, mount `users.json` into
the backend service (a commented line in `docker-compose.yml` shows how).

**Answer collection**: the answer page discloses that graded answers are stored
to improve the grading model; any key can opt out with `"log": false` in
`users.json`. Claude-graded answers land in `data/sessions/real_sessions.jsonl` as
gold pairs; free-tier answers land in `data/sessions/free_sessions.jsonl` *unlabeled*
(the local model's own score is never a training label) until a teacher-labeling
pass promotes them. Both files are gitignored.

Then open:

```text
http://127.0.0.1:8000
```

## Resume files → text

The upcoming mock-interview feature (see `docs/mock_interview_plan.md`) and
its transcription experiment need your resume as plain text. `resume_parser.py`
converts PDF and Word files:

```powershell
.venv\Scripts\python resume_parser.py path\to\resume.pdf      # or .docx / .txt / .md
```

Output goes to `data/resume/<name>.txt` (gitignored) with a preview so you can
check the extraction. PDFs are read with `pypdf`; `.docx` is parsed from its
XML with the standard library (paragraphs and tables in order); ligatures,
bullet glyphs and wrapped hyphens are normalized. Legacy `.doc` and scanned
(image-only) PDFs are not supported — save as `.docx` or export a text PDF.

## Data and privacy

Three kinds of data, three treatments:

- **Public** — code, the public-edition question banks (interview fields and
  retrieval metadata), the trained grader artifact, gold label files, and
  experiment results.
- **Private repository** — the complete banks with course-derived lesson text
  and the builder scripts, plus `grader/dataset.jsonl`. `tools/strip_chunks.py`
  turns the complete banks into the public ones; `RAG_FULL_DIR` points the
  training scripts at that checkout.
- **Never committed** — `.env`, `users.json`, and everything under `data/`
  (resume text, gathered interview questions, practice logs, quota state,
  recordings). See `data/README.md`.

This project was built with Claude Code as a pair programmer; the evaluation
design, every ship/no-ship decision, and the negative results are mine.

## How grading works

For knowledge-base questions, the server sends Claude the chunk's `model_answer`,
`key_points` (used as the rubric), `common_mistakes`, and `followups` alongside your
answer, so feedback is grounded in what the course actually teaches. See
`rag_ml/README.md` and `rag_ai/README.md` for the chunk schema.

For follow-up questions, the same chunk is kept but its key points are provided as
background context rather than a strict checklist (they belong to the original
question), together with the original question and your previous answer.

## Retrieval smoke test

`tests/retrieval_cases.json` holds curated focus-text queries for both banks
(`"corpus": "ml"` or `"ai"`) with the module or tags a good result should have.
Run it after changing `retrieval.py` or either corpus:

```powershell
.venv\Scripts\python tests\test_retrieval.py
```

It reports Recall@5 and MRR and fails below 90% Recall@5.

## Local ML grader (LLM distillation)

Mock mode's grading brain is a small trained model that approximates the Claude
grader - the LLM-judge distillation pattern at miniature scale. A pure-Python
feature extractor (`grader/features.py`) turns an (answer, rubric) pair into
rubric-coverage features (stemmed idf-weighted token overlap + char n-gram
cosine per key point, mistake similarity, question-echo, length/structure
stats), and a scikit-learn regressor trained on those features predicts the
1-10 score in milliseconds, deterministically, offline.

Build or rebuild it (free, no API calls):

```powershell
# generate_answers needs the complete banks (lesson text) from the private repository:
$env:RAG_FULL_DIR = "..\mle-aie-interview-coach-private"
.venv\Scripts\python grader\generate_answers.py   # synthetic answers with construction-known labels
.venv\Scripts\python grader\train.py              # train, evaluate, save grader\model.joblib
.venv\Scripts\python tests\test_grader.py         # sanity-check the artifact
```

`grader/dataset.jsonl` (the synthetic answers) is private too — its
content_extract tier quotes lesson text — so retraining needs that checkout;
the shipped `grader/model.joblib` and the gold label files are public.

The training data is manufactured from the corpora themselves: reference
answers (9), lesson-text extracts and key-point recalls (7), sentence subsets
(2-8), vague filler and question echoes (1-2), and fluent answers to the wrong
question (1), with style corruption applied at every quality level so messy
writing is not mistaken for low quality. `train.py` splits by chunk (never by
answer) so evaluation is always on unseen questions, and compares the model
against the old keyword-matching heuristic.

Optionally upgrade the labels from construction-derived to real Claude scores
(THIS ONE SPENDS API CREDITS - it prints a cost estimate and does nothing
without `--confirm`):

```powershell
.venv\Scripts\python grader\label_teacher.py                # dry run: cost estimate only
.venv\Scripts\python grader\label_teacher.py --confirm      # ~$10 at 300 labels with opus-4-8
.venv\Scripts\python grader\label_teacher.py --confirm --model claude-sonnet-5   # cheaper
.venv\Scripts\python grader\train.py                        # retrain on gold labels
```

### How good is it?

The shipped model was distilled from 598 Claude gold labels
(`grader/labels_teacher.jsonl` — kept in the repo; it joins `dataset.jsonl` by
`row_id`). On 121 held-out gold rows — answers to questions the model never saw
in training — agreement with the Claude teacher:

| Model | MAE | within ±1 | Spearman | QWK |
| --- | --- | --- | --- | --- |
| Keyword baseline | 2.05 | 45% | 0.680 | 0.574 |
| Distilled grader | **1.11** | **70%** | **0.730** | **0.784** |

The teacher itself is near-deterministic (95% exact agreement when regrading
the same answers), so the remaining gap is real model error, not label noise.
Known limitation, also measured on the gold set: lexical features judge whether
an answer covers the right points, not whether its claims are true —
confidently-wrong answers are the hardest tier. The per-key-point distillation
below is the first bite at that gap.

### Per-key-point distillation (dense supervision)

Overall scores cannot teach the student *which* rubric point it misjudged, so
`grader/label_keypoints.py` asks the teacher for a hit / partial / miss verdict
on every rubric key point of every gold-labeled row (598 rows × ~5 points, ~$8;
one call returns a whole row's verdicts). The labels are themselves
quality-measured: on a 25-row regrade, 95% exact verdict agreement and **zero**
hard hit↔miss flips. `train.py` distills them into a per-key-point classifier
that replaces the fixed 0.35 lexical threshold behind the hit/miss feedback
lists the app shows (601 held-out labeled points):

| Hit/miss judge | 3-class acc | macro-F1 | hit-F1 |
| --- | --- | --- | --- |
| Lexical threshold (0.35/0.6) | 70% | 0.62 | 0.76 |
| Distilled kp classifier | **78%** | **0.67** | **0.87** |

Honest negative result, same experiment: a stacked overall model fed the
classifier's coverage aggregates did **not** improve gold agreement (MAE 1.10
vs 1.11, QWK 0.770 vs 0.784), so `train.py`'s ship-gate declines it and the
plain model keeps scoring. The dense labels earn their keep in the
*explanations* — which points you hit, which you only brushed — not the number.

The construction-label metrics `train.py` prints are optimistic (those labels
partly share signal with the features); the gold-label section of its output is
the honest one. The third check is real usage: every real (non-mock) graded
session is logged to `data/sessions/real_sessions.jsonl` (kept out of git), and
`grader/evaluate_on_real.py` reports Claude-vs-local agreement on your actual
answers as they accumulate.

### Could a cheaper judge replace Claude? (measured)

`grader/judge_agreement.py` re-grades the same 121 held-out gold rows through
the same evaluation prompt with candidate judge models (DeepSeek V4, via their
OpenAI-compatible API — needs `DEEPSEEK_API_KEY` in `.env`; ~$0.35, dry-run by
default, `--confirm` to spend). August 2026 results:

| Judge | MAE | within ±1 | QWK | regrade consistency (exact) |
| --- | --- | --- | --- | --- |
| Distilled student | 1.11 | 70% | 0.78 | deterministic |
| deepseek-v4-flash | 0.59 | **94%** | 0.93 | 57% |
| deepseek-v4-pro | 0.57 | **96%** | 0.93 | 53% |

Both V4 judges track the Claude teacher about as closely as the teacher tracks
itself (95% exact on regrade), and they fix the distilled model's known blind
spots (confidently-wrong answers: 100% within ±1 vs 67%; heavy paraphrase: 88 to
100% vs 25%) — at roughly 1/200th of Opus-tier cost. Their weakness is score
jitter: regrading the same answer reproduces the exact score only 53-57% of the
time (vs Claude's 95%), and ~2.6% of calls returned malformed JSON (a retry
handles it). Conclusion — now shipped: V4-Flash is the paid tier's quota-free
workhorse judge (see the tiers section above), but Claude stays the
distillation teacher — gold labels need the reproducibility — and serves
"Always Claude" requests under the daily quota.
