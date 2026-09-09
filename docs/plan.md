# Project plan — what was built, what it measured, what comes next

> This file replaces `docs/dense_retrieval_plan.md` and
> `docs/mock_interview_plan.md` (merged 2026-09-07). Those two were lab
> notebooks kept verbatim; this is the condensed record plus the roadmap.
> The full notebooks survive in git history (last at commit `2469704`).
>
> Conventions: every number names the file it comes from; "README only"
> means the producing script prints the value and no results file is
> committed; "assistant labels" means the fairness judgments were made by
> the AI pair programmer at the author's request; the author reviewed a
> stratified 40-pair sample on 2026-09-08 and agreed on 39 (Cohen's kappa
> 0.95, see 1.9). Cell sizes are small everywhere (23 to 121 items), so
> differences under about ten points are direction, not ranking.

---

## Part 1 — Finished and failed phases (2026-07 to 2026-09-06)

### 1.1 Foundation — practice coach and the distilled grader

Built before the plans below: the practice track (question by track,
module and level; BM25 retrieval; Claude grading against a rubric), the
freemium tiers (`users.json`, paid keys, daily Claude quota), the DeepSeek
Flash workhorse, and a scikit-learn grader distilled from Claude labels so
the free tier and `--mock` mode grade offline.

| Measurement | Result | Source |
| --- | --- | --- |
| Distilled grader vs Claude teacher, 121 held-out gold rows (split grouped by question, seed 42) | MAE 1.11, 70% within ±1, QWK 0.784 (keyword baseline 2.05 / 45% / 0.574) | README only; labels in `grader/labels_teacher.jsonl` (598 rows) |
| Per-key-point classifier, 601 held-out points | 78% accuracy, macro-F1 0.67, hit-F1 0.87 (lexical threshold 70% / 0.62 / 0.76) | README only; labels in `grader/labels_keypoints.jsonl` |
| DeepSeek Flash as judge vs teacher | MAE 0.59, 94% within ±1, QWK 0.93; regrade consistency 17/30 exact | raw rows `grader/judge_agreement_results.jsonl`, summary README only |
| Cascade rule (local grader answers when confident) | 12% of evaluations local at 100% within ±1 | README only |

Cost of the labels: about $8 for the per-key-point pass. The training
answers (`grader/dataset.jsonl`, 3,866 rows) are derived from lesson text
and live only in the private checkout; the public repo holds the labels
and the trained artifact, so the grader table cannot be reproduced from
the public repo alone (roadmap step 1 fixes this).

### 1.2 Mock interview, Phase 0 — speech-to-text on technical vocabulary (done; rule not met, fallback shipped)

**Question.** Transcription mangles exactly the words that carry signal
("QWK", "LightGBM", "Recall@5"). How much, and how much does it move the
grade?

**Setup.** 88 items (68 sentences, 20 whole answers) over a 339-term
lexicon, read once by the author (26.5 min) and once by a TTS voice. Seven
conditions, from the browser's Web Speech API to ElevenLabs Scribe with
keyterm prompting to local faster-whisper on the RTX 5080. Pre-registered
rule: ship the cheapest condition with ≤ 5% of grades moved ≥ 1 point and
≤ 3% lenient term error rate on the human set.

**Result** (`grader/stt_eval_results.json`, rendered in
`docs/stt_evaluation.md`, human set):

| Condition | WER | Term error, lenient | Grades moved ≥ 1 (raw / normalised) |
| --- | ---: | ---: | ---: |
| Browser Web Speech (old path) | 19.9% | 29.4% | 60% / 45% |
| Scribe batch + full lexicon keyterms — **final transcript** | 5.9% | 3.0% | 35% / 25% |
| Scribe Realtime + policy-chosen 50 keyterms — **live transcript** | 12.0% | 9.7% | 35% / 35% |
| Local faster-whisper turbo + 50-term prompt — **offline** | 8.4% | 13.4% | 20% / 15% |

No condition met the rule, so the pre-registered fallback applied: ship
the best and state the residual. Findings that were not in the plan:
post-hoc fuzzy correction is a negative result (63% of its rewrites were
false); the sklearn grader moves 50% of grades when the reference is
merely lowercased, so the report grades normalised text on both sides;
the Flash judge's noise floor on identical text is 35 to 50%, so only the
deterministic grader can measure damage at this scale. Total spend about
$1.50.

### 1.3 Mock interview, Phase 1 — text loop and the report (done)

`coach/mock/` (templates, schemas, planning, turns, metrics, report,
routes) and `public/mock.html`. A hidden plan is built once from the
resume (role profile, 5 to 8 probe targets, rubric chunks attached from
the banks by BM25); a phase machine in code (warm-up, walkthrough, deep
dive, behavioral, closing); the model streams spoken text first and a JSON
trailer after a marker. The report has three blocks: computed
communication metrics, transcription quality, and an LLM scorecard with
six fixed dimensions plus role signals, every justification quoting the
transcript.

| Measurement | Result | Source |
| --- | --- | --- |
| Report consistency, same session regraded (5 sessions × 6 dimensions × 2) | Claude 73.3% exact, MAE 0.3; Flash 43.3%, MAE 0.8; hiring call stable for both (all "no hire", so uninformative) | `grader/report_consistency_results.json` (~$1.50) |
| Rubric grounding on the first live plan | 7/7 probes matched bank chunks at BM25 12.8 to 19.7 | plan notebook |

Decision: DeepSeek Flash runs the turns, Claude writes the report when a
key is present.

### 1.4 Mock interview, Phase 2 — live voice loops and the vendor comparison (done; hosted variant optional)

A DIY voice loop (`coach/voice/`, `server.py --voice`): browser audio over
one WebSocket, Silero VAD endpointing, streaming LLM with thinking off,
sentence-level TTS, barge-in with spoken-prefix truncation. Three audio
backends behind one switch: local (faster-whisper + Kokoro), ElevenLabs
(Scribe Realtime + Flash TTS), Deepgram (Nova-3 + Aura-2). Each answer is
recorded and re-transcribed after the session with the full lexicon, so
the report grades the final transcript and shows what the live one lost.

Equivalence harness (`grader/loop_eval.py`): the 20 real Phase 0 answers
replayed against a deterministic interviewer, so only the audio path
varies (`grader/loop_eval_results.json`):

| Backend | WER | Term error, lenient | First audio p50 / p95 | Grades moved |
| --- | ---: | ---: | ---: | ---: |
| **local (ships as main usage)** | 7.4% | 9.3% | 0.77 / 1.73 s | 3/18 |
| ElevenLabs (best cloud row) | 7.6% | 3.7% | 0.48 / 0.63 s | 5/18 |
| Deepgram (rejected; keyless-ElevenLabs fallback) | 12.1% | 14.8% | 1.35 / 4.48 s | 6/18 |

End-of-turn silence was swept under a rule (cut-off ≤ 5%): 1.2 s cut half
the answers mid-thought, 1.8 s cut 15%, 2.0 s ships at 5%. Barge-in
interrupts in 0.32 s under real LLM turns (3-answer run). Getting honest
rows found and fixed seven real bugs (blocking commit in the receive path,
ping starvation, Scribe's multi-segment commits, Deepgram's lazy finals
without interim results, a Silero v5 context-window omission, an
answer-continuation stall, and the harness streaming audio at 4×).

The hosted ElevenLabs Speech Engine path (Level 1) is wired and
config-verified (`tools/level1_up.py`, sidecar, tunnel) and taught two
undocumented protocol rules (echo the transcript's event id; end every
response with an empty final chunk), but was never run live; it is an
optional comparator, no longer on the recommended path. A blind TTS
preference test rides on that session and is likewise unrun.

### 1.5 Mock interview, Phase 3 — caching, logging, rubric tie-in (done)

Content-hash cache of role and plan results (`coach/mock/plan_cache.py`);
opt-in session logging (default off; resumes are personal data); prompt
caching measured (`grader/cache_check_results.json`): the second Claude
turn read 1,123 tokens from cache and paid full price on 374, about
−75% turn input; DeepSeek's automatic cache hit 512 prefix tokens. Found
live and fixed: `temperature` returns 400 on current Claude models. Missed
probes in the report link to "practice this exact question"
(`/?practice=<chunk_id>`).

### 1.6 Real-question bank, `rag_exp` (done)

Hand-collected interview experience posts (local-only spreadsheets that
name third parties; never committed anywhere) go through
`grader/ingest_questions.py`: dedupe, translate, rubric by the Claude
teacher. 57 chunks for $1.77; the bank is private (loader warn-skips when
absent); the mock grounds only on technical, experience and system-design
rounds. Retrieval suite stayed 23/23 with the third bank loaded.

### 1.7 Dense retrieval experiment — R1 pass, R2 fail, R3 not earned (2026-09-04, re-checked 2026-09-06)

**Why.** The README's "100% Recall@5 on 23 curated queries" was measured
but saturated: it could never show an embedding model winning. Arms fixed
before running: BM25 (incumbent), bge-small dense (ONNX on CPU, 127 MB),
Chroma (same vectors, store overhead only), hybrid (reciprocal rank fusion
of BM25 and dense). Sets: A = the 23 curated queries (ceiling check); B =
61 candidate-speak paraphrases with the target's tag vocabulary filtered
out (the real signal); C = 77 mock probes with hand-labeled rubric
fairness.

**Rules, frozen before the first run.**
- R1 (practice retrieval): replace BM25 only if set A stays 23/23 with MRR
  within 0.05, set B gains ≥ 10 points Recall@5, p95 ≤ 50 ms, model
  < 200 MB.
- R2 (mock rubric grounding): switch only if coverage rises ≥ 15 points at
  no loss of hand-labeled precision.
- R3 (the store): no shipping rule; report Chroma's overhead honestly.

**Results** (`grader/retrieval_eval_results.json`, rendered in
`docs/retrieval_evaluation.md`; current run on the grown banks):

| Arm | Set A Recall@5 / MRR | Set B Recall@5 | p95 | R1 |
| --- | --- | --- | ---: | --- |
| BM25 | 23/23 / 0.915 | 44/61 (72.1%) | 0.5 ms | baseline |
| dense | 23/23 / 0.946 | 48/61 (78.7%) | 2.3 ms | fail (+6.6) |
| **hybrid** | 23/23 / 0.935 | **51/61 (83.6%)** | 2.9 ms | **pass (+11.5)** |

The first run (2026-09-04, before bank growth) was 41 / 47 / 49 of 61;
growth raised every arm by one to three hits. Hybrid serves the practice
track (`retrieval_dense.py`, `coach/kb.py`); BM25 is the fallback when
the model is missing. Memory grows by about 200 MB with the model loaded.

- **R2 failed for a reason the plan did not expect.** Probe queries are
  long resume sentences, so BM25 already attaches a rubric to 77/77
  probes; coverage cannot rise. The real problem is precision of what is
  attached: BM25 55.8%, hybrid 61.0%, dense 64.9% on assistant labels
  (`grader/grounding_eval_results.json`). Grounding stayed on BM25.
- **R3 not earned**, as predicted: Chroma returned the identical top-5 on
  84/84 queries, +0.75 ms p95, 5.6 MB on disk vs 1.2 MB for the numpy
  array. No vector database.
- **Chunk harvest not triggered**: 9 of 77 probes (12%) had no fair chunk
  in any bank, below the pre-registered one-third line.
- **Product decision on this evidence (author, 2026-09-04):** mock probes
  come from the resume only; the job description decides which claims get
  probed and how deep, never adds topics (resume-project probes grounded
  fairly 69 to 74% of the time, job-theme probes 26 to 43%). The
  beyond-resume path is frozen behind `MOCK_BEYOND_RESUME`.
- A post-hoc "attach only when BM25 and dense agree" finding (25/25 fair)
  was written down as a hypothesis to test on fresh probes, and was
  falsified there (1.9).

### 1.8 Bank growth — lists, lesson-text expansion, primary docs (2026-09-04 to 09-06)

Rule: add, never rewrite. Chunk ids are joined by the grader's gold labels,
browser bookmarks and the plan cache, so growth appends new ids and every
kept chunk is reviewed (`tools/review_bank.py` stamps keep / fix / retire;
the loader skips retired chunks). Lesson text stays private; the public
banks are regenerated by `tools/strip_chunks.py`.

| Source | What happened | Chunks | Spend | Review |
| --- | --- | ---: | ---: | --- |
| GitHub question lists (MIT / Apache-2.0 repos, attribution recorded) → `rag_lists` | 534 parsed, 331 rubrics by the Claude teacher; author review found a 53% fix rate (over-strong claims carried from synthesized answers); a fix pass sent each flagged chunk back with the reviewer's note | 331 (319 serving) | $11.78 + $9.08 | 144 keep / 175 fixed / 12 retired (author) |
| AIE lesson-text expansion → `rag_ai` | DeepSeek proposed 289 finer sub-questions with verbatim supporting excerpts; 131 kept at review (drops: restatements, worked examples, course logistics, API details that date); teacher rubrics appended | 91 → 222 | ≈ $4.35 | 131 keep, 8 rewordings of course-internal references (assistant) |
| Primary MLOps documentation → `rag_docs` (scikit-learn, NannyML, Feast, MLflow, promptfoo, Kubernetes, Google ML guides; open licenses, sections pinned to commits) | 20 sections, 116 proposals aimed at the grounding gap list, 72 kept | 72 | ≈ $2.39 | 72 keep, 3 text glitches fixed (assistant) |
| MLE lesson-text expansion | 25 proposals, 22 kept by the author; the rubric run was **not** executed | 0 | ≈ $0.73 pending | — |

Verbatim probes showed the teacher rewrote rather than copied (median 0%
shared 8-grams; five AIE chunks paraphrase closely, accepted by the author
on 2026-09-06: only the original lesson files are private, rubric text
derived from them is not). R1 re-checked after every addition: set A
23/23, hybrid still ≥ +10 on set B.

### 1.9 Grounding experiment R4 — two runs, no policy passed (2026-09-06)

**Question.** Which attachment policy should the mock use for bank rubrics?
Four policies on 77 fresh resume-only probes: `bm25@10` (today), `agree`
(attach only when BM25 and dense pick the same chunk), `dense ≥ 0.70`,
`hybrid`. Rule R4, frozen: a policy replaces `bm25@10` only at ≥ 90%
precision and ≥ 40% coverage. Labels were made blind to policy by the
assistant at the author's request; the author reviewed 40 stratified
pairs with the assistant's reasons shown (2026-09-08): 39/40 agreement,
Cohen's kappa 0.95, one fair-to-unfair change where a technique-list chunk
had been attached to a decision probe (`docs/grounding_r4_grown.md`,
"Author spot-check"); the standard is
recorded in `grader/grounding_r4_labels*.json`.

| Policy, all banks | First run (204 pairs): fair / attached | After growth (237 pairs): fair / attached |
| --- | --- | --- |
| bm25@10 | 32/77 (41.6%) | 34/77 (44.2%) |
| agree | 19/27 (70.4%, coverage 24.7%) | 16/21 (76.2%, coverage 20.8%) |
| dense ≥ 0.70 | 34/70 (48.6%) | 31/73 (42.5%) |
| hybrid | 24/77 (31.2%) | 26/77 (33.8%) |
| probes with any fair candidate | 48/77 | 50/77 |

Sources: `grader/grounding_r4_results.json`, `grader/grounding_r4_results_grown.json`;
reports `docs/grounding_r4.md`, `docs/grounding_r4_grown.md`. Every
prediction written before the run was wrong (agree was predicted to pass;
the grown AIE bank was predicted to lift AIE fairness above 60%, it
reached 53%). `bm25@10` stays and the report keeps disclosing the rubric
tier per probe.

**What the labels say.** The unfair majority is a level mismatch, not
off-topic retrieval: probes ask for a decision or a playbook spanning
several claims ("why 90% precision", "what the CI gate looked like") and
the nearest chunk states one prerequisite claim or is a generic project
rubric. 203 reviewed new chunks moved fair probes by two, so chunk-level
growth has diminishing returns for grounding. The next lever is on the
mock's side: grade against the probe's own expected points with chunks as
hints, or compose a rubric from the top few chunks' key points. That is a
product decision, recorded and not made.

Housekeeping: the pool file quotes private bank text and is gitignored
(synced to the private repo); the first pool was removed from public
history by the author.

### 1.10 Deployment and the private-service track — not started

Planned twice (an EC2 t3.small with TLS at about $17/month; a private
retrieval service gated by access keys or a Tailscale network) and parked
both times: the t3.micro was closed, about $80 of AWS credit remains, and
the voice page still builds a plain `ws://` URL (`public/mock.js:483`).
Roadmap step 1 replaces both plans.

### 1.11 Project report (2026-09-06; kept locally since 2026-09-09)

The project report is the author's own audit of the repository. Since
2026-09-09 it lives outside the public tree at `data/project_report.md`
(gitignored): this plan and the README are the public record, the report
is personal; the 2026-09-06 edition is in history at 2469704. Scores out
of 10 on 2026-09-06: technical depth 8, correctness 7, code quality 6,
testing 6, documentation 7, reproducibility 6, deployment readiness 4,
portfolio strength 8. Its two critical risks (the backend Docker image
omitted four runtime modules and three banks; grading endpoints were
unauthenticated in the single-user configuration) and its nine
documentation-versus-code mismatches were fixed in roadmap step 1. The
2026-09-09 update re-scores after steps 1 and 3 (deployment readiness 7,
portfolio strength 9, testing, documentation and reproducibility 7–8)
and records the step 3 trade-offs and the live box's security posture.

### 1.12 Decisions that stand

1. Practice retrieval is hybrid (BM25 + bge-small); BM25 is the fallback;
   no vector database until a measurement asks for one.
2. Mock rubric grounding stays `bm25@10`; the report discloses the rubric
   tier per probe; the "agree" policy is retracted.
3. Mock probes come from the resume only; beyond-resume probing waits for
   a bank that can ground it.
4. Banks grow by appending reviewed chunks; ids are never rewritten; the
   original lesson files and the lesson excerpts stay private; rubrics and
   close paraphrases may be public.
5. Voice: local audio is main usage; Scribe is the best cloud row while
   the ElevenLabs balance lasts; Deepgram is the cloud option without an
   ElevenLabs key; end-of-turn silence 2.0 s; the report grades the final
   transcript, normalised on both sides.
6. Models: DeepSeek Flash for turns and paid-tier grading, Claude for the
   mock report and as the teacher, the distilled sklearn grader for the
   free tier, `--mock` mode and per-key-point verdicts.
7. Nothing paid runs without a dry run and `--confirm`; every rule is
   written before its experiment runs; predictions are checked against
   results and kept when wrong.
8. Assistant-made labels are flagged as such everywhere; the R4 labels
   were spot-checked by the author on 2026-09-08 (39/40), the R2 set C
   labels were not.

### 1.13 Loose ends carried into the roadmap

- 16 commits unpushed to the public remote.
- No results file for the distilled grader, judge summary and cascade
  numbers; the README tables are typed by hand and drifted (nine items).
- Docker image incomplete; no gate on paid endpoints when bound publicly;
  no request-body cap; hybrid CI gate passes vacuously without the model.
- 51 files with CRLF endings and no `.gitattributes`; no lock file; three
  Python versions named across docs, CI and Docker.
- Author spot-check of the R2 (set C) assistant labels; R4 done 2026-09-08.
- Optional, unchanged: MLE expansion rubric run (≈ $0.73); Level 1 hosted
  voice live session and the blind TTS preference (needs the mic and the
  ElevenLabs balance); the grounding lever (product decision).

---

## Part 2 — Roadmap

Order: 1 → 2 → 3 → 4 → 5. Step 2 is cheap and runs in parallel with step 1
because it waits on other people. Steps 4 and 5 are gated on step 1.

### Machine and account inventory (checked 2026-09-07)

| Item | State |
| --- | --- |
| Desktop | Windows 11, Ryzen 7 9800X3D (16 threads), RTX 5080 16 GB, NVIDIA driver 595.97 (CUDA 13.2 capable), 2.4 TB free |
| Python | 3.13.7 venv; no torch installed |
| Containers | Docker Desktop with the WSL2 backend; kubectl bundled; no Ubuntu WSL distro; the docker-desktop distro was stopped at check time |
| Missing CLIs | terraform, gcloud, aws, node, ffmpeg, OBS |
| Repos | public `origin` (16 commits unpushed) and `private` (complete banks, `grader/dataset.jsonl`, R4 pool); local-only `data/interview_exp/` spreadsheets that never leave this machine |
| Credits | AWS ≈ $80; Deepgram ≈ $199; ElevenLabs ≈ $7.85; Anthropic and DeepSeek paid keys in `.env` |

### Step 1 — Make the Coach the proof: a live, self-consistent public system

**Goal.** A public URL that runs the documented feature set — practice,
the text mock and the live voice mock — with every number in the README
traceable to a committed file. Zero new skills, highest return: it backs
every claim on the résumé. (The two-minute recording planned at first was
dropped on 2026-09-08 in favour of live voice on the box: a visitor can
try it rather than watch it.)

**Status 2026-09-07 (evening).** Done and pushed: the push itself (all
prior commits on origin); the results files (`grader/train_results.json`,
`cascade_results.json`, `judge_agreement_summary.json` — the shipped
artifact reproduces the README numbers exactly), `tools/render_readme.py`
with the eight marker blocks and its CI check, and the nine stale items;
the demo spend cap, public-bind refusal, body caps and generic 500s
(`tests/test_users.py`); the complete Docker image, Caddy override, wss
voice path and `tests/test_container.sh` (green locally, its own CI job);
`docs/deployment.md` with the recording script; the spot-check tooling
(commands below). Found on the way: CI had been red since 2026-09-06
because an unquoted colon in a step name made the workflow file invalid
YAML — fixed. Spot-check done 2026-09-08 (39/40, kappa 0.95).

**Status 2026-09-08.** Live at https://coach.cyfang.org: a t3.small in
us-east-2 behind an Elastic IP, the domain at GoDaddy with an A record for
`coach`, Caddy's certificate issued on first start, the four public banks
loaded (LISTS serves 319 of 331 chunks — 12 retired), hybrid retrieval up.
Every section-4 check of the runbook passed from outside: anonymous
grading goes to the local model, the demo key goes to DeepSeek and its
allowance counts down (60 → 59, server 200 → 199, Claude quota untouched),
a 40-request burst gets 34 429s, a 2 MB body and a 50 MB upload get 413,
ports 8000/8765 do not answer from outside. Then the author chose live
voice on the box over a recording: the image now carries the cloud voice
subset (`requirements-voice-cloud.txt`), `AUDIO_BACKEND=deepgram` runs the
loop on the free Deepgram credit, and voice is metered like the LLM calls
— `daily_voice_minutes` per key, `VOICE_DAILY_MINUTES` per server,
`VOICE_SESSION_MAX_MINUTES` per session, one LLM unit per interviewer turn,
re-transcription paid-key only and charged by clip length; covered by
`tests/test_users.py::test_voice_budget` and
`tests/test_voice.py::test_voice_session_budget`. Deployed the same day
(the author added the Deepgram key, `AUDIO_BACKEND` and the caps to the
box's `.env` and `daily_voice_minutes` to the demo key) and verified from
outside with one scripted session over `wss://coach.cyfang.org/ws/voice`:
`ready` 0.3 s after the hello (engine deepseek, Nova-3 + Aura-2 at 24 kHz),
first spoken sentence at 0.8 s, the opening's two sentences as 522 KB of
audio, `listening` at 3.6 s; afterwards the counters read voice 30 → 29
and 120 → 119 minutes, LLM 59 → 56 (roles, plan, connect). The first
verification pass found a regression the deploy had shipped — an inner
import shadowing `users` in `coach/mock/routes.py` made every mock POST
route answer 500 — fixed in 2e47431 with a route-level test. The author's
first browser session then found setup slow: measured from outside, a
fresh roles call took 178 s (79 s from the home machine) and a fresh plan
65 s, because the structured DeepSeek calls sent no thinking control and
V4 Flash reasoned at length (6,301 reasoning tokens of 9,557 on one roles
call). With thinking off for the two setup calls — extraction from the
resume, not reasoning — both take about 12 s and return the same role set
and a fully grounded 7-probe plan; grading and the report keep thinking
on, the setting the judge agreement was measured under. Each structured
call now logs its seconds and token counts. The author's spoken session
then worked (live transcript, interviewer follow-up) and found two
things: the endpointer cut a 2-3 s thinking pause mid-answer, and the
page forgot the pasted resume. Both addressed the same day: a text-aware
hold in the endpointer (`looks_unfinished` on the live transcript at
end-of-turn — a conjunction, article, filler or comma keeps the turn open
for `VOICE_HOLD_MS` more silence, 2.5 s, at most twice per answer; unit
tests, not yet re-measured on the 20 real recordings), and the resume and
JD remembered in the browser's localStorage.

**Step 1 closed 2026-09-08 (evening).** The author confirmed the box
rebuilt on 2858942, no `ANTHROPIC_API_KEY` in its `.env`, DeepSeek
auto-recharge off, the AWS budget alert set, and reached the URL from a
phone on a cellular network. Every "done when" criterion below is met on
the live box. Carried to the loose ends: re-measuring the end-of-turn
hold on the 20 real recordings, and the domain's renewal price.

**Steps.**

1. *Push.* Final secret scan of `origin/main..main`, then `git push origin
   main`. Half a day of review at most; the mirror branch in the private
   repo is already current.
2. *Results files for the grader claims* (the only numbers a reader cannot
   recompute today).
   - `grader/train.py` writes `grader/train_results.json` (dataset and
     split sizes, per-model MAE / within ±1 / Spearman / QWK, key-point
     classifier metrics, timestamp, sklearn version) and stops
     overwriting `grader/model.joblib` unless `--save` is passed. It must
     run from the private checkout because `dataset.jsonl` lives there;
     the JSON holds numbers only and is copied to the public repo.
   - `grader/judge_agreement.py --report` writes
     `grader/judge_agreement_summary.json` from the committed raw rows
     (no API spend).
   - `grader/cascade_analysis.py` writes `grader/cascade_results.json`
     (also private-checkout input).
   - `tools/render_readme.py` rewrites the README tables between marker
     comments from these files plus `retrieval_eval_results.json`,
     `loop_eval_results.json` and `stt_eval_results.json`; the CI workflow
     runs it with `--check` and fails on drift. Fix the nine stale items
     from the report in the same pass (set B table, `rag_ai/README.md`,
     `docs/backend.md`, module count, Python version, spec counts, the
     two dense thresholds, the TER range, the STT decision string).
   - Author spot-check of 40 stratified R4 pairs on
     `data/review/grounding_r4_grown.html`, with the agreement number
     written into `docs/grounding_r4_grown.md`. One hour; it turns
     "assistant labels" into "assistant labels, author agreement N%".
     Commands: `grader/grounding_r4.py --spotcheck --run grown` draws the
     blind sample page (`data/review/grounding_r4_spotcheck_grown.html`),
     and `grader/grounding_r4.py --spotcheck-apply PATH --run grown`
     scores the exported decisions and writes the section. The author
     chose `--prefill` (2026-09-08): the page shows the assistant's label
     and reason on every pair for confirmation or change, so the report
     will say "review, not blind" and count untouched pairs.
3. *Complete the Docker image.* `docker/backend.Dockerfile` adds
   `retrieval_dense.py`, `resume_parser.py`, `grader/stt_text.py` and
   `grader/stt_lexicon.json`, the public `rag_lists` and `rag_docs`
   banks, and the embedding model (bake `data/models/fastembed/` into the
   image, 127 MB, or persist it in the `coach-data` volume and let
   fastembed download on first start). `rag_exp` stays off the public demo
   (it is built from other people's interview posts) and is mounted as a
   volume only on a private instance. Add `tests/test_container.sh`:
   build, start, poll `/api/meta`, assert the bank list and the retrieval
   backend, run one `--mock` evaluation.
4. *Gate the paid endpoints.* When `users.json` is absent and the server
   binds to anything but loopback, refuse to start unless
   `--allow-anonymous-llm` is passed. Cap request bodies in
   `coach/web.py:read_json` (1 MB for JSON, a separate limit for resume
   uploads) and stop echoing exception text. On the demo box do **not**
   set `ANTHROPIC_API_KEY`: a demo key in `users.json` then routes every
   turn, grade and report to DeepSeek Flash (cents per session), the mock
   works, and the worst case of abuse is DeepSeek cents. Add nginx
   `limit_req` on `/api/`.
   *Demo spend cap (added 2026-09-07).* Today's daily quota counts Claude
   calls only; DeepSeek is quota-free by design. Add a per-key daily LLM
   budget in `users.json` (`"daily_llm_calls": 60`, counted for every
   engine, checked before any LLM call in practice grading and every
   mock route) and a server-wide daily cap from an environment variable
   (`LLM_DAILY_CAP`), both surfaced in `/api/meta`; a refused call
   returns a clear "demo allowance used up for today" message. At Flash
   list prices 60 calls a day is about four mock sessions and under $2
   in the worst month. The platform side is the hard stop: a separate
   DeepSeek key for the demo (revocable on its own, attributed on its
   own), a small prepaid balance, auto-recharge off. The key goes into
   the demo box's `.env`, never into a chat or a commit.
5. *Stand it up.* One t3.small (2 vCPU, 2 GB; backend RSS is about 230 MB
   with the model) on Ubuntu 24.04 with Docker; security group 80/443
   only; 2 GB swap; billing alarm at $20. TLS with Caddy in front of the
   existing nginx (automatic Let's Encrypt) via a `docker-compose.prod.yml`
   override; a domain (a cheap registrar or a free DuckDNS name). Fix
   `public/mock.js:483` to use `wss://` under HTTPS and a proxied path
   (`/ws/voice`) instead of a bare port. Voice on the box: no GPU, so
   `AUDIO_BACKEND=deepgram` behind the demo key (decided 2026-09-08; the
   text-only alternative was the fallback).
6. *Voice with an allowance* (replaced the recording, 2026-09-08). A
   `requirements-voice-cloud.txt` layer in the image (loop server + Silero
   VAD; the local stack stays out), the Deepgram backend that already
   existed, and metering that mirrors the LLM caps: minutes per key and
   per server, a hard session length, a spoken goodbye and the normal
   report when an allowance runs out, one LLM unit per interviewer turn,
   and the batch re-transcription gated to paid keys and charged by clip
   length. The measured trade-off stands in the README: Deepgram loses on
   technical-term accuracy and first-audio latency to the local stack and
   ElevenLabs, but it is free credit on one vendor. The README's first
   screen links the live URL instead of a video.

**Difficulties to expect.**
- Shell scripts and Dockerfiles with CRLF endings break inside Linux
  containers. Add `.gitattributes` (`* text=auto`, `*.sh text eol=lf`)
  before the first image build.
- huggingface.co was blocked on this machine (the model came from a GCS
  tarball). The server may download fine; if the image bakes the model,
  the local build needs the tarball path.
- The hybrid gate in CI currently exits 0 without the model; make it
  strict in the same change or the container test is the only real gate.
- `train.py` retrained today may not reproduce the August artifact
  bit-for-bit if scikit-learn moved; the results file must say which
  artifact it measured.
- A public mock spends DeepSeek per visitor. Rate limit, cap the daily
  quota per key, and watch the DeepSeek balance for the first week.

**Prepare.** AWS account with the credit; a domain or DuckDNS token;
Docker Desktop running locally; the private checkout for the two
private-input scripts; the demo `users.json` (one paid-tier demo key, one
author key); a `.env` for the server with `DEEPSEEK_API_KEY`, the demo
`DEEPGRAM_API_KEY` (its own Deepgram project, free credit, no card),
`AUDIO_BACKEND=deepgram` and the two daily caps.

**Cost and time.** About $17/month while it runs (four months of credit);
DeepSeek cents; Deepgram a few cents a day against the free credit, at
most about $1.40 at the 120-minute server cap. Three to four working days:
results files and README one day, Docker and gating one day, deploy and
TLS one day, cloud voice and its metering half a day.

**Done when.** A stranger opens the URL, runs a practice question without
a key of their own, runs a short text mock and a live voice mock on the
demo key, cannot make Claude spend or run the vendors past their daily
allowances, and every README number matches a committed file that CI
checks.

### Step 2 — A collaboration signal that is not a project

**Goal.** One substantive contribution to a repository other people
maintain, or a real co-maintainer on the Coach. It is the only item that
touches the "solo" problem, and it costs mostly waiting.

**Candidates, best first.**
1. The two undocumented ElevenLabs Speech Engine protocol rules learned in
   Phase 2 (echo the transcript's event id; end every response with an
   empty final chunk) as a documentation pull request to the public
   ElevenLabs docs repository, with the fake-websocket regression test as
   the reproducer. Real, small, already verified.
2. fastembed: the project needed a hand-placed model directory because
   the download host was blocked; an "offline model directory" example or
   a clearer error is a genuine contribution with a reproducer.
3. Deepgram SDK or docs: the lazy-final behaviour without interim results
   that cost a debugging day, documented with the measured 15 to 25 s lag.
4. The Canonical forums the author was pointed to: check the CLA
   requirement first; only take it if a concrete issue is in reach.
5. Co-maintainer on the Coach: only counts if the person actually reviews.
   Turn on branch protection with one required review and route step 1's
   pull requests through them; a name in CODEOWNERS with no review
   history reads as decoration.

**Difficulties.** Maintainer response times run weeks, so start in the
first week of step 1. Contributions get declined; keep the reproducer
public in your own repo either way, and link accepted ones from a short
"Upstream" section in the README.

**Prepare.** GitHub account with two-factor auth; a fork of the target
repo; the reproducer script committed in `tools/` or `tests/`; CLA
signature where required. Half a day per contribution plus waiting.

**Status 2026-09-08.** Reordered by the author: step 3 starts now and
this step runs alongside it rather than before it. Two things still
happen in step 3's first week because their cost is an hour and their
latency is weeks: the two drafted pull requests (candidates 1 and 2,
drafts in `data/notes/upstream/`, verified against upstream on
2026-09-07) get opened, and a camp colleague is asked to be the reviewer
of record — candidate 5 counts only with branch protection, one required
review, and step 3's pull requests actually going through them.

### Step 3 — A fine-tuned small-model grader against the sklearn grader (an experiment, not a replacement)

**Goal.** Measure whether a LoRA-fine-tuned small language model, served
by vLLM, grades closer to the Claude teacher than the distilled sklearn
grader on the same 598 gold labels and the same grouped split, and report
QWK, MAE, latency and cost side by side. One project closes fine-tuning,
Hugging Face, vLLM, GPU inference and PyTorch depth, and gives a
classical-versus-SLM distillation comparison. The sklearn grader stays the
CPU fallback whatever happens: the public clone and the t3.small have no
GPU.

**Data.** Private `grader/dataset.jsonl` (3,866 rows: `answer`,
`chunk_id`, `corpus`, `style`, `tier`, `row_id`; answers 58 words median,
164 max) joined to `grader/labels_teacher.jsonl` (598 rows: overall 1 to
10 plus four subscores) and `grader/labels_keypoints.jsonl` (per-key-point
verdicts); the rubric (question, key points) comes from the private banks
by `chunk_id`. Model input: question + key points + answer; output: the
overall score, optionally subscores and verdicts as JSON. Split:
`GroupShuffleSplit(test_size=0.2, random_state=42)` by chunk, exactly as
`train.py`, giving the same 121 held-out gold rows; plus four more seeds so
the comparison has an interval.

**Arms, fixed before running.**
- sklearn distilled grader (incumbent; numbers from step 1's results file).
- Encoder regression head (DeBERTa-v3-base or similar, full fine-tune):
  the cheap middle ground, often strong on scoring.
- Decoder LoRA: Qwen3-1.7B and Qwen3-4B (Apache-2.0, ungated) with PEFT;
  score read as the expected value over the logits of the tokens "1" to
  "10", which avoids parsing failures.
- Optional: QLoRA on a 7 to 8B model if VRAM allows, as an upper bound.

**Pre-registered rule (proposal, edit before the first run).** An SLM
enters the cascade as the local tier only if, on all five grouped splits,
its QWK beats the sklearn grader's by ≥ 0.05 with lower MAE, and vLLM
serves it at p95 ≤ 300 ms per answer on the RTX 5080. Otherwise the result
is a comparison table in the README. Serving cost is reported both ways:
GPU wall-clock here and dollars per thousand answers at a cloud L4 price.

**Steps.**
1. Environment: install Ubuntu 24.04 under WSL2, enable GPU in Docker
   Desktop, install torch with CUDA 12.8-or-newer wheels, PEFT, TRL,
   transformers, evaluate; confirm the Blackwell card (sm_120) is seen.
2. `grader/slm/prepare.py` builds the prompt records from the private
   files; `train.py` runs the arms with early stopping on a fold carved
   from the training side only; `eval.py` scores the held-out rows and
   writes `grader/slm_results.json` (numbers only, public).
3. Serve the best LoRA merged into the base with the `vllm/vllm-openai`
   image, measure latency, and add an `engine: "slm"` route behind an env
   var in `coach/grading.py` if the rule passes.
4. README section and a résumé line written from the results file.

**Difficulties.**
- vLLM has no native Windows support; everything runs in WSL2 or a
  container. Verify the vLLM wheel supports sm_120 before planning the
  serving step; if not, serve with transformers for the numbers and note
  vLLM as the intended path.
- huggingface.co was blocked here; check reachability from WSL2 first, or
  pull Qwen from a mirror.
- 598 examples overfit fast: small learning rate, two to three epochs,
  early stopping on a training-side fold, never on the held-out rows.
  Report per-`style` results, because answer styles repeat across chunks.
- 16 GB VRAM fits 4B in bf16 with LoRA and gradient checkpointing; 7 to
  8B needs 4-bit weights, and bitsandbytes on Blackwell needs a recent
  build. Flash-attention wheels may be missing; SDPA is fine.
- The weights are trained on lesson-derived answers: keep them in the
  private repo or a private model hub; publish metrics only.
- QWK on 121 rows has a wide interval; do not claim a winner on one
  split.

**Prepare.** About 40 GB of disk for models and containers; the private
checkout; a `grader/slm/` folder in the public repo with data paths
pointing at the private checkout; a Hugging Face account only if a gated
model is chosen (the plan avoids one).

**Cost and time.** $0 on the local GPU; optionally one cloud L4 hour
(about $1) for the cost line. Three to five days including the
environment.

**Status 2026-09-08 (started; taken ahead of step 2, see there).**
Environment done the same evening: Ubuntu 26.04 under WSL2 (kernel
6.18), the RTX 5080 visible with driver 595.97, `uv`-managed Python
3.12 at `~/.venvs/slm` with torch 2.11.0+cu128 (sm_120 in the compiled
arch list; bf16 matmul verified), transformers 5.16, peft 0.20, trl 1.12,
scikit-learn 1.9; huggingface.co and PyPI reachable from WSL2 (the Hub
was blocked on the Windows side, so no mirror is needed); 955 GB free;
`.wslconfig` caps WSL2 at 16 GB and 8 threads so the desktop stays
usable during runs. vLLM 0.28.0 ships a Linux wheel for 3.12 and will get
its own venv at the serving step. Private inputs confirmed in the sibling
checkout (3,866 rows, 598 teacher labels, 623 key-point labels). Docker
Desktop's WSL integration and a GPU container check followed the same
evening; the author read the rule as written and told the assistant to run
the step through without check-ins.

**Result 2026-09-08 (evening), rule PASSED.** `grader/slm/` (protocol in
its README; results in `grader/slm_results.json`, one run file per arm and
seed under `grader/slm/runs/`). Five chunk-grouped seeds, gold rows, mean ±
sd: sklearn incumbent retrained per seed QWK 0.794 ± 0.022, MAE 1.08;
DeBERTa-v3-base full fine-tune 0.792 ± 0.049, MAE 1.07 (level, noisier);
Qwen3-1.7B-Base + LoRA 0.927 ± 0.010, MAE 0.58; Qwen3-4B-Base + LoRA
0.941 ± 0.007, MAE 0.52, 95% within ±1. The 4B arm clears the bar on every
seed (+0.10 to +0.17 QWK, lower MAE each time). Served by vLLM 0.28 on the
RTX 5080 the merged 4B model answers at p50 20 ms / p95 30 ms, 81 answers/s
at eight clients, and reproduces its training-time grades (served QWK
0.932). Exploratory: adding the construction-labelled rows to the 1.7B arm
hurt (0.896 vs 0.914 on seed 42). Per tier the gain sits where lexical
features fail (heavy paraphrase, extracted text, vague answers); the top
tier is under-graded by about a point. Shipped: `coach/slm.py` and the
`SLM_URL` route (the local tier's overall grade from the SLM; verdicts,
subscores and the cascade stay with sklearn; silent server degrades),
`tests/test_slm.py`, the README section and its rendered table. Not
shipped: the weights (private, lesson-derived training answers). Found on
the way, all in the code: transformers 5 loads a checkpoint's own dtype
(fp16 DeBERTa diverged on step one until loaded fp32); vLLM 0.28 on WSL2
needs `VLLM_WSL2_ENABLE_PIN_MEMORY=1`, a C compiler, and the FlashInfer
sampler off (it JIT-compiles with nvcc). Open, carried to the loose ends:
the transfer to real spoken answers (the free-tier log), the L4 hour for a
measured cost line, and the résumé line, which is the author's.

**What differs between the arms (recorded 2026-09-09).** The four graders
differ in kind, not only in size. The table is what each one is and how
the experiment used it; the measured columns are from
`grader/slm_results.json` (five seeds, mean ± sd on the gold rows).

| | sklearn HGB | DeBERTa-v3-base | Qwen3-1.7B-Base | Qwen3-4B-Base |
| --- | --- | --- | --- | --- |
| Kind | gradient-boosted trees on 16 hand-made lexical features | encoder-only Transformer, reads both directions | decoder-only language model, reads left to right | same as 1.7B, deeper and wider |
| Parameters | thousands | 0.18 B | 1.7 B | 4 B |
| Pretraining | none | replaced-token detection ("spot the swapped word") on ~160 GB of general English, 2021 | next-token prediction on ~36 T tokens incl. code, math and STEM text, 2025 | same |
| Input | feature vector | chunk + answer, up to 512 tokens | instruction + chunk + answer + "Grade:", up to 640 tokens | same |
| Output | a number | a new regression head, trained from zero | probabilities of the ten digit tokens, read as 1 + E[digit] | same |
| What was trained | everything | all 184 M weights, fp32, eager attention | LoRA adapters (r 16, α 32, every projection), base frozen in bf16 | same |
| Training rows per seed | ≈ 3,100 (all training-side rows, teacher rows ×3) | 397–422 teacher-labelled rows | 397–422 | 397–422 |
| Best epoch (of the cap) | — | 8, 7, 8, 5, 8 of 8 | 3–4 of 4 | 3–4 of 4 |
| Train time, peak VRAM | 1 s, CPU | 59 s, 3.6 GB | 114 s, 4.6 GB | 239 s, 9.1 GB |
| p95 per answer, plain transformers | sub-ms | 12.5 ms | 35.8 ms | 67.8 ms (30 ms under vLLM) |
| QWK | 0.794 ± 0.022 | 0.792 ± 0.049 | 0.927 ± 0.010 | 0.941 ± 0.007 |

Reading. With about 400 labelled rows no model can learn the subject from
the labels; it must bring it. DeBERTa read general English and learned to
spot corrupted words; Qwen3 read textbooks and code, so it already treats
"gradients shrink through the layers" and "vanishing gradient" as the same
claim. DeBERTa lands on the sklearn grader for the reason sklearn stops
there: without subject knowledge both fall back to surface overlap. Size
mattered less than knowledge — 0.18 B to 1.7 B added 0.135 QWK, 1.7 B to
4 B added 0.014. Qwen's output head was not built from scratch (the digit
tokens after "Grade:" mean something before training starts), which is
why 400 rows suffice; the encoder's fresh regression head had to learn
what a grade is from those same rows. LoRA left the base frozen, so it
could not forget its pretraining; DeBERTa's full fine-tune moved every
weight on a tiny dataset and shows it in a seed spread twice any other
arm's. DeBERTa ran in fp32 with eager attention for practical reasons
(transformers 5 loaded its fp16 checkpoint and produced NaNs; DebertaV2
has no SDPA kernel), Qwen in bf16 with SDPA; neither choice affects the
ranking. DeBERTa's best epoch was the last one on four of five seeds, so
it was still improving at the cap; more epochs or the large variant might
add a little, and nothing in its curve suggests closing 0.14. The
trade-off the table states: the arm that could have served from a CPU box
— fastest, smallest — is the one that did not gain; the arms that gained
need a GPU.

### Step 4 — Postgres for state, then pgvector as a measured retrieval arm

**Goal.** Move the file-based state (access keys and tiers in
`users.json`, quota in `data/usage.json`, practice and mock session logs,
the plan cache) behind a store interface with an opt-in Postgres backend;
then add pgvector as a fourth arm of the retrieval harness. The vectors
alone do not justify a database: Chroma already showed an identical top-5
at a few hundred chunks, and the current numpy index is 1.2 MB. The state
does justify one, and once Postgres exists the vector column is a one-day
addition that turns "vector search" into a named production store
honestly.

**Steps.**
1. `coach/store.py`: one interface (users, usage, sessions, plan cache)
   with the file backend as the default and a Postgres backend selected by
   `DATABASE_URL` (psycopg 3 with a small pool; the stdlib server is
   threaded). Quota reservation becomes a transaction instead of a file
   rewrite.
2. Schema and `tools/migrate_to_postgres.py` that imports an existing
   `data/` folder; a dry run that prints counts.
3. Tests: the file backend in CI as now; the Postgres backend against a
   `services: postgres` container (`pgvector/pgvector:pg16`) in the
   workflow.
4. `docker-compose.yml` gains a `db` service and volume; the demo box from
   step 1 switches over after a migration and a smoke test.
5. `PgVectorRetriever` in `grader/dense_retrieval.py` and a `pgvector` arm
   in `grader/retrieval_eval.py`. Rule: identical top-5 to the numpy arm
   on sets A and B and p95 ≤ 50 ms; ship it as the vector store only when
   `DATABASE_URL` is set and the rule passes. The numpy index stays the
   default for clones with no database.

**Difficulties.** Editing `users.json` to revoke a leaked key becomes an
`UPDATE`, so the runbook changes; the public clone must keep running with
zero services, so the file backend cannot be dropped; CI gets a service
container and slower runs; local development needs Docker Desktop running;
Postgres on the t3.small adds about 100 MB of RAM (fine at 2 GB).

**Prepare.** Docker Desktop; `psycopg[binary,pool]` in a new
`requirements-db.txt`; a copy of the local `data/` folder to test the
migration; the retrieval harness from 1.7.

**Cost and time.** $0 locally; on the demo box no extra instance. Two to
three days; the pgvector arm is one of them.

### Step 5 — Kubernetes and Terraform for the two services, then tear it down

**Goal.** The same two-service compose running on a managed cluster
created by Terraform, reachable over HTTPS, recorded, then destroyed. It
ranks last because the product does not need it; it closes the K8s and
IaC lines that MLE postings list and SWE postings require.

**Prerequisites.** Step 1 (complete image, gated endpoints, a domain) and
step 4 or an accepted single replica with a persistent volume claim for
the file state.

**Choice.** GKE Autopilot over EKS: GKE's free tier covers one cluster's
management fee, and Autopilot bills per pod (about $20 to $30 a month for
a 0.5 vCPU / 1 GB backend and a tiny frontend); the EKS control plane
alone is about $73 a month and would consume the AWS credit in one
month.

**Steps.**
1. Install terraform and gcloud (winget), the GKE auth plugin for kubectl,
   and create a GCP project with billing and a $10 budget alert.
2. `infra/terraform/`: project APIs, Artifact Registry, the Autopilot
   cluster, a static IP, a DNS record, remote state in a GCS bucket with
   restricted access. API keys never enter Terraform variables or state;
   the Kubernetes Secret is created with kubectl from the server `.env`.
3. `infra/k8s/`: backend Deployment (resources 0.5 vCPU / 1 GB, readiness
   probe on `/api/meta`), frontend Deployment, Services, Ingress with a
   ManagedCertificate, a BackendConfig raising the timeout for the voice
   WebSocket, a PVC or the Cloud SQL connection from step 4.
4. Build `linux/amd64` images with buildx, push to Artifact Registry,
   apply, wait for the certificate, run the container smoke test against
   the public host, record one minute, `terraform destroy`.
5. `docs/deployment.md` runbook with the exact commands and the bill.

**Difficulties.** Managed certificates take 15 to 60 minutes and require
DNS to point at the IP first; the default Ingress timeout kills WebSockets
at 30 s; Autopilot enforces minimum pod resources; a forgotten cluster
costs money every day, so the destroy step is part of the plan, not an
afterthought; Windows image builds must target `linux/amd64`; Terraform
state is plaintext, hence no secrets in it.

**Prepare.** GCP account (a new one carries a trial credit) or the AWS
credit if EKS is chosen anyway; the domain from step 1; the images from
step 1; the `infra/` folder in the repo.

**Cost and time.** Under $30 if destroyed within a week. Three to five
days.

### Not on the roadmap, still open

- The SLM grader's transfer to real answers: every step 3 arm was trained
  and measured on the synthetic answer constructions; grade a sample of
  the free-tier log (`data/sessions/free_sessions.jsonl`) with the teacher
  and compare the sklearn and SLM grades on it before trusting the gain
  outside the constructions.
- One cloud L4 hour for a measured serving-cost line (the README's cost
  figure is an estimate from the RTX 5080 throughput).
- Re-measure the live loop's end-of-turn hold (`VOICE_HOLD_MS`, added
  2026-09-08 from one live session) on the 20 real Phase 0 recordings
  with `grader/loop_eval.py`, so the README's cut-off rate is measured
  under the shipped rule rather than the silence-only one.
- cyfang.org renews at $23.99 after the first year (bought 2026-09-08 at
  $4.99): turn auto-renew off or transfer before then.
- The grounding lever for the mock (grade against the probe's own expected
  points with chunks as hints, or compose a rubric from the top chunks):
  a product decision; when made, re-run R4 with author labels.
- MLE expansion rubric run (22 keeps, about $0.73).
- Level 1 hosted voice live session and the blind TTS preference test
  (needs the microphone and the ElevenLabs balance).
- Confidence intervals on the small-n tables (bootstrap over queries,
  probes and answers) so the reports state uncertainty instead of prose
  caveats.
