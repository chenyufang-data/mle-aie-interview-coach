# Plan — AI Mock Interview (resume-tailored, voice, downloadable report)

Status: **planned, not started** (written 2026-08-10; work resumes next
session). Decision context: the user wants this primarily for their own job
preparation, secondarily as a portfolio feature. See the assessment summary in
§1 and the open decisions in §9 before starting.

## 1. Goal and verdict

Add a Zara-style mock interview: the AI interviewer reads the user's resume
plus optional extra material (e.g. [interview_prep.md](interview_prep.md)),
walks through one experience/project (random or user-chosen), deep-dives with
adaptive technical probes, checks communication, and ends with a feedback
report. The user's voice is recorded and downloadable; so is the report.

Verdict from the assessment: **feasible and worthwhile.** Most building blocks
exist (structured LLM calls and schemas, engine routing, voice transcription,
the evaluation UI). The genuinely new piece is a multi-turn interviewer state
machine — also the portfolio upside, since every LLM call today is single-shot.
The main risk is diluting the project's "every number is measured" story;
mitigated by splitting the report into deterministic communication metrics
(computed, not judged) and rubric-grounded content grading.

## 2. Scope

In scope (phase-tagged, see §8):
- Materials input: paste resume + extra material; `.txt`/`.md` upload. (P1)
- Project detection and picker (random or chosen). (P1)
- Adaptive multi-turn interviewer with phases and hard turn caps. (P1)
- Text and voice answers (existing Web Speech API path). (P1)
- End-of-session structured report, rendered + downloadable as `.md`. (P1)
- Interviewer text-to-speech; hands-free mode. (P2)
- Continuous audio recording of the session, downloadable `.webm`. (P2)
- Computed communication metrics (WPM, filler rate, latency, balance). (P2)
- Transcript download; print-to-PDF report view. (P2)
- Tier routing, consent-aware optional logging, rubric tie-in with the prep
  doc, README/docs. (P3)

Out of scope for now: PDF resume parsing (needs `pypdf`; paste first),
server-side audio storage, multi-user session persistence, video.

## 3. User flow

1. **Setup** (`mock.html`): paste resume; optionally paste extra material;
   choose track (MLE/AIE), target level, interviewer style (neutral /
   friendly / tough), length preset (short ≈ 8 turns, standard ≈ 12); engine
   (Claude recommended, DeepSeek Flash cheaper); toggles: voice answers, read
   questions aloud, record audio. Click **Detect projects** → picker appears
   (list from the resume) with a **Random** option. **Start**.
2. **Interview**: one interviewer message at a time (optionally spoken). The
   user answers by typing or speaking (live transcript, editable before
   submit). A phase indicator shows warm-up / walkthrough / deep dive /
   behavioral / closing. **End early** is always available.
3. **Report**: rendered in-page; download buttons for report (`.md`),
   transcript (`.md`), recording (`.webm`). Nothing is stored server-side
   unless the user ticks "save this session" (P3).

## 4. Backend design (`server.py`, stateless)

Session state lives client-side (`sessionStorage`) and is sent with each
request — consistent with the existing stateless server and the frontend
contract. Three new endpoints plus one helper:

| Endpoint | Request | Response |
| --- | --- | --- |
| `POST /api/mock/projects` | `materials {resume, extra}` | `projects: [{name, one_line}]` |
| `POST /api/mock/start` | `materials, role, level, style, length, project` (name or `"random"`), `force_llm` | `plan` (hidden interview plan), `message` (first interviewer turn), `phase` |
| `POST /api/mock/turn` | `materials, plan, transcript [{phase, question, answer, seconds}]`, `answer`, `force_llm` | `message, phase, probe_target, done` |
| `POST /api/mock/report` | `materials, plan, transcript, metrics` (client-computed) | report JSON (§6) |

**Hidden interview plan** (generated once at start, steers every turn, bounds
length): `{project, opening, probe_targets: [5–8 technical points drawn from
the resume/material — decisions, trade-offs, failures, metrics, the problems
mentioned], behavioral_targets: [2–3], max_turns}`.

**Phase machine** (enforced in code, not left to the model):
`warmup (1)` → `walkthrough (1: "walk me through <project>")` →
`deepdive (4–6, adaptive)` → `behavioral (1–2)` → `closing (1)` → `done`.
The server decides the phase from turn counts and caps; the model decides the
*content* of the next question within the phase. Per turn the model receives
the plan, the transcript, and an instruction to either follow up on a weakness
in the last answer or move to the next unprobed target — one question only.

**Turn schema** (structured output, enforced):
`{interviewer_message, probe_target, rationale (hidden, ≤20 words), done}`.

**Engine**: reuse `grading_route()` — paid default → DeepSeek Flash,
`force_llm` → Claude under quota; free tier cannot run a mock (needs an LLM),
the page says so. Recommendation: **Claude by default for the interviewer and
the report** — adaptive probing (catching a contradiction with an answer two
turns ago) is where its edge over Flash shows; Flash remains the cheap option.
P3 optimization: mark the materials + plan block as a cached prompt prefix so
the growing transcript is the only uncached input per turn.

**Prompts**: a dedicated interviewer system prompt (persona, one question per
turn, never answer for the candidate, adapt to what was actually said, do not
invent facts about the resume), plus a report prompt that must quote the
candidate when citing a weakness (anti-hallucination) and grade content
against the plan's probe targets and any rubric-like material supplied.

**Logging**: off by default for mock sessions (the resume is personal data).
With the P3 opt-in, sessions go to a separate `grader/mock_sessions.jsonl`
under the existing consent policy (`"log": false` honored).

## 5. Frontend design (`public/mock.html`, `public/mock.js`)

Zero dependencies, same conventions as `app.js`. State in `sessionStorage`:
materials, plan, transcript with timings, settings. Audio blobs stay in
memory for the session (downloaded by the user; not persisted).

- **Voice answers**: reuse the `setupVoiceInput` pattern (Web Speech API,
  continuous, hidden outside secure contexts — needs HTTPS or localhost).
  Live transcript lands in the textarea; the user can edit before submit.
- **Recording (P2)**: one continuous `MediaRecorder` for the whole session
  (start at interview start, stop at end) → a single `.webm` download.
  Per-answer start/stop timestamps are kept in the transcript so a moment can
  be located in the recording. (Per-answer clips are possible but concatenating
  separate `.webm` blobs is unreliable; the continuous recorder is the robust
  choice.)
- **TTS (P2)**: `speechSynthesis.speak(message)`; pause recognition while
  speaking so the interviewer's voice is not transcribed; in hands-free mode,
  start listening when TTS ends.
- **Timings**: answer duration (question shown / TTS ended → submit) and
  response latency (→ first recognized word). Both feed the metrics.
- **Downloads**: `a[download]` with blob URLs — works in a normal browser
  tab (the app is served by nginx/Python, not a sandbox).
- **Report view**: rendered sections (§6); a print stylesheet so
  "Print → Save as PDF" produces a clean document.

## 6. Report structure

Two kinds of content, kept visibly separate:

**Computed communication metrics** (deterministic, client-side, P2):
words per minute; filler words per minute (regex list: um, uh, like, you
know, sort of, kind of, basically, actually, right?); average answer seconds;
longest/shortest answer; average response latency; answer-length balance
(coefficient of variation).

**LLM assessment** (structured output, enforced schema):
```json
{ "overall_impression": "...",
  "scores": { "communication_clarity": 7, "structure": 6,
              "technical_depth": 7, "ownership_specificity": 8, "concision": 5 },
  "per_turn": [ { "phase": "deepdive", "question": "...", "answer_summary": "...",
                  "what_was_good": ["..."], "what_was_missing": ["..."],
                  "stronger_answer": "...", "score": 6 } ],
  "red_flags": ["contradiction or vague claim, quoting the candidate"],
  "keep_doing": ["..."], "top_actions": ["...", "...", "..."] }
```
Scores 1–10; `stronger_answer` concise and interview-ready; every weakness
must quote or closely paraphrase what was said.

Rubric tie-in (P3): if the extra material contains Q&A pairs (as
`interview_prep.md` does), the report prompt treats them as the key points a
strong answer should hit for matching probes — the same rubric-grounded
grading the app already does for course questions.

## 7. Testing plan

- **Offline unit tests** (fake engine via monkeypatched `call_model`):
  phase progression and caps for both length presets; `done` set exactly at
  the cap or when the model signals closing; turn/report schema validation;
  metrics functions (WPM, filler rate, latency) on fixed transcripts;
  free-tier refusal; logging stays off without opt-in.
- **Live smoke**: one short session on DeepSeek Flash (cents; already
  authorized for DeepSeek). A Claude session needs fresh spend authorization
  (a standard session is on the order of tens of cents on Opus-tier).
- **UI**: headless Edge screenshots of setup, mid-interview, and report
  states via a sessionStorage-seeded harness (established pattern).

## 8. Phases and checklists

**Phase 1 — core loop (the bulk of the prep value)**
- [ ] `mock.html` setup form + materials paste/upload + project picker
- [ ] `/api/mock/projects`, `/start`, `/turn`, `/report` + schemas + prompts
- [ ] phase machine with caps; single-question enforcement
- [ ] text + voice answers; end-early
- [ ] report rendering + `.md` download
- [ ] offline unit tests; one Flash smoke session
- [ ] docs/frontend.md + docs/backend.md contract updates

**Phase 2 — Zara-style polish**
- [ ] interviewer TTS + hands-free mode (pause recognition while speaking)
- [ ] continuous session recording + `.webm` download
- [ ] computed communication metrics in the report
- [ ] transcript download; print-to-PDF report view

**Phase 3 — portfolio integration**
- [ ] consent-aware opt-in logging to `grader/mock_sessions.jsonl`
- [ ] prompt-cached materials prefix
- [ ] rubric tie-in with prep-doc Q&A
- [ ] README section + interview_prep additions (this is itself a talking point)

## 9. Open decisions (confirm at the start of next session)

1. Default engine for the interviewer: **Claude (recommended)** vs Flash.
2. Length presets: short ≈ 8 turns / standard ≈ 12 — adjust?
3. Recording: continuous session recording (recommended) vs per-answer clips.
4. Interviewer styles offered: neutral / friendly / tough — enough?
5. Should the mock be paid-tier only (recommended: yes, it needs an LLM) or
   also usable in single-user mode without `users.json` (yes, that already
   routes to Claude)?
6. PDF resume upload now (adds `pypdf`) or paste-only first (recommended).

## 10. Risks and mitigations

- Browser support: Web Speech API is Chromium-only in practice → note in UI;
  typing always works.
- HTTPS requirement for mic → run locally for personal prep; public demo
  waits on the domain/HTTPS item.
- Token growth over long sessions → turn caps; P3 caching; if needed,
  summarize early turns.
- TTS feeding back into the mic → pause recognition while speaking.
- Model asking several questions per turn or drifting off the resume →
  schema enforces one message; hidden plan and probe targets anchor it.
- Report hallucinating claims the candidate never made → "quote when citing a
  weakness" rule + a per-turn `answer_summary` the user can check.
- Personal data (resume) → client-side only by default; logging opt-in.

## 11. Next-session start checklist

- Say "build phase 1" and confirm the §9 decisions (or accept the
  recommendations).
- Run locally (`python server.py` with `users.json` present, open
  http://127.0.0.1:8000/mock.html) so the microphone works.
- `.env` already holds both `ANTHROPIC_API_KEY` and `DEEPSEEK_API_KEY`.
- Any Claude-engine live test needs explicit spend authorization; Flash
  smoke tests are covered by the existing DeepSeek authorization.
