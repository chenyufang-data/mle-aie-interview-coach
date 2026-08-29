# Plan — AI Mock Interview, revision 2 (voice as an evaluation problem)

Status: **planned, not started.** Revision 1 (2026-08-10) framed this as a
feature ("add voice I/O to the coach"). Revision 2 (2026-08-29) reframes it
around the evaluation problem hiding inside voice — and fixes the models,
APIs, and vendor (ElevenLabs) before any code. Read §1–§5 before starting;
§6–§9 carry the implementation detail; §10 lists the decisions still open.

## 1. The reframing

The obvious version — add voice I/O — is a feature. The version that matters
is upstream of the model: **speech-to-text will mangle exactly the words that
carry the signal.** "QWK", "LightGBM", "MAPE", "quantization", "F1",
"heteroscedasticity", "Recall@5" — if the transcript is wrong, the grader
scores a corrupted answer and the candidate is penalized for the pipeline's
failure, not their own. That is a data-quality problem, it is measurable, and
deciding whether and how to correct it (keyterm prompting, a domain lexicon,
post-hoc correction) is a far more interesting result than "I added voice".

So the plan now has a **Phase 0 that ships before any voice UI**: measure
word error rate on ML-interview vocabulary across transcription conditions,
measure the *downstream damage* to grades, and choose a correction policy
by a pre-registered decision rule (§2). Everything after that (the live
interviewer, latency, recording, the report) is built on what Phase 0 finds.

Vendor: **ElevenLabs** — Scribe v2 for STT (batch + realtime, keyterm
prompting), Flash v2.5 / v3 Conversational for TTS, and Speech Engine for the
live loop (§4). Researched 2026-08-29; facts and prices in §5.

## 2. Phase 0 — STT on technical vocabulary: the experiment

**Question.** How badly do transcription conditions corrupt ML-interview
vocabulary, how much does that corruption move the grade, and which
correction policy is worth its cost?

**Test set.**
- *Sentences* (~80–120): dense technical utterances drawn from three
  sources — rubric key points and model answers from the two question banks
  (real domain sentences), answers from `interview_prep.md`, and a curated
  hard-term set. Reference text is known exactly.
- *Lexicon* (~150–300 terms, the "signal words"): metric names (QWK, MAE,
  MAPE, F1, AUC/ROC, RMSE, Recall@5, MRR, QWK spelled out as "quadratic
  weighted kappa"), models/libraries (LightGBM, XGBoost, scikit-learn,
  HistGradientBoosting, BM25, PyTorch, DeepSeek, Claude), concepts
  (quantization, regularization, collinearity, heteroscedasticity, leakage,
  diarization, tokenization, embeddings, logits, softmax, k-fold, L1/L2,
  Ridge/Lasso, PCA, SVM, RAG), letter–number mixes (v2.5, t3.micro, int8,
  4-bit). Built from the corpora + the resume, so it is reproducible.
- *Audio, two sources, both reported*: **human** — the user reads the
  sentences once (~10–15 min; the real test distribution: accent, pace,
  hesitation); **synthetic** — ElevenLabs TTS reads the same sentences in
  2–3 voices (scalable paired audio, but biased: TTS pronounces terms
  cleanly). Decisions are made on the human set; synthetic shows scale.

**Conditions.**
1. Browser Web Speech API (today's path; no vocabulary control) — baseline.
2. Scribe v2 batch, no keyterms.
3. Scribe v2 batch + keyterm prompting with the full lexicon (≤1000 terms).
4. Condition 2 + post-hoc lexicon correction (fuzzy/phonetic match of
   transcript tokens to lexicon entries) — measures whether cheap
   post-processing closes the gap, **and its false-correction rate**.
5. Scribe v2 Realtime + 50 keyterms (the live constraint: 50 terms × 20
   chars) — and therefore a *policy for choosing the 50*: per-session, from
   the target role + chosen project (§7a). The keyterms are tailored per
   interview the same way the questions are.
6. **Local Whisper** (faster-whisper `large-v3-turbo` on the user's GPU, with
   `initial_prompt` vocabulary biasing — Whisper's weak form of keyterm
   prompting). This is the offline fallback stack's STT (§4a), so the
   experiment answers "is local good enough?" with the same numbers.

**Metrics.**
- **WER** — standard word-level Levenshtein after documented normalization
  (case, punctuation, number/acronym rules).
- **Term error rate (TER)** — over lexicon-term occurrences in the
  reference, the fraction not recovered; reported strict (exact) and lenient
  (canonicalized: "light gbm" ≡ "LightGBM"), with a per-term breakdown of
  *how* terms fail ("QWK" → "quick"; "MAPE" → "map").
- **Downstream damage — the number that matters**: grade the reference text
  and the transcript with the same grader (distilled model: free,
  deterministic; plus one LLM judge) and report |score(ref) − score(hyp)|,
  the share of answers whose grade moves ≥1 point *from transcription
  alone*, and per-key-point verdict flips from the kp classifier. This is
  "penalized for the pipeline's failure", quantified.

**Pre-registered decision rule.** Ship the cheapest condition whose
downstream ≥1-point damage rate is ≤5% and lenient TER is ≤3% on the human
set; if none qualifies, ship the best and state the residual damage in the
report to the user. Costs are reported next to every condition.

**Deliverables.** `grader/stt_eval.py` (evidence-script style: dry-run cost
estimate, `--confirm`), `grader/stt_lexicon.json`, results table committed,
README section. Estimated cost: well under $5 (Scribe batch $0.22/hr on
<1 hr of audio; ~10k TTS characters ≈ $0.50–1.00).

## 3. Latency budget and turn-taking

Natural conversational gaps are ~200–600 ms; beyond ~1 s the exchange feels
laggy, beyond ~2 s awkward. Budget from **end of user speech → first
interviewer audio**, target p50 ≤ 1.0 s, p95 ≤ 1.8 s, measured in the browser
(last-user-audio → first-agent-audio) and reported with p50/p95, not means.

| Stage | Typical | Design choice |
| --- | --- | --- |
| End-of-turn detection | 300–700 ms | **Patient** turn eagerness: candidates pause to think; cutting a thinking pause is the worst possible interviewer behaviour. Turn timeout 10–30 s. |
| STT finalization (Scribe v2 Realtime) | ~150 ms (30–80 ms inside Agents) | vendor-managed |
| LLM first token | 300–900 ms | stream tokens; keep interviewer turns to 1–2 sentences |
| TTS first byte (Flash v2.5) | ~75–150 ms | stream sentence-by-sentence |
| Network + playback | 100–200 ms | WebRTC via the vendor SDK |

A design consequence: **a JSON structured-output turn cannot be streamed to
TTS.** The interviewer's spoken text must stream as plain text. So the turn
protocol becomes: the model emits the spoken question first, then a trailer
after a marker (`\n---\n{json}` with `probe_target`, `rationale`) that the
server strips before TTS and parses afterwards; phase decisions stay
server-side and deterministic. Filler audio ("mm-hm") only if the LLM exceeds
a soft timeout (~3 s).

**Interruptions.**
- *Candidate interrupts the interviewer*: allowed. The transcript must
  record only what was actually spoken — the state machine needs the spoken
  prefix; if the question was cut before its core clause, re-ask briefly
  rather than proceed. (Whether Speech Engine reports the spoken prefix is an
  open verification item, §10.)
- *Interviewer interrupts the candidate*: never by barge-in. A gentle
  time-box ("let's move on") is a turn-timeout policy, not an interruption.

## 4. Architecture decision: who runs the loop

Three options were weighed:

| Option | What it gives | What it costs | Verdict |
| --- | --- | --- | --- |
| **Hosted ElevenAgents** (agent configured on their platform, optional custom-LLM URL) | Fastest to a demo; telephony; conversation history + audio retrievable | Conversation logic lives in their config; the phase machine would need a custom-LLM server anyway; least loop control | not chosen |
| **ElevenLabs Speech Engine** (bring-your-own agent: they do STT, turn-taking, barge-in, TTS, playback; your server receives transcripts + history and streams text back; Python and JS SDKs; `engine.serve()` or ASGI integration) | Solved turn-taking and barge-in; the interview state machine, prompts, grading, and report stay entirely in this project; built for exactly this shape | $0.08/min; a browser SDK (first external frontend dependency); some abstraction unknowns (keyterms per session, spoken prefix on interruption) | **chosen for the live loop** |
| **DIY** (Scribe v2 Realtime WebSocket + own VAD/endpointing + LLM streaming + TTS WebSocket + own barge-in) | Full control and full instrumentation; cheapest per minute (~$0.25 per 15-min session) | Turn-taking and barge-in are the hard part; weeks of tuning | kept as the measured fallback path; Phase 0 is DIY by nature |

Recommendation: **Speech Engine for the live conversation; own server for
everything that is the product** (state machine, prompts, grading, report).
Latency is still measured end-to-end from the browser regardless, so the
budget in §3 is a measured number, not a vendor claim. The existing browser
Web Speech + `speechSynthesis` path stays as the free/offline fallback and
is condition 1 of the experiment.

### 4a. Local mode — the offline fallback for the audio models

Ollama cannot serve this: it is an LLM runtime and has no STT/TTS models.
The right shape is the same idea one layer over — a local, OpenAI-compatible
**audio** server — plus the LLM the project already runs on Ollama:

| Role | Local choice | Why | Zero-install fallback |
| --- | --- | --- | --- |
| STT | **faster-whisper** `large-v3-turbo` (CTranslate2; realtime on an RTX 5080; word timestamps; `initial_prompt` for vocabulary bias) | best accuracy/speed trade-off among open models; NVIDIA Parakeet-TDT is the faster English-only alternative | browser Web Speech API (note: Chromium sends audio to Google — it is *free*, not *local*) |
| TTS | **Kokoro-82M** (Apache-2.0; near-realtime even on CPU; natural enough for an interviewer) | Piper is faster but robotic; XTTS/Chatterbox heavier, only needed for voice cloning | browser `speechSynthesis` |
| Endpointing / VAD | **Silero VAD** (tiny, MIT) | standard; drives end-of-turn and barge-in in the DIY loop | fixed silence timer |
| LLM | **DeepSeek V4 Flash** (cloud; already the paid-tier workhorse; ≈ $0.01 per session) | far better adaptive probing than a local 8–14B model, at negligible cost; needs streaming and thinking-off for low first-token latency (verification item, §10) | Ollama (`--ollama`) — the true offline fallback when there is no internet |
| Packaging | **Speaches** (formerly faster-whisper-server): one Docker image exposing OpenAI-style `/v1/audio/transcriptions` and `/v1/audio/speech` backed by faster-whisper + Kokoro | one container, one env var (`AUDIO_BASE_URL`); the server code talks to ElevenLabs or Speaches through the same two calls | direct Python libs if Docker is unwanted |

So "local mode" is precisely **local audio + cloud LLM**: the microphone
audio, the transcripts, and the recording never leave the machine; only the
text of the conversation (which includes resume content) goes to DeepSeek.
Cost ≈ $0.01 per session (GPU time otherwise). Latency is comparable to the
cloud path (Whisper turbo ≈ 200–400 ms per utterance, Kokoro first audio
≈ 150–300 ms; DeepSeek first token ≈ 300–800 ms streamed, with thinking
disabled for turns — with V4's hybrid thinking left on, a turn can add
seconds, which is why it must be off for conversation and on only for the
report). What local mode gives up: Speech Engine's tuned turn-taking and
barge-in (the DIY loop is ours to tune), and Scribe's keyterm prompting
(Whisper's `initial_prompt` is weaker — condition 6 measures how much).

Fallback chains, in order — audio: ElevenLabs → local server → browser API
→ text; LLM: Claude → DeepSeek Flash → Ollama. Every level is measured in
Phase 0 rather than assumed.

A design element that falls out of the two-transcript reality: record the
session locally (`MediaRecorder`) and, after the interview, **re-transcribe
the recording with Scribe v2 batch and the full 1000-term lexicon**. The
report then grades the *final* transcript (what you said), shows the *live*
transcript (what the interviewer heard), and quantifies the difference —
"transcription cost you N points" — as a per-session instance of Phase 0.

## 5. Models and APIs — the complete inventory

| Role | Choice | Fallback / alternative | Cost basis | Status |
| --- | --- | --- | --- | --- |
| Interviewer LLM (live turns, streamed) | Claude `claude-opus-4-8` | DeepSeek `deepseek-v4-flash` (paid-tier default per existing routing) | per token; Flash ≈ 1/200th | keys in `.env` |
| Report / content grading LLM | Claude (teacher) or Flash per routing | distilled grader for rubric-shaped parts | per token | existing |
| Rubric grading, offline | distilled sklearn `grader/model.joblib` (score + kp classifier) | — | zero | existing |
| Question/rubric retrieval | BM25 (`retrieval.py`) | — | zero | existing |
| STT, live | Scribe v2 Realtime, ≤50 keyterms (inside Speech Engine) | browser Web Speech API (free, no vocabulary control) | $0.39/hr direct, or within $0.08/min Speech Engine | new |
| STT, batch (Phase 0 + post-session re-transcription) | Scribe v2 batch, ≤1000 keyterms | — | $0.22/hr + $0.05/hr keyterms | new |
| TTS, live | Flash v2.5 (~75 ms) via Speech Engine | v3 Conversational (quality, same price); browser `speechSynthesis` (free fallback) | $0.05 / 1k chars | new |
| TTS, synthetic test audio | Multilingual v2 or Eleven v3 | — | $0.10 / 1k chars | new (Phase 0) |
| Turn-taking, barge-in, playback | ElevenLabs Speech Engine (WebRTC browser SDK + Python server SDK) | DIY VAD + WebSockets | $0.08/min (burst $0.16) | new |
| Session audio | browser `MediaRecorder`, local download | — | zero | new |
| Lexicon correction | own fuzzy matcher (difflib / rapidfuzz) | — | zero | new (Phase 0) |

ElevenLabs plan facts (2026-08-29): Free — pay-as-you-go, 15 Speech Engine
minutes, no card; Starter $6/mo — 10k TTS chars, 4.5 hrs STT, 75 minutes;
Creator $22/mo — 220k chars, 100 hrs STT, 275 minutes. LLM usage is billed
separately from the per-minute rate. Starter covers Phase 0 plus a handful
of mock sessions; Creator covers regular practice.

### 5a. Cost estimates (ElevenLabs list prices, 2026-08-29)

**Phase 0, one-time.** Human set ≈ 15 min of audio; synthetic set = 100
sentences × 3 voices ≈ 40 min of audio and ≈ 36k TTS characters.

| Item | Quantity | Rate | Cost |
| --- | --- | --- | --- |
| Synthetic test audio (Multilingual v2 / v3 quality voices) | 36k chars | $0.10 / 1k | $3.60 |
| Scribe v2 batch, no keyterms | 0.92 hr | $0.22 / hr | $0.20 |
| Scribe v2 batch + keyterms | 0.92 hr | $0.27 / hr | $0.25 |
| Scribe v2 Realtime + keyterms | 0.92 hr | ≈ $0.44 / hr | $0.40 |
| Consistency re-runs | ×2 of the STT rows | | ≈ $0.85 |
| **Total** | | | **≈ $5.30 pay-as-you-go**; ≈ $8.60 on Starter ($6 + 26k chars over the included 10k) |

**Per mock session** (12 turns ≈ 15 min; short 8-turn ≈ 10 min in brackets).

| Path | STT / TTS / loop | LLM | Re-transcription | Total |
| --- | --- | --- | --- | --- |
| Speech Engine | $1.20 [$0.80] | Flash ≈ $0.01; Claude ≈ $0.30 | $0.07 | **≈ $1.30–1.60** [$0.90–1.15] |
| DIY cloud (Scribe Realtime + Flash v2.5 TTS) | $0.11 + $0.13 | same | $0.07 | ≈ $0.30–0.60 |
| Local audio mode (§4a) | $0 | Flash ≈ $0.01 (Ollama $0) | local Whisper $0 | **≈ $0.01** |

**Monthly, Speech Engine path** (plan minutes are the binding constraint;
STT hours in every plan cover re-transcription many times over; LLM extra).

| Sessions / month | Minutes | Best plan | ≈ Cost |
| --- | --- | --- | --- |
| 5 | 75 | Starter ($6, 75 min included) | $6 |
| 10 | 150 | Starter + 75 overage min × $0.08 | $12 |
| 18 | 270 | Creator ($22, 275 min; $11 first month) | $22 |
| 30 | 450 | Creator + 175 × $0.08 | $36 |
| 60 | 900 | Creator + 625 × $0.08 = $72, or Pro $99 (1,238 min) | $72–99 |

Reading: for personal prep at a few sessions a week, **Creator at $22/month
is the realistic ceiling**; Phase 0 plus the first sessions fit in Starter.
Local mode makes unlimited practice free once the experiment says it is
good enough.

## 6. User flow (unchanged in shape)

1. **Setup** (`mock.html`): paste resume + optional material; track, level,
   interviewer style (neutral / friendly / tough), length (short ≈ 8 turns,
   standard ≈ 12); engine; toggles: live voice (Speech Engine) / browser
   voice / text; record audio. **Detect roles** → pick a target role (or
   paste a job description) → pick a supporting project or "interviewer's
   choice" (§7a).
2. **Interview**: one interviewer message at a time; phase indicator;
   **End early** always available.
3. **Report**: rendered in-page; downloads for report (`.md`), live and
   final transcripts (`.md`), recording (`.webm`); nothing stored server-side
   without the opt-in.

## 7. Backend design (`server.py` + a Speech Engine sidecar)

Session state stays client-side and is sent per request (stateless server,
consistent with the frontend contract).

| Endpoint | Purpose |
| --- | --- |
| `POST /api/mock/roles` | propose target roles from the resume / job description, each with supporting projects and probe themes (§7a) |
| `POST /api/mock/start` | build the hidden interview plan; first turn |
| `POST /api/mock/turn` | next turn (text path); same logic the sidecar calls |
| `POST /api/mock/report` | final report from transcript(s) + client metrics |
| `POST /api/mock/keyterms` | choose the ≤50 realtime keyterms for this session from the resume + project (policy from Phase 0) |
| `POST /api/mock/transcribe` | post-session batch re-transcription of the uploaded recording with the full lexicon |

### 7a. Role-driven sessions and the 50-keyterm policy

Instead of "pick a project at random", the session is organized the way a
real interview is — **around a target role**:

1. `POST /api/mock/roles`: from the resume (and an optional pasted job
   description, which overrides), the LLM proposes 3–5 plausible target
   roles ("fraud-detection MLE", "LLM application engineer", "forecasting
   data scientist"…), each with the resume projects that support it, ranked
   by relevance, and the role's typical probe themes (fraud detection: class
   imbalance, precision/recall at a threshold, label delay, leakage, drift).
2. The user picks the role, then a supporting project — or "interviewer's
   choice". The interviewer persona is set by the role (company type,
   seniority), not just by tone.
3. The hidden plan draws probe targets from the chosen project **and** the
   role's themes, and keeps a small "wander" budget (1–2 probes from adjacent
   areas) so the mock is not perfectly predictable — real interviewers drift.
4. Where a role theme matches a question-bank chunk (BM25 over
   `rag_ml`/`rag_ai`), that chunk's rubric seeds the probe and grades the
   answer — rubric-grounded grading for the technical part of the mock, the
   same mechanism the app already has.

The same choice fixes the **50-keyterm selection** for Scribe Realtime.
Candidate terms = union(role vocabulary, chosen-project vocabulary, resume
terms, lexicon). Rank by: (a) Phase 0 measured failure rate *without*
keyterms — the 50 slots go to terms that demonstrably need help, not to
words Scribe already gets right; (b) presence in the resume/project; (c)
rarity in general English. Multi-word terms count once but must fit 20
characters, so canonical short forms are kept in the lexicon. The policy is
itself evaluated in Phase 0 (condition 5 vs a naive "first 50" baseline).

**Hidden interview plan** (once, at start): `{role, project, opening,
probe_targets: [5–8 technical points from the project + role themes],
rubric_chunks: [...], behavioral_targets, wander_budget, max_turns}`. **Phase machine** enforced in code: warm-up (1) →
walkthrough (1) → deep dive (4–6, adaptive) → behavioral (1–2) → closing (1)
→ done. The model chooses content within the phase: follow up on a weakness
in the last answer or move to the next unprobed target — one question only.

**Turn protocol**: spoken text streamed first; trailer `\n---\n{json}`
stripped before TTS, parsed after. **Speech Engine sidecar**: a small ASGI
process using the Python SDK (`on_transcript` → the same turn function →
`session.send_response(stream)`), so the text path and the voice path share
one state machine. **Engine routing** reuses `grading_route()`; free tier
cannot run a mock (needs an LLM). **Logging** off by default (resumes are
personal data); P3 opt-in writes `grader/mock_sessions.jsonl` under the
existing consent policy.

## 8. Frontend (`public/mock.html`, `public/mock.js`) and the report

Vanilla JS as elsewhere. The ElevenLabs browser SDK is loaded only on the
mock page and only when live voice is enabled — the documented exception to
the zero-dependency rule. Browser voice (Web Speech + `speechSynthesis`)
and text remain dependency-free. `MediaRecorder` runs for the whole session
→ one `.webm`; per-answer timestamps in the transcript locate moments.
Timings captured: answer duration, response latency, and the §3
last-user-audio → first-agent-audio measurement.

**Report** — three visibly separate blocks:
- *Computed communication metrics* (deterministic): words per minute, filler
  words per minute, average answer seconds, longest/shortest, response
  latency, answer-length balance.
- *Transcription quality* (from the two transcripts): term corrections
  applied, WER between live and final, grade difference attributable to
  transcription.
- *LLM assessment* (enforced schema): overall impression; scores 1–10 for
  communication clarity, structure, technical depth, ownership/specificity,
  concision; per-turn `{question, answer_summary, what_was_good,
  what_was_missing, stronger_answer, score}`; red flags that **quote** the
  candidate; keep-doing; top three actions. Rubric tie-in (P3): Q&A pairs in
  the supplied material act as key points for matching probes.

## 9. Phases and checklists

**Phase 0 — STT evaluation (ships first; the centerpiece)**
- [ ] `grader/stt_lexicon.json` built from corpora + resume; sentence set
- [ ] user records the human set (~15 min); synthetic set via TTS
- [ ] `grader/stt_eval.py`: conditions 1–5, WER/TER/downstream damage, costs
- [ ] keyterm selection policy for the 50-term realtime limit
- [ ] results table + README section; decision recorded

**Phase 1 — core loop, text + browser voice**
- [ ] `mock.html` setup, materials, project picker
- [ ] `/api/mock/projects|start|turn|report`, plan, phase machine, turn protocol
- [ ] report rendering + downloads; offline unit tests (fake engine); one
      Flash smoke session

**Phase 2 — live loop on Speech Engine**
- [ ] Python sidecar + browser SDK; token endpoint; patient turn-taking
- [ ] per-session keyterms; interruption policy (verify spoken-prefix)
- [ ] latency instrumentation, p50/p95 reported
- [ ] local recording + post-session re-transcription; two-transcript report

**Phase 3 — polish and portfolio**
- [ ] consent-aware opt-in logging; prompt-cached materials prefix
- [ ] rubric tie-in with prep-doc Q&A; README + interview_prep additions

## 10. Open decisions and verification items

Decisions:
1. ElevenLabs plan: Starter ($6) is enough to start.
2. The human test set: the user records ~100 sentences (~15 min).
3. Local fallback stack (§4a): Speaches container (faster-whisper + Kokoro)
   vs direct Python libraries — recommendation: Speaches, one container.
   Condition 6 (local Whisper) is now in the experiment by default.
4. Accept the ElevenLabs browser SDK as the mock page's one dependency.
5. Default interviewer LLM: Claude (recommended) vs Flash.
6. Decision-rule thresholds (≤5% damage, ≤3% lenient TER) — confirm.

Verify against DeepSeek docs before Phase 1: the streaming (`stream: true`,
SSE) path for the OpenAI-compatible endpoint — `call_deepseek` is currently
non-streaming — and the parameter that disables V4's hybrid thinking per
request, so interviewer turns get a fast first token while the report keeps
thinking on. Measure first-token latency both ways.

Verify against ElevenLabs docs before Phase 2 (not answered by the docs read
on 2026-08-29): whether Speech Engine accepts per-session STT keyterms;
whether the server learns the spoken prefix on interruption; whether the
TTS voice/model is selectable per session; whether session audio is
retrievable (local recording makes this non-blocking).

## 11. Risks and mitigations

- Transcription corrupts grades → Phase 0 measures it; the report shows it;
  the final transcript uses the full lexicon.
- Keyterm limit (50 realtime) → per-session selection policy, evaluated.
- Endpointing cuts thinking pauses → patient eagerness, long turn timeout,
  measured barge-in and cut-off rates.
- Vendor abstraction leaks (keyterms, spoken prefix) → verification list;
  DIY path kept as fallback.
- Browser support: Web Speech is Chromium-only; Speech Engine SDK needs
  WebRTC → typing always works.
- HTTPS for the microphone → local runs for prep; public demo waits on the
  domain/HTTPS item.
- Personal data (resume, voice) → client-side by default; logging opt-in;
  ElevenLabs sees the audio in the live path (state this in the UI).

## 12. Next-session start checklist

- Start with Phase 0: say "build phase 0"; confirm the §10 decisions or
  accept the recommendations. Have an ElevenLabs API key ready (Starter
  plan or credits) — it goes in `.env` as `ELEVENLABS_API_KEY`.
- Phase 0 needs ~15 minutes of the user's time to record the human set.
- `.env` already holds `ANTHROPIC_API_KEY` and `DEEPSEEK_API_KEY`; DeepSeek
  spend is authorized; any Claude live test needs fresh authorization;
  ElevenLabs spend needs authorization once the key exists.

## Sources (ElevenLabs, read 2026-08-29)

- API pricing: https://elevenlabs.io/pricing/api — Agents pricing:
  https://elevenlabs.io/pricing/agents
- Scribe v2 launch and features: https://elevenlabs.io/blog/introducing-scribe-v2 ;
  https://elevenlabs.io/blog/scribe-v2-just-got-an-upgrade ;
  STT capabilities: https://elevenlabs.io/docs/overview/capabilities/speech-to-text ;
  batch endpoint reference: https://elevenlabs.io/docs/api-reference/speech-to-text/convert
- Speech Engine: https://elevenlabs.io/docs/overview/capabilities/speech-engine ;
  quickstart: https://elevenlabs.io/docs/eleven-api/guides/cookbooks/speech-engine
- Conversation flow (turn-taking, interruptions):
  https://elevenlabs.io/docs/eleven-agents/customization/conversation-flow
- TTS models: https://elevenlabs.io/docs/overview/models
