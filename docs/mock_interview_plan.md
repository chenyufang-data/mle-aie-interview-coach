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

**Status (2026-08-30).** Everything above is built and has run end to end on
the synthetic set (`synth_matilda`, 88 items, 16.9 min; ≈ $0.9 of credits for
TTS + every Scribe condition). Results: `docs/stt_evaluation.md` and the README
section. Headline numbers (synthetic audio, so optimistic): Scribe batch loses
7.0% of lexicon terms (lenient) at 4.1% WER; the full-lexicon keyterms cut that
to 2.3% (strict 12.4% → 3.3%); Realtime + 50 keyterms 9.0% → 5.7–6.0%; local
Whisper turbo 14.7% (13.0% with `initial_prompt`); post-hoc fuzzy correction is a
**negative result** (73% of its rewrites were false); Scribe Realtime finalizes
in 0.11 s p50 / 0.25 s p95. Downstream damage under the distilled grader is
15–30% of answers moved ≥ 1 point (5–15% when both sides are normalized) — and
no condition meets the pre-registered rule on this set. Two things the
experiment surfaced that were not in the plan: the **grader itself moves 50% of
grades when the untouched reference is merely lowercased and stripped of
punctuation** (surface-form brittleness the mock report must neutralize by
grading normalized text on both sides, or by hardening the grader), and the
**Flash judge's noise floor is 35%** moved ≥ 1 on identical text, so it cannot
measure damage at this scale.

**Human set run and decision recorded (2026-08-30).** 88/88 items read once at
natural pace (26.5 min; run cost ≈ $0.56, Phase 0 total ≈ $1.5 of $10). Real
speech doubles every error rate vs TTS audio. Web Speech — the app's old voice
path — loses 29.4% of lexicon terms (WER 19.9%): disqualified by measurement.
Scribe batch + full keyterms: 3.0% lenient / 4.0% strict. Realtime: 15.4% →
11.7% (naive 50) → 9.7% (policy 50) — the policy beats naive on human audio.
Post-hoc fix stays dropped (63% false). Whisper: lowest grade damage (fluent
smoothing suits the brittle grader) but 13.4–16.1% term loss. Flash judge noise
floor 50% on these answers. No condition met the pre-registered rule, so per
its fallback the best ships with residual damage stated: **final transcript =
Scribe v2 batch + full-lexicon keyterms; live transcript = Scribe v2 Realtime +
policy-50 keyterms; local Whisper offline, re-transcribed via batch + keyterms
when online; the report grades normalized text on both sides and shows both
transcripts.** Residual: ~3% of terms lost in the final transcript; 25–35% of
spoken answers grade ≥ 1 point off their written reference (mixes true damage
with speaking-vs-writing deviation and grader surface-form brittleness —
hardening the grader on style is a follow-up for Phase 1).

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

Recommendation: **ElevenLabs Speech Engine is Level 1 — the reference loop
and the premium option; own server for everything that is the product**
(state machine, prompts, grading, report). The expected *main* usage is
Level 2 (local audio + DeepSeek Flash, §4a) **if** the equivalence test in
Phase 2 shows it performs like Level 1; Level 1 then remains the benchmark
and the occasional premium session. Latency is measured end-to-end from the
browser on every level, so the budget in §3 is a measured number, not a
vendor claim.

### 4a. Fallback levels for the ElevenLabs stack

Degradation is organized in **levels**, each a complete working
configuration, not per component:

| Level | Audio (STT / TTS / turn-taking) | LLM | Needs | ≈ Cost / session | Role |
| --- | --- | --- | --- | --- | --- |
| **1** | ElevenLabs: Scribe v2 Realtime (+50 keyterms), Flash v2.5 TTS, Speech Engine loop | Claude or DeepSeek Flash | internet, ElevenLabs credits | $1.30–1.60 | reference loop; premium sessions |
| **2** | Local audio server (Speaches: faster-whisper turbo + Kokoro) + Silero VAD, DIY loop | DeepSeek Flash | GPU; internet for the LLM only | $0.01 | **expected main usage** if the equivalence test passes |
| **3** | same local audio | Ollama | nothing — fully offline | $0 | offline |
| degraded | browser Web Speech + `speechSynthesis`, or text only | any | a browser | — | last resort |

All levels share one loop implementation and one turn function; only the
endpoints differ (`AUDIO_BACKEND`, `LLM_ENGINE`). The *final* transcript
(the one the report grades) is where accuracy matters most, and Scribe
batch with the 1000-term lexicon costs ≈ $0.07 per session — so a Level-2
session with a Level-1 final transcript is the cheapest high-accuracy
combination whenever the machine is online; Level 3 uses local Whisper for
the final transcript too.

Why the local audio layer is not "Ollama for audio": Ollama is an LLM
runtime with no STT/TTS models. The right shape is the same idea one layer
over — a local, OpenAI-compatible **audio** server. The components of
Levels 2–3:

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

Whether Level 2 is "as good as" Level 1 is not assumed: Phase 0 measures
the STT half, and the Phase 2 **equivalence test** (§9) measures the whole
loop — latency, cut-offs, barge-in, live term accuracy, TTS preference —
under a pre-registered rule.

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
| Synthetic test audio (Multilingual v2 / v3 quality voices) | 36k chars | $0.10 / 1k | $3.60 — *measured 2026-08-30: Flash v2.5 on pay-as-you-go bills ≈ 0.1 credit/char, so one 88-item voice cost $0.38* |
| Scribe v2 batch, no keyterms | 0.92 hr | $0.22 / hr | $0.20 |
| Scribe v2 batch + keyterms | 0.92 hr | $0.27 / hr (SDK: +20% surcharge; >100 keyterms ⇒ 20 s minimum per request) | $0.25 |
| Scribe v2 Realtime + keyterms | 0.92 hr | ≈ $0.44 / hr (measured: 410 credits ≈ $0.11 per 17-min pass, i.e. ≈ $0.39/hr, keyterms free) | $0.40 |
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

Reading: for personal prep at a few sessions a week on Level 1, **Creator at
$22/month is the realistic ceiling**; Phase 0 plus the first sessions fit in
Starter. Level 2 makes unlimited practice essentially free once the
experiments say it is good enough.

### 5b. Plan or pay-as-you-go?

Plans are prepaid bundles at a discount; overage and the Free plan's
pay-as-you-go use the same list rates. Starter's $6 buys ≈ $8 of list value
(75 Speech Engine minutes ≈ $6, 4.5 STT hours ≈ $1, 10k TTS chars ≈ $1) —
but only if all three are used. So the answer follows the usage level:

| If main usage is… | ElevenLabs consumption | Better choice |
| --- | --- | --- |
| Level 1 (Speech Engine sessions) | ≥ 75 min/month | Starter; Creator above ~275 min |
| Level 2 (local audio + DeepSeek) | Phase 0 once (≈ $5); optional Scribe final transcripts ($0.07/session); occasional Level-1 benchmark sessions ($1.27 each) | **Pay-as-you-go** — no monthly fee; a typical month is under $3 |

Break-even is one line: Starter pays for itself only in a month with more
than 75 Speech Engine minutes (75 × $0.08 = $6). Since the expected main
usage is Level 2, **start on pay-as-you-go** and move to Starter/Creator
only in a heavy Level-1 month; plans are monthly and cancellable, so the
hedge is cheap. Verification item: confirm that Free-plan pay-as-you-go
covers Scribe, TTS, and Speech Engine minutes beyond the 15 included (§10).

### 5c. One live session, by setup

Assumptions (standard session, 12 turns ≈ 15 min; short 8-turn ≈ 10 min
scales to ~⅔): per turn ≈ 4k input tokens (persona + plan + resume + growing
transcript) and ~150 output; start (roles + plan) ≈ 7k in / 1.5k out;
report ≈ 8.5k in / ~4k out incl. thinking. Prices: Claude opus-4-8 $5 / $25
per M (cache reads ≈ 10%); DeepSeek Flash $0.14 / $0.28 per M; Speech Engine
$0.08/min; Scribe Realtime + keyterms ≈ $0.44/hr; Flash v2.5 TTS $0.05 per
1k chars (≈ 2.5k chars of interviewer speech per session); Scribe batch
final transcript with keyterms ≈ $0.07 per session.

LLM component: all-Claude ≈ $0.50 (≈ $0.34 with prompt caching); Flash
turns + Claude report ≈ $0.15; all-Flash ≈ $0.01; Ollama $0.

| Setup | Audio + loop | LLM | Final transcript | ≈ per session |
| --- | --- | --- | --- | --- |
| A. Level 1, all Claude | Speech Engine $1.20 | $0.50 | Scribe $0.07 | **$1.77** (≈ $1.60 cached) |
| B. Level 1, recommended (Flash turns, Claude report) | $1.20 | $0.15 | $0.07 | **$1.42** |
| C. Level 1, all Flash | $1.20 | $0.01 | $0.07 | $1.28 |
| D. "Level 1.5": DIY loop on cloud audio (Scribe Realtime + Flash TTS), Flash turns, Claude report | $0.11 + $0.13 | $0.15 | $0.07 | **$0.46** |
| E. Level 2 (local audio), Flash turns, Claude report, Scribe final transcript | $0 | $0.15 | $0.07 | **$0.22** |
| F. Level 2, Flash turns, Claude report, local final transcript | $0 | $0.15 | $0 | $0.15 |
| G. Level 2, all Flash, local final transcript | $0 | $0.01 | $0 | $0.01 |
| H. Level 3, fully offline (Ollama) | $0 | $0 | $0 | $0 |
| I. Text-only mock (no voice), Flash turns, Claude report | — | $0.15 | — | $0.15 |

Readings: the Speech Engine minute rate is ~85% of a Level 1 session — the
LLM is never the cost driver unless everything runs on Claude. Setup D falls
out of the Level 2 build almost for free (the DIY loop with cloud audio
endpoints) and is ~3× cheaper than Speech Engine for the same models, at
the price of our own turn-taking. At 12 sessions a month: B ≈ $17, D ≈ $5.5,
E ≈ $2.6, G ≈ $0.12. The current $10 ElevenLabs balance covers Phase 0
(≈ $5) plus roughly 3–4 Level 1 sessions, or Phase 0 plus ~70 sessions of
Scribe final transcripts on Level 2.

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

### 7b. Job-description-driven interviewer and the 面经 question bank

Scope decision (2026-08-29): the mock focuses on the **experience / project
deep-dive round**. Two inputs make it specific to a real target:

**The job description → the interviewer.** A pasted JD is turned by the LLM
into a structured *role profile*: `{title, level, domain, must_have_skills,
responsibilities, stack, what_they_evaluate}`. That profile (a) writes the
interviewer's persona prompt — what this team cares about, which of your
projects matter to them, the seniority bar; (b) supplies the probe themes;
(c) seeds the per-session keyterms. Without a JD, the role picker (§7a)
plays the same part from the resume alone.

**A third question bank from 面经.** Real interview-experience posts
(一亩三分地 and similar) hand-collected by the user — no crawler — become
`rag_exp/`, a bank in the same chunk schema the app already serves
(the first source already exists: `data/interview_exp/`, gitignored because it
names a third party, holds the live interview questions the user gathers —
a question-frequency log with columns question / frequency / interviewee /
company / industry / round, and a per-interview record with JD links and
rounds; it will keep growing. The ingest reads `.xlsx` via openpyxl and
uses the frequency column to rank probes)
(`id / interview{question, model_answer, key_points, common_mistakes,
followups} / metadata{company, role_family, round, date, language, source}`).
Ingest pipeline `grader/ingest_questions.py`: paste file → normalize and
translate (questions in English for the interview, original text kept) →
dedupe against all banks by BM25 → **rubric generation with the Claude
teacher** (key points, model answer, follow-ups — the part 面经 posts never
contain, and the part grading depends on) → tag `round` (experience /
technical discussion / behavioral / coding / system design) so the mock
retrieves only rounds it simulates. Raw pastes stay local and gitignored;
the bank stores rephrased questions and generated rubrics.

**Retrieval at deep-dive time**: query = role-profile keywords + chosen
project keywords + the last answer; metadata filter by round (and company
when the JD names one and the bank has it); BM25 top-k → the interviewer
chooses adaptively (drill vs move on) → the chunk's rubric grades the answer.
Probes that come from a bank are marked *rubric-grounded* in the report;
probes generated from the resume alone are graded against an on-the-fly
rubric written at plan time (§8a).

Cost: rubric generation ≈ $3–5 per 100 questions with the teacher; retrieval
is free. Quality filter: 面经 mixes coding and design rounds — those are
tagged and excluded here, not deleted.

**Hidden interview plan** (once, at start): `{role_profile, project,
opening, probe_targets: [5–8 technical points from the project + role
themes], rubric_chunks: [...], on_the_fly_rubrics: {...}, behavioral_targets,
wander_budget, max_turns}`.

### 7c. Default job descriptions

When no JD is pasted, the session still runs through the JD path: a small
library of **default JD templates** (`public/jd_templates.json` or served
from the backend), one per role family × level — Machine Learning Engineer,
AI / LLM Application Engineer, Data Scientist, ML Platform / Infra, Applied
Scientist; junior / mid / senior. Each is a short, realistic posting
(responsibilities, must-haves, stack, what the team evaluates) written from
common real postings, with an optional *domain* slot the resume fills
("fraud detection", "recommendations", "forecasting") so probe themes stay
specific. The role picker (§7a) maps each proposed role to a template; the
user can edit the template text before starting, and the UI labels it as a
generic default. One code path — JD text → role profile — whether the JD
was pasted or defaulted, which is the point: the persona and probe structure
are always built the same way.

### 7d. Which model does what

"The grader" is four different jobs in the mock, and they do not all want
the same model:

| Job | Level 1 / 2 default | On demand ("Always Claude") | Level 3 (offline) | Why |
| --- | --- | --- | --- | --- |
| Live interviewer turns (12 × per session, latency-critical) | DeepSeek Flash | Claude | Ollama | cheap, fast, measured 94% agreement with the teacher as a judge |
| End-of-session report / scorecard (1 × per session, quality-critical) | **Claude** when a key is present, else Flash | — | Ollama (labelled lower-confidence) | one call, tens of cents; the scorecard is tracked across sessions, and Claude's 95% regrade consistency vs Flash's 53–57% is what makes session-to-session comparison meaningful |
| Per-key-point verdicts on bank-grounded probes | distilled kp classifier (free, deterministic) with the LLM's verdicts alongside | — | same | reuse of the measured classifier; also the instrument for the transcription-damage number |
| Communication metrics | computed, no model | — | same | deterministic by design |

The distilled sklearn grader is *not* a general grader for the mock — it was
distilled on course rubrics — so it plays the supporting roles above rather
than replacing the scorecard. Phase 1 includes a small consistency test
(regrade 5 sessions twice with Flash and with Claude) so the report default
is a measured choice: Flash takes over the report only if its scorecard
consistency comes within a few points of Claude's. **Phase machine** enforced in code: warm-up (1) →
walkthrough (1) → deep dive (4–6, adaptive) → behavioral (1–2) → closing (1)
→ done. The model chooses content within the phase: follow up on a weakness
in the last answer or move to the next unprobed target — one question only.

**Turn protocol**: spoken text streamed first; trailer `\n---\n{json}`
stripped before TTS, parsed after. **Speech Engine sidecar**: a small ASGI
process using the Python SDK (`on_transcript` → the same turn function →
`session.send_response(stream)`), so the text path and the voice path share
one state machine. **Engine routing** reuses `grading_route()`; free tier
cannot run a mock (needs an LLM). **Logging** off by default (resumes are
personal data); P3 opt-in writes `data/sessions/mock_sessions.jsonl` under the
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

### 8a. The grading rubric

Two levels, mirroring how a real hiring scorecard works.

**Level A — per question (content).** Every probe carries a rubric: from
the bank chunk when retrieved (§7b), otherwise written at plan time from the
resume — the *expected specifics* for that probe: the decision, the
alternative considered, the metric and its value, the outcome, the lesson.
Each key point is judged **hit / partial / miss** on meaning (the same
verdict scheme the app's grader was distilled on), with `common_mistakes`
checked against the answer → a 1–10 question score and the
good / missing / stronger-answer lines in the report.

**Level B — session scorecard**, six dimensions scored 1–10 with anchors,
each justification required to quote the transcript:

| Dimension | 3 looks like | 6 looks like | 9 looks like |
| --- | --- | --- | --- |
| Technical depth | buzzwords, no mechanism | correct but stays at "what" | precise "why", trade-offs, limits |
| Ownership & specificity | "we did", no numbers | own role clear, some numbers | "I decided X because Y", concrete figures, honest about others' parts |
| Structure | wanders, answers a different question | mostly context → action → result | tight STAR, answers exactly what was asked |
| Communication clarity | undefined jargon, lost thread | understandable with effort | a smart non-expert follows it |
| Concision & time management | 3-minute answers to 30-second questions | mostly proportionate | proportionate; computed metrics corroborate |
| Reflection | no failures, nothing to change | names a problem | what failed, what was learned, what would be done differently |

Plus **red flags** (contradictions between answers, claims that dissolve
under follow-up, inability to explain one's own resume line, blame-shifting)
and a hiring-committee style **overall call** — *strong hire / hire / lean
no / no* — with the two sentences an interviewer would actually write.

**Layering — global rubric + role layer.** The six dimensions above are the
**global rubric** and never change: a stable scorecard is what lets progress
be tracked across sessions (the same dimension, the same anchors, week over
week). On top of it, a **role layer** is generated once the resume and the
role are chosen, from the role profile's "what they evaluate": 1–3 extra
*role signals* with their own anchors — e.g. MLE: production judgement
(serving, monitoring, failure modes) and experimentation rigor; AI/LLM
engineer: evaluation design, prompt/agent robustness, cost–latency
trade-offs; Data Scientist: statistical rigor and business framing; Applied
Scientist: novelty and literature awareness — plus a **level shift** in what
"9" means (senior: scope, ambiguity, influence on others; junior: correctness
and learning speed). Role signals are scored and shown separately from the
global six and folded into the overall call, so the global scores stay
comparable across roles. Per-question rubrics (Level A) are inherently
role-specific already.

Honesty about validity: unlike the course grader there is no gold label for
"a good mock-interview answer". What can be measured is *consistency*
(regrade the same session; the app's teacher regrades at 95%) and the
rubric-grounded share of probes; validity is ultimately judged by the user
against real interview outcomes.

## 9. Phases and checklists

**Phase 0 — STT evaluation (ships first; the centerpiece)**
- [x] `grader/stt_testset.py` → `grader/stt_lexicon.json` (339 terms: 215
      curated + 124 from the banks; private overlay `data/stt/lexicon_extra.json`)
      and `grader/stt_sentences.jsonl` (88 items: 32 curated hard-term
      sentences, 36 bank sentences, 20 whole answers for downstream damage;
      ~2,400 words ≈ 16 min of reading)
- [x] recording page `public/stt_record.html` (MediaRecorder take per item +
      Web Speech in parallel = condition 1; localhost-only `/api/stt/*` routes)
- [x] human set recorded (88/88, 26.5 min) and evaluated
- [x] synthetic set via TTS: `synth_matilda` (Flash v2.5, 88 items, 16.9 min,
      1,438 credits ≈ $0.38 — pay-as-you-go TTS bills ~0.1 credit/char, not 0.5)
- [x] `grader/stt_eval.py`: all conditions incl. 6 (faster-whisper
      large-v3-turbo on the RTX 5080, float16 via pip CUDA wheels), WER, strict /
      lenient term error rate with a per-term "became" table, downstream damage
      (distilled grader on the raw transcript + DeepSeek Flash judge with its
      measured noise floor), post-hoc correction with false-correction audit,
      realtime finalization latency, measured credits; dry-run cost + `--confirm`
- [x] keyterm policy (`stt_text.select_keyterms`, cross-fitted by fold) vs the
      naive first-50 — evaluated as conditions `scribe_rt_policy50` /
      `scribe_rt_naive50`
- [x] results: `grader/stt_eval_results.json` + `docs/stt_evaluation.md` +
      README section (both sets); **decision recorded** (§2 status: batch +
      keyterms final, Realtime + policy-50 live, Whisper offline)

**Phase 1 — core loop, text + browser voice**
- [x] `coach/mock/` package (templates, schemas, engine+fake, planning,
      turns, metrics, report, routes) + `public/mock.html`/`mock.js`
- [x] `/api/mock/templates|roles|start|turn|report`; hidden plan with
      BM25 rubric grounding (live smoke: 7/7 probes matched bank chunks at
      scores 12.8–19.7); phase machine in code; `---` JSON trailer protocol
- [x] report rendering + `.md` downloads; offline tests
      (`tests/test_mock.py`, fake engine, in CI); live Flash smoke session
      (turn latency 1.5–2.1 s; the tough persona produced adaptive
      follow-ups quoting the candidate). Report cap fix: V4 thinking spends
      completion tokens, 8k truncated a 7-turn report — now 12k + a
      finish_reason==length retry
- [x] browser voice (Web Speech dictation + speechSynthesis) — shipped with
      the Phase 2 voice UI (mock.js voice modes)
- [x] report-consistency test (grader/report_consistency.py, run
      2026-08-31, ~$1.50): 5 identical sessions regraded twice per engine —
      Claude 73.3% exact dimension scores / MAE 0.3, Flash 43.3% / MAE 0.8;
      the hiring call was stable 5/5 for both. Claude stays the report
      default (§7d) — now a measured choice, not an assumption

**Phase 2 — live loops and the Level 1 vs Level 2 equivalence test**
- [x] DIY loop (Levels 2–3): coach/voice/ — browser audio over one
      WebSocket per session (`server.py --voice`; mock.js AudioWorklet mic
      + per-sentence playback with `played` acks), Silero VAD patient
      endpointing (end-silence measured below), streaming LLM
      (coach/llm.call_chat_stream, thinking off) → sentence chunker → TTS,
      barge-in with spoken-prefix truncation; `AUDIO_BACKEND` switch
      local / speaches / elevenlabs. Live smoke on real audio + real Flash:
      Whisper 0.85 s → first token 0.70 s → Kokoro 0.54 s = 2.3 s
      end-of-speech → first agent audio, adaptive tough-persona follow-up
- [x] Speaches container verified (faster-whisper turbo CT2 + Kokoro:
      warm STT 1.6 s per 31 s answer, TTS 0.36 s); local backend passes the
      session keyterms as Whisper's `initial_prompt`
- [x] latency instrumentation on every level: server per-stage (stt / llm
      first token / tts first byte / first audio) + client
      last-user-audio → first-agent-audio; session p50/p95 in `done`
- [x] local per-answer recording + POST /api/mock/transcribe (Scribe batch
      + full lexicon when keyed, local Whisper otherwise); two-transcript
      report block: WER live-vs-final, terms recovered, kp-verdict flips —
      both sides normalized per the Phase 0 decision
- [x] **Equivalence test, Level 2/3 half** (grader/loop_eval.py; the 20
      real Phase 0 answers as the scripted candidate, 12.4 min, deterministic
      fake-engine interviewer so only the audio path varies). Endpointing
      iterated per the rule: 1.2 s end-silence cut 50% of answers
      mid-thought, 1.8 s cut 15%, 2.0 s ships at **cut-off 5%** (= the bar).
      Shipped-config numbers: live TER 5.6–9.3% lenient across runs (54
      occurrences — small-n variance), WER ~7%, stt p50 0.35 s,
      **first-audio p50 0.77–0.83 s / p95 ≤ 1.74 s** (≤ 2.0 s bar),
      distilled-grader movement 3/18 answers (16.7%, both sides
      normalized; Phase 0 context: 25–35% spoken-vs-written); re-verified
      under the final non-blocking-commit architecture. The harness
      caught two real bugs now fixed: the Silero v5 context-window omission
      (VAD scored real speech ~0.0) and the answer-continuation stall
      (candidate talks past the committed answer → appended as an
      afterthought, interviewer regenerates)
- [x] DIY-cloud row measured (`--backend elevenlabs`, 2026-08-31, ~$0.68
      across the debug series): live TER 3.7% lenient / WER 7.6%, Scribe
      finalize p50 0.11 s, first-audio p50 0.48 / p95 0.63 s, cut-off 5%,
      grader movement 5/18 — cloud audio wins terms and latency, local
      wins grader stability (Whisper's fluent smoothing: the Phase 0
      finding, reproduced through the whole live loop). Getting here
      surfaced and fixed FIVE production bugs in the loop/STT path: a
      blocking commit and a blocking connect in the receive path (deaf
      VAD), protocol-ping starvation tearing the socket down during slow
      commits, Scribe's empty-commit handshake race, and Scribe's
      unsolicited multi-segment commits (only the final clause of long
      answers survived — 68% WER until segments were accumulated). Under
      the rule with this as the closest Level-1 proxy, **Level 2 stays
      main usage**: its grader damage beats the cloud path's
- [x] Deepgram row MEASURED (2026-08-31, five instrumented runs ≈ $0.95
      of the ~$200 signup credit; decision 9): Nova-3 streaming + Aura-2
      behind `AUDIO_BACKEND=deepgram` (`STT_BACKEND`/`TTS_BACKEND` split
      the bundle), Nova-3 batch + capped-100 keyterms as final-transcript
      fallback. Getting an honest row surfaced that Nova-3 WITHOUT
      interim results both recognizes and finalizes lazily (the
      finalized transcript trailed realtime input by 15–25 s; a Finalize
      drains only ~2–3 s per final) — fixed with interim_results=true,
      commits assembling finals + the open window's interim, and a
      watermark against cross-turn leaks — plus one harness bug
      (stream_pcm's def-time default ignored the per-backend speed;
      answers streamed at 4×). Final numbers: WER 12.0%, TER 14.8%
      lenient, STT finalize ~0.00 s, first-audio p50 1.35 / p95 4.48 s,
      cut-off 5%, damage 6/18. VERDICT: fails the 2.0 s p95 bar and
      loses to BOTH measured rows on text quality (Scribe 3.7%, local
      5.6–9.3% lenient TER) — Deepgram is the keyless-cloud fallback,
      not the recommendation; keyterm-off and batch variants untested
- [ ] Level 1 live run: sidecar + tunnel + page + engine are all wired and
      config-verified (engine created; `asr.keywords` and patient turn
      accepted; quirk: English agents take eleven_flash_v2 — v2_5 is a
      400 on update and a 500 inside create). `tools/level1_up.py` brings
      the tunnel + engine config up in one command and the mock page has
      the "Level 1 — hosted voice" mode; the session itself needs a human
      speaking. The blind TTS preference test rides on that session.
      First live try 2026-08-31: silent while self-advancing — the sidecar
      violated two upstream rules (`agent_response.event_id` must ECHO the
      answered user_transcript's id, ElevenLabs discards mismatches; every
      response terminates with an empty `is_final: true` chunk). Fixed +
      fake-websocket regression test. Since decision 9 Level 1 is an
      OPTIONAL comparator, no longer on the recommended path
- [ ] barge-in scripted probe: the mechanism is verified end to end (a
      fault-injection run produced `interrupted` + clean spoken-prefix
      recovery), but the probe cannot reliably hit the fake engine's ~1 s
      speaking window; re-probe under real LLM turns (3–8 s windows)

**Phase 3 — polish and portfolio**
- [ ] resume-analysis cache (user request 2026-08-31): hash-keyed
      server-side cache of `/api/mock/roles` and `/api/mock/start` results
      in `data/mock_cache/` (gitignored — resumes are personal), so repeat
      practice across roles skips the re-analysis wait and the LLM spend;
      a "fresh plan" bypass for new questions on the same role. Pairs with
      the prompt-cached materials prefix below
- [ ] consent-aware opt-in logging; prompt-cached materials prefix
- [ ] rubric tie-in with prep-doc Q&A; README + interview_prep additions

**Infra track — private RAG service (§13)**: see the checklist there; it
runs alongside Phases 1–3 and is the prerequisite for sharing the mock with
authorized users.

## 10. Open decisions and verification items

Decisions:
1. ElevenLabs billing: start pay-as-you-go (§5b); a plan only in a heavy
   Level-1 month. Verify that Free-plan pay-as-you-go covers Scribe,
   TTS, and Speech Engine minutes.
2. The human test set: the user records ~100 sentences (~15 min).
3. Local fallback stack (§4a): Speaches container (faster-whisper + Kokoro)
   vs direct Python libraries — recommendation: Speaches, one container.
   Condition 6 (local Whisper) is now in the experiment by default.
4. Accept the ElevenLabs browser SDK as the mock page's one dependency.
5. Default interviewer LLM: Claude (recommended) vs Flash.
6. Decision-rule thresholds (≤5% damage, ≤3% lenient TER) — confirm. On the
   synthetic set only batch + keyterms clears the TER bar (2.3%); the damage bar
   is not met by any condition and is entangled with grader brittleness (§2
   status). Decide after the human set whether the damage bar is measured on
   normalized text (word errors only) — the recommendation — or on raw text.
   → RESOLVED with the human set (§2 status): rule not met by any condition;
   pre-registered fallback applied; the report grades normalized text.
7. Private service access shape (§13): Tailscale private network
   (recommended for "me plus a few people") vs public HTTPS + access keys
   (needed for a public demo and browser voice on EC2). Both can coexist.
8. Private data storage: the private git repository (now) vs a versioned S3
   bucket (when data outgrows git or others need deploy-time access).
9. Audio vendor (2026-08-31; supersedes the ElevenLabs framing of 1 and 4):
   the user dropped the ElevenLabs-application consideration — "just
   choose better options". **Deepgram** is the recommended cloud vendor:
   Nova-3 streaming/batch STT (first-class keyterm prompting, ~$0.008/min)
   + Aura-2 TTS, one key (`DEEPGRAM_API_KEY`, ~$200 signup credit).
   ElevenLabs stays as the measured legacy row until its ~$7.85 balance
   runs out; Scribe batch + full lexicon remains the final-transcript
   default while its key exists (the measured Phase 0 winner), `FINAL_STT`
   overrides. Hosted Level 1 is not re-bought elsewhere: the DIY loop
   already beat it on grader damage.
   → MEASURED the same day (Phase 2 checklist): the swap hypothesis did
   not survive the harness — Deepgram TER 14.8% lenient vs Scribe 3.7%,
   first-audio p95 4.48 s vs 0.63 s, damage 6/18 vs 5/18. Standing order
   after the data: local = main usage, Scribe = best cloud row while the
   ElevenLabs balance lasts, Deepgram = the cloud option without an
   ElevenLabs key (~$200 credit; STT finalize ~0.00 s is its one win).
   The final-transcript order (Scribe first) already matches.

Verify against DeepSeek docs before Phase 1: the streaming (`stream: true`,
SSE) path for the OpenAI-compatible endpoint — `call_deepseek` is currently
non-streaming — and the parameter that disables V4's hybrid thinking per
request, so interviewer turns get a fast first token while the report keeps
thinking on. Measure first-token latency both ways.
→ MEASURED 2026-08-30: SSE streaming works; `thinking: {"type": "disabled"}`
gives ~0.8 s to first content token vs ~1.4 s enabled (short prompt, n=3
each); `coach/llm.py call_chat()` implements the switch and the mock's turns
run with thinking off, the report with it on.

Verify against ElevenLabs docs before Phase 2 (not answered by the docs read
on 2026-08-29): whether Speech Engine accepts per-session STT keyterms;
whether the server learns the spoken prefix on interruption; whether the
TTS voice/model is selectable per session; whether session audio is
retrievable (local recording makes this non-blocking).
-> RESOLVED 2026-08-30 against SDK 2.65's typed API (speech_engine.create/
update): STT keyterms YES (`asr.keywords`, provider `scribe_realtime`); TTS
voice/model YES (`tts.voice_id`, `tts.model_id`); turn-taking YES
(`turn.turn_eagerness: patient`, `turn_timeout` up to 30 s). Spoken prefix
NO - the upstream protocol is text-only (init / user_transcript in,
agent_response out; verified from the SDK's protocol types), so on barge-in
a Speech Engine server never learns how much of its question was actually
spoken, while the DIY loop knows exactly (sentence `played` acks). Also:
`ws_url` must be publicly reachable, so a local Level 1 session needs a
tunnel. Both findings push the expected main usage further toward Level 2.
Live addendum 2026-08-31: two more upstream rules, learned when the first
Level 1 session sat silent — `agent_response.event_id` must echo the
answered user_transcript's id (their barge-in correlation key; mismatches
are silently discarded), and every response terminates with an
empty-content `is_final: true` chunk. Both in coach/voice/sidecar.py with
a fake-websocket regression test.

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
  the cloud audio vendor (Deepgram or ElevenLabs) sees the audio on cloud
  backends (state this in the UI); the local backend keeps audio
  on-machine.

## 12. Next-session start checklist

- Start with Phase 0: say "build phase 0"; confirm the §10 decisions or
  accept the recommendations. Have an ElevenLabs API key ready
  (pay-as-you-go, or credits) — it goes in `.env` as `ELEVENLABS_API_KEY`.
- Phase 0 needs ~15 minutes of the user's time to record the human set.
- Resume as text: `.venv\Scripts\python resume_parser.py <resume.pdf|.docx>`
  → `data/resume/<name>.txt` (gitignored). Phase 0's lexicon and Phase 1's role
  picker read it from there; the setup page will accept the same upload.
- `.env` already holds `ANTHROPIC_API_KEY` and `DEEPSEEK_API_KEY`; DeepSeek
  spend is authorized; any Claude live test needs fresh authorization;
  ElevenLabs spend needs authorization once the key exists.

## 13. Private RAG service — data private, code public, service gated

Decision (2026-08-29): the retrieval-and-grading backend becomes a **private
service** that the author and authorized users reach, while the code stays
public. This is the industry pattern the repo split already set up: code in
the public repository, data in private storage, secrets in the environment,
and a deployed instance that is authenticated — not a new component.

**What is actually private at runtime.** The app never needs a chunk's
lesson text (`content`) to serve — the public banks are sufficient for the
course tracks. The private data the service exists to serve is (a) the
`rag_exp/` bank built from gathered 面经 (§7b), (b) per-user data (session
logs, mock-interview sessions, resumes in flight), and (c) access keys. So
the private service = the existing backend + `rag_exp/` mounted from private
storage + authentication. The complete course banks stay in the private
repository for training only.

**Design.**
- *Data registry*: private banks and the grader dataset live in the private
  repository (or a versioned private S3 bucket later); a deploy step pulls
  the current version into the `coach-data` volume. Data is versioned
  separately from code; the public image contains no private data.
- *Service boundary*: a `POST /api/retrieve` endpoint (`{track, query,
  filters, k}` → ranked chunks with rubric fields) so every client — the
  practice page, the mock interviewer, a future MCP connector for claude.ai —
  uses the same retrieval; grading stays behind `/api/evaluate`.
- *Access*, one of two shapes, chosen in §10:
  - **Private network**: Tailscale on the EC2 host — no public port, invited
    users join the tailnet; zero exposure, free for personal use.
  - **Public but gated**: HTTPS (domain + Caddy/Let's Encrypt) with the
    existing access keys, optionally Cloudflare Access in front. Required
    anyway for browser voice on the public deployment.
- *Per-user authorization*: `users.json` grows a `banks` allow-list (which
  private banks a key may query) — the same fail-closed loader.
- *Observability*: per-request log line (engine, bank, latency, cost
  estimate) into `data/`, and the `/api/meta` health check already in compose.

**What it is not.** No vector database or managed RAG service: at 282 chunks
BM25 measures 100% Recall@5, and adding an index without a number asking for
it would contradict the project's own evidence. The trigger is measured — the
retrieval suite showing BM25 recall dropping as `rag_exp/` grows, or the mock
needing semantic matching from JD themes to probes — and the first response
is embeddings + a local index (pgvector or Qdrant), not a hosted service.

**Checklist (infra track, can run alongside Phases 1–3).**
- [ ] `POST /api/retrieve` + contract entry in docs/backend.md
- [ ] deploy step that pulls private banks into the volume (script +
      compose mount); public image verified to contain no private data
- [ ] access shape chosen and set up (Tailscale, or domain + HTTPS + keys)
- [ ] `users.json` bank allow-list, fail-closed, unit-tested
- [ ] request log line; `docs/deployment.md` runbook

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
