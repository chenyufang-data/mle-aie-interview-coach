# Frontend requirements

Scope: everything under `public/`. The frontend is a static browser app served by
the backend's file handler. It owns presentation and session flow — nothing else.

## Runtime requirements

- Any modern browser (ES2017+: `async/await`, `fetch`, `sessionStorage`).
- **Zero dependencies** — no npm, no build step, no framework. The files in
  `public/` are served as-is by the backend — or, in the Docker setup, by a
  dedicated nginx image (`docker/frontend.Dockerfile`) that serves `public/`
  and proxies `/api/` to the backend service, keeping everything same-origin.
- No configuration and no secrets. The frontend never sees an API key and never
  talks to Anthropic or Ollama directly — only to its own backend, same-origin.

## Responsibilities (must)

- **Setup page** (`index.html`): collect track (MLE / AIE), level, topic, and
  optional focus text. The "course knowledge base" topic groups are built at load
  time from `GET /api/meta` — module names and chunk counts come from the server.
  An optional access-key field (stored in `sessionStorage`, sent as the
  `X-Access-Key` header on every API call) identifies paid users when the server
  has tiers enabled; a badge under the field shows the current tier, the paid
  tier's default judge (`meta.user.paid_grader`, e.g. DeepSeek Flash), and the
  remaining daily Claude quota, and stays hidden when `meta.user.tiers_enabled`
  is false. The frontend never decides the tier — it only reports what the
  server said.
- **Interview page** (`interview.html`): render the question and guidance, run the
  elapsed timer with pause/resume, count words in the answer box, submit the
  answer, and render the evaluation (overall score, four subscores, summary,
  strengths, gaps, suggested answer, next steps, follow-up question, plus a
  "Graded by" chip naming the model behind the score — the server's
  `graded_by` field, never inferred client-side). Two answer-box extras: a 🎤
  voice-input button (browser Web Speech API, client-side only — audio never
  reaches the server; the button hides itself outside secure contexts) and an
  "Always Claude" checkbox that sends `force_llm: true` to skip the smart
  cascade and the DeepSeek workhorse, forcing a metered Claude evaluation.
- **Session flow**: "Practice the follow-up" keeps the parent question, previous
  answer, and parent `chunk_id` so the backend can grade in context; "New
  question" accumulates served `chunk_id`s and sends them as `exclude` so
  knowledge-base questions do not repeat within a session. Session state lives in
  `sessionStorage` only.

## Must not

- Hard-code anything corpus-specific: module names, chunk counts, or question
  text. All of it comes from the backend, so corpora can change without touching
  `app.js`.
- Grade answers, rank or select questions, or otherwise duplicate backend logic.
- Store secrets or call any external API.

## Backend API consumed

See [backend.md](backend.md#api-contract) for the authoritative contract. The
frontend uses exactly three endpoints:

| Endpoint | When |
| --- | --- |
| `GET /api/meta` | Setup page load — builds knowledge-base topic groups |
| `POST /api/question` | Start of a session and "New question" |
| `POST /api/evaluate` | Answer submission (first answers and follow-ups) |

`public/mock.html` + `mock.js` is the mock-interview page: setup (resume +
JD or default template, style, length, voice mode) → role cards from
`/api/mock/roles` → the interview → report rendered in blocks (LLM
scorecard, computed metrics, rubric-grounded verdicts, and — for voice
sessions — the live-vs-final transcription-quality block) with `.md`
downloads. Three voice modes:

- **text** — the Phase 1 chat flow over `/api/mock/turn` (`answer_ms`
  measured question-shown → send);
- **browser** — Web Speech dictation into the answer box plus
  `speechSynthesis` questions; free, Chromium-only, still dependency-free;
- **live** — the `coach/voice` WebSocket loop: `mock-audio-worklet.js`
  captures the microphone (AudioWorklet, downsampled to 16 kHz PCM16),
  the server VAD/STT/LLM/TTS pipeline streams back per-sentence audio the
  page plays and acks (`played` messages are the ground truth for what an
  interruption actually cut off). The page shows loop state and measured
  end-of-speech → first-audio latency.

In browser and live modes each answer is also recorded (`MediaRecorder`,
opt-out checkbox) and re-transcribed via `POST /api/mock/transcribe`; the
result rides on the transcript entry as `answer_final`, which the report
grades. Session state lives in the page and is lost on reload (documented
tradeoff of the stateless server).

`public/stt_record.html` + `stt_record.js` is a separate developer page (not
linked from the app) for recording the Phase 0 STT test set: one
`MediaRecorder` take per item, saved through the localhost-only
`/api/stt/*` routes, with the browser Web Speech API run on the same take so
condition 1 of the experiment ("today's voice path") is captured for free. It
shares `styles.css` and the same no-dependency rule.
