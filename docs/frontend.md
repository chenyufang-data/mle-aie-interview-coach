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
  has tiers enabled; a badge under the field shows the current tier and
  remaining daily Claude quota, and stays hidden when `meta.user.tiers_enabled`
  is false. The frontend never decides the tier — it only reports what the
  server said.
- **Interview page** (`interview.html`): render the question and guidance, run the
  elapsed timer with pause/resume, count words in the answer box, submit the
  answer, and render the evaluation (overall score, four subscores, summary,
  strengths, gaps, suggested answer, next steps, follow-up question).
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
