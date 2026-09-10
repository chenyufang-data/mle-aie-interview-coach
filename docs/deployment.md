# Deployment runbook — the public demo on one small VM

Scope: the practice track, the text mock interview and the live voice mock
over HTTPS on a t3.small, a demo access key that strangers may use, every
LLM call routed to DeepSeek Flash under daily budgets, every minute of
voice routed to Deepgram under daily budgets, and nothing on the box that
can spend Claude credit. This is roadmap step 1 in [plan.md](plan.md); the
pieces it relies on are the Docker files, `docker-compose.prod.yml`,
`tests/test_container.sh`, and the budgets in `coach/users.py`.

## 0. What the public URL exposes

| Visitor | What they get | What they cannot do |
| --- | --- | --- |
| Anonymous | Bank questions on every public track, instant grading by the distilled local model | Reach any LLM: with `users.json` present, no key means the free tier |
| Demo key (`"tier": "paid", "daily_llm_calls": 60`) | AI questions, DeepSeek Flash grading, the full text mock with report | Exceed 60 LLM calls a day on that key, or `LLM_DAILY_CAP` for the whole box; reach Claude (no `ANTHROPIC_API_KEY` on the server) |
| Demo key, live voice (`"daily_voice_minutes": 30`) | The live voice mock on Deepgram (Nova-3 streaming + Aura-2 TTS), each answer re-transcribed for the report | Exceed 30 voice minutes a day on that key, `VOICE_DAILY_MINUTES` for the whole box, or `VOICE_SESSION_MAX_MINUTES` in one session — the interviewer says goodbye and the socket closes within a minute of the allowance running out |
| Anyone | 10 requests/second per address with a burst of 20 (nginx answers 429 beyond that); bodies capped per route (413) | Upload more than 16 MB of resume or 48 MB of audio |

The private `rag_exp` bank is never copied into the image. `rag_lists` and
`rag_docs` are generated banks (gitignored); copy their `all_chunks.jsonl`
to the server before building if you want the Lists and Docs tracks on the
demo — both were built from open-licensed sources with attribution in their
READMEs — otherwise the image serves the two course banks.

## 1. Prepare locally (once)

1. **Keys for the server.** Generate two access keys:

   ```powershell
   .venv\Scripts\python -c "import secrets; print(secrets.token_urlsafe(24))"
   ```

   Put them in a `users.json` you will copy to the server (never commit it):

   ```json
   {
     "<owner-key>": { "name": "Chenyu", "tier": "paid" },
     "<demo-key>":  { "name": "Demo", "tier": "paid", "daily_llm_calls": 60,
                      "daily_voice_minutes": 30, "log": false }
   }
   ```

   `"log": false` keeps strangers' answers out of the session logs;
   `daily_voice_minutes` caps the live voice mock per key (0 or absent =
   unlimited, right for the owner's key).

2. **`.env` for the server**, six lines and no Anthropic key:

   ```text
   DEEPSEEK_API_KEY=<the separate DeepSeek key created for the demo>
   LLM_DAILY_CAP=200
   DEEPGRAM_API_KEY=<a Deepgram key created for the demo, in its own project>
   AUDIO_BACKEND=deepgram
   VOICE_DAILY_MINUTES=120
   DOMAIN=<your hostname>
   ```

   Without `ANTHROPIC_API_KEY` the "Always Claude" checkbox has nothing to
   call; the server degrades those requests to the DeepSeek workhorse
   (`coach/grading.py`). Keep the DeepSeek account balance small (about $5)
   and auto-recharge off; the balance is the hard stop. The Deepgram key
   turns the live voice mock on (section 6): make it in a project of its
   own on the Deepgram console so it can be revoked and read on its own,
   and leave the account on the free credit with no card on file — at
   Nova-3 streaming plus Aura-2 prices a 30-minute allowance is about
   $0.35 a day at most, and `VOICE_DAILY_MINUTES` bounds the whole box.

3. **A hostname.** Any registrar's A record, or a free DuckDNS name; Let's
   Encrypt works with both. The record must point at the VM's public IP
   before Caddy starts, or the certificate request fails.

4. **Local rehearsal.** `bash tests/test_container.sh` builds both images,
   boots the stack in `--mock` mode and checks `/api/meta`, one question and
   one graded answer through nginx. Docker Desktop must be running. On a
   network where the model download is blocked,
   `CONTAINER_TEST_MODEL_DIR=data/models/fastembed bash tests/test_container.sh`
   mounts the local copy instead.

## 2. The VM (AWS console)

- EC2, Ubuntu Server 24.04 LTS, **t3.small** (2 vCPU, 2 GB; the backend
  holds about 230 MB with the embedding model loaded), 20 GB gp3.
- Security group: 22 from your own IP only; 80 and 443 from anywhere.
  Nothing else — 8000, 8080 and 8765 stay inside the compose network.
- An Elastic IP if you use a registrar record (DuckDNS can follow a
  changing IP with its updater instead).
- Billing → Budgets: a $20/month budget with an email alert.

On the box:

```bash
sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER && newgrp docker
docker compose version
```

## 3. Deploy

```bash
git clone https://github.com/chenyufang-data/mle-aie-interview-coach.git
cd mle-aie-interview-coach
# from your machine: scp users.json .env ubuntu@<host>:mle-aie-interview-coach/
# optional: scp rag_lists/all_chunks.jsonl rag_docs/all_chunks.jsonl into the same folders
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
docker compose -f docker-compose.yml -f docker-compose.prod.yml logs -f backend
```

The backend log should say `Tiers: ON - 2 paid key(s)`, `LLM budgets:
server-wide 200 calls/day; 1 key(s) carry a daily_llm_calls cap`, `Voice
budgets: server-wide 120 min/day; 1 key(s) carry a daily_voice_minutes cap`,
`Voice loop: ws://127.0.0.1:8765 (stt=deepgram, tts=deepgram)` and, after
the one-time model download, `Retrieval: hybrid BM25 + bge-small`. A
`Voice loop: off - ...` line names what is missing (the backend's key, or
a local backend the image cannot run). The embedding model, the document
vectors and the Silero VAD model persist in the `coach-data` volume
(`FASTEMBED_CACHE_DIR`, `RETRIEVAL_INDEX_DIR`, `SILERO_VAD_PATH` in
`docker-compose.yml`), so a rebuild does not download again. Caddy obtains
the certificate on first start and keeps it in `caddy-data`.

## 4. Verify from a clean browser or shell

```bash
D=https://<your hostname>
curl -s $D/api/meta | python3 -m json.tool | head -30          # banks, "retrieval": "hybrid"
curl -s -X POST $D/api/evaluate -H 'Content-Type: application/json' \
  -d '{"answer":"precision is the share of flagged items that are positive","question":"What is precision?"}' \
  | python3 -c "import json,sys; r=json.load(sys.stdin); print(r.get('graded_by'), r.get('summary_label'))"
# anonymous -> the local grader ("Free tier"); repeat with -H 'X-Access-Key: <demo-key>' -> DeepSeek
curl -s $D/api/meta -H 'X-Access-Key: <demo-key>' | python3 -c "import json,sys; print(json.load(sys.stdin)['user'])"
# llm_left_today counts down; at 0 the page says "Daily LLM allowance used" and the mock refuses with a message
for i in $(seq 1 40); do curl -s -o /dev/null -w '%{http_code} ' $D/api/meta & done; wait; echo   # some 429s
curl -s $D/api/mock/voice | python3 -m json.tool   # "enabled": true, deepgram both sides, "ws_path": "/ws/voice"
# the demo key's meta also carries voice_cap / voice_left_today (minutes) and the server-wide pair
```

Then the real check: paste `example_resume.txt` into the mock page with the
demo key, run two turns, end early, and read the report. Then once more in
live voice mode: the status line says `Live loop ready — deepgram_nova3 +
deepgram TTS`, the interviewer speaks the opening, an answer of a few
sentences is transcribed live and followed up on, and interrupting the
interviewer mid-question cuts the audio. The account chip's tooltip shows
the voice minutes counting down; at 0 the interviewer says goodbye, the
report is written, and a new session is refused with a message.

## 5. Operate

- **Update**: `git pull` then the same `up -d --build` line. Data, keys and
  certificates live in volumes and bind mounts, not in the image.
- **Revoke or change the demo key**: edit `users.json` on the box; the
  server reloads it on the next request, no restart. With the Postgres
  store on (section 9) the keys live in the database instead: revoke with
  `docker compose ... exec db psql -U coach -d coach -c "UPDATE access_keys SET revoked_at = now() WHERE name = 'Demo'"`
  (effective within 5 s), add or change one by editing `users.json` and
  re-running the migration tool (it upserts). Revoke the DeepSeek key
  itself on the DeepSeek platform if it ever leaks.
- **Watch spend**: the DeepSeek usage page and the Deepgram project's usage
  page (both demo keys are separate, so their traffic is attributable), and
  the counters:
  `docker compose -f docker-compose.yml -f docker-compose.prod.yml exec backend cat /data/usage.json`
  (rows are keyed by a digest of each key; `_server` is the instance total;
  `llm` counts calls, `voice` counts seconds).
- **Logs**: `docker compose ... logs --tail=200 backend`. Unhandled errors
  print their traceback there; the client only sees the exception class.
  Every structured DeepSeek call prints one `deepseek json:` line with its
  seconds, whether thinking was on, and its completion and reasoning
  tokens — the first thing to read when a visitor says setup is slow.
  Measured 2026-09-08: roles and plan take about 12 s each with thinking
  off (the shipped setting); with thinking on they took 65–178 s, most of
  it 6k+ reasoning tokens. The report keeps thinking on and takes about a
  minute.
- **Cost**: about $15/month for the instance and $2 for the disk on
  on-demand pricing; DeepSeek at 60 calls a day is under $2 a month even if
  every call is a mock turn; Deepgram at the 120-minute server cap is at
  most about $1.40 a day against the free credit, and a typical day is a
  few cents.
- **Tear down**: `docker compose ... down -v`, terminate the instance,
  release the Elastic IP, delete the DNS record.

## 6. Voice on the server (cloud, in the image)

The image installs `requirements.txt` plus `requirements-voice-cloud.txt`:
the loop server and the Silero VAD runtime, which is all a cloud audio
stack needs. The local stack (faster-whisper, CUDA wheels, Kokoro) stays
out — far too heavy for a t3.small, and there is no GPU on it. With
`AUDIO_BACKEND=deepgram` and `DEEPGRAM_API_KEY` in the server's `.env`,
the loop starts beside the HTTP API (the log line above); without a cloud
backend the server says why the loop is off, the mock page greys the live
option out with that reason, and `/ws/voice` answers 502.
`docker-compose.prod.yml` publishes `VOICE_WS_PATH=/ws/voice`, nginx
proxies that path to port 8765, and the page dials `wss://` on its own
origin, so one certificate covers the API and the socket.

Deepgram was measured in August against the local stack and ElevenLabs
(README, "Live voice"): it loses on technical-term accuracy (14.8% lenient
term loss vs 3.7% for Scribe) and is slower to first audio (p95 4.5 s), but
it is one vendor for STT and TTS, runs on the free credit, and streams
over a single WebSocket the box can hold. That is the right trade for a
free public demo; the README states the measured numbers next to it.

What a visitor can spend: only the voice allowances. The loop requires a
paid key (the demo key), charges one LLM unit at connect and one per
interviewer turn — exactly as the text mock's start and turn routes do —
and meters the session's wall-clock time in one-minute ticks against the
key's `daily_voice_minutes` and the box's `VOICE_DAILY_MINUTES`. A spent
allowance ends the session inside a minute with a spoken goodbye and the
normal report; a new session is refused with a message. One session cannot
run past `VOICE_SESSION_MAX_MINUTES` (20). The per-answer re-transcription
(`POST /api/mock/transcribe`, Nova-3 batch on this box) is paid-key only
and charged to the same allowance by clip length. `tests/test_users.py`
and `tests/test_voice.py::test_voice_session_budget` cover the metering.

Turn-taking knobs, all in the server's `.env` and read at startup
(`up -d` after a change; no rebuild): `VOICE_END_SILENCE_MS` (default
2000, the measured end-of-answer silence), `VOICE_HOLD_MS` (default 2500:
how much longer the loop waits when the live transcript ends mid-sentence,
0 to disable). The first live session on the box was cut during a 2-3 s
thinking pause; the hold is the answer to that, and a patient interviewer
for a demo audience may also want `VOICE_END_SILENCE_MS=2500`.

To use ElevenLabs on the box instead (the better measured cloud row, but
paid): add `elevenlabs>=2.65` to `requirements-voice-cloud.txt`, rebuild,
and set `AUDIO_BACKEND=elevenlabs` with `ELEVENLABS_API_KEY`.

## 7. What a visitor sees

The README's first screen links the live URL. A visitor with no key can
practise on every public track with the local grader; with the demo key
(handed out on request — never printed in the repository) they can run the
text mock and, in live voice mode, talk to the interviewer: the opening,
an answer transcribed as they speak, a follow-up on the weakest part of
it, barge-in by interrupting, and the report with the live and final
transcripts side by side. The two-minute walk-through, in order: a
practice question graded; the mock from resume paste to two turns and the
report; the live voice mock with one interruption; the README results
tables and `docs/plan.md`, which record the negative results too.

## 8. Demo-day checklist

- [ ] `users.json` on the box has the demo key with `daily_llm_calls`, `daily_voice_minutes` and `"log": false`
- [ ] `.env` on the box has no `ANTHROPIC_API_KEY`; it has `AUDIO_BACKEND=deepgram`, the demo Deepgram key and `VOICE_DAILY_MINUTES`
- [ ] DeepSeek balance small, auto-recharge off, the demo key is the separate one; Deepgram on the free credit, no card, own project
- [ ] `/api/meta` shows `retrieval.backend: "hybrid"` and the expected banks; `/api/mock/voice` shows `"enabled": true`
- [ ] A burst returns 429s; a 2 MB JSON body returns 413
- [ ] Budget alert set; instance type t3.small; only 22/80/443 open
- [ ] One live voice session run end to end from a phone or another machine
- [ ] The README links the live URL
- [ ] If the Postgres store is on (section 9): `/api/meta` shows `store.backend: "postgres"` and `retrieval.vectors: "pgvector"`, and the demo key's allowance counts down there

## 9. Postgres on the box (optional, roadmap step 4)

The file store serves the demo fine; this switch is for the state
guarantees (a row-locked budget, history, revocation by `UPDATE`) and to
run what step 4 measured. About 100 MB of RAM on the t3.small, no extra
instance.

1. On the box, add a line to `.env`: `POSTGRES_PASSWORD=<a long random string>`.
   Nothing else changes in `.env`; the compose override builds
   `DATABASE_URL` from it.
2. Pull and rebuild with the override, which starts the database and a
   backend that still reads the files (the override is what sets
   `DATABASE_URL`, so build first, import next, then start):

   ```bash
   git pull
   docker compose -f docker-compose.yml -f docker-compose.db.yml -f docker-compose.prod.yml build
   docker compose -f docker-compose.yml -f docker-compose.db.yml -f docker-compose.prod.yml up -d db
   ```

3. Dry-run the import of the existing volume and the keys, then import:

   ```bash
   docker compose -f docker-compose.yml -f docker-compose.db.yml -f docker-compose.prod.yml run --rm backend \
     python tools/migrate_to_postgres.py --data /data --users /app/users.json --dry-run
   docker compose -f docker-compose.yml -f docker-compose.db.yml -f docker-compose.prod.yml run --rm backend \
     python tools/migrate_to_postgres.py --data /data --users /app/users.json
   ```

   The dry run prints what the folder holds and what the tables hold; the
   import prints before and after counts. Running it again changes nothing.
4. Start everything on the store: the same `up -d --build` line as section
   3 with `-f docker-compose.db.yml` added. The backend log's `State store:`
   line names the database and its counts, and the retrieval line ends
   with `vectors served by Postgres + pgvector`.
5. Verify from outside as in section 4: `/api/meta` shows
   `store.backend: "postgres"` and `retrieval.vectors: "pgvector"`; one
   graded answer with the demo key moves `llm_left_today` down by one.
   The counters are now read with
   `docker compose ... exec db psql -U coach -d coach -c "SELECT * FROM usage_counters ORDER BY day DESC, row_id"`.
6. Keep the `coach-data` volume: the embedding model, the vector cache and
   the Silero model still live there, and the files are the fallback if you
   ever drop the override (the store never writes to them once Postgres is
   on, so counters would restart from the last file state).

From then on `users.json` on the box is only the import source: add or
change a key there and re-run step 3's second command; revoke as in
section 5. Back up the database with
`docker compose ... exec db pg_dump -U coach coach > coach-$(date +%F).sql`.
