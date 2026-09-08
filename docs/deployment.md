# Deployment runbook — the public demo on one small VM

Scope: the practice track and the text mock interview over HTTPS on a
t3.small, a demo access key that strangers may use, every LLM call routed
to DeepSeek Flash under daily budgets, and nothing on the box that can
spend Claude credit. Voice stays a local demonstration (section 6). This is
roadmap step 1 in [plan.md](plan.md); the pieces it relies on are the
Docker files, `docker-compose.prod.yml`, `tests/test_container.sh`, and the
budgets in `coach/users.py`.

## 0. What the public URL exposes

| Visitor | What they get | What they cannot do |
| --- | --- | --- |
| Anonymous | Bank questions on every public track, instant grading by the distilled local model | Reach any LLM: with `users.json` present, no key means the free tier |
| Demo key (`"tier": "paid", "daily_llm_calls": 60`) | AI questions, DeepSeek Flash grading, the full text mock with report | Exceed 60 LLM calls a day on that key, or `LLM_DAILY_CAP` for the whole box; reach Claude (no `ANTHROPIC_API_KEY` on the server) |
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
     "<demo-key>":  { "name": "Demo", "tier": "paid", "daily_llm_calls": 60, "log": false }
   }
   ```

   `"log": false` keeps strangers' answers out of the session logs.

2. **`.env` for the server**, three lines and no Anthropic key:

   ```text
   DEEPSEEK_API_KEY=<the separate DeepSeek key created for the demo>
   LLM_DAILY_CAP=200
   DOMAIN=<your hostname>
   ```

   Without `ANTHROPIC_API_KEY` the "Always Claude" checkbox has nothing to
   call; the server degrades those requests to the DeepSeek workhorse
   (`coach/grading.py`). Keep the DeepSeek account balance small (about $5)
   and auto-recharge off; the balance is the hard stop.

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
server-wide 200 calls/day; 1 key(s) carry a daily_llm_calls cap`, and,
after the one-time model download, `Retrieval: hybrid BM25 + bge-small`.
The model and the document vectors persist in the `coach-data` volume
(`FASTEMBED_CACHE_DIR`, `RETRIEVAL_INDEX_DIR` in `docker-compose.yml`), so a
rebuild does not download again. Caddy obtains the certificate on first
start and keeps it in `caddy-data`.

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
```

Then the real check: paste `example_resume.txt` into the mock page with the
demo key, run two turns, end early, and read the report.

## 5. Operate

- **Update**: `git pull` then the same `up -d --build` line. Data, keys and
  certificates live in volumes and bind mounts, not in the image.
- **Revoke or change the demo key**: edit `users.json` on the box; the
  server reloads it on the next request, no restart. Revoke the DeepSeek key
  itself on the DeepSeek platform if it ever leaks.
- **Watch spend**: the DeepSeek usage page (the demo key is separate, so its
  traffic is attributable), and the counters:
  `docker compose -f docker-compose.yml -f docker-compose.prod.yml exec backend cat /data/usage.json`
  (rows are keyed by a digest of each key; `_server` is the instance total).
- **Logs**: `docker compose ... logs --tail=200 backend`. Unhandled errors
  print their traceback there; the client only sees the exception class.
- **Cost**: about $15/month for the instance and $2 for the disk on
  on-demand pricing; DeepSeek at 60 calls a day is under $2 a month even if
  every call is a mock turn.
- **Tear down**: `docker compose ... down -v`, terminate the instance,
  release the Elastic IP, delete the DNS record.

## 6. Voice on the server (optional; not in the image)

The image installs `requirements.txt` only. The voice loop needs
`requirements-stt.txt`, whose local stack (faster-whisper, CUDA wheels,
Kokoro) is far too heavy for a t3.small; the cloud backends need only the
light part of it, but the image does not install them, so `/ws/voice`
answers 502 until you add a layer. The plan keeps the public URL text-only
and demonstrates voice from the local machine (section 7). If you do want
cloud voice on the box: add a `pip install` layer for the cloud subset,
set `AUDIO_BACKEND=deepgram` and `DEEPGRAM_API_KEY` in `.env`, and start the
backend with `--voice`; `docker-compose.prod.yml` already publishes
`VOICE_WS_PATH=/ws/voice`, nginx proxies that path to port 8765, and the
page dials `wss://` on its own origin.

## 7. The two-minute recording

Local setup: `.venv\Scripts\python server.py --voice` with `users.json` in
place, the browser at 1280×720, Windows Game Bar (Win+G) or OBS recording
the browser window; install `ffmpeg` (`winget install ffmpeg`) to trim.

| Time | Shot | What to say |
| --- | --- | --- |
| 0:00–0:20 | Practice page: pick MLE, a topic, answer in two sentences, submit | "Questions come from five reviewed banks; grading is a rubric — here the local distilled model, on a paid key DeepSeek or Claude." |
| 0:20–1:20 | Mock page: paste `example_resume.txt`, pick the role, start; two turns; End early; the report | "The plan is built from the resume only; each probe is graded against a rubric; the report shows which points I hit." |
| 1:20–1:40 | Voice: start live voice, answer, interrupt the interviewer mid-question | "Local Whisper and Kokoro, 2 s end-of-turn silence, barge-in under half a second — all measured." |
| 1:40–2:00 | README results tables, then `docs/plan.md` | "Every number is rendered from a committed results file; the plan records the negative results too." |

Upload unlisted to YouTube and link it from the README's first screen.

## 8. Demo-day checklist

- [ ] `users.json` on the box has the demo key with `daily_llm_calls` and `"log": false`
- [ ] `.env` on the box has no `ANTHROPIC_API_KEY`
- [ ] DeepSeek balance small, auto-recharge off, the demo key is the separate one
- [ ] `/api/meta` shows `retrieval.backend: "hybrid"` and the expected banks
- [ ] A burst returns 429s; a 2 MB JSON body returns 413
- [ ] Budget alert set; instance type t3.small; only 22/80/443 open
- [ ] The README links the live URL and the recording
