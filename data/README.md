# data/ — personal and runtime data (gitignored)

Everything the app reads or writes that is personal or generated at runtime
lives here, so one `.gitignore` rule (`data/*`) keeps it all out of the repo.
Only this README is tracked. In Docker the same layout is mounted at `/data`
from the `coach-data` volume.

```
data/
├── resume/          resume text produced by resume_parser.py (data/resume/<name>.txt)
├── interview_exp/   interview questions you gather (spreadsheets, notes) — the
│                    source for the rag_exp/ question bank; keep adding to it
├── sessions/        practice logs written by server.py:
│                      real_sessions.jsonl   LLM-graded answers (Claude rows = gold pairs
│                                            for grader/evaluate_on_real.py)
│                      free_sessions.jsonl   free-tier answers, stored UNLABELED
│                      mock_sessions.jsonl   mock-interview sessions (opt-in, planned)
├── stt_audio/       Phase 0 transcription experiment (grader/stt_eval.py), one
│                    folder per audio set:
│                      human/            takes from public/stt_record.html
│                      synth_<voice>/    ElevenLabs TTS readings of the same items
│                    each with audio/, manifest.json (durations, browser transcript),
│                    transcripts.jsonl (every condition's output, the re-run cache)
│                    and judge.jsonl (cached DeepSeek grades)
├── stt/             lexicon_extra.json — private terms (resume, employers,
│                    product names) merged into grader/stt_lexicon.json when present
└── usage.json       per-key daily Claude quota state
```

Paths are overridable with `REAL_SESSIONS_PATH`, `FREE_SESSIONS_PATH`,
`USAGE_PATH` (see `docs/backend.md`). Secrets stay at the repo root, not
here: `.env` (API keys) and `users.json` (access keys) — both gitignored.
