"""Environment, paths, constants, and runtime mode flags."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
PUBLIC_DIR = BASE_DIR / "public"
ENV_PATH = BASE_DIR / ".env"

# One question-bank corpus per track; all share the id/interview/metadata
# schema. EXP (real gathered interview questions, grader/ingest_questions.py)
# is PRIVATE - gitignored, mounted from private storage on deploys - so
# kb.load_chunks simply skips it where the file is absent and the app runs
# as a two-track install.
CORPUS_PATHS = {
    "MLE": BASE_DIR / "rag_ml" / "all_chunks.jsonl",
    "AIE": BASE_DIR / "rag_ai" / "all_chunks.jsonl",
    "EXP": BASE_DIR / "rag_exp" / "all_chunks.jsonl",
    # Public GitHub interview lists (MIT / Apache-2.0), rubrics by the
    # teacher; separate from the course banks so the grounding experiment
    # can measure with and without it (docs/plan.md §1.8).
    "LISTS": BASE_DIR / "rag_lists" / "all_chunks.jsonl",
    # rag_docs: rubrics written from sections of primary documentation on
    # the MLOps topics the course banks lack (grader/ingest_docs.py); same
    # private/generated treatment as rag_lists.
    "DOCS": BASE_DIR / "rag_docs" / "all_chunks.jsonl",
}
DEFAULT_MODEL = "claude-opus-4-8"

# Trained distilled grader for mock mode (grader/train.py artifact).
GRADER_PATH = BASE_DIR / "grader" / "model.joblib"

# Every real (non-mock) graded exchange is appended here: each row is a gold
# (answer, teacher score) pair for evaluating and later retraining the
# distilled grader, and the practice history for progress features.
# Overridable so Docker can point it at a persistent volume.
REAL_SESSIONS_PATH = Path(
    os.environ.get("REAL_SESSIONS_PATH", BASE_DIR / "data" / "sessions" / "real_sessions.jsonl")
)

# Free-tier answers land here UNLABELED (their only score is the student
# model's own prediction, which must never be used as a training label -
# that would be self-distillation of the model's own errors). They become
# training data only if later graded by the teacher (label_teacher.py-style
# active learning). Kept separate from the gold-pair log for that reason.
FREE_SESSIONS_PATH = Path(
    os.environ.get("FREE_SESSIONS_PATH", BASE_DIR / "data" / "sessions" / "free_sessions.jsonl")
)

# Phase 3 mock-interview log - OPT-IN per session, unlike the practice
# logs above: a mock's plan and transcript embed resume-derived content,
# so nothing is written unless the request itself consents (plan section 7).
MOCK_SESSIONS_PATH = Path(
    os.environ.get("MOCK_SESSIONS_PATH", BASE_DIR / "data" / "sessions" / "mock_sessions.jsonl")
)

# Freemium tiers (Claude mode only): see coach/users.py.
USERS_PATH = Path(os.environ.get("USERS_PATH", BASE_DIR / "users.json"))
USAGE_PATH = Path(os.environ.get("USAGE_PATH", BASE_DIR / "data" / "usage.json"))
PAID_DAILY_QUOTA = int(os.environ.get("PAID_DAILY_QUOTA", "30"))
# Daily LLM-call budgets (2026-09-07). PAID_DAILY_QUOTA meters Claude only;
# DeepSeek was quota-free by design - right for the owner's key, wrong for
# a demo key on a public URL. Two caps close that: a per-key
# "daily_llm_calls" in users.json (every engine; 0 or absent = unlimited)
# and this server-wide cap over all keys (0 = unlimited). A refused call
# grades locally with the reason "budget"; see coach/users.py take_call.
LLM_DAILY_CAP = int(os.environ.get("LLM_DAILY_CAP", "0"))
# Daily live-voice budgets (2026-09-08, cloud voice on the public demo).
# The voice loop streams audio to a paid vendor (Deepgram on the demo box)
# for as long as a session runs, so it is metered in minutes, not calls: a
# per-key "daily_voice_minutes" in users.json (0 or absent = unlimited),
# this server-wide cap over all keys (0 = unlimited), and a hard length per
# session. A session is charged in ticks while it runs and ends, with a
# spoken goodbye, within one tick of an allowance running out; a
# re-transcription clip (POST /api/mock/transcribe) is charged by its
# length. See coach/users.py take_voice and coach/voice/loop.py meter.
VOICE_DAILY_MINUTES = int(os.environ.get("VOICE_DAILY_MINUTES", "0"))
VOICE_SESSION_MAX_MINUTES = int(os.environ.get("VOICE_SESSION_MAX_MINUTES", "20"))
# Binding beyond localhost without users.json would grade every stranger's
# request with the owner's keys; server.py refuses that unless this (or
# --allow-anonymous-llm) says the network is trusted.
ALLOW_ANONYMOUS_LLM = os.environ.get("ALLOW_ANONYMOUS_LLM", "0") == "1"

# Smart cascade for paid users: answers the student grades reliably are served
# locally without spending an LLM call (Claude quota, or a DeepSeek request
# when that is the paid workhorse). The rule is MEASURED, not guessed
# (grader/cascade_analysis.py, 121 held-out gold rows): routing only
# clearly-below-rubric answers keeps ~12% of evaluations local at 100%
# within-+/-1 teacher agreement (MAE 0.48). The tempting high-score route
# measured at 47% agreement — polished-looking answers are exactly where
# lexical features get fooled — so only the low side ships. A request can
# force the full Claude evaluation with "force_llm": true.
PAID_CASCADE = os.environ.get("PAID_CASCADE", "1") != "0"

# Optional SLM grader for the local tier (roadmap step 3, coach/slm.py): a
# vLLM server holding the fine-tuned Qwen3 model measured in grader/slm/.
# Off unless SLM_URL is set; a silent or slow server degrades to the sklearn
# grade. The public demo box has no GPU and leaves this unset.
SLM_URL = os.environ.get("SLM_URL", "").strip()
SLM_MODEL = os.environ.get("SLM_MODEL", "slm")
SLM_TIMEOUT_S = float(os.environ.get("SLM_TIMEOUT_S", "2"))
SLM_BACKOFF_S = float(os.environ.get("SLM_BACKOFF_S", "30"))
CASCADE_PRED_MAX = 2.5
CASCADE_FRAC_HIT_MAX = 0.25

# "claude" (default, needs ANTHROPIC_API_KEY), "mock" (free, offline), or
# "ollama" (free local model, needs Ollama running at localhost:11434).
# Reassigned by server.main(); always read as config.MODE.
MODE = "claude"
# True when the live loop WebSocket is up (auto-detected at startup, or
# forced with --voice); read as config.VOICE_ENABLED by GET /api/mock/voice.
# When it is False, VOICE_DISABLED_REASON says why (missing optional deps,
# --no-voice, port taken) so the mock page can explain the disabled option.
VOICE_ENABLED = False
VOICE_DISABLED_REASON = None


def voice_ws_path():
    """Voice loop behind a reverse proxy. When VOICE_WS_PATH is set (e.g.
    "/ws/voice", the path docker/nginx.conf forwards to the loop's port),
    GET /api/mock/voice adds it to the capabilities and the browser dials
    wss://<page host><path> on its own origin instead of ws://<host>:8765,
    which an https page cannot open. Read at call time, like deepseek_model,
    so a value from .env (loaded after import) counts too. None when unset
    or not an absolute path."""
    path = os.environ.get("VOICE_WS_PATH", "").strip()
    return path if path.startswith("/") else None
# Practice-question retrieval. "auto" (default) serves the hybrid BM25 +
# bge-small retriever (retrieval_dense.py) when fastembed and the model are
# available and falls back to BM25 with a stated reason; "bm25" forces the
# fallback; "hybrid" forces the stack and fails loudly. The swap was earned
# by the pre-registered rule in docs/plan.md (results:
# docs/retrieval_evaluation.md). RETRIEVAL_ACTIVE reports what is serving.
RETRIEVAL_BACKEND = os.environ.get("RETRIEVAL_BACKEND", "auto").lower()
RETRIEVAL_ACTIVE = "bm25"
RETRIEVAL_DISABLED_REASON = None

# Mock-interview probes come from the RESUME only; the job description
# decides which resume claims get probed and how deep (jd_emphasis), never
# adds topics the resume lacks. FROZEN OFF 2026-09-04 (user decision): the
# retrieval experiment's set C showed JD "role theme" probes ground fairly
# in only 26-43% of cases and hold 7 of 9 content gaps, because the banks
# hold nothing beyond what a resume claims. Unfreeze (MOCK_BEYOND_RESUME=1)
# only after a bank of beyond-resume questions exists to ground them.
MOCK_BEYOND_RESUME = os.environ.get("MOCK_BEYOND_RESUME", "0") == "1"
OLLAMA_MODEL = "llama3.2"
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"

# Paid-tier workhorse judge (tiered Claude mode only). With DEEPSEEK_API_KEY
# in .env, paid evaluations default to DeepSeek V4 Flash - measured against
# the Claude teacher on the 121 held-out gold rows (grader/judge_agreement.py):
# 94% within-+/-1, QWK 0.93, at ~1/200th of Opus-tier cost. Claude remains the
# distillation teacher and serves "Always Claude" requests under the daily
# quota. Without the key, paid routing behaves exactly as before (all Claude).
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"


def deepseek_model():
    return os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")


def deepseek_available():
    return MODE == "claude" and bool(os.environ.get("DEEPSEEK_API_KEY"))


def load_env_file():
    if not ENV_PATH.exists():
        return

    for raw_line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            # Tolerate a bare Anthropic key pasted without a variable name.
            if line.startswith("sk-ant-"):
                os.environ.setdefault("ANTHROPIC_API_KEY", line)
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)
