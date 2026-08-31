"""Environment, paths, constants, and runtime mode flags."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
PUBLIC_DIR = BASE_DIR / "public"
ENV_PATH = BASE_DIR / ".env"

# One question-bank corpus per track; both share the id/interview/metadata schema.
CORPUS_PATHS = {
    "MLE": BASE_DIR / "rag_ml" / "all_chunks.jsonl",
    "AIE": BASE_DIR / "rag_ai" / "all_chunks.jsonl",
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

# Freemium tiers (Claude mode only): see coach/users.py.
USERS_PATH = Path(os.environ.get("USERS_PATH", BASE_DIR / "users.json"))
USAGE_PATH = Path(os.environ.get("USAGE_PATH", BASE_DIR / "data" / "usage.json"))
PAID_DAILY_QUOTA = int(os.environ.get("PAID_DAILY_QUOTA", "30"))

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
CASCADE_PRED_MAX = 2.5
CASCADE_FRAC_HIT_MAX = 0.25

# "claude" (default, needs ANTHROPIC_API_KEY), "mock" (free, offline), or
# "ollama" (free local model, needs Ollama running at localhost:11434).
# Reassigned by server.main(); always read as config.MODE.
MODE = "claude"
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
