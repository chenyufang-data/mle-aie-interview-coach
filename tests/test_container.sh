#!/usr/bin/env bash
# Container smoke test for the Docker packaging (docker/, docker-compose.yml).
#
# Builds both images, starts the stack with the backend in --mock mode (no
# API key: the offline ML grader), and checks through the nginx frontend:
#   - GET /api/meta answers within CONTAINER_TEST_TIMEOUT seconds (180)
#   - the public banks are listed and the private rag_exp bank is NOT
#   - retrieval.backend is "hybrid" (the bge-small model downloaded). Set
#     CONTAINER_TEST_ALLOW_BM25=1 to downgrade that to a warning on networks
#     where the download is blocked, or CONTAINER_TEST_MODEL_DIR=<dir> to
#     mount a local model copy (e.g. data/models/fastembed) read-only;
#     CONTAINER_TEST_RETRIEVAL_BACKEND=bm25 forces the fallback, to check
#     the downgrade path on a machine where the download works
#   - POST /api/question then POST /api/evaluate answer 200 with a score
# then tears the stack down, volumes included. Prints PASS/FAIL/WARN lines
# and exits non-zero on any FAIL. Uses its own compose project name, so it
# never touches a real deployment's containers or its coach-data volume.
#
#   bash tests/test_container.sh
#   CONTAINER_TEST_ALLOW_BM25=1 bash tests/test_container.sh
set -u
set -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 2
PROJECT="${COMPOSE_PROJECT_NAME:-coach-container-test}"
BASE_URL="${CONTAINER_TEST_URL:-http://localhost:8080}"
TIMEOUT_S="${CONTAINER_TEST_TIMEOUT:-180}"
ALLOW_BM25="${CONTAINER_TEST_ALLOW_BM25:-0}"
MODEL_DIR="${CONTAINER_TEST_MODEL_DIR:-}"
RETRIEVAL_BACKEND="${CONTAINER_TEST_RETRIEVAL_BACKEND:-}"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/coach-container.XXXXXX")"
OVERRIDE="$WORK/override.yml"
FAILURES=0
pass() { echo "PASS: $*"; }
fail() { echo "FAIL: $*"; FAILURES=$((FAILURES + 1)); }
warn() { echo "WARN: $*"; }

# A Python for the JSON checks: the repo venv when present (Windows or
# Linux layout), else whatever the runner provides.
py() {
  if [ -x "$ROOT/.venv/Scripts/python.exe" ]; then "$ROOT/.venv/Scripts/python.exe" "$@"
  elif [ -x "$ROOT/.venv/bin/python" ]; then "$ROOT/.venv/bin/python" "$@"
  elif command -v python3 >/dev/null 2>&1; then python3 "$@"
  else python "$@"
  fi
}

compose() { docker compose -p "$PROJECT" -f docker-compose.yml -f "$OVERRIDE" "$@"; }

cleanup() {
  status=$?
  if [ "$FAILURES" -gt 0 ] || [ "$status" -ne 0 ]; then
    echo "---- backend logs (last 60 lines) ----"
    compose logs --no-color --tail=60 backend 2>/dev/null
    echo "---- frontend logs (last 20 lines) ----"
    compose logs --no-color --tail=20 frontend 2>/dev/null
  fi
  echo "== tearing down (docker compose down -v)"
  compose down -v --remove-orphans >/dev/null 2>&1 || true
  rm -rf "$WORK"
}
trap cleanup EXIT

# Override: --mock backend, and no users.json bind mount (a missing host
# file would be created as a directory named users.json in the repo).
{
  echo "services:"
  echo "  backend:"
  echo '    command: ["python", "server.py", "--mock"]'
  if [ -n "$RETRIEVAL_BACKEND" ]; then
    echo "    environment:"
    echo "      RETRIEVAL_BACKEND: $RETRIEVAL_BACKEND"
  fi
  echo "    volumes: !override"
  echo "      - coach-data:/data"
  if [ -n "$MODEL_DIR" ]; then
    host_dir="$MODEL_DIR"
    case "$host_dir" in /*) ;; *) host_dir="$ROOT/$host_dir" ;; esac
    if command -v cygpath >/dev/null 2>&1; then host_dir="$(cygpath -w "$host_dir")"; fi
    printf "      - '%s:/data/models/fastembed:ro'\n" "$host_dir"
  fi
} > "$OVERRIDE"

echo "== building images (compose project $PROJECT)"
if ! compose build; then fail "docker compose build"; exit 1; fi
pass "images built"

echo "== starting the stack (backend: python server.py --mock)"
if ! compose up -d; then fail "docker compose up"; exit 1; fi

echo "== waiting up to ${TIMEOUT_S}s for $BASE_URL/api/meta"
META="$WORK/meta.json"
deadline=$(( $(date +%s) + TIMEOUT_S ))
until curl -fsS -o "$META" "$BASE_URL/api/meta" 2>/dev/null; do
  if [ "$(date +%s)" -ge "$deadline" ]; then
    fail "GET /api/meta did not answer within ${TIMEOUT_S}s"
    exit 1
  fi
  sleep 3
done
pass "GET /api/meta answered"

# Banks: MLE and AIE are tracked; rag_lists and rag_docs are generated
# locally (gitignored), so they are expected only when this checkout has
# them. rag_exp is private and must never be in the image.
EXPECT="MLE AIE"
for pair in "LISTS:rag_lists" "DOCS:rag_docs"; do
  track="${pair%%:*}"; dir="${pair##*:}"
  if [ -f "$dir/all_chunks.jsonl" ]; then
    EXPECT="$EXPECT $track"
  else
    warn "$dir/all_chunks.jsonl is not in this checkout (generated bank): the $track track is not expected"
  fi
done

py - "$META" "$EXPECT" "$ALLOW_BM25" <<'PY' || FAILURES=$((FAILURES + 1))
import json, sys
meta = json.load(open(sys.argv[1], encoding="utf-8"))
expected = sys.argv[2].split()
allow_bm25 = sys.argv[3] == "1"
kb = meta.get("kb", {})
banks = sorted(kb)
print(f"banks listed: {banks}")
failures = 0
missing = [b for b in expected if b not in banks]
if missing:
    print(f"FAIL: banks missing from /api/meta: {missing}"); failures += 1
else:
    print(f"PASS: public banks present: {expected}")
if "EXP" in banks:
    print("FAIL: the private rag_exp bank is inside the image"); failures += 1
else:
    print("PASS: private rag_exp bank absent")
empty = [b for b in banks if not kb[b].get("chunks")]
if empty:
    print(f"FAIL: banks with no chunks: {empty}"); failures += 1
retrieval = meta.get("retrieval", {})
backend, reason = retrieval.get("backend"), retrieval.get("reason")
if backend == "hybrid":
    print("PASS: retrieval backend is hybrid")
elif allow_bm25:
    print(f"WARN: retrieval backend is {backend!r} ({reason}); accepted because CONTAINER_TEST_ALLOW_BM25=1")
else:
    print(f"FAIL: retrieval backend is {backend!r}, expected hybrid ({reason})"); failures += 1
sys.exit(1 if failures else 0)
PY

# One graded answer: fetch a bank question, then evaluate an answer to it
# (in --mock mode the local distilled grader scores it offline).
Q="$WORK/question.json"
code="$(curl -sS -o "$Q" -w '%{http_code}' -H 'Content-Type: application/json' \
  -X POST "$BASE_URL/api/question" \
  -d '{"source":"kb","role":"MLE","topic":"Any course topic","level":"Mid-level","focus":"precision recall imbalanced classes","exclude":[]}')"
if [ "$code" = "200" ]; then
  pass "POST /api/question -> 200"
else
  fail "POST /api/question -> $code: $(head -c 300 "$Q" 2>/dev/null)"
fi

REQ="$WORK/evaluate_req.json"
py - "$Q" "$REQ" <<'PY'
import json, sys
try:
    q = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception:
    q = {}
body = {
    "role": "MLE", "level": "Mid-level",
    "topic": q.get("what_interviewer_is_testing") or "ML system design",
    "question": q.get("question") or "How do you evaluate a classifier on imbalanced data?",
    "chunk_id": q.get("chunk_id"), "source": "kb", "timeUsed": "1:30",
    "answer": ("Precision is the share of flagged items that are truly positive and "
               "recall is the share of positives we catch. On imbalanced data I look at "
               "the precision-recall curve and PR-AUC rather than accuracy, pick the "
               "threshold from the business cost of false positives versus misses, "
               "use class weights or resampling during training, and validate with "
               "stratified or time-based splits so the estimate is not leaked."),
}
json.dump(body, open(sys.argv[2], "w", encoding="utf-8"))
PY

EVAL="$WORK/evaluate.json"
code="$(curl -sS -o "$EVAL" -w '%{http_code}' -H 'Content-Type: application/json' \
  -X POST "$BASE_URL/api/evaluate" --data-binary "@$REQ")"
if [ "$code" = "200" ]; then
  if py - "$EVAL" <<'PY'
import json, sys
r = json.load(open(sys.argv[1], encoding="utf-8"))
score = r.get("overall_score")
ok = isinstance(score, (int, float)) and not isinstance(score, bool)
print(f"evaluate: overall_score={score!r} graded_by={r.get('graded_by')!r}")
sys.exit(0 if ok else 1)
PY
  then pass "POST /api/evaluate -> 200 with a numeric overall_score"
  else fail "POST /api/evaluate -> 200 but no numeric overall_score: $(head -c 300 "$EVAL")"
  fi
else
  fail "POST /api/evaluate -> $code: $(head -c 300 "$EVAL" 2>/dev/null)"
fi

# Voice capabilities: the image carries the cloud voice subset, so the loop
# is on when the .env names a cloud backend and its key (the demo box) and
# otherwise off with a stated reason (CI has no .env). Either way the page
# must learn which from this endpoint.
VOICE="$WORK/voice.json"
code="$(curl -sS -o "$VOICE" -w '%{http_code}' "$BASE_URL/api/mock/voice")"
if [ "$code" = "200" ] && py - "$VOICE" <<'PY'
import json, sys
caps = json.load(open(sys.argv[1], encoding="utf-8"))
if caps.get("enabled") is True:
    print(f"voice: on - {caps.get('stt_backend')} + {caps.get('tts_backend')}, ws_path={caps.get('ws_path')!r}")
elif caps.get("enabled") is False and caps.get("reason"):
    print(f"voice: off - {caps['reason']}")
else:
    print(f"unexpected voice caps: {caps}"); sys.exit(1)
PY
then pass "GET /api/mock/voice states whether the live loop is up"
else fail "GET /api/mock/voice -> $code: $(head -c 300 "$VOICE" 2>/dev/null)"
fi

if [ "$FAILURES" -gt 0 ]; then
  echo "RESULT: FAIL ($FAILURES failure(s))"
  exit 1
fi
echo "RESULT: PASS"
exit 0
