#!/usr/bin/env bash
# The whole GPU side of the SLM experiment, in order, on WSL2 (~/.venvs/slm).
#   bash grader/slm/run_all.sh [LOG_FILE]
# Every step appends to the log and a failed step does not stop the rest;
# the report at the end aggregates whatever run files exist. Prompt records
# (grader/slm/prepare.py) and the sklearn arm (grader/slm/sklearn_arm.py)
# run on the Windows side first, since they need the main checkout's venv.
set -u
REPO="${REPO:-/mnt/c/Users/chenyu/Documents/mle-aie-interview-coach}"
PRIVATE="${PRIVATE:-/mnt/c/Users/chenyu/Documents/mle-aie-interview-coach-private}"
WEIGHTS="${WEIGHTS:-$HOME/slm_runs}"
LOG="${1:-$WEIGHTS/run_all.log}"
SEEDS="${SEEDS:-42 1 2 3 4}"
ARMS="${ARMS:-deberta qwen1.7b qwen4b}"
mkdir -p "$WEIGHTS" "$(dirname "$LOG")"
source "$HOME/.venvs/slm/bin/activate"
cd "$REPO"
run() {
  echo "== $(date +%H:%M:%S) $*" | tee -a "$LOG"
  "$@" 2>&1 | tee -a "$LOG"
  echo "== exit ${PIPESTATUS[0]}" | tee -a "$LOG"
}
for arm in $ARMS; do
  for seed in $SEEDS; do
    run python grader/slm/train.py --arm "$arm" --seed "$seed" --private-dir "$PRIVATE" --weights-dir "$WEIGHTS"
  done
done
run python grader/slm/train.py --arm qwen1.7b --seed 42 --silver --private-dir "$PRIVATE" --weights-dir "$WEIGHTS"
run python grader/slm/report.py
echo "== $(date +%H:%M:%S) ALL DONE" | tee -a "$LOG"
