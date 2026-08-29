"""Measure local-grader agreement with Claude on REAL practice answers.

Run:  .venv\\Scripts\\python grader\\evaluate_on_real.py

Reads data/sessions/real_sessions.jsonl (written by server.py whenever
a real, non-mock evaluation happens) and compares the local distilled grader's
prediction against the score Claude actually gave for the same answer. This is
the honest test-distribution evaluation: your writing style, under time
pressure, on questions you actually practiced.

Rows are used only when they were graded by Claude (Ollama scores are not the
teacher), reference a knowledge-base chunk (the local grader needs a rubric),
and are not follow-ups (follow-ups have no rubric of their own).
"""

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

import joblib

from grader import features  # noqa: F401  (unpickling needs FeatureExtractor importable)

LOG_PATH = BASE_DIR / "data" / "sessions" / "real_sessions.jsonl"
ARTIFACT_PATH = BASE_DIR / "grader" / "model.joblib"
MIN_ROWS_FOR_RANK = 5


def main():
    if not ARTIFACT_PATH.exists():
        print("No grader artifact - run grader\\generate_answers.py then grader\\train.py.")
        return 1
    if not LOG_PATH.exists():
        print("No real sessions logged yet. Practice with the real server (no --mock) "
              "on knowledge-base questions and rows will accumulate automatically.")
        return 1

    chunks_by_id = {}
    for folder in ("rag_ml", "rag_ai"):
        with (BASE_DIR / folder / "all_chunks.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    chunk = json.loads(line)
                    chunks_by_id[chunk["id"]] = chunk

    artifact = joblib.load(ARTIFACT_PATH)
    extractor, model = artifact["extractor"], artifact["model"]

    usable, skipped = [], 0
    with LOG_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            chunk = chunks_by_id.get(row.get("chunk_id") or "")
            if (row.get("graded_by") != "claude" or chunk is None
                    or row.get("source") == "followup" or not row.get("answer")):
                skipped += 1
                continue
            usable.append((row, chunk))

    if not usable:
        print(f"No usable rows yet ({skipped} skipped: mock/ollama/follow-up/non-KB). "
              "Practice knowledge-base questions with the real server to collect gold pairs.")
        return 1

    print(f"{'claude':>7s} {'local':>6s} {'diff':>5s}  question")
    diffs, pairs = [], []
    for row, chunk in usable:
        claude_score = row["evaluation"]["overall_score"]
        local = float(model.predict([extractor.extract(row["answer"], chunk)])[0])
        local_rounded = max(1, min(10, round(local)))
        diffs.append(abs(local_rounded - claude_score))
        pairs.append((claude_score, local))
        print(f"{claude_score:7d} {local_rounded:6d} {local_rounded - claude_score:+5d}  "
              f"{row['question'][:70]}")

    n = len(diffs)
    mae = sum(diffs) / n
    within1 = sum(1 for d in diffs if d <= 1) / n
    print(f"\nRows: {n} ({skipped} skipped)   MAE: {mae:.2f}   within +/-1: {within1:.0%}")
    if n >= MIN_ROWS_FOR_RANK:
        from scipy.stats import spearmanr

        rank = spearmanr([p[0] for p in pairs], [p[1] for p in pairs]).statistic
        print(f"Spearman: {rank:.3f}")
    else:
        print(f"(Spearman reported once {MIN_ROWS_FOR_RANK}+ rows accumulate.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
