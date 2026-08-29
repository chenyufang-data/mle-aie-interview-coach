"""Measure candidate cascade rules on the held-out gold rows.

Run:  .venv\\Scripts\\python grader\\cascade_analysis.py

The cascade question: for which answers can the distilled student grade
INSTEAD of Claude without the paid user noticing a quality drop? A rule
routes an answer local when the student is confident; everything else
escalates to the teacher. This script replays the exact grouped split from
train.py, takes the held-out Claude-labeled rows the model never trained on,
and reports, for each candidate rule: how many answers it would keep local
(coverage) and how well the student agrees with Claude on exactly those
answers (within +/-1, MAE).

Pick the operating point, then keep the chosen thresholds in server.py in
sync. Rules are measured on synthetic answers - the population is not real
users, so re-run against data/sessions/real_sessions.jsonl once it has volume.
"""

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

import joblib
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from grader.features import FEATURE_NAMES, FeatureExtractor  # noqa: F401

CORPORA = ("rag_ml", "rag_ai")
DATASET_PATH = BASE_DIR / "grader" / "dataset.jsonl"
TEACHER_PATH = BASE_DIR / "grader" / "labels_teacher.jsonl"
ARTIFACT_PATH = BASE_DIR / "grader" / "model.joblib"


def main():
    chunks_by_id = {}
    for folder in CORPORA:
        with (BASE_DIR / folder / "all_chunks.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    chunk = json.loads(line)
                    chunks_by_id[chunk["id"]] = chunk

    rows = [json.loads(line) for line in DATASET_PATH.open(encoding="utf-8") if line.strip()]
    teacher = {}
    for line in TEACHER_PATH.open(encoding="utf-8"):
        if line.strip():
            record = json.loads(line)
            teacher[record["row_id"]] = record["teacher_score"]

    artifact = joblib.load(ARTIFACT_PATH)
    extractor, model = artifact["extractor"], artifact["model"]

    X, y, groups, is_gold = [], [], [], []
    for row in rows:
        chunk = chunks_by_id.get(row["chunk_id"])
        if chunk is None:
            continue
        X.append(extractor.extract(row["answer"], chunk))
        gold = row["row_id"] in teacher
        y.append(teacher[row["row_id"]] if gold else row["label"])
        is_gold.append(gold)
        groups.append(row["chunk_id"])
    X, y, is_gold = np.array(X), np.array(y, dtype=float), np.array(is_gold)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, test_idx = next(splitter.split(X, y, groups))
    gold_idx = np.array([i for i in test_idx if is_gold[i]])
    print(f"Gold held-out rows: {len(gold_idx)}")

    pred = model.predict(X[gold_idx])
    truth = y[gold_idx]
    frac_hit = X[gold_idx, FEATURE_NAMES.index("kp_frac_hit")]
    mistake = X[gold_idx, FEATURE_NAMES.index("mistake_sim_max")]

    def report(name, mask):
        n = int(mask.sum())
        if n == 0:
            print(f"{name:52s}  coverage  0%")
            return
        rounded = np.clip(np.rint(pred[mask]), 1, 10)
        within1 = float(np.mean(np.abs(rounded - np.clip(np.rint(truth[mask]), 1, 10)) <= 1))
        mae = float(np.mean(np.abs(pred[mask] - truth[mask])))
        print(f"{name:52s}  coverage {n:3d}/{len(gold_idx)} ({n/len(gold_idx):4.0%})   "
              f"within+/-1 {within1:4.0%}   MAE {mae:.2f}")

    print("\nBaseline (route everything local):")
    report("all rows", np.ones_like(truth, dtype=bool))

    print("\nCandidate rules (route local when ...):")
    for hi in (7.5, 8.0):
        for fh in (0.6, 0.75):
            report(f"pred>={hi} & frac_hit>={fh}", (pred >= hi) & (frac_hit >= fh))
    for lo in (2.5, 3.0, 3.5):
        for fl in (0.15, 0.25):
            report(f"pred<={lo} & frac_hit<={fl}", (pred <= lo) & (frac_hit <= fl))
    for lo, hi in ((3.0, 8.0), (3.5, 7.5), (3.0, 7.5)):
        both = ((pred <= lo) & (frac_hit <= 0.25)) | ((pred >= hi) & (frac_hit >= 0.6))
        report(f"extremes lo<={lo} hi>={hi} (frac-guarded)", both)
        guarded = both & (mistake <= np.median(mistake))
        report(f"  + mistake_sim below median", guarded)

    print("\nEscalation check - agreement on the rows a rule would SEND TO CLAUDE")
    print("(low agreement there is GOOD: it means the rule escalates the hard ones):")
    for lo, hi in ((3.0, 8.0), (3.5, 7.5)):
        local = ((pred <= lo) & (frac_hit <= 0.25)) | ((pred >= hi) & (frac_hit >= 0.6))
        report(f"escalated complement of lo<={lo} hi>={hi}", ~local)
    return 0


if __name__ == "__main__":
    sys.exit(main())
