"""Train the distilled answer grader.

Run:  .venv\\Scripts\\python grader\\train.py

Fits two students on the synthetic dataset - a Ridge regression baseline
(e-rater style linear model) and a HistGradientBoostingRegressor - compares
them against the old keyword-coverage heuristic on a held-out test set, and
saves the better one plus the fitted FeatureExtractor to grader/model.joblib.

The split is grouped BY CHUNK: all answers to a question land on the same side
of the split, so the model is always evaluated on questions whose rubric it
never saw during training. Splitting by row instead would leak the rubric.

If grader/labels_teacher.jsonl exists (from grader/label_teacher.py), those
gold teacher scores override the construction labels and get 3x sample weight.
Until then, metrics are optimistic: construction labels are partly derived
from the same coverage signal the features measure.
"""

import json
import sys
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

import joblib
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import cohen_kappa_score, mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from grader.features import FEATURE_NAMES, FeatureExtractor

CORPORA = {"ml": "rag_ml", "ai": "rag_ai"}
DATASET_PATH = BASE_DIR / "grader" / "dataset.jsonl"
TEACHER_PATH = BASE_DIR / "grader" / "labels_teacher.jsonl"
ARTIFACT_PATH = BASE_DIR / "grader" / "model.joblib"
TEACHER_WEIGHT = 3.0
SUBSCORE_NAMES = ("technical_depth", "structure", "practical_judgment", "communication")


def clip_scores(values):
    return np.clip(np.rint(values), 1, 10).astype(int)


def metrics(y_true, y_pred):
    rounded = clip_scores(y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "within1": float(np.mean(np.abs(rounded - clip_scores(y_true)) <= 1)),
        "spearman": float(spearmanr(y_true, y_pred).statistic),
        "qwk": float(cohen_kappa_score(
            clip_scores(y_true), rounded, weights="quadratic", labels=list(range(1, 11))
        )),
    }


def print_table(results):
    print(f"{'model':26s} {'MAE':>6s} {'+/-1':>6s} {'Spearman':>9s} {'QWK':>6s}")
    for name, scores in results.items():
        print(f"{name:26s} {scores['mae']:6.2f} {scores['within1']:6.0%} "
              f"{scores['spearman']:9.3f} {scores['qwk']:6.3f}")


def main():
    chunks_by_id = {}
    for folder in CORPORA.values():
        with (BASE_DIR / folder / "all_chunks.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    chunk = json.loads(line)
                    chunks_by_id[chunk["id"]] = chunk

    if not DATASET_PATH.exists():
        print("No dataset found - run grader\\generate_answers.py first.")
        return 1
    with DATASET_PATH.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]

    teacher, teacher_sub = {}, {}
    if TEACHER_PATH.exists():
        with TEACHER_PATH.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    teacher[record["row_id"]] = record["teacher_score"]
                    scores = record.get("scores") or {}
                    if all(name in scores for name in SUBSCORE_NAMES):
                        teacher_sub[record["row_id"]] = scores

    extractor = FeatureExtractor().fit(chunks_by_id.values())

    X, y, groups, weights, sources, row_ids = [], [], [], [], [], []
    for row in rows:
        chunk = chunks_by_id.get(row["chunk_id"])
        if chunk is None:
            continue
        row_ids.append(row["row_id"])
        X.append(extractor.extract(row["answer"], chunk))
        if row["row_id"] in teacher:
            y.append(teacher[row["row_id"]])
            weights.append(TEACHER_WEIGHT)
            sources.append("teacher")
        else:
            y.append(row["label"])
            weights.append(1.0)
            sources.append("construction")
        groups.append(row["chunk_id"])
    X = np.array(X)
    y = np.array(y, dtype=float)
    weights = np.array(weights)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))

    frac_hit = X[:, FEATURE_NAMES.index("kp_frac_hit")]
    baseline_pred = np.clip(np.rint(2 + 8 * frac_hit[test_idx]), 1, 10)

    candidates = {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
            l2_regularization=1.0, random_state=42,
        ),
    }

    results = {"keyword_baseline": metrics(y[test_idx], baseline_pred)}
    fitted = {}
    for name, model in candidates.items():
        if name == "ridge":
            model.fit(X[train_idx], y[train_idx], ridge__sample_weight=weights[train_idx])
        else:
            model.fit(X[train_idx], y[train_idx], sample_weight=weights[train_idx])
        fitted[name] = model
        results[name] = metrics(y[test_idx], model.predict(X[test_idx]))

    print(f"Dataset: {len(y)} rows over {len(set(groups))} chunks "
          f"({Counter(sources)['teacher']} teacher-labeled, weight {TEACHER_WEIGHT}x)")
    print(f"Split by chunk: {len(train_idx)} train / {len(test_idx)} test rows\n")
    print_table(results)

    # The Level-2 evaluation: agreement with real Claude scores, restricted to
    # held-out teacher-labeled rows (their chunks were never trained on).
    sources_arr = np.array(sources)
    gold_idx = np.array([i for i in test_idx if sources_arr[i] == "teacher"])
    gold_results = None
    if len(gold_idx):
        gold_baseline = np.clip(np.rint(2 + 8 * frac_hit[gold_idx]), 1, 10)
        gold_results = {"keyword_baseline": metrics(y[gold_idx], gold_baseline)}
        for name, model in fitted.items():
            gold_results[name] = metrics(y[gold_idx], model.predict(X[gold_idx]))
        print(f"\nGold teacher agreement ({len(gold_idx)} held-out Claude-labeled rows):")
        print_table(gold_results)

    best_name = max(candidates, key=lambda name: results[name]["spearman"])

    # Multi-output subscores: the teacher's four subscores are stored with
    # every gold label, so per-subscore models can replace the display
    # heuristics (overall +/- word-count offsets). Trained on teacher rows
    # only; the "overall-as-proxy" column is what the heuristic effectively
    # showed, so the model must beat it to earn its place.
    sub_models, sub_metrics = {}, {}
    sub_ok = np.array([rid in teacher_sub for rid in row_ids])
    sub_train = np.array([i for i in train_idx if sub_ok[i]])
    sub_test = np.array([i for i in test_idx if sub_ok[i]])
    if len(sub_train) >= 100 and len(sub_test) >= 20:
        overall_proxy = fitted[best_name].predict(X[sub_test])
        print(f"\nSubscore models ({len(sub_train)} teacher rows train / {len(sub_test)} test):")
        for name in SUBSCORE_NAMES:
            target = np.array([teacher_sub[row_ids[i]][name] for i in sub_train], dtype=float)
            truth = np.array([teacher_sub[row_ids[i]][name] for i in sub_test], dtype=float)
            model = HistGradientBoostingRegressor(
                max_iter=200, learning_rate=0.06, max_leaf_nodes=31,
                l2_regularization=1.0, random_state=42,
            )
            model.fit(X[sub_train], target)
            mae_model = float(np.mean(np.abs(model.predict(X[sub_test]) - truth)))
            mae_proxy = float(np.mean(np.abs(overall_proxy - truth)))
            sub_models[name] = model
            sub_metrics[name] = {"mae": mae_model, "mae_overall_proxy": mae_proxy}
            print(f"  {name:20s} MAE {mae_model:.2f}  (overall-as-proxy {mae_proxy:.2f})")

    artifact = {
        "model": fitted[best_name],
        "model_name": best_name,
        "extractor": extractor,
        "feature_names": FEATURE_NAMES,
        "metrics": results,
        "gold_metrics": gold_results,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "label_sources": dict(Counter(sources)),
        "subscore_models": sub_models or None,
        "subscore_metrics": sub_metrics or None,
    }
    joblib.dump(artifact, ARTIFACT_PATH)
    print(f"\nSaved {best_name} to {ARTIFACT_PATH.relative_to(BASE_DIR)}")
    if not teacher:
        print("Note: labels are construction-only, so these metrics are optimistic. "
              "Run grader\\label_teacher.py for gold labels, then retrain.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
