"""Train the distilled answer grader.

Run:  .venv\\Scripts\\python grader\\train.py                       (retrain + save the artifact)
      .venv\\Scripts\\python grader\\train.py --no-save --evaluate-artifact
                                    (metrics only: retrain in memory AND score the
                                     shipped grader/model.joblib on the same rows)

Every run writes grader/train_results.json - the numbers the README cites.
The synthetic dataset lives in the private checkout; point at it with
--dataset PATH or GRADER_DATASET=PATH.

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

import argparse
import hashlib
import json
import os
import platform
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

import joblib
import numpy as np
import sklearn
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import cohen_kappa_score, f1_score, mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from grader.features import FEATURE_NAMES, KP_FEATURE_NAMES, FeatureExtractor

CORPORA = {"ml": "rag_ml", "ai": "rag_ai"}
DATASET_PATH = BASE_DIR / "grader" / "dataset.jsonl"
TEACHER_PATH = BASE_DIR / "grader" / "labels_teacher.jsonl"
KEYPOINTS_PATH = BASE_DIR / "grader" / "labels_keypoints.jsonl"
ARTIFACT_PATH = BASE_DIR / "grader" / "model.joblib"
RESULTS_PATH = BASE_DIR / "grader" / "train_results.json"
TEACHER_WEIGHT = 3.0
SUBSCORE_NAMES = ("technical_depth", "structure", "practical_judgment", "communication")
KP_AGG_NAMES = ("kp_pred_frac_hit", "kp_pred_frac_partial",
                "kp_pred_phit_mean", "kp_pred_phit_min")


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


def parse_args():
    parser = argparse.ArgumentParser(description="Train / evaluate the distilled grader")
    parser.add_argument("--dataset", default=os.environ.get("GRADER_DATASET") or str(DATASET_PATH),
                        help="synthetic answers (grader/dataset.jsonl; private checkout)")
    parser.add_argument("--results", default=str(RESULTS_PATH),
                        help="where to write the metrics JSON")
    parser.add_argument("--no-save", action="store_true",
                        help="do not overwrite grader/model.joblib")
    parser.add_argument("--evaluate-artifact", action="store_true",
                        help="also score the shipped grader/model.joblib on the held-out rows")
    return parser.parse_args()


def kp_eval(pred, truth):
    return {
        "acc3": float(np.mean(pred == truth)),
        "macro_f1": float(f1_score(truth, pred, average="macro")),
        "hit_f1": float(f1_score(truth == "hit", pred == "hit")),
    }


def evaluate_artifact(artifact, kept, chunks_by_id, gold_idx, y, kp_labels, test_set):
    """Score the SHIPPED artifact with its own fitted extractor on the same
    held-out gold rows (and held-out key points) the retrain is scored on."""
    extractor, model = artifact["extractor"], artifact["model"]
    X = np.array([extractor.extract(kept[i]["answer"], chunks_by_id[kept[i]["chunk_id"]])
                  for i in gold_idx])
    truth = y[gold_idx]
    frac_hit = X[:, artifact["feature_names"].index("kp_frac_hit")]
    out = {
        "model_name": artifact["model_name"],
        "n_gold": int(len(gold_idx)),
        "gold": {
            "keyword_baseline": metrics(truth, np.clip(np.rint(2 + 8 * frac_hit), 1, 10)),
            artifact["model_name"]: metrics(truth, model.predict(X)),
        },
    }
    kp_classifier = artifact.get("kp_classifier")
    if kp_classifier is not None and kp_labels:
        feats, verdicts = [], []
        for i in test_set:
            row = kept[i]
            chunk = chunks_by_id[row["chunk_id"]]
            v = kp_labels.get(row["row_id"])
            if not v or len(v) != len(chunk["interview"]["key_points"]):
                continue
            for f, verdict in zip(extractor.keypoint_features(row["answer"], chunk), v):
                feats.append(f)
                verdicts.append(verdict)
        if feats:
            kp_X, kp_truth = np.array(feats), np.array(verdicts)
            cover = kp_X[:, artifact["kp_feature_names"].index("cover_best")]
            lexical = np.where(cover >= 0.6, "hit", np.where(cover >= 0.35, "partial", "miss"))
            out["keypoints"] = {"classifier": kp_eval(kp_classifier.predict(kp_X), kp_truth),
                                "lexical_threshold": kp_eval(lexical, kp_truth),
                                "n_test": int(len(kp_truth))}
        stacked = artifact.get("stacked_model")
        if stacked is not None:
            classes = list(kp_classifier.classes_)
            hit_col = classes.index("hit")
            agg = np.zeros((len(gold_idx), len(artifact["kp_agg_names"])))
            for k, i in enumerate(gold_idx):
                f = extractor.keypoint_features(kept[i]["answer"], chunks_by_id[kept[i]["chunk_id"]])
                if not f:
                    continue
                proba = kp_classifier.predict_proba(np.array(f))
                pred = np.array(classes)[np.argmax(proba, axis=1)]
                agg[k] = [float(np.mean(pred == "hit")), float(np.mean(pred == "partial")),
                          float(proba[:, hit_col].mean()), float(proba[:, hit_col].min())]
            out["gold"]["stacked"] = metrics(truth, stacked.predict(np.hstack([X, agg])))
    return out


def main():
    args = parse_args()
    dataset_path = Path(args.dataset)
    chunks_by_id = {}
    for folder in CORPORA.values():
        with (BASE_DIR / folder / "all_chunks.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    chunk = json.loads(line)
                    chunks_by_id[chunk["id"]] = chunk

    if not dataset_path.exists():
        print(f"No dataset at {dataset_path} - run grader\\generate_answers.py first, "
              "or point --dataset / GRADER_DATASET at the private checkout's copy.")
        return 1
    with dataset_path.open(encoding="utf-8") as handle:
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

    X, y, groups, weights, sources, row_ids, kept = [], [], [], [], [], [], []
    for row in rows:
        chunk = chunks_by_id.get(row["chunk_id"])
        if chunk is None:
            continue
        kept.append(row)
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

    # Per-key-point semantic coverage classifier, trained on the teacher's
    # hit/partial/miss verdicts (grader/label_keypoints.py). It replaces the
    # fixed 0.35 lexical threshold behind the hit/miss lists, and its per-row
    # aggregates feed a stacked overall model. Both must beat their baseline
    # on the held-out gold rows to ship.
    kp_classifier = kp_metrics = stacked_model = stacked_metrics = None
    kp_labels = {}
    if KEYPOINTS_PATH.exists():
        with KEYPOINTS_PATH.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    if record.get("pass", "main") == "main":
                        kp_labels[record["row_id"]] = record["verdicts"]

        train_set, test_set = set(train_idx), set(test_idx)
        kp_train, kp_test = ([], []), ([], [])
        for i, row in enumerate(kept):
            verdicts = kp_labels.get(row["row_id"])
            chunk = chunks_by_id[row["chunk_id"]]
            if not verdicts or len(verdicts) != len(chunk["interview"]["key_points"]):
                continue
            side = kp_train if i in train_set else kp_test if i in test_set else None
            if side is None:
                continue
            for feats, verdict in zip(
                    extractor.keypoint_features(row["answer"], chunk), verdicts):
                side[0].append(feats)
                side[1].append(verdict)

        if len(kp_train[0]) >= 500 and len(kp_test[0]) >= 100:
            kp_classifier = HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                l2_regularization=1.0, random_state=42,
            )
            kp_classifier.fit(np.array(kp_train[0]), np.array(kp_train[1]))
            kp_X_test = np.array(kp_test[0])
            kp_truth = np.array(kp_test[1])
            kp_pred = kp_classifier.predict(kp_X_test)
            cover = kp_X_test[:, KP_FEATURE_NAMES.index("cover_best")]
            base_pred = np.where(cover >= 0.6, "hit",
                                 np.where(cover >= 0.35, "partial", "miss"))

            kp_metrics = {"classifier": kp_eval(kp_pred, kp_truth),
                          "lexical_threshold": kp_eval(base_pred, kp_truth),
                          "n_train": len(kp_train[0]), "n_test": len(kp_test[0])}
            print(f"\nKey-point coverage ({kp_metrics['n_train']} labeled points "
                  f"train / {kp_metrics['n_test']} held-out):")
            for name in ("classifier", "lexical_threshold"):
                m = kp_metrics[name]
                print(f"  {name:18s} 3-class acc {m['acc3']:4.0%}  "
                      f"macro-F1 {m['macro_f1']:.2f}  hit-F1 {m['hit_f1']:.2f}")

            # Stacked overall model: base features + the classifier's per-row
            # coverage aggregates. Gold rows decide whether it replaces the
            # plain model for scoring (the cascade keeps the plain model - its
            # thresholds were measured against it).
            classes = list(kp_classifier.classes_)
            hit_col = classes.index("hit")
            agg = np.zeros((len(kept), len(KP_AGG_NAMES)))
            for i, row in enumerate(kept):
                feats = extractor.keypoint_features(
                    row["answer"], chunks_by_id[row["chunk_id"]])
                if not feats:
                    continue
                proba = kp_classifier.predict_proba(np.array(feats))
                pred = np.array(classes)[np.argmax(proba, axis=1)]
                agg[i] = [float(np.mean(pred == "hit")),
                          float(np.mean(pred == "partial")),
                          float(proba[:, hit_col].mean()),
                          float(proba[:, hit_col].min())]
            X_ext = np.hstack([X, agg])
            stacked = HistGradientBoostingRegressor(
                max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                l2_regularization=1.0, random_state=42,
            )
            stacked.fit(X_ext[train_idx], y[train_idx],
                        sample_weight=weights[train_idx])
            if len(gold_idx):
                plain_gold = gold_results[best_name]
                stacked_gold = metrics(y[gold_idx], stacked.predict(X_ext[gold_idx]))
                print(f"\nStacked overall model (gold rows): "
                      f"MAE {stacked_gold['mae']:.2f} vs {plain_gold['mae']:.2f}, "
                      f"within+/-1 {stacked_gold['within1']:.0%} vs {plain_gold['within1']:.0%}, "
                      f"QWK {stacked_gold['qwk']:.3f} vs {plain_gold['qwk']:.3f}")
                stacked_metrics = {"gold": stacked_gold, "gold_plain": plain_gold}
                if (stacked_gold["within1"] + stacked_gold["qwk"]
                        > plain_gold["within1"] + plain_gold["qwk"]):
                    stacked_model = stacked
                    print("  -> stacked model SHIPS (scores the UI; cascade keeps the plain model).")
                else:
                    print("  -> stacked model does NOT ship (no gold improvement).")

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
        "kp_classifier": kp_classifier,
        "kp_feature_names": KP_FEATURE_NAMES,
        "kp_metrics": kp_metrics,
        "kp_agg_names": KP_AGG_NAMES,
        "stacked_model": stacked_model,
        "stacked_metrics": stacked_metrics,
    }
    if args.no_save:
        print(f"\n--no-save: {ARTIFACT_PATH.relative_to(BASE_DIR)} left untouched.")
    else:
        joblib.dump(artifact, ARTIFACT_PATH)
        print(f"\nSaved {best_name} to {ARTIFACT_PATH.relative_to(BASE_DIR)}")

    # ---- results file: what the README cites --------------------------
    doc = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "script": "grader/train.py",
        "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__},
        "dataset": {"file": dataset_path.name, "rows": int(len(y)),
                    "chunks": int(len(set(groups))),
                    "teacher_rows": int(Counter(sources)["teacher"]),
                    "construction_rows": int(Counter(sources)["construction"]),
                    "teacher_weight": TEACHER_WEIGHT},
        "split": {"rule": "GroupShuffleSplit by chunk_id, test_size 0.2, random_state 42",
                  "train": int(len(train_idx)), "test": int(len(test_idx)),
                  "gold_test": int(len(gold_idx))},
        "retrained": {"best": best_name, "all_test": results, "gold": gold_results,
                      "subscores": sub_metrics or None, "keypoints": kp_metrics,
                      "stacked": stacked_metrics, "stacked_ships": stacked_model is not None},
        "artifact_saved": not args.no_save,
    }
    if args.evaluate_artifact and ARTIFACT_PATH.exists():
        shipped = joblib.load(ARTIFACT_PATH)
        doc["artifact"] = {
            "path": str(ARTIFACT_PATH.relative_to(BASE_DIR)),
            "sha256": hashlib.sha256(ARTIFACT_PATH.read_bytes()).hexdigest()[:16],
            **evaluate_artifact(shipped, kept, chunks_by_id, gold_idx, y, kp_labels,
                                set(test_idx)),
        }
        art = doc["artifact"]
        print(f"\nShipped artifact ({art['model_name']}, {art['sha256']}) on the same "
              f"{art['n_gold']} gold rows:")
        print_table(art["gold"])
        if art.get("keypoints"):
            for name in ("classifier", "lexical_threshold"):
                m = art["keypoints"][name]
                print(f"  kp {name:18s} 3-class acc {m['acc3']:4.0%}  "
                      f"macro-F1 {m['macro_f1']:.2f}  hit-F1 {m['hit_f1']:.2f}")
    results_path = Path(args.results)
    results_path.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(f"Results written to {results_path}")
    if not teacher:
        print("Note: labels are construction-only, so these metrics are optimistic. "
              "Run grader\\label_teacher.py for gold labels, then retrain.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
