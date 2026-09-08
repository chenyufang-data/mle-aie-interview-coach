"""The incumbent, retrained per seed: grader/train.py's HistGradientBoosting
student on the same features, targets and sample weights, evaluated on the
gold rows of each seed's split. Runs on the CPU in the main checkout (it
needs grader/features.py's dependencies).

Run:  .venv\\Scripts\\python grader\\slm\\sklearn_arm.py [--private-dir PATH] [--seeds 42 1 2 3 4]

Writes grader/slm/runs/sklearn_seed<N>.json. Seed 42 reproduces the gold
row of grader/train_results.json ("retrained" section) up to float noise.
"""

import argparse
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from grader.features import FEATURE_NAMES, FeatureExtractor  # noqa: E402
from grader.slm import common  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="SLM experiment: sklearn incumbent per seed")
    parser.add_argument("--private-dir", default=str(common.default_private_dir()))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(common.SEEDS))
    args = parser.parse_args()
    import sklearn
    from sklearn.ensemble import HistGradientBoostingRegressor

    chunks = common.load_chunks()
    rows = common.keep_rows(common.load_rows(Path(args.private_dir)), chunks)
    teacher = common.load_teacher()

    started = time.perf_counter()
    extractor = FeatureExtractor().fit(chunks.values())
    X = np.array([extractor.extract(row["answer"], chunks[row["chunk_id"]]) for row in rows])
    y = np.array([teacher[r["row_id"]]["teacher_score"] if r["row_id"] in teacher else r["label"]
                  for r in rows], dtype=float)
    weights = np.array([common.TEACHER_WEIGHT if r["row_id"] in teacher else 1.0 for r in rows])
    print(f"features: {X.shape} in {time.perf_counter() - started:.0f}s")
    frac_hit = X[:, FEATURE_NAMES.index("kp_frac_hit")]

    for seed in args.seeds:
        train_idx, test_idx = common.split_indices(rows, seed)
        gold = common.gold_indices(rows, teacher, test_idx)
        t0 = time.perf_counter()
        model = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
            l2_regularization=1.0, random_state=42)      # train.py's fixed params
        model.fit(X[train_idx], y[train_idx], sample_weight=weights[train_idx])
        pred = model.predict(X[gold])
        baseline = np.clip(np.rint(2 + 8 * frac_hit[gold]), 1, 10)
        run = {
            "arm": "sklearn", "seed": seed, "generated": datetime.now().isoformat(timespec="seconds"),
            "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__},
            "n_train": int(len(train_idx)), "n_test": int(len(test_idx)), "n_gold": int(len(gold)),
            "train_seconds": round(time.perf_counter() - t0, 1),
            "config": {"model": "HistGradientBoostingRegressor", "max_iter": 300,
                       "learning_rate": 0.06, "max_leaf_nodes": 31, "l2_regularization": 1.0,
                       "training_rows": "all train-side rows: construction labels, teacher "
                                        "labels overriding at 3x weight (grader/train.py)"},
            "gold": common.metrics(y[gold], pred),
            "keyword_baseline": common.metrics(y[gold], baseline),
            "predictions": [{"row_id": rows[i]["row_id"], "y": float(y[i]), "pred": float(p)}
                            for i, p in zip(gold, pred)],
        }
        common.write_json(common.RUNS_DIR / f"sklearn_seed{seed}.json", run)
        g = run["gold"]
        print(f"seed {seed:>2}: gold {len(gold)} rows  MAE {g['mae']:.3f}  within1 {g['within1']:.3f}  "
              f"spearman {g['spearman']:.3f}  QWK {g['qwk']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
