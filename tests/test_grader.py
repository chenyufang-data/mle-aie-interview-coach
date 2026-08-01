"""Sanity checks for the distilled grader artifact.

Run:  .venv\\Scripts\\python tests\\test_grader.py

Loads grader/model.joblib and checks basic ordering invariants on a few
chunks: the reference answer must outscore an off-topic answer and a vague
answer, and every prediction must clamp into 1..10. These are cheap
guardrails against a broken artifact, not an accuracy benchmark - accuracy
numbers come from grader/train.py's held-out split.
"""

import json
import random
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

import joblib

from grader import features  # noqa: F401  (unpickling needs FeatureExtractor importable)

ARTIFACT_PATH = BASE_DIR / "grader" / "model.joblib"
SAMPLES = 8
VAGUE = ("Good question. I would follow best practices, try a few standard "
         "approaches, and evaluate carefully. It depends on the data.")


def main():
    if not ARTIFACT_PATH.exists():
        print("No grader artifact - run grader\\generate_answers.py then grader\\train.py.")
        return 1
    artifact = joblib.load(ARTIFACT_PATH)
    extractor, model = artifact["extractor"], artifact["model"]

    chunks = []
    for folder in ("rag_ml", "rag_ai"):
        with (BASE_DIR / folder / "all_chunks.jsonl").open(encoding="utf-8") as handle:
            chunks.extend(json.loads(line) for line in handle if line.strip())

    rng = random.Random(7)
    failures = 0
    for chunk in rng.sample(chunks, SAMPLES):
        other = rng.choice([c for c in chunks if c["id"] != chunk["id"]])
        answers = {
            "reference": chunk["interview"]["model_answer"],
            "vague": VAGUE,
            "off_topic": other["interview"]["model_answer"],
        }
        scores = {
            name: float(model.predict([extractor.extract(text, chunk)])[0])
            for name, text in answers.items()
        }
        checks = [
            ("reference > off_topic", scores["reference"] > scores["off_topic"]),
            ("reference > vague", scores["reference"] > scores["vague"]),
            ("predictions in range", all(0 <= s <= 11 for s in scores.values())),
        ]
        bad = [name for name, ok in checks if not ok]
        failures += len(bad)
        status = "PASS" if not bad else f"FAIL ({', '.join(bad)})"
        print(f"[{status}] {chunk['id']}: ref={scores['reference']:.1f} "
              f"vague={scores['vague']:.1f} off_topic={scores['off_topic']:.1f}")

    test_metrics = artifact["metrics"][artifact["model_name"]]
    print(f"\nArtifact: {artifact['model_name']} "
          f"(held-out MAE {test_metrics['mae']:.2f}, "
          f"Spearman {test_metrics['spearman']:.3f}, QWK {test_metrics['qwk']:.3f}; "
          f"labels: {artifact['label_sources']})")
    if failures:
        print(f"{failures} ordering checks failed.")
        return 1
    print("All ordering checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
