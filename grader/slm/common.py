"""Shared pieces of the SLM grader experiment: data loading, the split that
mirrors grader/train.py exactly, the prompt, the digit-token score decode,
the metrics, and the pre-registered rule.

Everything here is CPU-only and import-light so the WSL2 training venv (torch
+ transformers + peft, no anthropic/fastembed) and the Windows venv (the
sklearn arm) can both use it.
"""

import json
import os
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2]
CORPORA = {"ml": "rag_ml", "ai": "rag_ai"}          # grader/train.py CORPORA
TEACHER_PATH = BASE_DIR / "grader" / "labels_teacher.jsonl"
RUNS_DIR = BASE_DIR / "grader" / "slm" / "runs"
RESULTS_PATH = BASE_DIR / "grader" / "slm_results.json"

SEEDS = (42, 1, 2, 3, 4)          # 42 reproduces train.py's 121 gold rows
TEST_SIZE = 0.2                    # grader/train.py
DEV_SIZE = 0.15                    # carved from the TRAIN side only, by chunk
TEACHER_WEIGHT = 3.0               # grader/train.py TEACHER_WEIGHT

# Pre-registered rule (docs/plan.md step 3), frozen before the first run:
RULE = {
    "qwk_margin": 0.05,            # SLM QWK >= sklearn QWK + margin, every seed
    "mae_must_be_lower": True,     # and SLM MAE < sklearn MAE, every seed
    "p95_ms_max": 300.0,           # vLLM p95 per answer on the RTX 5080
}

# The 1..10 grade is read from ONE digit token: digit d = grade - 1, so the
# grade is 1 + E[d] over the ten digits' softmax (docs/plan.md step 3; the
# Qwen tokenizer splits "10" into two tokens, hence the shift).
DIGITS = [str(d) for d in range(10)]


def default_private_dir():
    env = os.environ.get("COACH_PRIVATE_DIR")
    if env:
        return Path(env)
    return BASE_DIR.parent / (BASE_DIR.name + "-private")


# ------------------------------------------------------------------ data

def load_chunks(base_dir=BASE_DIR):
    """The two course banks (public edition: question + key points)."""
    chunks = {}
    for folder in CORPORA.values():
        path = base_dir / folder / "all_chunks.jsonl"
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    chunk = json.loads(line)
                    chunks[chunk["id"]] = chunk
    return chunks


def load_rows(private_dir):
    path = Path(private_dir) / "grader" / "dataset.jsonl"
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_teacher(path=TEACHER_PATH):
    teacher = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                teacher[record["row_id"]] = record
    return teacher


def keep_rows(rows, chunks):
    """grader/train.py drops rows whose chunk is not in the banks; today none."""
    return [row for row in rows if row["chunk_id"] in chunks]


# ----------------------------------------------------------------- split

def split_indices(rows, seed, test_size=TEST_SIZE):
    """Chunk-grouped split over ALL rows, exactly as grader/train.py:
    GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed) on the
    rows in file order, groups = chunk_id. Seed 42 is the shipped split."""
    from sklearn.model_selection import GroupShuffleSplit
    groups = [row["chunk_id"] for row in rows]
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(np.zeros(len(rows)), groups=groups))
    return train_idx, test_idx


def gold_indices(rows, teacher, test_idx):
    """Held-out rows with a teacher label, in train.py's order."""
    return np.array([i for i in test_idx if rows[i]["row_id"] in teacher], dtype=int)


def labeled_indices(rows, teacher, idx):
    return np.array([i for i in idx if rows[i]["row_id"] in teacher], dtype=int)


def dev_split(rows, idx, seed, dev_size=DEV_SIZE):
    """Early-stopping fold carved from the training side, grouped by chunk so
    the dev rubrics are unseen too. Never touches the test side."""
    from sklearn.model_selection import GroupShuffleSplit
    idx = np.asarray(idx)
    groups = [rows[i]["chunk_id"] for i in idx]
    splitter = GroupShuffleSplit(n_splits=1, test_size=dev_size, random_state=seed)
    fit_pos, dev_pos = next(splitter.split(np.zeros(len(idx)), groups=groups))
    return idx[fit_pos], idx[dev_pos]


# ---------------------------------------------------------------- prompt

INSTRUCTION = ("You grade one interview answer against its rubric. Give a single "
               "digit from 0 (missing, wrong or off-topic) to 9 (complete, "
               "precise and well judged).")


def build_prompt(chunk, answer):
    interview = chunk["interview"]
    points = "\n".join(f"- {p}" for p in interview.get("key_points", []))
    return (f"{INSTRUCTION}\n\n"
            f"Question: {interview['question'].strip()}\n\n"
            f"Rubric (what a strong answer covers):\n{points}\n\n"
            f"Candidate answer:\n{answer.strip()}\n\n"
            f"Score:")


def encoder_text(chunk, answer):
    """Pair text for the encoder arm: rubric side and answer side."""
    interview = chunk["interview"]
    points = " ".join(f"- {p}" for p in interview.get("key_points", []))
    return f"{interview['question'].strip()} {points}", answer.strip()


def make_records(rows, chunks, teacher):
    """One record per dataset row: prompt, silver label, teacher grade if any."""
    records = []
    for row in rows:
        chunk = chunks[row["chunk_id"]]
        entry = teacher.get(row["row_id"])
        records.append({
            "row_id": row["row_id"],
            "chunk_id": row["chunk_id"],
            "corpus": row.get("corpus"),
            "tier": row.get("tier"),
            "style": row.get("style") or [],
            "silver": int(row["label"]),
            "teacher": int(entry["teacher_score"]) if entry else None,
            "prompt": build_prompt(chunk, row["answer"]),
            "rubric_text": encoder_text(chunk, row["answer"])[0],
            "answer": row["answer"].strip(),
        })
    return records


# ----------------------------------------------------------------- score

def expected_grade(digit_logits):
    """digit_logits: array [..., 10] of logits over the tokens "0".."9".
    Returns 1 + E[d], a continuous grade in [1, 10]."""
    logits = np.asarray(digit_logits, dtype=np.float64)
    logits = logits - logits.max(axis=-1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=-1, keepdims=True)
    return 1.0 + probs @ np.arange(10, dtype=np.float64)


def clip_scores(values):
    return np.clip(np.rint(values), 1, 10).astype(int)


def metrics(y_true, y_pred):
    """Mirrors grader/train.py metrics(): MAE on the raw prediction, the rest
    on the rounded, clipped grade; QWK over the labels 1..10."""
    from scipy.stats import spearmanr
    from sklearn.metrics import cohen_kappa_score, mean_absolute_error
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rounded = clip_scores(y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "within1": float(np.mean(np.abs(rounded - clip_scores(y_true)) <= 1)),
        "spearman": float(spearmanr(y_true, y_pred).statistic),
        "qwk": float(cohen_kappa_score(clip_scores(y_true), rounded,
                                       weights="quadratic", labels=list(range(1, 11)))),
    }


# ------------------------------------------------------------------ rule

def rule_verdict(slm_by_seed, sklearn_by_seed, p95_ms=None, rule=RULE):
    """Apply the pre-registered rule to per-seed gold metrics dicts
    {seed: {"qwk":..., "mae":...}}. Returns a dict with the per-seed checks
    and the overall pass/fail; latency None means "not measured" (fails)."""
    seeds = sorted(set(slm_by_seed) & set(sklearn_by_seed))
    checks = []
    for seed in seeds:
        s, k = slm_by_seed[seed], sklearn_by_seed[seed]
        qwk_ok = s["qwk"] >= k["qwk"] + rule["qwk_margin"]
        mae_ok = (s["mae"] < k["mae"]) if rule["mae_must_be_lower"] else True
        checks.append({"seed": seed, "qwk_slm": s["qwk"], "qwk_sklearn": k["qwk"],
                       "mae_slm": s["mae"], "mae_sklearn": k["mae"],
                       "qwk_ok": bool(qwk_ok), "mae_ok": bool(mae_ok)})
    quality = bool(seeds) and all(c["qwk_ok"] and c["mae_ok"] for c in checks)
    latency_ok = p95_ms is not None and p95_ms <= rule["p95_ms_max"]
    return {"seeds": seeds, "checks": checks, "quality_pass": quality,
            "p95_ms": p95_ms, "latency_pass": bool(latency_ok),
            "pass": bool(quality and latency_ok), "rule": rule}


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
