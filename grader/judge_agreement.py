"""Measure candidate judge models against the Claude teacher on the gold rows.

Run:  .venv\\Scripts\\python grader\\judge_agreement.py            (cost estimate only)
      .venv\\Scripts\\python grader\\judge_agreement.py --confirm  (spends DeepSeek credits)
      .venv\\Scripts\\python grader\\judge_agreement.py --report   (metrics from saved results)

The substitution question: could a cheaper model (DeepSeek V4) replace Claude
as the grading judge without a quality drop? This script replays the exact
grouped split from train.py, takes the held-out Claude-labeled rows, and has
each candidate re-grade them through the same evaluation prompt the app uses.
It reports agreement with the Claude teacher (MAE, within +/-1, Spearman,
QWK) next to the distilled student on the same rows, plus a self-consistency
pass (regrade the same answers, count exact matches — the Claude teacher
measures 95% there).

Results append to grader/judge_agreement_results.jsonl keyed by
(model, pass, row_id), so the run is resumable.

THIS SCRIPT COSTS MONEY (DeepSeek credits, not Anthropic — roughly $0.35 for
both models at defaults). Nothing is sent without --confirm.
"""

import argparse
import json
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

import joblib
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import GroupShuffleSplit

import server
from grader.features import FeatureExtractor  # noqa: F401 (unpickling)

DATASET_PATH = BASE_DIR / "grader" / "dataset.jsonl"
TEACHER_PATH = BASE_DIR / "grader" / "labels_teacher.jsonl"
ARTIFACT_PATH = BASE_DIR / "grader" / "model.joblib"
OUT_PATH = BASE_DIR / "grader" / "judge_agreement_results.jsonl"

API_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_MODELS = ["deepseek-v4-flash", "deepseek-v4-pro"]
# USD per 1M tokens (input, output), estimate printout only.
PRICES = {"deepseek-v4-flash": (0.14, 0.28), "deepseek-v4-pro": (0.435, 0.87)}
EST_OUTPUT_TOKENS = 1400  # thinking + JSON, observed ~1350 on v4-flash


def load_gold_rows():
    """Replay train.py's grouped split; return held-out teacher-labeled rows."""
    server.load_chunks()
    rows = [json.loads(line) for line in DATASET_PATH.open(encoding="utf-8")
            if line.strip()]
    teacher = {}
    for line in TEACHER_PATH.open(encoding="utf-8"):
        if line.strip():
            record = json.loads(line)
            teacher[record["row_id"]] = record["teacher_score"]

    kept = [row for row in rows if row["chunk_id"] in server.CHUNKS_BY_ID]
    y = [teacher.get(row["row_id"], row["label"]) for row in kept]
    groups = [row["chunk_id"] for row in kept]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, test_idx = next(splitter.split(np.zeros(len(kept)), y, groups))
    gold = [kept[i] for i in test_idx if kept[i]["row_id"] in teacher]
    for row in gold:
        row["teacher_score"] = teacher[row["row_id"]]
    return sorted(gold, key=lambda r: r["row_id"])


def build_prompt(row):
    chunk = server.CHUNKS_BY_ID[row["chunk_id"]]
    data = {
        "role": "AIE" if row["corpus"] == "ai" else "MLE",
        "level": "Mid-level",
        "topic": chunk["metadata"]["module"],
        "question": chunk["interview"]["question"],
        "answer": row["answer"],
        "timeUsed": "02:00",
    }
    prompt = server.build_evaluation_prompt(data, chunk)
    return (prompt + "\n\nRespond with ONLY a JSON object matching this JSON "
            "schema:\n" + json.dumps(server.EVALUATION_SCHEMA))


def call_judge(model, prompt, api_key):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": server.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": 8000,
        "temperature": 0,
    }
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    result = json.loads(body["choices"][0]["message"]["content"])
    usage = body.get("usage", {})
    return result, usage


def metrics(pred, truth):
    pred, truth = np.asarray(pred, float), np.asarray(truth, float)
    ip = np.clip(np.rint(pred), 1, 10).astype(int)
    it = np.clip(np.rint(truth), 1, 10).astype(int)
    return {
        "MAE": float(np.mean(np.abs(pred - truth))),
        "within1": float(np.mean(np.abs(ip - it) <= 1)),
        "exact": float(np.mean(ip == it)),
        "spearman": float(spearmanr(pred, truth).statistic),
        "QWK": float(cohen_kappa_score(ip, it, weights="quadratic",
                                       labels=list(range(1, 11)))),
    }


def report(gold, results):
    truth = {row["row_id"]: row["teacher_score"] for row in gold}
    tiers = {row["row_id"]: row["tier"] for row in gold}

    artifact = joblib.load(ARTIFACT_PATH)
    extractor, student = artifact["extractor"], artifact["model"]
    X = np.array([extractor.extract(row["answer"],
                                    server.CHUNKS_BY_ID[row["chunk_id"]])
                  for row in gold])
    student_pred = {row["row_id"]: float(p)
                    for row, p in zip(gold, student.predict(X))}

    lines = {}
    for line in results:
        lines.setdefault((line["model"], line["pass"]), {})[line["row_id"]] = line

    print(f"\nGold held-out rows: {len(gold)} "
          f"(teacher: Claude, self-consistency 95% exact)")
    header = f"{'judge':22s} {'n':>4s} {'MAE':>6s} {'within+/-1':>10s} {'exact':>6s} {'rho':>6s} {'QWK':>6s}"
    print("\nAgreement with the Claude teacher on the gold rows:")
    print(header)
    ids_all = [row["row_id"] for row in gold]
    m = metrics([student_pred[i] for i in ids_all], [truth[i] for i in ids_all])
    print(f"{'distilled student':22s} {len(ids_all):4d} {m['MAE']:6.2f} "
          f"{m['within1']:10.0%} {m['exact']:6.0%} {m['spearman']:6.2f} {m['QWK']:6.2f}")
    for model in sorted({model for model, p in lines if p == "main"}):
        graded = lines[(model, "main")]
        ids = [i for i in ids_all if i in graded]
        if not ids:
            continue
        m = metrics([graded[i]["overall"] for i in ids], [truth[i] for i in ids])
        print(f"{model:22s} {len(ids):4d} {m['MAE']:6.2f} {m['within1']:10.0%} "
              f"{m['exact']:6.0%} {m['spearman']:6.2f} {m['QWK']:6.2f}")

    print("\nSelf-consistency (regrade the same answers, exact overall match):")
    for model in sorted({model for model, p in lines if p == "retry"}):
        first, second = lines[(model, "main")], lines[(model, "retry")]
        ids = [i for i in second if i in first]
        if not ids:
            continue
        same = np.mean([first[i]["overall"] == second[i]["overall"] for i in ids])
        w1 = np.mean([abs(first[i]["overall"] - second[i]["overall"]) <= 1
                      for i in ids])
        print(f"{model:22s} {len(ids):4d} rows   exact {same:4.0%}   within+/-1 {w1:4.0%}")

    print("\nPer-tier within+/-1 vs teacher (hardest tiers are the decision):")
    tier_names = sorted({tiers[i] for i in ids_all})
    judges = [("distilled student", student_pred)] + [
        (model, {i: g["overall"] for i, g in lines[(model, "main")].items()})
        for model, p in sorted(lines) if p == "main"]
    print(f"{'tier':28s}" + "".join(f"{name[:18]:>20s}" for name, _ in judges))
    for tier in tier_names:
        ids_t = [i for i in ids_all if tiers[i] == tier]
        cells = []
        for _, preds in judges:
            ids = [i for i in ids_t if i in preds]
            if ids:
                w1 = np.mean([abs(round(preds[i]) - round(truth[i])) <= 1
                              for i in ids])
                cells.append(f"{w1:>19.0%} ")
            else:
                cells.append(f"{'-':>19s} ")
        print(f"{tier:28s}" + "".join(cells))


def main():
    parser = argparse.ArgumentParser(description="Judge-agreement experiment")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--consistency", type=int, default=30,
                        help="Rows to regrade for self-consistency (default 30)")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--confirm", action="store_true",
                        help="Actually call the DeepSeek API")
    parser.add_argument("--report", action="store_true",
                        help="Only print metrics from saved results")
    args = parser.parse_args()

    server.load_env_file()
    gold = load_gold_rows()

    done = []
    if OUT_PATH.exists():
        with OUT_PATH.open(encoding="utf-8") as handle:
            done = [json.loads(line) for line in handle if line.strip()]
    if args.report:
        report(gold, done)
        return 0

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    done_keys = {(line["model"], line["pass"], line["row_id"]) for line in done}
    retry_ids = {row["row_id"] for row in gold[:args.consistency]}
    todo = [(model, pass_, row)
            for model in models
            for pass_, pool in (("main", gold),
                                ("retry", [r for r in gold
                                           if r["row_id"] in retry_ids]))
            for row in pool
            if (model, pass_, row["row_id"]) not in done_keys]

    est_in = sum(len(build_prompt(row)) // 4 + len(server.SYSTEM_PROMPT) // 4
                 for _, _, row in todo)
    cost = sum((len(build_prompt(row)) // 4 + len(server.SYSTEM_PROMPT) // 4)
               / 1e6 * PRICES[model][0] + EST_OUTPUT_TOKENS / 1e6 * PRICES[model][1]
               for model, _, row in todo if model in PRICES)
    print(f"Candidates: {', '.join(models)}")
    print(f"To grade: {len(todo)} calls ({len(done)} already saved) "
          f"over {len(gold)} gold rows + {args.consistency} consistency rows/model")
    print(f"Estimated: ~{est_in:,} tokens in / ~{len(todo) * EST_OUTPUT_TOKENS:,} out "
          f"-> ~${cost:.2f} in DeepSeek credits")
    if not args.confirm:
        print("\nDry run - nothing sent. Re-run with --confirm.")
        return 0

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("DEEPSEEK_API_KEY not set (add it to .env).")
        return 1

    failed = 0
    with OUT_PATH.open("a", encoding="utf-8") as out:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(call_judge, model, build_prompt(row), api_key):
                (model, pass_, row)
                for model, pass_, row in todo
            }
            for i, future in enumerate(as_completed(futures), start=1):
                model, pass_, row = futures[future]
                try:
                    result, usage = future.result()
                    overall = int(result["overall_score"])
                except Exception as exc:
                    failed += 1
                    print(f"[{i}/{len(todo)}] FAILED {model} {pass_} "
                          f"{row['row_id']}: {exc}", flush=True)
                    continue
                out.write(json.dumps({
                    "model": model, "pass": pass_, "row_id": row["row_id"],
                    "overall": overall, "scores": result.get("scores"),
                    "tokens_out": usage.get("completion_tokens"),
                }, ensure_ascii=False) + "\n")
                out.flush()
                print(f"[{i}/{len(todo)}] {model} {pass_} {row['tier']:15s} "
                      f"teacher={row['teacher_score']} judge={overall}", flush=True)

    print(f"\nDone, {failed} failed. Metrics:")
    with OUT_PATH.open(encoding="utf-8") as handle:
        report(gold, [json.loads(line) for line in handle if line.strip()])
    if failed:
        print(f"\n{failed} calls failed - re-run with --confirm to fill gaps "
              "(saved rows are skipped).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
