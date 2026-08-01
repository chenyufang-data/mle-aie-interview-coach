"""Label a sample of the synthetic dataset with the real Claude grader.

Run:   .venv\\Scripts\\python grader\\label_teacher.py            (prints cost estimate only)
       .venv\\Scripts\\python grader\\label_teacher.py --confirm  (actually spends API credits)

Grades a stratified sample of grader/dataset.jsonl through the exact same
prompt + schema the app uses for real evaluations, so the student model is
distilled from the deployed teacher. Results append to
grader/labels_teacher.jsonl; already-labeled rows are skipped, so the script
is resumable and the sample can be grown incrementally with a larger --limit.

By default the teacher is the app's grading model (ANTHROPIC_MODEL from .env,
falling back to claude-opus-4-8). Pass --model claude-sonnet-5 for cheaper
labeling - fine for bootstrapping, but note the student then approximates
Sonnet's grading, not the model the app actually grades with.

THIS SCRIPT COSTS MONEY. Nothing is sent without --confirm.
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

import server

DATASET_PATH = BASE_DIR / "grader" / "dataset.jsonl"
OUT_PATH = BASE_DIR / "grader" / "labels_teacher.jsonl"
SEED = 42
EST_OUTPUT_TOKENS = 1200  # thinking + JSON evaluation, rough average

# USD per million tokens (input, output), for the estimate printout only.
PRICES = {
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


def build_prompt(row, chunk):
    data = {
        "role": "AIE" if row["corpus"] == "ai" else "MLE",
        "level": "Mid-level",
        "topic": chunk["metadata"]["module"],
        "question": chunk["interview"]["question"],
        "answer": row["answer"],
        "timeUsed": "02:00",
    }
    return server.build_evaluation_prompt(data, chunk)


def stratified_sample(rows, limit, rng):
    """Round-robin across tiers so every quality level is represented."""
    by_tier = defaultdict(list)
    for row in rows:
        by_tier[row["tier"]].append(row)
    for tier_rows in by_tier.values():
        rng.shuffle(tier_rows)
    sample = []
    while len(sample) < limit and any(by_tier.values()):
        for tier in sorted(by_tier):
            if by_tier[tier] and len(sample) < limit:
                sample.append(by_tier[tier].pop())
    return sample


def main():
    parser = argparse.ArgumentParser(description="Gold-label answers with the Claude grader")
    parser.add_argument("--limit", type=int, default=300,
                        help="Total teacher labels to have (default 300)")
    parser.add_argument("--model", help="Override the teacher model "
                        "(e.g. claude-sonnet-5 for cheaper labeling)")
    parser.add_argument("--confirm", action="store_true",
                        help="Actually call the API (otherwise just prints the estimate)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Concurrent API calls (default 4)")
    parser.add_argument("--tiers", help="Comma-separated tier names to restrict "
                        "sampling to (e.g. mistake,paraphrase_good)")
    args = parser.parse_args()

    server.load_env_file()
    if args.model:
        os.environ["ANTHROPIC_MODEL"] = args.model
    model = os.environ.get("ANTHROPIC_MODEL", server.DEFAULT_MODEL)
    server.load_chunks()

    with DATASET_PATH.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if args.tiers:
        wanted = {tier.strip() for tier in args.tiers.split(",")}
        rows = [row for row in rows if row["tier"] in wanted]

    done = set()
    if OUT_PATH.exists():
        with OUT_PATH.open(encoding="utf-8") as handle:
            done = {json.loads(line)["row_id"] for line in handle if line.strip()}

    rng = random.Random(SEED)
    sample = stratified_sample(rows, args.limit, rng)
    todo = [row for row in sample if row["row_id"] not in done]

    prompts = {row["row_id"]: build_prompt(row, server.CHUNKS_BY_ID[row["chunk_id"]])
               for row in todo}
    input_tokens = sum(len(p) // 4 for p in prompts.values()) \
        + len(todo) * len(server.SYSTEM_PROMPT) // 4
    output_tokens = len(todo) * EST_OUTPUT_TOKENS
    price = PRICES.get(model)
    cost = (f"~${input_tokens / 1e6 * price[0] + output_tokens / 1e6 * price[1]:.2f}"
            if price else "unknown (model not in price table)")
    print(f"Teacher model: {model}")
    print(f"To label: {len(todo)} answers ({len(done)} already done, target {args.limit})")
    print(f"Estimated tokens: ~{input_tokens:,} in / ~{output_tokens:,} out -> {cost}")

    if not args.confirm:
        print("\nDry run - nothing sent. Re-run with --confirm to label.")
        return 0

    labeled = failed = 0
    # Workers only call the API; the labels file is written from this thread.
    with OUT_PATH.open("a", encoding="utf-8") as out:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(server.call_claude, prompts[row["row_id"]],
                            server.EVALUATION_SCHEMA): row
                for row in todo
            }
            for i, future in enumerate(as_completed(futures), start=1):
                row = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    failed += 1
                    print(f"[{i}/{len(todo)}] FAILED {row['row_id']}: {exc}")
                    continue
                out.write(json.dumps({
                    "row_id": row["row_id"],
                    "teacher_score": result["overall_score"],
                    "scores": result["scores"],
                    "model": model,
                }, ensure_ascii=False) + "\n")
                out.flush()
                labeled += 1
                print(f"[{i}/{len(todo)}] {row['tier']:15s} construction={row['label']} "
                      f"teacher={result['overall_score']}", flush=True)

    print(f"\nLabeled {labeled}, failed {failed}. Now retrain: "
          ".venv\\Scripts\\python grader\\train.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
