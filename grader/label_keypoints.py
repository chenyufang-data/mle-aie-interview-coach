"""Per-key-point teacher labels: which rubric points did each answer cover?

Run:   .venv\\Scripts\\python grader\\label_keypoints.py            (cost estimate only)
       .venv\\Scripts\\python grader\\label_keypoints.py --confirm  (spends API credits)

The distilled grader's measured blind spots are semantic: paraphrases its
lexical features cannot see, and confidently-wrong answers that look lexically
right. Overall scores cannot teach it which key point it misjudged - so this
script asks the Claude teacher for a per-key-point verdict (hit / partial /
miss, judged on meaning) for every row that already has a gold overall label.
One API call yields K key-point labels, so this is the cheapest dense
supervision the teacher can give.

Labels append to grader/labels_keypoints.jsonl keyed by row_id (resumable).
--consistency N regrades N already-labeled rows once more and reports per-kp
verdict agreement, so the label quality is itself a measured number.

THIS SCRIPT COSTS MONEY. Nothing is sent without --confirm.
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

import server

DATASET_PATH = BASE_DIR / "grader" / "dataset.jsonl"
TEACHER_PATH = BASE_DIR / "grader" / "labels_teacher.jsonl"
OUT_PATH = BASE_DIR / "grader" / "labels_keypoints.jsonl"

SYSTEM = ("You are a strict, precise grading assistant. Judge whether an "
          "interview answer conveys the MEANING of each rubric key point. "
          "Wording does not matter; substance does.")

VERDICT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "verdict": {"type": "string",
                                "enum": ["hit", "partial", "miss"]},
                    "why": {"type": "string",
                            "description": "at most 15 words"},
                },
                "required": ["verdict", "why"],
            },
        }
    },
    "required": ["verdicts"],
}

# USD per 1M tokens (input, output) for the estimate printout.
PRICES = {"claude-opus-4-8": (5.00, 25.00), "claude-sonnet-5": (3.00, 15.00)}
EST_OUTPUT_TOKENS = 450  # ~5 short verdict+why objects, no extended thinking


def build_prompt(row, chunk):
    interview = chunk["interview"]
    points = "\n".join(f"{i}. {p}" for i, p in enumerate(interview["key_points"]))
    return f"""Which rubric key points does this candidate answer cover?

Interview question: {interview["question"]}

Candidate answer: {row["answer"]}

Rubric key points:
{points}

For EACH key point, in order, give a verdict:
- "hit": the answer conveys this point's substance (paraphrase counts fully)
- "partial": touches the idea but incomplete, muddled, or only implied
- "miss": absent - or stated incorrectly (a confident wrong claim about this
  point is a miss, even if it uses the right vocabulary)

Return exactly {len(interview["key_points"])} verdicts, index order, each with
a "why" of at most 15 words."""


def call_teacher(prompt, n_points, model):
    for _attempt in range(2):
        response = server.get_client().messages.create(
            model=model,
            max_tokens=1500,
            system=SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            output_config={"format": {"type": "json_schema",
                                      "schema": VERDICT_SCHEMA}},
        )
        text = next((b.text for b in response.content if b.type == "text"), "")
        result = json.loads(text)
        if len(result["verdicts"]) == n_points:
            return result["verdicts"]
    raise RuntimeError(f"verdict count != {n_points} twice")


def main():
    parser = argparse.ArgumentParser(description="Per-key-point gold labels")
    parser.add_argument("--model", default=os.environ.get("ANTHROPIC_MODEL",
                                                          "claude-opus-4-8"))
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--consistency", type=int, default=0,
                        help="Regrade N already-labeled rows to measure "
                             "label stability")
    args = parser.parse_args()

    server.load_env_file()
    server.load_chunks()

    teacher_ids = set()
    for line in TEACHER_PATH.open(encoding="utf-8"):
        if line.strip():
            teacher_ids.add(json.loads(line)["row_id"])
    rows = [json.loads(line) for line in DATASET_PATH.open(encoding="utf-8")
            if line.strip()]
    rows = [row for row in rows if row["row_id"] in teacher_ids
            and row["chunk_id"] in server.CHUNKS_BY_ID]

    done = {}
    if OUT_PATH.exists():
        for line in OUT_PATH.open(encoding="utf-8"):
            if line.strip():
                record = json.loads(line)
                done.setdefault(record["pass"], set()).add(record["row_id"])

    todo = [("main", row) for row in rows
            if row["row_id"] not in done.get("main", set())]
    if args.consistency:
        labeled = sorted(done.get("main", set()))
        redo = set(labeled[:args.consistency]) - done.get("retry", set())
        todo += [("retry", row) for row in rows if row["row_id"] in redo]

    prompts = {row["row_id"]: build_prompt(row, server.CHUNKS_BY_ID[row["chunk_id"]])
               for _, row in todo}
    input_tokens = sum(len(p) // 4 + len(SYSTEM) // 4 for p in prompts.values())
    output_tokens = len(todo) * EST_OUTPUT_TOKENS
    price = PRICES.get(args.model)
    cost = (f"~${input_tokens / 1e6 * price[0] + output_tokens / 1e6 * price[1]:.2f}"
            if price else "unknown model")
    print(f"Teacher model: {args.model}")
    print(f"To label: {len(todo)} rows "
          f"({len(done.get('main', set()))} already done of {len(rows)} gold rows)")
    print(f"Estimated: ~{input_tokens:,} in / ~{output_tokens:,} out -> {cost}")
    if not args.confirm:
        print("\nDry run - nothing sent. Re-run with --confirm to label.")
        return 0

    labeled = failed = 0
    with OUT_PATH.open("a", encoding="utf-8") as out:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {}
            for pass_, row in todo:
                chunk = server.CHUNKS_BY_ID[row["chunk_id"]]
                n = len(chunk["interview"]["key_points"])
                futures[pool.submit(call_teacher, prompts[row["row_id"]],
                                    n, args.model)] = (pass_, row)
            for i, future in enumerate(as_completed(futures), start=1):
                pass_, row = futures[future]
                try:
                    verdicts = future.result()
                except Exception as exc:
                    failed += 1
                    print(f"[{i}/{len(todo)}] FAILED {row['row_id']}: {exc}",
                          flush=True)
                    continue
                out.write(json.dumps({
                    "row_id": row["row_id"], "pass": pass_,
                    "verdicts": [v["verdict"] for v in verdicts],
                    "why": [v["why"] for v in verdicts],
                    "model": args.model,
                }, ensure_ascii=False) + "\n")
                out.flush()
                labeled += 1
                counts = "".join(v["verdict"][0] for v in verdicts)
                print(f"[{i}/{len(todo)}] {pass_} {row['tier']:15s} "
                      f"teacher_overall known, kp={counts}", flush=True)

    print(f"\nLabeled {labeled}, failed {failed}.")

    if args.consistency:
        first, second = {}, {}
        for line in OUT_PATH.open(encoding="utf-8"):
            if line.strip():
                record = json.loads(line)
                (first if record["pass"] == "main" else second)[record["row_id"]] = record["verdicts"]
        both = [rid for rid in second if rid in first
                and len(first[rid]) == len(second[rid])]
        if both:
            pairs = [(a, b) for rid in both
                     for a, b in zip(first[rid], second[rid])]
            exact = sum(1 for a, b in pairs if a == b) / len(pairs)
            hard_flip = sum(1 for a, b in pairs
                            if {a, b} == {"hit", "miss"}) / len(pairs)
            print(f"Consistency on {len(both)} rows / {len(pairs)} key points: "
                  f"exact {exact:.0%}, hit<->miss flips {hard_flip:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
