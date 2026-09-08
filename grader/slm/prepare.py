"""Build the prompt records for the SLM arms and check the split.

Run:  python grader/slm/prepare.py [--private-dir PATH]

Reads the private dataset (3,866 rows) and the public teacher labels and
banks, writes <private>/grader/slm_records.jsonl (prompts contain the
answers, so the file stays private), and prints the per-seed split sizes.
Seed 42 must give the 121 gold rows of grader/train_results.json.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from grader.slm import common  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="SLM experiment: build prompt records")
    parser.add_argument("--private-dir", default=str(common.default_private_dir()))
    args = parser.parse_args()
    private = Path(args.private_dir)

    chunks = common.load_chunks()
    rows = common.keep_rows(common.load_rows(private), chunks)
    teacher = common.load_teacher()
    records = common.make_records(rows, chunks, teacher)
    out = private / "grader" / "slm_records.jsonl"
    with out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    labeled = sum(1 for r in records if r["teacher"] is not None)
    words = sorted(len(r["prompt"].split()) for r in records)
    print(f"records: {len(records)} rows over {len({r['chunk_id'] for r in records})} chunks, "
          f"{labeled} teacher-labeled; prompt words median {words[len(words) // 2]}, "
          f"max {words[-1]} -> {out}")
    print("teacher grades:", dict(sorted(Counter(r["teacher"] for r in records
                                                 if r["teacher"] is not None).items())))
    shipped = None
    results_path = common.BASE_DIR / "grader" / "train_results.json"
    if results_path.exists():
        shipped = json.loads(results_path.read_text(encoding="utf-8"))["split"]
    for seed in common.SEEDS:
        train_idx, test_idx = common.split_indices(rows, seed)
        gold = common.gold_indices(rows, teacher, test_idx)
        fit_idx, dev_idx = common.dev_split(rows, common.labeled_indices(rows, teacher, train_idx), seed)
        note = ""
        if seed == 42 and shipped:
            same = (len(train_idx), len(test_idx), len(gold)) == (
                shipped["train"], shipped["test"], shipped["gold_test"])
            note = "  <- matches train_results.json" if same else "  !! differs from train_results.json"
        print(f"seed {seed:>2}: train {len(train_idx)} / test {len(test_idx)} rows, "
              f"gold {len(gold)}; labeled train {len(fit_idx) + len(dev_idx)} = "
              f"fit {len(fit_idx)} + dev {len(dev_idx)}{note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
