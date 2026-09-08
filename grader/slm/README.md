# SLM grader experiment (roadmap step 3)

Does a fine-tuned small language model grade closer to the Claude teacher
than the distilled scikit-learn grader, on the same held-out gold rows?
The answer lives in [`grader/slm_results.json`](../slm_results.json) and the
README's "Local ML grader" section; this folder is how it was produced.

## Protocol (fixed before the first run)

- **Rows and split.** The private `grader/dataset.jsonl` (3,866 answers over
  282 rubrics) with `grader/labels_teacher.jsonl` (598 Claude grades) joined by
  `row_id`; `GroupShuffleSplit` by `chunk_id`, test size 0.2, exactly as
  `grader/train.py`. Seed 42 is the shipped split (3,083 / 783 rows, 121 gold
  rows); seeds 1 to 4 give the interval. Gold rows = held-out rows with a
  teacher grade.
- **Early stopping** on a dev fold of 15% of the training side's labeled
  rows, grouped by chunk (`common.dev_split`), best epoch by QWK. The test
  side is never read during training.
- **Arms.** `sklearn` (the incumbent retrained per seed, same features, same
  targets: construction labels with teacher grades overriding at 3x weight);
  `deberta` (DeBERTa-v3-base, full fine-tune, regression head, MSE);
  `qwen1.7b` and `qwen4b` (Qwen3 base models + LoRA r16 on every projection).
  The SLM arms train on the teacher-labeled training rows only (about 400);
  the exploratory `qwen1.7b-silver` adds every train-side row with its
  construction label and is reported outside the verdict.
- **Grade decode.** The causal models see the prompt (`common.build_prompt`:
  instruction, question, rubric key points, answer, "Score:") and are trained
  with cross-entropy over the ten digit tokens at the last position, digit =
  grade - 1; the grade is read back as 1 + E[digit]. No text is generated or
  parsed, and vLLM serves the same thing as one token with logprobs.
- **Metrics** as `grader/train.py`: MAE on the raw prediction, within-1 and
  QWK on the rounded grade.
- **Rule** (`common.RULE`): the best SLM arm replaces the sklearn grader as
  the local tier only if, on every seed, its QWK is at least 0.05 above the
  sklearn arm's with a lower MAE, and vLLM serves it at p95 <= 300 ms per
  answer on the RTX 5080. Otherwise the result is a comparison table.

## Running it

Windows side (the main venv; CPU):

```powershell
.venv\Scripts\python grader\slm\prepare.py        # records -> private checkout, split check
.venv\Scripts\python grader\slm\sklearn_arm.py    # incumbent per seed -> runs/sklearn_seed*.json
```

WSL2 side (`~/.venvs/slm`: torch cu128, transformers, peft; see docs/plan.md
step 3 for the environment):

```bash
bash grader/slm/run_all.sh                        # deberta, qwen1.7b, qwen4b x 5 seeds, silver, report
python grader/slm/serve_latency.py --arm qwen4b   # merge + vLLM (~/.venvs/vllm) p95 per answer
python grader/slm/report.py                       # -> grader/slm_results.json with the verdict
```

`tests/test_slm.py` covers the split logic, the prompt, the decode, the
metrics against `grader/train.py`'s, and the rule, offline.

## What is where

- `runs/*.json` — one file per arm and seed: metrics, dev curve, timing, peak
  VRAM, per-row predictions (row ids only; the answers stay private). Committed.
- `<private>/grader/slm_records.jsonl` — the prompt records (they contain the
  answers). Private checkout only.
- `~/slm_runs/` on WSL2 — adapters, heads, merged models, vLLM logs. Never
  committed; the weights were trained on lesson-derived answers.
