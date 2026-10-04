# Experiments

Every measurement in the project, one folder per study. Each folder holds
its scripts, its inputs (labels, probes, test sets) and its committed
results files; the README tables are rendered from those files by
`tools/render_readme.py`, and CI fails when they drift. The rules each
experiment was judged by were written in `docs/plan.md` before it ran.

Scripts run from the repository root, for example
`.venv\Scripts\python experiments\distill\train.py`. Anything that spends
money prints a cost estimate and does nothing without `--confirm`.

| Folder | Question | Results | Report |
| --- | --- | --- | --- |
| `distill/` | Can a scikit-learn model trained on Claude labels grade offline? Which answers can stay local? Could DeepSeek replace Claude as the judge? | `train_results.json`, `cascade_results.json`, `judge_agreement_summary.json` | README, "Local ML grader" |
| `slm/` | Does a fine-tuned small language model beat the sklearn grader on the same gold rows? (roadmap step 3) | `slm_results.json`, `runs/` | `slm/README.md`, README |
| `jev/` | Can an API decision model (TypeSafe's Jev) carry that quality to a box with no GPU? (roadmap step 6) | `jev_results.json`, `runs/responses.jsonl` | README, step 6 section |
| `retrieval/` | Does dense or hybrid retrieval beat BM25 for practice questions? Does a vector store earn its place? (R1, R3, pgvector) | `retrieval_eval_results.json` | `docs/retrieval_evaluation.md` |
| `grounding/` | Which policy attaches a fair bank rubric to a mock-interview probe? (R2, R4, the author spot-check) | `grounding_eval_results.json`, `grounding_r4_results*.json`, `grounding_r4_spotcheck_grown.json` | `docs/grounding_r4*.md` |
| `speech/` | How much does transcription damage technical terms and grades (Phase 0)? How do the live-voice backends compare (Phase 2)? | `stt_eval_results.json`, `loop_eval_results.json` | `docs/stt_evaluation.md`, README |
| `mock/` | Is the mock report stable when regraded? Does the turn shape hit the prompt cache? | `report_consistency_results.json`, `cache_check_results.json` | README, "AI mock interview" |
| `coding/` | Does the coding tutor answer within the plan's latency target (hint p95 at most 5 s locally)? (roadmap step 7) | `tutor_latency_results.json` | `docs/plan.md` step 7, README |

Private inputs stay out of this repository: the synthetic answers the
grader trains on (`grader/dataset.jsonl` in the private checkout; a local
copy at `distill/dataset.jsonl` is gitignored), the SLM prompt records,
and the R4 labeling pool. The scripts say where they look and take a
path or an environment variable (`GRADER_DATASET`, `--private-dir`).
