# rag_lists - question bank built from licensed public interview lists

The fourth question bank (retrieval plan §12.2, item 4): technical
questions taken from public GitHub interview-preparation lists whose
licenses allow reuse, each rewritten into a grading rubric by the Claude
teacher. The list author's answer is given to the teacher as source
material and rewritten in the teacher's own words; the chunk stores the
question verbatim (`metadata.original`) and a source URL pinned to the
commit that was ingested, never the source answer.

These are **not** real interview reports (that is `rag_exp/`): they are
author-written or LLM-synthesized prep questions. The bank is kept
separate from the course banks on purpose - its own track ("Lists"),
switchable, and measurable with and without in the grounding experiment.

Build / update it locally (clones live under `data/interview_exp/github/`,
gitignored):

    .venv\Scripts\python grader\ingest_lists.py                     # free dry run + cost estimate
    .venv\Scripts\python grader\ingest_lists.py --generate --confirm  # paid teacher run

The ingest selects tiers (ombharatiya intermediate + advanced; Kalyan
capped at 30, internals first), dedupes lexically within the pool,
question-against-question against every bank, and by bge-small cosine
(>= 0.90) when the hybrid stack is available, then generates one rubric
chunk per surviving question in the same `id / interview / metadata`
schema the app serves. Re-runs are idempotent - existing ids are skipped.
First run (2026-09-04): 331 chunks, 301 from ombharatiya and 30 from
Kalyan, $11.78 of teacher calls; details in docs/plan.md §1.8.

Review the chunks with `tools/review_bank.py rag_lists` (local page under
`data/review/`), then `--apply` the saved decisions: retired chunks stay
in the file with `metadata.review.status = "retire"` and the app skips
them.

## Sources and attribution

| Source | License | Attribution carried in every chunk |
|---|---|---|
| [ombharatiya/AI-Engineer-Interview-Questions](https://github.com/ombharatiya/AI-Engineer-Interview-Questions) - folders 02-09, intermediate and advanced tiers | MIT, `licenses/ombharatiya-AI-Engineer-Interview-Questions.MIT.txt` | Copyright (c) 2026 Om |
| [KalyanKS-NLP/LLM-Interview-Questions-and-Answers-Hub](https://github.com/KalyanKS-NLP/LLM-Interview-Questions-and-Answers-Hub) - `Interview_QA/` | Apache-2.0, `licenses/KalyanKS-NLP-LLM-Interview-Questions-and-Answers-Hub.Apache-2.0.txt` | Authored by Kalyan KS |

The rubrics are derivative works of those lists, adapted (not copied) as
the licenses permit; the license texts are redistributed alongside as
they require. No endorsement by either author is implied. The bank file
`all_chunks.jsonl` is gitignored and backed up to the private repository
like `rag_exp/`; it could be published under the same licenses later.
