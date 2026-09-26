# Bank builders

Scripts that build and grow the question banks in `banks/`. Each has a
free dry run; calls to the Claude teacher print a cost estimate and need
`--confirm`. Every new chunk is reviewed before it serves
(`tools/review_bank.py`), and ids are appended, never rewritten.

| Script | Builds | Source |
| --- | --- | --- |
| `ingest_questions.py` | `banks/rag_exp/` | hand-collected interview reports (local only; never committed) |
| `ingest_lists.py` | `banks/rag_lists/` | licensed GitHub question lists (MIT / Apache-2.0) |
| `ingest_docs.py` | `banks/rag_docs/` | sections of primary documentation, pinned to commits |
| `expand_chunks.py` | finer sub-questions for `rag_ai` / `rag_ml` | lesson text in the private checkout |

Run from the repository root, for example
`.venv\Scripts\python ingest\ingest_lists.py --help`.
