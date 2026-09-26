# Question banks

Five banks of interview questions with rubrics (model answer, key points,
common mistakes, follow-ups). The server loads every bank whose file is
present and skips the rest with a warning; each folder's README has its
schema, provenance and license.

| Folder | Track | What | In this repository |
| --- | --- | --- | --- |
| `rag_ml/` | MLE | course bank, 191 chunks | tracked, public stripped edition |
| `rag_ai/` | AIE | course bank, 222 chunks | tracked, public stripped edition |
| `rag_exp/` | Real Qs | questions from real interview reports, rewritten | README only; the bank is private |
| `rag_lists/` | Lists | licensed GitHub question lists rewritten into rubrics | README and licenses; the bank is generated locally |
| `rag_docs/` | Docs | rubrics from primary documentation on MLOps gaps | README and licenses; the bank is generated locally |

The complete course banks, with the lesson text, live in the private
repository at its root; `tools/strip_chunks.py` produces the public
editions here. The scripts that build and grow the other banks are in
`ingest/`. Chunk ids are joined by gold labels, bookmarks and the plan
cache, so banks grow by appending reviewed chunks and ids are never
rewritten.
