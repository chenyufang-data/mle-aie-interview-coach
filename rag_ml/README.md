# rag_ml — MLE question bank (public edition)

191 interview chunks over 20 modules of classical ML and data analysis
(pandas, regression, regularization, cross-validation, collinearity, PCA,
categorical features, time series, trees, ensembles, classification,
logistic regression, SVM, clustering). Serves the **MLE** track.

## What a chunk contains

```json
{ "id": "06_linear_regression_model__03_impute_with_train_stats_no_leakage",
  "interview": {
    "question": "...",            // the interview question the coach asks
    "model_answer": "...",        // reference answer used for grading
    "key_points": ["..."],        // the rubric: what a strong answer must hit
    "common_mistakes": ["..."],   // what the grader checks the answer against
    "followups": ["..."] },       // probes the coach can continue with
  "metadata": { "module": "Regression", "topic": "...",
                "tags": ["..."], "difficulty": "mid" } }
```

`all_chunks.jsonl` is the only file the app, the retriever (`retrieval.py`),
the grader, and the tests read. Chunk ids are unique across both banks.

## Public vs complete edition

This is the public edition: the interview and retrieval fields only. The
complete edition adds each chunk's `content` (the course-derived lesson text
with code) and source references, plus the per-module chunk files and the
builder scripts that author the chunks. It lives in a private repository
because the lesson text derives from course material. Everything the app
does at runtime works on this edition; only training-data generation
(`grader/generate_answers.py`, which reads `content`) needs the complete
banks, passed via `RAG_FULL_DIR`. `tools/strip_chunks.py` regenerates this
edition from the complete one.
