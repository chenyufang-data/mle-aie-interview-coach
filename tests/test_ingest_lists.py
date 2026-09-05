"""Offline tests for the GitHub-list ingest (grader/ingest_lists.py).

Run:  .venv\\Scripts\\python tests\\test_ingest_lists.py

Pure parsing / routing / dedupe / chunk-assembly helpers on synthetic
markdown - no clones, no banks, no network.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grader.ingest_lists import (  # noqa: E402
    bank_overlap, build_chunk, candidate_id, cap_pool, module_for,
    parse_kalyan_file, parse_omb_questions, select_tiers,
)

OMB = """# RAG - Interview Questions

55 questions: 2 basic, 1 advanced.

## Basic

### 1. What is RAG, and what problem does it solve?

<details><summary><b>Answer</b></summary>

RAG fetches documents at query time. **Worth sketching.**

```mermaid
graph LR; A-->B
```

</details>

### 2. What is a chunk?

<details><summary><b>Answer</b></summary>

A chunk is a unit of text.

</details>

## Advanced

### 3. When does reranking pay for itself under a latency budget?

<details><summary><b>Answer</b></summary>

Only when first-stage recall is the bottleneck.

</details>
"""

KALYAN = """Authored by **Kalyan KS**.

## 📌 Q1: Explain how KV Cache accelerates LLM inference.

### ✅ Answer

The KV cache stores keys and values of previous tokens.

## 📌 Q2: What is a foundation model?

### ✅ Answer

A large pre-trained model.

---

## **👨🏻‍💻 LLM Engineer Toolkit**
"""


def test_parse_omb_tiers_and_answers():
    rows = parse_omb_questions(OMB, "04-rag-and-retrieval")
    assert [r["num"] for r in rows] == [1, 2, 3]
    assert [r["tier"] for r in rows] == ["basic", "basic", "advanced"]
    assert rows[0]["text"] == "What is RAG, and what problem does it solve?"
    assert "fetches documents" in rows[0]["answer"]
    assert "mermaid" not in rows[0]["answer"]
    assert rows[2]["answer"] == "Only when first-stage recall is the bottleneck."


def test_parse_kalyan():
    rows = parse_kalyan_file(KALYAN, "QA_1-3.md")
    assert [r["num"] for r in rows] == [1, 2]
    assert rows[0]["text"] == "Explain how KV Cache accelerates LLM inference."
    assert rows[0]["answer"] == "The KV cache stores keys and values of previous tokens."
    assert "Toolkit" not in rows[1]["answer"]


def test_tiers_modules_ids():
    rows = parse_omb_questions(OMB, "04-rag-and-retrieval")
    rows += parse_kalyan_file(KALYAN, "QA_1-3.md")
    kept = select_tiers(rows)
    # ombharatiya basic tier is out; Kalyan (untiered) stays.
    assert [r["text"][:12] for r in kept] == ["When does re", "Explain how ", "What is a fo"]
    assert module_for(kept[0]) == "RAG & Retrieval"
    assert module_for(kept[1]) == "Inference & Production"
    assert module_for(kept[2]) == "LLM Fundamentals"
    assert candidate_id(kept[0]) == "list_omb_04_003_when_does_reranking_pay"
    assert candidate_id(kept[1]) == "list_kal_001_explain_how_kv_cache"


def test_bank_overlap_is_question_vs_question():
    bank = [{"id": "10_pca__02_what_is_pca",
             "interview": {"question": "What is PCA?",
                           "key_points": ["what problem it can actually solve",
                                          "variance", "projection"]}},
            {"id": "kv_cache",
             "interview": {"question": "What is the KV cache and why does it speed up decoding?",
                           "key_points": ["keys and values are reused"]}}]
    rows = [{"text": "What is RAG, and what problem does it actually solve?"},
            {"text": "What are the limits of PCA on sparse data?"},
            {"text": "Why does the KV cache speed up decoding?"}]
    kept, dropped = bank_overlap(rows, bank)
    # Key points no longer make a 4-token question a duplicate of a long
    # rubric, and a 1-token bank question no longer absorbs every question
    # that mentions its term; a question mostly contained in a bank
    # question is the duplicate.
    assert [r["text"][:12] for r in kept] == ["What is RAG,", "What are the"]
    assert kept[0].get("bank_ref") == "10_pca__02_what_is_pca"
    assert dropped[0]["bank_dup"] == "kv_cache"


def test_cap_pool():
    rows = []
    for n in range(1, 5):
        rows.append({"src": "omb", "folder": "04-rag-and-retrieval", "num": n, "text": f"q{n}"})
    rows.append({"src": "kal", "num": 1, "text": "What is a foundation model?"})
    rows.append({"src": "kal", "num": 2, "text": "Explain KV cache and inference batching."})
    out = cap_pool(rows, cap=2, kalyan_cap=1)
    assert [r["text"] for r in out] == ["q1", "q2", "Explain KV cache and inference batching."]
    assert rows[2]["capped"] and rows[4]["capped"]


def test_build_chunk():
    row = {"src": "omb", "folder": "04-rag-and-retrieval", "tier": "advanced", "num": 3,
           "text": "When does reranking pay for itself?", "answer": "x",
           "file": "04-rag-and-retrieval/questions.md", "module": "RAG & Retrieval",
           "id": "list_omb_04_003_when_does_reranking_pay", "bank_ref": "03_rag__07"}
    rubric = {"question": "When does reranking pay for itself?", "topic": "reranking",
              "tags": ["rag"], "difficulty": "advanced", "round": "technical",
              "model_answer": "...", "key_points": ["a"], "common_mistakes": ["b"],
              "followups": ["c"]}
    chunk = build_chunk(row, rubric, "abc1234")
    assert chunk["id"] == row["id"]
    assert chunk["metadata"]["license"] == "MIT"
    assert chunk["metadata"]["source_url"].endswith("/blob/abc1234/04-rag-and-retrieval/questions.md")
    assert chunk["metadata"]["bank_ref"] == "03_rag__07"
    assert chunk["metadata"]["original"] == row["text"]
    assert "answer" not in chunk["metadata"]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all ingest_lists tests passed")
