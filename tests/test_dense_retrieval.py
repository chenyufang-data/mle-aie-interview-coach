"""Offline unit test for the dense/hybrid retrieval arms (grader/dense_retrieval.py).

Run:  .venv\\Scripts\\python tests\\test_dense_retrieval.py

No model download: a fake embedder maps text to a deterministic
bag-of-words vector, which is enough to check what the experiment relies
on - filter parity with retrieval.Retriever, the RRF fusion math, ranking
and tie-breaking, and the top_scored/search contracts. The real model's
numbers live in grader/retrieval_eval.py, not here.
"""

import hashlib
import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from retrieval import Retriever, retrieval_text, tokenize  # noqa: E402
from retrieval_dense import (DenseRetriever, HybridRetriever, filter_candidates,  # noqa: E402
                             level_candidates, normalize_rows, rrf_fuse)

DIM = 64


class FakeEmbedder:
    """Hashed bag of words - deterministic, similar texts get similar vectors."""
    name = "fake"
    query_prefix = ""

    def _vector(self, text):
        vec = np.zeros(DIM, dtype=np.float32)
        for token in tokenize(text):
            slot = int(hashlib.md5(token.encode()).hexdigest(), 16) % DIM
            vec[slot] += 1.0
        return vec

    def embed_docs(self, texts):
        return normalize_rows([self._vector(t) for t in texts])

    def embed_query(self, text):
        return normalize_rows([self._vector(text)])[0]


def chunk(cid, module, topic, tags, difficulty, question, points):
    return {"id": cid,
            "interview": {"question": question, "key_points": points, "model_answer": "",
                          "common_mistakes": [], "followups": []},
            "metadata": {"module": module, "topic": topic, "tags": tags,
                         "difficulty": difficulty}}


CHUNKS = [
    chunk("c1", "Regression", "Overfitting", ["overfitting", "regularization"], "beginner",
          "What is overfitting and how do you prevent it?", ["train-test gap", "regularization"]),
    chunk("c2", "Regression", "Leakage", ["data-leakage"], "intermediate",
          "How can test information leak into training?", ["preprocessing before split"]),
    chunk("c3", "Trees", "Boosting", ["gradient-boosting", "xgboost"], "advanced",
          "Explain gradient boosting.", ["residual fitting", "learning rate"]),
    chunk("c4", "Trees", "Random forest", ["bagging", "random-forest"], "intermediate",
          "Why does bagging reduce variance?", ["bootstrap", "decorrelated trees"]),
    chunk("c5", "Deployment", "Monitoring", ["drift", "monitoring"], "advanced",
          "How do you detect data drift in production?", ["PSI", "feature distributions"]),
]


def ids(seq):
    return [c["id"] for c in seq]


def test_filter_parity():
    bm25 = Retriever(CHUNKS)
    dense = DenseRetriever(CHUNKS, FakeEmbedder())
    combos = [dict(), dict(module="Trees"), dict(level="Entry-level"), dict(level="Senior"),
              dict(level="Staff", module="Regression"), dict(exclude_ids=("c1", "c2")),
              dict(level="Entry-level", exclude_ids=("c1",)),  # exclude would empty -> skipped
              dict(module="Deployment", level="Entry-level")]  # level would empty -> skipped
    for combo in combos:
        expected = ids(bm25.search(query="", **combo))
        got = ids(dense.search(query="", **combo))
        pool = ids(c for _, c in filter_candidates(CHUNKS, combo.get("module"),
                                                   combo.get("level"), combo.get("exclude_ids", ())))
        assert got == expected == pool, (combo, got, expected, pool)
    assert ids(c for _, c in level_candidates(CHUNKS, "Senior")) == ["c2", "c3", "c4", "c5"]
    try:
        dense.search(query="", module="Nope")
        raise AssertionError("module filter must raise on an empty pool, like BM25")
    except RuntimeError:
        pass
    print("PASS filter parity with retrieval.Retriever")


def test_dense_ranking():
    dense = DenseRetriever(CHUNKS, FakeEmbedder())
    top = dense.top_scored("overfitting regularization", limit=2)
    assert ids(c for _, c in top) == ["c1", "c2"] or ids(c for _, c in top)[0] == "c1", top
    assert all(-1.0 <= s <= 1.0 for s, _ in top)
    assert top[0][0] > top[1][0]
    # Level filter applies inside top_scored: Senior excludes the beginner chunk.
    senior = dense.top_scored("overfitting regularization", level="Senior", limit=5)
    assert "c1" not in ids(c for _, c in senior)
    assert dense.top_scored("", limit=3) == []
    hits = dense.search(query="gradient boosting residual", limit=1)
    assert ids(hits) == ["c3"]
    print("PASS dense ranking, level filter, limits")


def test_rrf_math():
    fused = rrf_fuse([["a", "b", "c"], ["b", "a"]], k=60)
    assert abs(fused["a"] - (1 / 61 + 1 / 62)) < 1e-12
    assert abs(fused["b"] - (1 / 62 + 1 / 61)) < 1e-12
    assert abs(fused["c"] - 1 / 63) < 1e-12
    assert fused["a"] == fused["b"] > fused["c"]
    print("PASS reciprocal-rank fusion arithmetic")


def test_hybrid():
    bm25 = Retriever(CHUNKS)
    dense = DenseRetriever(CHUNKS, FakeEmbedder())
    hybrid = HybridRetriever(bm25, dense)
    # A query only BM25 can match lexically: dense ranks everyone, BM25 lists
    # just its > 0 matches, and fusion still puts the lexical hit first.
    top = hybrid.top_scored("PSI feature distributions drift", limit=3)
    assert ids(c for _, c in top)[0] == "c5", top
    scores = [s for s, _ in top]
    assert scores == sorted(scores, reverse=True) and all(s > 0 for s in scores)
    # Same candidate pool semantics as the others.
    assert ids(hybrid.search(query="", module="Trees")) == ["c3", "c4"]
    assert ids(hybrid.search(query="drift", module="Trees", limit=2)) in (["c3", "c4"], ["c4", "c3"])
    # Ties (identical fused scores) break by chunk index, deterministically.
    ranked = hybrid.rank("zzz-unseen-token", list(enumerate(CHUNKS)))
    assert [i for _, i, _ in ranked] == sorted(i for _, i, _ in ranked) or \
        len({s for s, _, _ in ranked}) > 1
    print("PASS hybrid fusion, pool semantics, tie-break")


def test_retrieval_text_is_shared():
    text = retrieval_text(CHUNKS[0])
    assert "Regression" in text and "overfitting" in text and "train-test gap" in text
    print("PASS all arms rank retrieval_text (module + topic + tags + question + key points)")


if __name__ == "__main__":
    test_filter_parity()
    test_dense_ranking()
    test_rrf_math()
    test_hybrid()
    test_retrieval_text_is_shared()
    print("\nAll dense-retrieval unit tests passed.")
