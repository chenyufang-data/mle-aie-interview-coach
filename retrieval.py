"""Retrieval over the course question bank.

Filters candidates by metadata (module, candidate level, already-asked ids),
then ranks them with BM25 against a short per-question document built from the
interview fields - not the raw lesson content, which is long and noisy.
Pure Python; at ~200 chunks scoring every candidate per query is plenty fast.
"""

import math
import re
from collections import Counter


# Which knowledge-base difficulties fit each candidate level.
LEVEL_DIFFICULTY = {
    "Entry-level": {"beginner", "beginner-intermediate"},
    "Mid-level": {"beginner-intermediate", "intermediate"},
    "Senior": {"intermediate", "advanced"},
    "Staff": {"intermediate", "advanced"},
}

BM25_K1 = 1.5
BM25_B = 0.75


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def retrieval_text(chunk):
    meta = chunk["metadata"]
    interview = chunk["interview"]
    return " ".join(
        [
            meta["module"],
            meta["topic"],
            " ".join(meta["tags"]),
            interview["question"],
            " ".join(interview["key_points"]),
        ]
    )


class Retriever:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self._term_freqs = [Counter(tokenize(retrieval_text(c))) for c in self.chunks]
        self._doc_lengths = [sum(tf.values()) for tf in self._term_freqs]
        total = len(self.chunks)
        self._avg_length = (sum(self._doc_lengths) / total) if total else 0.0
        doc_freq = Counter()
        for tf in self._term_freqs:
            doc_freq.update(tf.keys())
        self._idf = {
            term: math.log(1 + (total - df + 0.5) / (df + 0.5))
            for term, df in doc_freq.items()
        }

    def _bm25(self, index, query_tokens):
        tf = self._term_freqs[index]
        length_norm = 1 - BM25_B + BM25_B * self._doc_lengths[index] / self._avg_length
        score = 0.0
        for term in query_tokens:
            freq = tf.get(term)
            if freq:
                score += self._idf[term] * freq * (BM25_K1 + 1) / (freq + BM25_K1 * length_norm)
        return score

    def top_scored(self, query, level=None, limit=3):
        """(score, chunk) pairs, best first — for callers that need the BM25
        score itself (the mock's probe→rubric matching applies a threshold,
        unlike search(), which serves *some* question no matter what)."""
        query_tokens = list(dict.fromkeys(tokenize(query))) if query else []
        if not self.chunks or not query_tokens:
            return []
        candidates = list(enumerate(self.chunks))
        allowed = LEVEL_DIFFICULTY.get(level)
        if allowed:
            leveled = [(i, c) for i, c in candidates if c["metadata"]["difficulty"] in allowed]
            if leveled:
                candidates = leveled
        scored = sorted(((self._bm25(i, query_tokens), i, c) for i, c in candidates),
                        key=lambda item: (-item[0], item[1]))
        return [(score, chunk) for score, _, chunk in scored[:limit] if score > 0]

    def search(self, query="", module=None, level=None, exclude_ids=(), limit=5):
        """Return candidate chunks, best first.

        The level and exclude filters skip themselves rather than empty the
        pool, so there is always at least one candidate. Without a query (or
        when nothing matches it) the whole filtered pool comes back so the
        caller can pick at random; otherwise only the top ``limit`` matches do.
        """
        if not self.chunks:
            raise RuntimeError("The question bank is empty.")

        candidates = list(enumerate(self.chunks))

        if module and module != "Any course topic":
            candidates = [(i, c) for i, c in candidates if c["metadata"]["module"] == module]
            if not candidates:
                raise RuntimeError(f"No knowledge-base questions for module {module!r}.")

        allowed = LEVEL_DIFFICULTY.get(level)
        if allowed:
            leveled = [(i, c) for i, c in candidates if c["metadata"]["difficulty"] in allowed]
            if leveled:
                candidates = leveled

        exclude_ids = set(exclude_ids or ())
        if exclude_ids:
            fresh = [(i, c) for i, c in candidates if c["id"] not in exclude_ids]
            if fresh:
                candidates = fresh

        query_tokens = list(dict.fromkeys(tokenize(query))) if query else []
        if not query_tokens:
            return [c for _, c in candidates]

        scored = sorted(
            ((self._bm25(i, query_tokens), i, c) for i, c in candidates),
            key=lambda item: (-item[0], item[1]),
        )
        matched = [c for score, _, c in scored if score > 0]
        if not matched:
            return [c for _, c in candidates]
        return matched[:limit]
