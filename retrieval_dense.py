"""Dense and hybrid retrieval over the question banks - the runtime half of
the retrieval experiment (docs/plan.md, results in
docs/retrieval_evaluation.md).

Shipped because the pre-registered rule R1 passed for `hybrid` on
2026-09-04: on the vocabulary-mismatch set it lifted Recall@5 from 41/61 to
49/61 (+13 points) with no regression on the curated set and a 2.6 ms p95.
Dense alone missed the bar by 0.2 points, so what ships is the fusion:

    DenseRetriever   bge-small-en-v1.5 (fastembed, ONNX, CPU), cosine over a
                     numpy matrix of the same `retrieval_text` BM25 indexes
    HybridRetriever  reciprocal-rank fusion (k = 60) of BM25 and dense ranks

Both expose retrieval.Retriever's two methods with the same signatures and
the same filter semantics (module strict; level and exclude skip themselves
rather than empty the pool), so coach/kb.py swaps them in with one line and
falls back to BM25 - with a stated reason - when fastembed or the model is
unavailable. The model (~127 MB) downloads once into data/models/fastembed/;
document vectors cache under data/index/ keyed by a content hash.
"""

import hashlib
import os
import time
from pathlib import Path

import numpy as np

from retrieval import LEVEL_DIFFICULTY, retrieval_text, tokenize

BASE_DIR = Path(__file__).resolve().parent
MODEL_NAME = "BAAI/bge-small-en-v1.5"
# Fixed before the experiment ran (plan §1): bge v1.5 recommends this prefix
# on the query side for short query -> passage retrieval; documents get none.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
# Both overridable so Docker can point them at the persistent volume
# (docker-compose.yml): the one-time ~127 MB model download and the document
# vectors then survive image rebuilds. Defaults are the local data/ layout.
MODEL_CACHE = Path(os.environ.get("FASTEMBED_CACHE_DIR")
                   or BASE_DIR / "data" / "models" / "fastembed")
INDEX_DIR = Path(os.environ.get("RETRIEVAL_INDEX_DIR") or BASE_DIR / "data" / "index")
RRF_K = 60


# ----------------------------------------------------------------- helpers

def normalize_rows(matrix):
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def _display_path(path):
    """Repo-relative when possible; the cache may live outside the repo
    (FASTEMBED_CACHE_DIR), where relative_to would raise."""
    try:
        return str(path.relative_to(BASE_DIR))
    except ValueError:
        return str(path)


def local_model_dir():
    """A model directory placed under MODEL_CACHE by hand (huggingface.co is
    unreachable from some networks; the fastembed tarball from Google Cloud
    Storage works there). None means "let fastembed download"."""
    if not MODEL_CACHE.exists():
        return None
    for path in sorted(MODEL_CACHE.iterdir()):
        if path.is_dir() and (path / "model_optimized.onnx").exists():
            return path
    return None


def filter_candidates(chunks, module=None, level=None, exclude_ids=()):
    """Exactly Retriever.search's candidate pool: module is strict (raises
    when empty), the level and exclude filters skip themselves rather than
    empty the pool. Returns (index, chunk) pairs."""
    if not chunks:
        raise RuntimeError("The question bank is empty.")
    candidates = list(enumerate(chunks))
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
    return candidates


def level_candidates(chunks, level=None):
    """Retriever.top_scored's pool: level filter only, skip-if-empty."""
    candidates = list(enumerate(chunks))
    allowed = LEVEL_DIFFICULTY.get(level)
    if allowed:
        leveled = [(i, c) for i, c in candidates if c["metadata"]["difficulty"] in allowed]
        if leveled:
            candidates = leveled
    return candidates


# ---------------------------------------------------------------- embedder

class Embedder:
    """bge-small via fastembed (ONNX on CPU). Query texts get QUERY_PREFIX,
    documents do not - fixed in the plan, not tunable."""

    def __init__(self, model_name=MODEL_NAME, query_prefix=QUERY_PREFIX, threads=None):
        os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
        from fastembed import TextEmbedding

        local = local_model_dir()
        kwargs = {"specific_model_path": str(local)} if local else {"cache_dir": str(MODEL_CACHE)}
        if threads:
            kwargs["threads"] = threads
        MODEL_CACHE.mkdir(parents=True, exist_ok=True)
        self.model = TextEmbedding(model_name, **kwargs)
        self.name = model_name
        self.query_prefix = query_prefix
        self.model_dir = Path(local) if local else None
        self.source = _display_path(local) if local else "fastembed download"

    def info(self):
        onnx = (self.model_dir / "model_optimized.onnx") if self.model_dir else None
        return {"model": self.name, "source": self.source,
                "onnx_mb": round(onnx.stat().st_size / 1048576, 1) if onnx and onnx.exists() else None,
                "query_prefix": self.query_prefix}

    def embed_docs(self, texts):
        return normalize_rows(list(self.model.embed(list(texts), batch_size=64)))

    def embed_query(self, text):
        return normalize_rows([next(self.model.embed([self.query_prefix + text]))])[0]


# ------------------------------------------------------------------- dense

class DenseRetriever:
    def __init__(self, chunks, embedder, doc_vectors=None):
        self.chunks = list(chunks)
        self.embedder = embedder
        if doc_vectors is None:
            doc_vectors = embedder.embed_docs([retrieval_text(c) for c in self.chunks])
        self.vectors = normalize_rows(doc_vectors)
        assert len(self.vectors) == len(self.chunks)

    def scores(self, query):
        """Cosine similarity of the query to every chunk (index-aligned)."""
        return self.vectors @ self.embedder.embed_query(query)

    def rank(self, query, candidates):
        """Candidate (index, chunk) pairs best first; ties broken by index."""
        scores = self.scores(query)
        return sorted(((float(scores[i]), i, c) for i, c in candidates),
                      key=lambda item: (-item[0], item[1]))

    def top_scored(self, query, level=None, limit=3):
        if not self.chunks or not (query or "").strip():
            return []
        ranked = self.rank(query, level_candidates(self.chunks, level))
        return [(score, chunk) for score, _, chunk in ranked[:limit]]

    def search(self, query="", module=None, level=None, exclude_ids=(), limit=5):
        candidates = filter_candidates(self.chunks, module, level, exclude_ids)
        if not (query or "").strip():
            return [c for _, c in candidates]
        return [c for _, _, c in self.rank(query, candidates)[:limit]]


# ---------------------------------------------------------------- pgvector

def vector_literal(vector):
    """A vector in pgvector's text form, "[x,y,...]", at float32 precision."""
    return "[" + ",".join(repr(float(x)) for x in np.asarray(vector, dtype=np.float32)) + "]"


def pgvector_pool(url, max_size=4, timeout=10.0):
    """A small psycopg pool for PgVectorRetriever when no state-store pool
    exists (the experiment harness). Raises RuntimeError with the reason."""
    try:
        from psycopg_pool import ConnectionPool
    except ImportError as exc:
        raise RuntimeError("psycopg is not installed (pip install -r requirements-db.txt)") from exc
    try:
        pool = ConnectionPool(url, min_size=1, max_size=max_size, open=True, timeout=timeout,
                              kwargs={"autocommit": True})
        pool.wait(timeout=timeout)
    except Exception as exc:
        raise RuntimeError(f"cannot connect: {exc}") from exc
    return pool


class PgVectorRetriever:
    """DenseRetriever's vectors served by Postgres + pgvector (roadmap step
    4): the same bge-small document vectors in a chunk_vectors table, keyed
    by corpus and the corpus fingerprint, so a changed bank re-embeds and a
    rebuilt container reuses. Every query is an exact cosine scan (`<=>`
    over a few hundred rows, no ANN index, so nothing approximates) and the
    candidate filters apply in Python to the returned order - the contract
    DenseRetriever keeps. The step 4 rule (grader/retrieval_eval.py) is what
    lets it serve: identical top-5 to the numpy arm on sets A and B and a
    p95 under 50 ms."""

    def __init__(self, chunks, embedder, doc_vectors, name, pool, table="chunk_vectors"):
        self.chunks = list(chunks)
        self.embedder = embedder
        self.name = name
        self.pool = pool
        self.table = table
        self.vectors = normalize_rows(doc_vectors)
        assert len(self.vectors) == len(self.chunks)
        self.fingerprint = corpus_fingerprint(self.chunks, embedder.name)
        self.dim = int(self.vectors.shape[1])
        self.build_seconds = None
        self._ensure()

    def _ensure(self):
        """Create the extension and table when missing; (re)load this
        corpus's rows unless the stored fingerprint and count already match."""
        with self.pool.connection() as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS {self.table} (corpus text NOT NULL, "
                f"fingerprint text NOT NULL, idx integer NOT NULL, chunk_id text NOT NULL, "
                f"embedding vector({self.dim}) NOT NULL, PRIMARY KEY (corpus, idx))")
            stored = conn.execute(
                f"SELECT fingerprint, count(*) FROM {self.table} WHERE corpus = %s "
                "GROUP BY fingerprint", (self.name,)).fetchall()
            if stored == [(self.fingerprint, len(self.chunks))]:
                return
            t0 = time.perf_counter()
            with conn.transaction():
                conn.execute(f"DELETE FROM {self.table} WHERE corpus = %s", (self.name,))
                with conn.cursor() as cur:
                    cur.executemany(
                        f"INSERT INTO {self.table} (corpus, fingerprint, idx, chunk_id, embedding) "
                        "VALUES (%s, %s, %s, %s, %s::vector)",
                        [(self.name, self.fingerprint, i, c["id"], vector_literal(v))
                         for i, (c, v) in enumerate(zip(self.chunks, self.vectors))])
            self.build_seconds = time.perf_counter() - t0

    def size_mb(self):
        """Table plus index bytes on disk, every corpus included."""
        with self.pool.connection() as conn:
            size = conn.execute("SELECT pg_total_relation_size(%s::regclass)",
                                (self.table,)).fetchone()[0]
        return size / 1048576

    def scores_ranked(self, query):
        """[(index, cosine)] best first over the whole corpus, ties by index."""
        literal = vector_literal(self.embedder.embed_query(query))
        with self.pool.connection() as conn:
            rows = conn.execute(
                f"SELECT idx, 1 - (embedding <=> %s::vector) FROM {self.table} "
                "WHERE corpus = %s ORDER BY embedding <=> %s::vector, idx",
                (literal, self.name, literal)).fetchall()
        return [(int(i), float(s)) for i, s in rows]

    def rank(self, query, candidates):
        """Candidate (index, chunk) pairs best first; ties broken by index."""
        scores = dict(self.scores_ranked(query))
        return sorted(((scores.get(i, -1.0), i, c) for i, c in candidates),
                      key=lambda item: (-item[0], item[1]))

    def top_scored(self, query, level=None, limit=3):
        if not self.chunks or not (query or "").strip():
            return []
        ranked = self.rank(query, level_candidates(self.chunks, level))
        return [(score, chunk) for score, _, chunk in ranked[:limit]]

    def search(self, query="", module=None, level=None, exclude_ids=(), limit=5):
        candidates = filter_candidates(self.chunks, module, level, exclude_ids)
        if not (query or "").strip():
            return [c for _, c in candidates]
        return [c for _, _, c in self.rank(query, candidates)[:limit]]


# ------------------------------------------------------------------ hybrid

def rrf_fuse(rank_lists, k=RRF_K):
    """Reciprocal-rank fusion: item -> sum over lists of 1 / (k + rank),
    rank starting at 1; items absent from a list contribute nothing."""
    fused = {}
    for ranked in rank_lists:
        for rank, item in enumerate(ranked, start=1):
            fused[item] = fused.get(item, 0.0) + 1.0 / (k + rank)
    return fused


class HybridRetriever:
    """BM25 + dense fused by RRF over the same candidate pool. BM25's list
    holds only chunks it scores > 0 (its own notion of a match); dense
    ranks every candidate. Ties break by chunk index. Scores are RRF
    values (~0.016-0.033), not BM25 units - the mock's rubric threshold
    keeps reading the BM25 retriever (coach/kb.py keeps both)."""

    def __init__(self, bm25, dense, k=RRF_K):
        assert [c["id"] for c in bm25.chunks] == [c["id"] for c in dense.chunks]
        self.bm25 = bm25
        self.dense = dense
        self.chunks = dense.chunks
        self.k = k

    def rank(self, query, candidates):
        tokens = list(dict.fromkeys(tokenize(query)))
        bm25_scored = sorted(((self.bm25._bm25(i, tokens), i) for i, _ in candidates),
                             key=lambda item: (-item[0], item[1]))
        bm25_list = [i for score, i in bm25_scored if score > 0]
        dense_list = [i for _, i, _ in self.dense.rank(query, candidates)]
        fused = rrf_fuse([bm25_list, dense_list], self.k)
        by_index = {i: c for i, c in candidates}
        return sorted(((fused.get(i, 0.0), i, by_index[i]) for i, _ in candidates),
                      key=lambda item: (-item[0], item[1]))

    def top_scored(self, query, level=None, limit=3):
        if not self.chunks or not (query or "").strip():
            return []
        ranked = self.rank(query, level_candidates(self.chunks, level))
        return [(score, chunk) for score, _, chunk in ranked[:limit] if score > 0]

    def search(self, query="", module=None, level=None, exclude_ids=(), limit=5):
        candidates = filter_candidates(self.chunks, module, level, exclude_ids)
        if not (query or "").strip():
            return [c for _, c in candidates]
        ranked = [c for score, _, c in self.rank(query, candidates) if score > 0]
        return ranked[:limit] if ranked else [c for _, c in candidates]


# --------------------------------------------------------- index build/cache

def corpus_fingerprint(chunks, model_name):
    digest = hashlib.sha256(model_name.encode("utf-8"))
    for chunk in chunks:
        digest.update(chunk["id"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(retrieval_text(chunk).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()[:16]


def load_or_build(name, chunks, embedder, cache_dir=None):
    """Document vectors for one corpus, cached under data/index/ keyed by a
    hash of (model, ids, document text) so rebuilds are deterministic and
    stale caches are impossible. Returns (vectors, seconds, from_cache)."""
    cache_dir = Path(cache_dir or INDEX_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = corpus_fingerprint(chunks, embedder.name)
    path = cache_dir / f"{name}.{fingerprint}.npz"
    if path.exists():
        t0 = time.perf_counter()
        vectors = np.load(path)["vectors"]
        return vectors, time.perf_counter() - t0, True
    t0 = time.perf_counter()
    vectors = embedder.embed_docs([retrieval_text(c) for c in chunks])
    seconds = time.perf_counter() - t0
    for stale in cache_dir.glob(f"{name}.*.npz"):
        stale.unlink()
    np.savez(path, vectors=vectors)
    return vectors, seconds, False


def hybrid_availability():
    """None when the hybrid stack can run here, else a short reason. Loads
    the model (downloading it on first use), so a None answer means the
    embedder is ready."""
    try:
        import fastembed  # noqa: F401
    except ImportError as exc:
        return f"missing dependency {getattr(exc, 'name', exc)!r} (pip install -r requirements.txt)"
    try:
        embedder = Embedder()
    except Exception as exc:  # network (model download), corrupt cache, ...
        return f"embedding model unavailable: {type(exc).__name__}: {str(exc)[:120]}"
    return embedder
