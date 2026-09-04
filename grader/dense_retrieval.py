"""Experiment-only pieces of the retrieval experiment (docs/dense_retrieval_plan.md).

The dense and hybrid retrievers themselves live in retrieval_dense.py at the
repo root since rule R1 passed and they ship in the runtime; this module
re-exports them for the harness and adds what the runtime does not need:

    ChromaRetriever  the same vectors served from a persistent Chroma
                     collection - the "does a vector store earn its place at
                     339 chunks?" arm (Q3); needs chromadb
    rss_mb / hardware / dir_size_mb   measurement helpers for the report
"""

import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from retrieval import LEVEL_DIFFICULTY, retrieval_text  # noqa: E402
from retrieval_dense import (INDEX_DIR, MODEL_CACHE, MODEL_NAME, QUERY_PREFIX,  # noqa: E402,F401
                             RRF_K, DenseRetriever, Embedder, HybridRetriever,
                             corpus_fingerprint, filter_candidates, level_candidates,
                             load_or_build, local_model_dir, normalize_rows, rrf_fuse)


# ------------------------------------------------------------ measurement

def rss_mb():
    """Resident set size of this process in MB (None if unmeasurable)."""
    try:
        import psutil  # optional
        return psutil.Process().memory_info().rss / 1048576
    except ImportError:
        pass
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        get_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), wintypes.DWORD]
        get_info.restype = wintypes.BOOL
        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        if get_info(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return counters.WorkingSetSize / 1048576
        return None
    try:
        import resource
        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return kb / 1024 if sys.platform != "darwin" else kb / 1048576
    except Exception:
        return None


def hardware():
    """CPU model + thread count, so latency is never reported bare."""
    name = platform.processor() or platform.machine()
    if os.name == "nt":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            name = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    else:
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith("model name"):
                    name = line.split(":", 1)[1].strip()
                    break
        except OSError:
            pass
    return {"cpu": name, "threads": os.cpu_count(), "platform": platform.platform(),
            "python": platform.python_version()}


def dir_size_mb(path):
    path = Path(path)
    if not path.exists():
        return 0.0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1048576


def clear_index(cache_dir=None):
    shutil.rmtree(Path(cache_dir or INDEX_DIR), ignore_errors=True)


# ------------------------------------------------------------------ chroma

class ChromaRetriever:
    """The same vectors served from a persistent Chroma collection. Filters
    map to `where` clauses (difficulty $in, module $eq) when the filtered
    pool is non-empty - the same outcome as filter_candidates - and exclude
    is applied by over-fetching. ef_search is raised so HNSW is effectively
    exact at this size; agreement with the numpy arm is measured anyway."""

    def __init__(self, chunks, embedder, doc_vectors, name, path=None, reset=True):
        os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
        import chromadb
        from chromadb.config import Settings

        self.chunks = list(chunks)
        self.embedder = embedder
        self.path = Path(path or (INDEX_DIR / "chroma"))
        self.path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.path),
                                                settings=Settings(anonymized_telemetry=False))
        if reset:
            try:
                self.client.delete_collection(name)
            except Exception:
                pass
        try:
            self.collection = self.client.get_or_create_collection(
                name, configuration={"hnsw": {"space": "cosine", "ef_search": 512,
                                              "ef_construction": 200}})
        except TypeError:  # older API: metadata-form settings
            self.collection = self.client.get_or_create_collection(
                name, metadata={"hnsw:space": "cosine", "hnsw:search_ef": 512})
        self.build_seconds = None
        if self.collection.count() != len(self.chunks):
            t0 = time.perf_counter()
            vectors = normalize_rows(doc_vectors)
            for start in range(0, len(self.chunks), 100):
                batch = self.chunks[start:start + 100]
                self.collection.upsert(
                    ids=[c["id"] for c in batch],
                    embeddings=vectors[start:start + len(batch)].tolist(),
                    documents=[retrieval_text(c) for c in batch],
                    metadatas=[{"module": c["metadata"]["module"],
                                "difficulty": c["metadata"]["difficulty"]} for c in batch],
                )
            self.build_seconds = time.perf_counter() - t0
        self.by_id = {c["id"]: c for c in self.chunks}

    def _where(self, module=None, level=None):
        clauses = []
        if module and module != "Any course topic":
            clauses.append({"module": {"$eq": module}})
        allowed = LEVEL_DIFFICULTY.get(level)
        if allowed:
            pool = filter_candidates(self.chunks, module, level)
            if all(c["metadata"]["difficulty"] in allowed for _, c in pool):
                clauses.append({"difficulty": {"$in": sorted(allowed)}})
        if not clauses:
            return None
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    def _query(self, query, where, n):
        n = max(1, min(n, len(self.chunks)))
        result = self.collection.query(
            query_embeddings=[self.embedder.embed_query(query).tolist()],
            n_results=n, where=where, include=["distances"])
        ids = result["ids"][0]
        dists = result["distances"][0]
        return [(1.0 - float(d), self.by_id[i]) for i, d in zip(ids, dists)]

    def top_scored(self, query, level=None, limit=3):
        if not self.chunks or not (query or "").strip():
            return []
        return self._query(query, self._where(level=level), limit)

    def search(self, query="", module=None, level=None, exclude_ids=(), limit=5):
        candidates = filter_candidates(self.chunks, module, level, exclude_ids)
        if not (query or "").strip():
            return [c for _, c in candidates]
        exclude_ids = set(exclude_ids or ())
        hits = self._query(query, self._where(module, level), limit + len(exclude_ids))
        allowed_ids = {c["id"] for _, c in candidates}
        return [c for _, c in hits if c["id"] in allowed_ids][:limit]


if __name__ == "__main__":
    # Smoke: build every bank once, print sizes and a sanity query.
    from coach import config

    embedder = Embedder()
    print(json.dumps({"hardware": hardware(), "model": embedder.info(), "rss_mb": rss_mb()}, indent=1))
    for role, path in config.CORPUS_PATHS.items():
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            chunks = [json.loads(line) for line in handle if line.strip()]
        vectors, seconds, cached = load_or_build(role, chunks, embedder)
        dense = DenseRetriever(chunks, embedder, vectors)
        top = dense.top_scored("my model does great on training data and badly on new data", limit=1)
        print(f"{role}: {len(chunks)} chunks embedded in {seconds:.2f}s (cache={cached}); "
              f"sanity top-1: {top[0][1]['metadata']['topic']!r} cos={top[0][0]:.3f}")
