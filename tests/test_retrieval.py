"""Retrieval smoke test over curated queries.

Run:  .venv\\Scripts\\python tests\\test_retrieval.py
      .venv\\Scripts\\python tests\\test_retrieval.py --backend dense --cases tests/retrieval_cases_paraphrase.json

Covers both question banks: rag_ml (corpus "ml", MLE track) and rag_ai
(corpus "ai", AIE track). Reports Recall@5 (is a relevant question in the
top five?) and MRR (how high does the first relevant result rank?). A case's
result is relevant when it matches the case's expected_module and shares a
tag with expected_tags (whichever of the two the case specifies).

The default (BM25, the curated cases) is the CI gate. --backend dense or
hybrid runs the experiment arms from grader/dense_retrieval.py and skips
cleanly when their optional dependencies (requirements-retrieval-eval.txt)
are absent; the measured comparison lives in docs/retrieval_evaluation.md.
"""

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from retrieval import Retriever

CORPUS_DIRS = {"ml": "rag_ml", "ai": "rag_ai"}
CORPUS_ROLE = {"ml": "MLE", "ai": "AIE"}  # vector-cache names, as coach/config.py
LIMIT = 5
RECALL_TARGET = 0.9


def is_relevant(chunk, case):
    meta = chunk["metadata"]
    module = case.get("expected_module")
    if module and meta["module"] != module:
        return False
    tags = case.get("expected_tags")
    if tags and not set(tags) & set(meta["tags"]):
        return False
    return bool(module or tags)


def load_chunks(corpus_dir):
    with (BASE_DIR / corpus_dir / "all_chunks.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_retrievers(backend):
    chunks = {name: load_chunks(path) for name, path in CORPUS_DIRS.items()}
    if backend == "bm25":
        return {name: Retriever(c) for name, c in chunks.items()}
    from retrieval_dense import DenseRetriever, HybridRetriever, hybrid_availability, load_or_build
    embedder = hybrid_availability()
    if isinstance(embedder, str):
        print(f"SKIP: --backend {backend} cannot run here - {embedder}")
        return None
    retrievers = {}
    for name, c in chunks.items():
        vectors, _, _ = load_or_build(CORPUS_ROLE[name], c, embedder)
        dense = DenseRetriever(c, embedder, vectors)
        retrievers[name] = dense if backend == "dense" else HybridRetriever(Retriever(c), dense)
    return retrievers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["bm25", "dense", "hybrid"], default="bm25")
    parser.add_argument("--cases", default=str(BASE_DIR / "tests" / "retrieval_cases.json"))
    parser.add_argument("--no-gate", action="store_true",
                        help="report only; never fail on the recall target (experiment sets)")
    args = parser.parse_args()

    retrievers = load_retrievers(args.backend)
    if retrievers is None:
        return 0
    with Path(args.cases).open(encoding="utf-8") as handle:
        cases = json.load(handle)

    hits = 0
    reciprocal_ranks = []
    per_corpus = {name: [0, 0] for name in CORPUS_DIRS}  # corpus -> [hits, total]
    for case in cases:
        corpus = case.get("corpus", "ml")
        results = retrievers[corpus].search(query=case["query"], limit=LIMIT)[:LIMIT]
        rank = next(
            (pos for pos, chunk in enumerate(results, start=1) if is_relevant(chunk, case)),
            None,
        )
        reciprocal_ranks.append(1 / rank if rank else 0.0)
        hits += 1 if rank else 0
        per_corpus[corpus][0] += 1 if rank else 0
        per_corpus[corpus][1] += 1
        print(f"[{'PASS' if rank else 'FAIL'}] [{corpus}] rank={rank or '-'}  {case['query']!r}")
        if not rank and results:
            top = results[0]["metadata"]
            print(f"       top result was: {top['module']} / {top['topic']} tags={top['tags']}")

    recall = hits / len(cases)
    mrr = sum(reciprocal_ranks) / len(cases)
    by_corpus = "   ".join(
        f"{name}: {done[0]}/{done[1]}" for name, done in per_corpus.items() if done[1]
    )
    print(f"\n[{args.backend}] Recall@{LIMIT}: {recall:.0%} ({hits}/{len(cases)})   MRR: {mrr:.2f}   [{by_corpus}]")
    if recall < RECALL_TARGET and not args.no_gate:
        print(f"Below the {RECALL_TARGET:.0%} Recall@{LIMIT} target - retrieval needs attention.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
