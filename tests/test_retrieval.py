"""Retrieval smoke test over curated queries.

Run:  .venv\\Scripts\\python tests\\test_retrieval.py

Covers both question banks: rag_ml (corpus "ml", MLE track) and rag_ai
(corpus "ai", AIE track). Reports Recall@5 (is a relevant question in the
top five?) and MRR (how high does the first relevant result rank?). A case's
result is relevant when it matches the case's expected_module and shares a
tag with expected_tags (whichever of the two the case specifies).
"""

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from retrieval import Retriever

CORPUS_DIRS = {"ml": "rag_ml", "ai": "rag_ai"}
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


def load_retriever(corpus_dir):
    with (BASE_DIR / corpus_dir / "all_chunks.jsonl").open(encoding="utf-8") as handle:
        return Retriever([json.loads(line) for line in handle if line.strip()])


def main():
    retrievers = {name: load_retriever(path) for name, path in CORPUS_DIRS.items()}

    with (BASE_DIR / "tests" / "retrieval_cases.json").open(encoding="utf-8") as handle:
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
    print(f"\nRecall@{LIMIT}: {recall:.0%} ({hits}/{len(cases)})   MRR: {mrr:.2f}   [{by_corpus}]")
    if recall < RECALL_TARGET:
        print(f"Below the {RECALL_TARGET:.0%} Recall@{LIMIT} target - retrieval needs attention.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
