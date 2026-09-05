"""Question-bank corpora and chunk selection."""

import json
import random

from retrieval import Retriever

from coach import config
from coach.config import CORPUS_PATHS

# role -> {"chunks": [...], "retriever": <serves practice questions>,
#          "bm25": Retriever, "modules": [...]}
# "retriever" is the hybrid BM25 + dense retriever when its stack is
# available (config.RETRIEVAL_BACKEND) and the BM25 Retriever otherwise.
# "bm25" is always the BM25 Retriever: the mock's rubric grounding reads
# it (coach/mock/planning.py) because its threshold is in BM25 score units
# and rule R2 of the retrieval plan has not moved it.
KB = {}
# Chunk ids are unique across both corpora, so evaluation can look them up globally.
CHUNKS_BY_ID = {}


def _hybrid_embedder():
    """The shared embedder, or None with config.RETRIEVAL_DISABLED_REASON set."""
    backend = config.RETRIEVAL_BACKEND
    if backend == "bm25":
        config.RETRIEVAL_DISABLED_REASON = "RETRIEVAL_BACKEND=bm25"
        return None
    try:
        from retrieval_dense import hybrid_availability
    except ImportError as exc:  # numpy missing: fastembed would be too
        outcome = f"missing dependency {getattr(exc, 'name', exc)!r}"
    else:
        outcome = hybrid_availability()
    if isinstance(outcome, str):
        if backend == "hybrid":
            raise SystemExit(f"RETRIEVAL_BACKEND=hybrid: cannot start - {outcome}")
        config.RETRIEVAL_DISABLED_REASON = outcome
        return None
    return outcome


def load_chunks():
    embedder = _hybrid_embedder()
    for role, path in CORPUS_PATHS.items():
        if not path.exists():
            print(f"Warning: {path} not found; {role} course knowledge base disabled.")
            continue
        with path.open(encoding="utf-8") as handle:
            chunks = [json.loads(line) for line in handle if line.strip()]
        # Author review (tools/review_bank.py) retires a chunk by flag, never
        # by deletion: ids stay stable for labels and bookmarks.
        live = [c for c in chunks
                if c["metadata"].get("review", {}).get("status") != "retire"]
        if len(live) != len(chunks):
            print(f"{role}: {len(chunks) - len(live)} retired chunk(s) skipped")
        chunks = live
        bm25 = Retriever(chunks)
        retriever = bm25
        if embedder is not None:
            from retrieval_dense import DenseRetriever, HybridRetriever, load_or_build

            vectors, _, _ = load_or_build(role, chunks, embedder)
            retriever = HybridRetriever(bm25, DenseRetriever(chunks, embedder, vectors))
        KB[role] = {
            "chunks": chunks,
            "retriever": retriever,
            "bm25": bm25,
            # Module names in corpus (class) order, for the topic dropdown.
            "modules": list(dict.fromkeys(chunk["metadata"]["module"] for chunk in chunks)),
        }
        CHUNKS_BY_ID.update({chunk["id"]: chunk for chunk in chunks})
    config.RETRIEVAL_ACTIVE = "hybrid" if embedder is not None else "bm25"


def select_chunk(role, module, level, focus, exclude_ids):
    kb = KB.get(role)
    if kb is None:
        path = CORPUS_PATHS.get(role)
        detail = f"{path.parent.name}/{path.name} missing" if path else f"unknown track {role!r}"
        raise RuntimeError(f"Course knowledge base for the {role} track is not loaded ({detail}).")

    candidates = kb["retriever"].search(
        query=focus,
        module=module,
        level=level,
        exclude_ids=exclude_ids,
        limit=5,
    )
    # Any of the top matches is a good serve; sampling keeps repeat sessions
    # from always landing on the same question.
    return random.choice(candidates)


def kb_question_payload(chunk):
    meta = chunk["metadata"]
    return {
        "question": chunk["interview"]["question"],
        "what_interviewer_is_testing": f"{meta['module']} - {meta['topic']} ({meta['difficulty']})",
        "answer_guidance": chunk["interview"]["key_points"],
        "chunk_id": chunk["id"],
        "source": "kb",
    }
