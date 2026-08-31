"""Question-bank corpora and chunk selection."""

import json
import random

from retrieval import Retriever

from coach.config import CORPUS_PATHS

# role -> {"chunks": [...], "retriever": Retriever, "modules": [...]}
KB = {}
# Chunk ids are unique across both corpora, so evaluation can look them up globally.
CHUNKS_BY_ID = {}


def load_chunks():
    for role, path in CORPUS_PATHS.items():
        if not path.exists():
            print(f"Warning: {path} not found; {role} course knowledge base disabled.")
            continue
        with path.open(encoding="utf-8") as handle:
            chunks = [json.loads(line) for line in handle if line.strip()]
        KB[role] = {
            "chunks": chunks,
            "retriever": Retriever(chunks),
            # Module names in corpus (class) order, for the topic dropdown.
            "modules": list(dict.fromkeys(chunk["metadata"]["module"] for chunk in chunks)),
        }
        CHUNKS_BY_ID.update({chunk["id"]: chunk for chunk in chunks})


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
