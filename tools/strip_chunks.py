"""Produce the public question banks from the complete (private) ones.

Run:  .venv\\Scripts\\python tools\\strip_chunks.py <private-repo-dir>

The complete banks (private repository) carry the course-derived lesson text
in every chunk's `content` field plus source references (`source_notebook`,
`source_pdf`, `cell_refs`, `page_refs`). The public repository ships the
same chunks with those fields removed: the `interview` block (question,
model answer, key points, common mistakes, follow-ups) and the retrieval
metadata (module, topic, tags, difficulty) are all the app, the retriever,
the grader, and the tests need. Only training-data generation
(`grader/generate_answers.py`, the content_extract tier) reads `content`,
and it takes the private directory via `RAG_FULL_DIR`.

The same function is used by the git-history rewrite that created the
public repository, so public history never contained the stripped fields.
"""

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
BANKS = ("rag_ml", "rag_ai")
DROP_TOP = ("content",)
DROP_META = ("source_notebook", "source_pdf", "cell_refs", "page_refs", "source_pages",
             "slide_refs", "class_no")


def strip_chunk(chunk):
    out = {k: v for k, v in chunk.items() if k not in DROP_TOP}
    meta = out.get("metadata")
    if isinstance(meta, dict):
        out["metadata"] = {k: v for k, v in meta.items() if k not in DROP_META}
    return out


def strip_jsonl_text(text):
    """Transform a whole all_chunks.jsonl file (used by the history rewrite)."""
    lines = []
    for line in text.splitlines():
        if line.strip():
            lines.append(json.dumps(strip_chunk(json.loads(line)), ensure_ascii=False))
    return "\n".join(lines) + "\n"


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 1
    private = Path(sys.argv[1])
    for bank in BANKS:
        src = private / bank / "all_chunks.jsonl"
        dst = BASE_DIR / bank / "all_chunks.jsonl"
        if not src.exists():
            print(f"missing {src}")
            return 1
        text = strip_jsonl_text(src.read_text(encoding="utf-8"))
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(text, encoding="utf-8")
        n = text.count("\n")
        print(f"{bank}: {n} chunks -> {dst.relative_to(BASE_DIR)} "
              f"({len(text):,} bytes; dropped {DROP_TOP + DROP_META})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
