"""Offline tests for the lesson-text expansion tool (grader/expand_chunks.py).

Run:  .venv\\Scripts\\python tests\\test_expand_chunks.py

Pure helpers only: excerpt grounding, lexical dedupe, child ids, chunk
assembly, parent selection on a synthetic private bank. No LLM, no
private checkout needed.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grader import expand_chunks as ex  # noqa: E402

LESSON = ("Rerankers re-score the top candidates with a cross-encoder. Rules of thumb "
          "from the deck: - rerank at most 50 candidates; - budget 30-80 ms per query. "
          "Skip the reranker when first-stage recall@20 is already above 95%.")


def test_excerpt_grounded():
    assert ex.excerpt_grounded("Skip the reranker when first-stage recall@20 is already above 95%.", LESSON)
    # Markdown / whitespace differences tolerated, invented sentences not.
    assert ex.excerpt_grounded("rerank at most **50** candidates; budget 30-80 ms per query", LESSON)
    assert not ex.excerpt_grounded("Rerankers always double end-to-end latency in production.", LESSON)
    assert not ex.excerpt_grounded("top", LESSON)


def test_lexical_dup():
    bank = [("rag_07", "When does a reranking stage justify its latency cost?"),
            ("rag_08", "How do you chunk PDFs for retrieval?")]
    assert ex.lexical_dup("When does reranking justify its latency cost?", bank) == "rag_07"
    assert ex.lexical_dup("How many candidates should a reranker score per query?", bank) is None
    assert ex.lexical_dup("Why rerank?", bank) is None  # too short to judge


def test_child_id_and_chunk():
    parent = {"id": "03_rag_and_retrieval__07_reranking", "content": LESSON,
              "interview": {"question": "What does a reranker do?", "key_points": ["cross-encoder"]},
              "metadata": {"source_pdf": "deck.pdf", "class_no": 3, "module": "RAG & Retrieval",
                           "topic": "reranking", "tags": ["rag"], "difficulty": "intermediate",
                           "slide_refs": [12]}}
    row = {"parent_id": parent["id"], "n": 2, "question": "How many candidates should a reranker score?",
           "excerpt": "rerank at most 50 candidates; budget 30-80 ms per query", "claim": "cap at 50",
           "seed_topic": "the reranking stage of a RAG pipeline", "difficulty": "intermediate"}
    row["id"] = ex.child_id(row)
    assert row["id"] == "03_rag_and_retrieval__07_reranking__x02_how_many_candidates_should"
    rubric = {"question": row["question"], "topic": "reranker budget", "tags": ["reranking", "latency"],
              "difficulty": "intermediate", "round": "technical", "model_answer": "...",
              "key_points": ["a", "b", "c"], "common_mistakes": ["d"], "followups": ["e"]}
    chunk = ex.build_chunk(row, parent, rubric)
    assert chunk["content"] == row["excerpt"]
    meta = chunk["metadata"]
    assert meta["expanded_from"] == parent["id"] and meta["origin"] == "expand"
    assert meta["source_pdf"] == "deck.pdf" and meta["slide_refs"] == [12] and meta["class_no"] == 3
    assert meta["tags"] == ["latency", "rag", "reranking"]
    assert meta["review"] == {"status": "unreviewed"} and meta["seed_topic"] == row["seed_topic"]
    assert meta["module"] == "RAG & Retrieval" and meta["topic"] == "reranker budget"


def test_parents_for_skips_children_and_filters_seeds():
    with tempfile.TemporaryDirectory() as tmp:
        bank_dir = Path(tmp) / "rag_ai"
        bank_dir.mkdir()
        rows = [
            {"id": "p1", "content": LESSON, "interview": {"question": "q1", "key_points": []},
             "metadata": {"module": "RAG"}},
            {"id": "p2", "content": "Tokens are the atomic unit an LLM reads.",
             "interview": {"question": "q2", "key_points": []}, "metadata": {"module": "LLM"}},
            {"id": "p1__x01_child", "content": "x", "interview": {"question": "c", "key_points": []},
             "metadata": {"module": "RAG", "expanded_from": "p1"}},
        ]
        (bank_dir / "all_chunks.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        old = ex.PRIVATE_DIR
        ex.PRIVATE_DIR = Path(tmp)
        try:
            # p1 already has a child -> skipped; children never expand.
            assert [p["id"] for p in ex.parents_for("ai")] == ["p2"]
            (bank_dir / "all_chunks.jsonl").write_text(
                "\n".join(json.dumps(r) for r in rows[:2]) + "\n", encoding="utf-8")
            assert [p["id"] for p in ex.parents_for("ai")] == ["p1", "p2"]
            assert [p["id"] for p in ex.parents_for("ai", seed_only=True)] == ["p1"]
            assert [p["id"] for p in ex.parents_for("ai", ids=["p2"])] == ["p2"]
            assert len(ex.parents_for("ai", limit=1)) == 1
        finally:
            ex.PRIVATE_DIR = old


def test_apply_decisions_undecided_means_drop():
    with tempfile.TemporaryDirectory() as tmp:
        scratch = Path(tmp)
        rows = [
            {"id": "p1__x01_a", "verdict": "keep", "reason": "", "excerpt": "x y z", "question": "a?"},
            {"id": "p1__x02_b", "verdict": "keep", "reason": "", "excerpt": "x y z", "question": "b?"},
            {"id": "p1__x03_c", "verdict": "keep", "reason": "", "excerpt": "x y z", "question": "c?"},
            {"verdict": "drop", "reason": "excerpt not found in the lesson text", "excerpt": "q",
             "question": "d?"},
        ]
        (scratch / "expand_ai_proposals.json").write_text(
            json.dumps({"bank": "ai", "rows": rows}), encoding="utf-8")
        (scratch / "d.json").write_text(json.dumps({"bank": "ai", "decided": [
            {"id": "p1__x01_a", "v": "keep", "note": "good"},
            {"id": "p1__x02_b", "v": "drop", "note": "trivia"}]}), encoding="utf-8")
        old = ex.PROPOSALS_DIR
        ex.PROPOSALS_DIR = scratch
        try:
            ex.apply_decisions("ai", scratch / "d.json")
            data = json.loads((scratch / "expand_ai_proposals.json").read_text(encoding="utf-8"))
            verdicts = {r.get("id", "auto"): (r["verdict"], r["reason"]) for r in data["rows"]}
            assert verdicts["p1__x01_a"] == ("keep", "author: keep - good")
            assert verdicts["p1__x02_b"] == ("drop", "author: drop - trivia")
            assert verdicts["p1__x03_c"] == ("drop", "author: undecided")
            # Automatic drops are left alone.
            assert verdicts["auto"] == ("drop", "excerpt not found in the lesson text")
            (scratch / "wrong.json").write_text(json.dumps({"bank": "ml", "decided": []}), encoding="utf-8")
            try:
                ex.apply_decisions("ai", scratch / "wrong.json")
                raise AssertionError("bank mismatch not caught")
            except SystemExit:
                pass
        finally:
            ex.PROPOSALS_DIR = old


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all expand_chunks tests passed")
