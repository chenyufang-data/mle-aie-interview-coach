# rag_ai — AIE question bank (public edition)

91 interview chunks over 6 modules of LLM and agent engineering (LLM
foundations and prompting, LLM APIs and prompting in code, RAG and
retrieval, AI workflows and agents, the agent loop and context engineering,
harness engineering and evaluation). Serves the **AIE** track.

Same schema as [rag_ml](../rag_ml/README.md): each chunk carries an
`interview` block (question, model answer, key points, common mistakes,
follow-ups) and retrieval `metadata` (module, topic, tags, difficulty).
`all_chunks.jsonl` is the only file the app reads; ids are unique across
both banks.

This is the public edition — interview and retrieval fields only. The
complete edition with each chunk's course-derived `content`, source
references, per-module files, and builder scripts lives in a private
repository; `tools/strip_chunks.py` regenerates this edition from it.
