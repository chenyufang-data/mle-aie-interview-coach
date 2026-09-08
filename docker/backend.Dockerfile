# Backend: Python API only (question selection, retrieval, grading, logging).
# Build from the repo root: docker build -f docker/backend.Dockerfile .
FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Runtime files only - training scripts, datasets, and gold labels stay out.
# Top-level modules the runtime imports: retrieval (BM25), retrieval_dense
# (hybrid BM25 + bge-small, coach/kb.py), resume_parser (/api/mock/parse_file).
COPY server.py retrieval.py retrieval_dense.py resume_parser.py ./
COPY coach/ coach/
# grader/: the distilled grader artifact plus the two modules the runtime
# imports (features.py so the artifact unpickles; stt_text.py for the voice
# keyterm policy) and the two inputs that policy reads (coach/voice/keyterms.py).
COPY grader/__init__.py grader/features.py grader/stt_text.py grader/model.joblib \
     grader/stt_lexicon.json grader/stt_failure_rates.json grader/
# Question banks. rag_exp is PRIVATE and is never copied (the infra plan
# mounts it). rag_lists and rag_docs are generated locally and gitignored,
# so a CI checkout has only their README: the wildcard beside it lets the
# build succeed either way (BuildKit rejects a wildcard that matches nothing
# on its own) and the server skips a missing bank with a warning at startup.
COPY rag_ml/all_chunks.jsonl rag_ml/
COPY rag_ai/all_chunks.jsonl rag_ai/
COPY rag_lists/README.md rag_lists/all_chunks.jsonl* rag_lists/
COPY rag_docs/README.md rag_docs/all_chunks.jsonl* rag_docs/
# The backend can also serve the static frontend, so this image works standalone;
# behind the nginx frontend service these files are simply never requested.
COPY public/ public/

# HOST=0.0.0.0 binds both the HTTP server and, when the optional voice stack
# (requirements-stt.txt) is installed, the voice-loop WebSocket on 8765.
# FASTEMBED_CACHE_DIR / RETRIEVAL_INDEX_DIR (retrieval_dense.py) are set by
# docker-compose.yml to the persistent volume so the one-time model download
# and the document vectors survive rebuilds.
ENV HOST=0.0.0.0 \
    PYTHONUNBUFFERED=1

EXPOSE 8000 8765
CMD ["python", "server.py"]
