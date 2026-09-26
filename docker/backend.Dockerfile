# Backend: Python API only (question selection, retrieval, grading, logging).
# Build from the repo root: docker build -f docker/backend.Dockerfile .
FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt requirements-voice-cloud.txt requirements-db.txt ./
# requirements-voice-cloud.txt is the light part of the voice stack (the
# loop server + Silero VAD): with AUDIO_BACKEND=deepgram and its key in the
# server's .env the live voice mode is on; otherwise the server states why
# the loop is off and serves text only (docs/deployment.md section 6).
# requirements-db.txt is the Postgres driver for the state store: used when
# docker-compose.db.yml sets DATABASE_URL, idle otherwise (coach/store.py).
RUN pip install --no-cache-dir -r requirements.txt -r requirements-voice-cloud.txt \
    -r requirements-db.txt

# Runtime files only - experiments, datasets, and gold labels stay out.
# coach/ is the whole runtime: the retrievers, the resume parser, the
# distilled grader's features, and coach/assets/ (the grader artifact plus the
# lexicon and failure rates the voice keyterm policy reads).
COPY server.py ./
COPY coach/ coach/
# The migration tool, so an existing data/ volume can be imported into the
# Postgres store from inside the container (docker-compose.db.yml).
COPY tools/migrate_to_postgres.py tools/
# Question banks. rag_exp is PRIVATE and is never copied (the infra plan
# mounts it). rag_lists and rag_docs are generated locally and gitignored,
# so a CI checkout has only their README: the wildcard beside it lets the
# build succeed either way (BuildKit rejects a wildcard that matches nothing
# on its own) and the server skips a missing bank with a warning at startup.
COPY banks/rag_ml/all_chunks.jsonl banks/rag_ml/
COPY banks/rag_ai/all_chunks.jsonl banks/rag_ai/
COPY banks/rag_lists/README.md banks/rag_lists/all_chunks.jsonl* banks/rag_lists/
COPY banks/rag_docs/README.md banks/rag_docs/all_chunks.jsonl* banks/rag_docs/
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
