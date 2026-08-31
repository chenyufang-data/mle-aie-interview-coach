# Backend: Python API only (question selection, retrieval, grading, logging).
# Build from the repo root: docker build -f docker/backend.Dockerfile .
FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Runtime files only — training scripts, datasets, and gold labels stay out.
COPY server.py retrieval.py ./
COPY coach/ coach/
COPY grader/__init__.py grader/features.py grader/model.joblib grader/
COPY rag_ml/all_chunks.jsonl rag_ml/
COPY rag_ai/all_chunks.jsonl rag_ai/
# The backend can also serve the static frontend, so this image works standalone;
# behind the nginx frontend service these files are simply never requested.
COPY public/ public/

ENV HOST=0.0.0.0 \
    PYTHONUNBUFFERED=1

EXPOSE 8000
CMD ["python", "server.py"]
