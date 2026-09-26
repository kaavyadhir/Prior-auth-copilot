FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# CPU-only torch, installed first so pip treats the dependency as satisfied.
# sentence-transformers pulls in the default CUDA build otherwise, which adds
# roughly 2 GB of GPU libraries to an image that will never see a GPU.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY scripts/ ./scripts/
COPY data/policies/ ./data/policies/

# Bake the embedding model into the image so the first request is not a cold
# model download.
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Build the policy index at build time so the container starts ready to serve.
RUN python scripts/ingest_policies.py

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
