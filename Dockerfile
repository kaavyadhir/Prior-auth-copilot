FROM python:3.11-slim

# Run as a non-root user. Hugging Face Spaces executes containers as UID 1000,
# and a service running as root is poor practice on any host.
RUN useradd --create-home --uid 1000 user

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface

USER user
WORKDIR /home/user/app

# CPU-only torch, installed first so pip treats the dependency as satisfied.
# sentence-transformers pulls the default CUDA build otherwise, adding roughly
# 2 GB of GPU libraries to an image that will never see a GPU.
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user app/ ./app/
COPY --chown=user scripts/ ./scripts/
COPY --chown=user data/policies/ ./data/policies/

# Bake the embedding model into the image so the first request is not a cold
# model download.
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Build the policy index at build time so the container starts ready to serve.
RUN python scripts/ingest_policies.py

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
