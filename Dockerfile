FROM python:3.11-slim

# Run as a non-root user. A web service has no reason to run as root, and some
# container hosts execute images as UID 1000 regardless of what the image says.
RUN useradd --create-home --uid 1000 user

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

USER user
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user app/ ./app/
COPY --chown=user scripts/ ./scripts/
COPY --chown=user data/policies/ ./data/policies/

# COPY has to create the intermediate ./data directory, and whether --chown
# reaches a parent the builder created is builder-dependent. It did not here:
# ./data came out root-owned, so the service could not create ./data/index. The
# failure was quiet - the index lives in memory, /health reported the right
# passage count, and the whole corpus was silently re-embedded through the API
# on every restart. Set the ownership explicitly rather than relying on builder
# behaviour.
USER root
RUN mkdir -p /home/user/app/data/index \
    && chown -R user:user /home/user/app/data
USER user

# The policy index is built at startup, not here: embeddings come from the API
# and the build has no credentials. That also keeps the API key out of the image.
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
