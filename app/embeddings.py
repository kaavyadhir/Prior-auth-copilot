"""Embeddings via the Gemini embedding API.

Originally this ran sentence-transformers locally. That was replaced because
the model plus PyTorch needed roughly 500 MB resident, which does not fit the
memory budget of any free container host. Calling the provider's embedding API
drops the image from ~1.5 GB to ~300 MB and runtime memory to about 150 MB.

The trade-off is real and worth stating: retrieval now needs a network round
trip and depends on the same upstream as generation. It is mitigated by the
shared retry/fallback policy in `app.upstream`, and the corpus is embedded once
at startup rather than per request.
"""
from __future__ import annotations

import numpy as np

from app.config import require_api_key, settings
from app.upstream import run_with_fallback

# Indexed passages and search queries are embedded with different task types.
# Asymmetric retrieval embeddings measurably beat using one type for both.
DOCUMENT = "RETRIEVAL_DOCUMENT"
QUERY = "RETRIEVAL_QUERY"


def _client():
    from google import genai

    return genai.Client(api_key=require_api_key())


def _embed_batch(client, model: str, texts: list[str], task_type: str) -> list[list[float]]:
    """Embed a batch, verifying we got one vector per input.

    This check is not paranoia. Some embedding models return a single
    *aggregated* vector for a multi-input request rather than one per input.
    Silently accepting that would misalign every passage with its vector and
    corrupt the index in a way that looks like bad retrieval, not a bug.
    """
    from google.genai import types

    response = client.models.embed_content(
        model=model,
        contents=texts,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=settings.embedding_dimensions,
        ),
    )
    vectors = [list(e.values) for e in response.embeddings]
    if len(vectors) == len(texts):
        return vectors

    # The model aggregated. Fall back to one request per text.
    return [
        list(
            client.models.embed_content(
                model=model,
                contents=text,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=settings.embedding_dimensions,
                ),
            ).embeddings[0].values
        )
        for text in texts
    ]


def embed(texts: list[str], task_type: str = DOCUMENT) -> np.ndarray:
    if not texts:
        return np.zeros((0, settings.embedding_dimensions), dtype=np.float32)

    client = _client()
    size = settings.embedding_batch_size
    vectors: list[list[float]] = []

    for start in range(0, len(texts), size):
        batch = texts[start : start + size]
        result, _model = run_with_fallback(
            lambda model, b=batch: _embed_batch(client, model, b, task_type),
            models=settings.embedding_model_chain,
            max_attempts=settings.llm_max_attempts,
            backoff_seconds=settings.llm_backoff_seconds,
        )
        vectors.extend(result)

    return np.asarray(vectors, dtype=np.float32)
