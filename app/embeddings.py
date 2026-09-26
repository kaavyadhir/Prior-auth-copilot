"""Sentence-transformer embeddings.

Loaded once at import-time cost and reused - model loading, not inference, is
the expensive part of a request path like this.
"""
from __future__ import annotations

import numpy as np

from app.config import settings

_model = None


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(settings.embedding_model)
    return _model


def embed(texts: list[str]) -> np.ndarray:
    return np.asarray(get_model().encode(texts, show_progress_bar=False))
