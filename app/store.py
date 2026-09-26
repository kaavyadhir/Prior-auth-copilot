"""Vector storage.

Two backends behind one interface:

* `NumpyStore`   - zero-setup, file-backed. Fine at this corpus size (a few
                   thousand policy passages) and keeps local dev dependency-free.
* `PgVectorStore` - Postgres + pgvector, used by docker-compose.

Keeping storage behind an interface means the retrieval and decision layers
never change when the backend does.
"""
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class Passage:
    text: str
    source_document: str
    page: int | None


class VectorStore(ABC):
    @abstractmethod
    def add(self, passages: list[Passage], vectors: np.ndarray) -> None: ...

    @abstractmethod
    def search(self, vector: np.ndarray, k: int) -> list[tuple[Passage, float]]: ...

    @abstractmethod
    def count(self) -> int: ...


class NumpyStore(VectorStore):
    """Cosine similarity over an in-memory matrix, persisted to disk."""

    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        self._vectors: np.ndarray | None = None
        self._passages: list[Passage] = []
        self._load()

    @property
    def _vec_path(self) -> str:
        return os.path.join(self.index_dir, "vectors.npy")

    @property
    def _meta_path(self) -> str:
        return os.path.join(self.index_dir, "passages.json")

    def _load(self) -> None:
        if os.path.exists(self._vec_path) and os.path.exists(self._meta_path):
            self._vectors = np.load(self._vec_path)
            with open(self._meta_path, encoding="utf-8") as fh:
                self._passages = [Passage(**row) for row in json.load(fh)]

    def add(self, passages: list[Passage], vectors: np.ndarray) -> None:
        vectors = _normalize(vectors)
        if self._vectors is None:
            self._vectors = vectors
        else:
            self._vectors = np.vstack([self._vectors, vectors])
        self._passages.extend(passages)
        self._persist()

    def _persist(self) -> None:
        os.makedirs(self.index_dir, exist_ok=True)
        np.save(self._vec_path, self._vectors)
        with open(self._meta_path, "w", encoding="utf-8") as fh:
            json.dump([asdict(p) for p in self._passages], fh, ensure_ascii=False)

    def search(self, vector: np.ndarray, k: int) -> list[tuple[Passage, float]]:
        if self._vectors is None or len(self._passages) == 0:
            return []
        query = _normalize(vector.reshape(1, -1))[0]
        scores = self._vectors @ query          # cosine, both sides unit-norm
        top = np.argsort(-scores)[:k]
        return [(self._passages[i], float(scores[i])) for i in top]

    def count(self) -> int:
        return len(self._passages)


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    return matrix / norms
