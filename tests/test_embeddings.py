"""Embedding batching, and the aggregation trap.

No network: a fake client replays provider behaviour, including the failure
mode where a model returns one aggregated vector instead of one per input.
"""
import numpy as np

from app import embeddings
from app.embeddings import DOCUMENT


class FakeModels:
    def __init__(self, aggregate=False, dims=4):
        self.aggregate = aggregate
        self.dims = dims
        self.requests: list[int] = []

    def embed_content(self, model, contents, config):
        texts = contents if isinstance(contents, list) else [contents]
        self.requests.append(len(texts))
        count = 1 if (self.aggregate and len(texts) > 1) else len(texts)
        return type("R", (), {
            "embeddings": [
                type("E", (), {"values": [float(i)] * self.dims})() for i in range(count)
            ]
        })()


class FakeClient:
    def __init__(self, **kw):
        self.models = FakeModels(**kw)


def patch(monkeypatch, client):
    monkeypatch.setattr(embeddings, "_client", lambda: client)
    monkeypatch.setattr(embeddings.settings, "embedding_dimensions", client.models.dims)
    monkeypatch.setattr(embeddings, "run_with_fallback",
                        lambda fn, **kw: (fn("fake-model"), "fake-model"))


def test_empty_input_returns_an_empty_matrix(monkeypatch):
    patch(monkeypatch, FakeClient())
    assert embeddings.embed([], DOCUMENT).shape == (0, 4)


def test_one_vector_per_text(monkeypatch):
    client = FakeClient()
    patch(monkeypatch, client)
    out = embeddings.embed(["a", "b", "c"], DOCUMENT)
    assert out.shape == (3, 4)


def test_aggregating_model_falls_back_to_one_request_per_text(monkeypatch):
    """Some models return a single vector for a multi-input request.

    Accepting that silently would misalign every passage with its vector and
    corrupt the index in a way that looks like poor retrieval, not a bug.
    """
    client = FakeClient(aggregate=True)
    patch(monkeypatch, client)
    out = embeddings.embed(["a", "b", "c"], DOCUMENT)

    assert out.shape == (3, 4)
    # one batched attempt, then three individual calls
    assert client.models.requests == [3, 1, 1, 1]


def test_batches_respect_the_configured_size(monkeypatch):
    client = FakeClient()
    patch(monkeypatch, client)
    monkeypatch.setattr(embeddings.settings, "embedding_batch_size", 2)
    out = embeddings.embed(["a", "b", "c", "d", "e"], DOCUMENT)

    assert out.shape == (5, 4)
    assert client.models.requests == [2, 2, 1]


def test_output_is_float32(monkeypatch):
    patch(monkeypatch, FakeClient())
    assert embeddings.embed(["a"], DOCUMENT).dtype == np.float32
