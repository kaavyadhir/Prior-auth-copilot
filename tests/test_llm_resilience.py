"""Retry and model-fallback behaviour for upstream calls.

Upstream failure handling is the code most likely to be wrong and least likely
to be exercised by hand, so it is tested against a fake provider with a fake
clock - no network, no waiting.
"""
import pytest

from app.upstream import UpstreamUnavailable, run_with_fallback


class ApiError(Exception):
    def __init__(self, code):
        super().__init__(f"status {code}")
        self.code = code


class FakeProvider:
    """Replays a scripted sequence of outcomes per model."""

    def __init__(self, script: dict[str, list]):
        self.script = {k: list(v) for k, v in script.items()}
        self.calls: list[str] = []

    def __call__(self, model):
        self.calls.append(model)
        outcome = self.script[model].pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def run(provider, models, max_attempts=3, sleeps=None):
    return run_with_fallback(
        provider, models=models, max_attempts=max_attempts, backoff_seconds=1.0,
        sleep=(sleeps.append if sleeps is not None else lambda _: None),
    )


def test_returns_first_success_without_retrying():
    p = FakeProvider({"a": ["ok"]})
    assert run(p, ["a"]) == ("ok", "a")
    assert p.calls == ["a"]


def test_transient_error_is_retried_on_the_same_model():
    p = FakeProvider({"a": [ApiError(503), ApiError(503), "ok"]})
    assert run(p, ["a"]) == ("ok", "a")
    assert p.calls == ["a", "a", "a"]


def test_backoff_is_exponential():
    p = FakeProvider({"a": [ApiError(503), ApiError(503), "ok"]})
    sleeps = []
    run(p, ["a"], sleeps=sleeps)
    assert sleeps == [1.0, 2.0]


def test_exhausted_retries_fall_through_to_the_next_model():
    p = FakeProvider({"a": [ApiError(503)] * 3, "b": ["ok"]})
    assert run(p, ["a", "b"]) == ("ok", "b")
    assert p.calls == ["a", "a", "a", "b"]


def test_retired_model_skips_immediately_without_burning_retries():
    """404 means the model is gone. Retrying it wastes time and quota."""
    p = FakeProvider({"a": [ApiError(404)], "b": ["ok"]})
    sleeps = []
    assert run(p, ["a", "b"], sleeps=sleeps) == ("ok", "b")
    assert p.calls == ["a", "b"]
    assert sleeps == []


@pytest.mark.parametrize("status", [400, 401, 403])
def test_client_errors_raise_immediately_and_are_not_masked_by_fallback(status):
    """A bad request or bad key is a bug. Failing over would hide it."""
    p = FakeProvider({"a": [ApiError(status)], "b": ["ok"]})
    with pytest.raises(ApiError):
        run(p, ["a", "b"])
    assert p.calls == ["a"]


def test_all_models_failing_raises_upstream_unavailable():
    p = FakeProvider({"a": [ApiError(503)] * 3, "b": [ApiError(503)] * 3})
    with pytest.raises(UpstreamUnavailable) as excinfo:
        run(p, ["a", "b"])
    assert "a" in str(excinfo.value) and "b" in str(excinfo.value)


def test_model_chain_deduplicates_and_preserves_order():
    from app.config import Settings

    s = Settings(llm_model="x", llm_fallback_models="y, x ,z,")
    assert s.model_chain == ["x", "y", "z"]


def test_embedding_chain_is_separate_from_the_generation_chain():
    from app.config import Settings

    s = Settings(llm_model="gen", llm_fallback_models="gen2",
                 embedding_model="emb", embedding_fallback_models="emb2")
    assert s.model_chain == ["gen", "gen2"]
    assert s.embedding_model_chain == ["emb", "emb2"]
