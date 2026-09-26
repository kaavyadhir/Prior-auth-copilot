"""Retry and model-fallback behaviour.

Upstream failure handling is the code most likely to be wrong and least likely
to be exercised by hand, so it is tested against a fake client with a fake
clock - no network, no waiting.
"""
import pytest

from app.llm import LLMUnavailable, generate_with_fallback


class ApiError(Exception):
    def __init__(self, code):
        super().__init__(f"status {code}")
        self.code = code


class FakeClient:
    """Replays a scripted sequence of outcomes per model."""

    def __init__(self, script: dict[str, list]):
        self.script = {k: list(v) for k, v in script.items()}
        self.calls: list[str] = []
        self.models = self

    def generate_content(self, model, contents, config):
        self.calls.append(model)
        outcome = self.script[model].pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def run(client, models, max_attempts=3, sleeps=None):
    return generate_with_fallback(
        client, "prompt", None,
        models=models, max_attempts=max_attempts, backoff_seconds=1.0,
        sleep=(sleeps.append if sleeps is not None else lambda _: None),
    )


def test_returns_first_success_without_retrying():
    client = FakeClient({"a": ["ok"]})
    response, model = run(client, ["a"])
    assert (response, model) == ("ok", "a")
    assert client.calls == ["a"]


def test_transient_error_is_retried_on_the_same_model():
    client = FakeClient({"a": [ApiError(503), ApiError(503), "ok"]})
    sleeps = []
    response, model = run(client, ["a"], sleeps=sleeps)
    assert (response, model) == ("ok", "a")
    assert client.calls == ["a", "a", "a"]


def test_backoff_is_exponential():
    client = FakeClient({"a": [ApiError(503), ApiError(503), "ok"]})
    sleeps = []
    run(client, ["a"], sleeps=sleeps)
    assert sleeps == [1.0, 2.0]


def test_exhausted_retries_fall_through_to_the_next_model():
    client = FakeClient({"a": [ApiError(503)] * 3, "b": ["ok"]})
    response, model = run(client, ["a", "b"])
    assert (response, model) == ("ok", "b")
    assert client.calls == ["a", "a", "a", "b"]


def test_retired_model_skips_immediately_without_burning_retries():
    """404 means the model is gone. Retrying it wastes time and budget."""
    client = FakeClient({"a": [ApiError(404)], "b": ["ok"]})
    sleeps = []
    response, model = run(client, ["a", "b"], sleeps=sleeps)
    assert (response, model) == ("ok", "b")
    assert client.calls == ["a", "b"]
    assert sleeps == []


@pytest.mark.parametrize("status", [400, 401, 403])
def test_client_errors_raise_immediately_and_are_not_masked_by_fallback(status):
    """A bad request or bad key is a bug. Failing over would hide it."""
    client = FakeClient({"a": [ApiError(status)], "b": ["ok"]})
    with pytest.raises(ApiError):
        run(client, ["a", "b"])
    assert client.calls == ["a"]


def test_all_models_failing_raises_llm_unavailable():
    client = FakeClient({"a": [ApiError(503)] * 3, "b": [ApiError(503)] * 3})
    with pytest.raises(LLMUnavailable) as excinfo:
        run(client, ["a", "b"])
    assert "a" in str(excinfo.value) and "b" in str(excinfo.value)


def test_model_chain_deduplicates_and_preserves_order():
    from app.config import Settings

    settings = Settings(llm_model="x", llm_fallback_models="y, x ,z,")
    assert settings.model_chain == ["x", "y", "z"]
