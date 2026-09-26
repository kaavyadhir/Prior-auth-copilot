"""Shared handling for a flaky upstream provider.

Both generation and embedding call the same API and fail the same ways, so the
retry policy lives here rather than being written twice.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

# Worth retrying against the SAME model - the request was fine, the service was busy.
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
# The model is gone or not granted to this key. Retrying cannot help; move on.
MODEL_UNAVAILABLE_STATUS = {404}


class UpstreamUnavailable(RuntimeError):
    """Every model in the chain failed for a reason worth retrying later."""


def status_code(exc: Exception) -> int | None:
    return getattr(exc, "code", None)


def call_with_retry(
    operation: Callable[[], Any],
    *,
    label: str,
    max_attempts: int,
    backoff_seconds: float,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """Retry one operation against one model, then give up.

    Raises the original exception so the caller can decide whether to try a
    different model or fail outright.
    """
    for attempt in range(max_attempts):
        try:
            return operation()
        except Exception as exc:
            code = status_code(exc)
            if code in RETRYABLE_STATUS and attempt < max_attempts - 1:
                delay = backoff_seconds * (2**attempt)
                logger.warning(
                    "%s returned %s; retrying in %.1fs (attempt %d/%d)",
                    label, code, delay, attempt + 1, max_attempts,
                )
                sleep(delay)
                continue
            raise
    raise AssertionError("unreachable")


def run_with_fallback(
    make_call: Callable[[str], Any],
    *,
    models: list[str],
    max_attempts: int,
    backoff_seconds: float,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[Any, str]:
    """Call the first model that answers, returning (result, model_used).

    Three failure modes, handled differently:

    * Transient (429/5xx) - the request was valid and the service was busy.
      Retry the same model with exponential backoff, then fall through.
    * Model unavailable (404) - retrying cannot help, so skip to the next model.
      Providers retire models on their own schedule and keep advertising them in
      list-models, so this is not an edge case.
    * Anything else (400, 401, 403) - a malformed request or a bad key. Raise
      immediately rather than letting fallback mask a bug.
    """
    last_error: Exception | None = None

    for model in models:
        try:
            return call_with_retry(
                lambda m=model: make_call(m),
                label=model,
                max_attempts=max_attempts,
                backoff_seconds=backoff_seconds,
                sleep=sleep,
            ), model
        except Exception as exc:
            code = status_code(exc)
            last_error = exc
            if code in RETRYABLE_STATUS or code in MODEL_UNAVAILABLE_STATUS:
                logger.warning("giving up on %s (status %s)", model, code)
                continue
            raise

    raise UpstreamUnavailable(
        f"No model in {models} could be reached. Last error: {last_error}"
    ) from last_error
