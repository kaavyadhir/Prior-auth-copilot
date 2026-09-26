"""Application settings, loaded from environment or .env."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    gemini_api_key: str = ""
    llm_model: str = "gemini-3.8-flash"
    # Tried in order when the primary is overloaded or has been retired.
    llm_fallback_models: str = (
        "gemini-3.7-flash,gemini-3.5-flash,gemini-3-flash-preview,"
        "gemini-3.5-flash-lite,gemini-3.1-flash-lite,gemini-flash-lite-latest"
    )
    llm_max_attempts: int = 2
    llm_backoff_seconds: float = 1.0
    embedding_model: str = "gemini-embedding-001"
    # Deliberately empty. The obvious alternative, gemini-embedding-2, does not
    # accept retrieval task types and returns ONE aggregated vector for a
    # multi-input request. Falling back to it would silently build an index with
    # different semantics from the queries run against it - worse than failing,
    # because retrieval would just quietly degrade. An embedding outage instead
    # leaves the index empty, and every request escalates to a human.
    embedding_fallback_models: str = ""
    # 768 keeps the index small and is ample for a corpus this size.
    embedding_dimensions: int = 768
    embedding_batch_size: int = 16

    # If the best-matching policy passage scores below this, we do not let the
    # model reason at all - we route the case to a human reviewer.
    min_retrieval_similarity: float = 0.35
    top_k: int = 5

    index_dir: str = "data/index"
    policy_dir: str = "data/policies"


    @property
    def embedding_model_chain(self) -> list[str]:
        return _dedupe([self.embedding_model] + _split(self.embedding_fallback_models))

    @property
    def model_chain(self) -> list[str]:
        """Primary model first, then declared fallbacks, de-duplicated."""
        return _dedupe([self.llm_model] + _split(self.llm_fallback_models))


def _split(value: str) -> list[str]:
    return [name.strip() for name in value.split(",") if name.strip()]


def _dedupe(names: list[str]) -> list[str]:
    seen: set[str] = set()
    return [n for n in names if not (n in seen or seen.add(n))]


settings = Settings()


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or obviously malformed."""


def require_api_key() -> str:
    """Validate the Gemini key before any network call.

    Deliberately checks only that a key is *present* and not the template
    placeholder. Key formats change - Google moved AI Studio keys from the
    "AIza" prefix to "AQ." - so validating the shape of a credential here would
    reject valid keys the moment the provider rotates its format. The API is the
    only authority on whether a key is good; our job is to catch the far more
    common local mistake of not filling it in.
    """
    key = (settings.gemini_api_key or "").strip()
    if not key or key == "your_key_here":
        raise ConfigError(
            "GEMINI_API_KEY is not set.\n"
            "  1. Create a key at https://aistudio.google.com/apikey\n"
            "  2. Copy .env.example to .env  (.env is gitignored; .env.example is NOT)\n"
            "  3. Put the key in .env - editing .env.example has no effect at runtime"
        )
    return key
