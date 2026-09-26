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
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # If the best-matching policy passage scores below this, we do not let the
    # model reason at all - we route the case to a human reviewer.
    min_retrieval_similarity: float = 0.35
    top_k: int = 5

    index_dir: str = "data/index"
    policy_dir: str = "data/policies"


    @property
    def model_chain(self) -> list[str]:
        """Primary model first, then declared fallbacks, de-duplicated."""
        chain = [self.llm_model] + [
            name.strip() for name in self.llm_fallback_models.split(",") if name.strip()
        ]
        seen: set[str] = set()
        return [m for m in chain if not (m in seen or seen.add(m))]


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
