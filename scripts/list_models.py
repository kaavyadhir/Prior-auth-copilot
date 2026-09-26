"""Print the models your Gemini API key can use, so LLM_MODEL can be set correctly."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import ConfigError, require_api_key  # noqa: E402


def main() -> int:
    try:
        api_key = require_api_key()
    except ConfigError as exc:
        print(f"Configuration problem:\n{exc}", file=sys.stderr)
        return 1

    from google import genai
    from google.genai import errors

    client = genai.Client(api_key=api_key)
    try:
        models = list(client.models.list())
    except errors.ClientError as exc:
        print(f"The API rejected this key: {exc}", file=sys.stderr)
        return 1

    for model in models:
        actions = getattr(model, "supported_actions", None) or []
        if not actions or "generateContent" in actions:
            print(model.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
