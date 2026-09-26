"""Diagnose the Gemini integration one layer at a time.

The service deliberately swallows LLM errors and escalates, which is right for
production and unhelpful for debugging. This walks the same path in isolation
and reports exactly which step fails.
"""
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import ConfigError, require_api_key, settings  # noqa: E402
from app.llm import LLMUnavailable, generate_with_fallback     # noqa: E402
from app.schemas import CriteriaEvaluation, PriorAuthRequest   # noqa: E402

EXCERPT = (
    "Open and laparoscopic Roux-en-Y gastric bypass (RYGBP) are covered for Medicare "
    "beneficiaries who have a body-mass index >= 35, have at least one co-morbidity "
    "related to obesity, and have been previously unsuccessful with medical treatment "
    "for obesity."
)
REQUEST = PriorAuthRequest(
    member_id="M1",
    procedure_code="43644",
    diagnosis_codes=["E66.01", "E11.9"],
    clinical_note=(
        "52F, BMI 41.2. Type 2 diabetes mellitus diagnosed 2019. Completed 14 months of "
        "physician-supervised diet and exercise with maximum 6 kg loss, regained."
    ),
)


def step(number: int, label: str) -> None:
    print(f"\n[{number}] {label}")


def main() -> int:
    step(1, "Checking GEMINI_API_KEY")
    try:
        key = require_api_key()
    except ConfigError as exc:
        print(f"    FAILED\n{exc}")
        return 1
    print(f"    OK - key present, {len(key)} chars, starts {key[:3]!r}")

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=key)

    step(2, "Listing models your key advertises")
    try:
        models = list(client.models.list())
    except Exception:
        print("    FAILED")
        traceback.print_exc()
        return 1

    names = []
    for model in models:
        actions = getattr(model, "supported_actions", None) or []
        if not actions or "generateContent" in actions:
            names.append(model.name)

    noise = ("tts", "image", "embedding", "aqa", "gemma")
    text_models = [n for n in names if not any(word in n for word in noise)]
    print(f"    {len(names)} generateContent models ({len(text_models)} text-only shown)")
    for name in text_models:
        marker = "  <-- LLM_MODEL" if name.endswith(settings.llm_model) else ""
        print(f"      {name}{marker}")
    if not any(n.endswith(settings.llm_model) for n in names):
        print(f"    WARNING: LLM_MODEL={settings.llm_model!r} is not in this list.")
        print("             Set LLM_MODEL in .env to one of the names above")
        print("             (drop the leading 'models/').")
    print("    NOTE: being listed does not prove your key can call it - step 3 does.")

    chain = settings.model_chain
    step(3, f"Plain text generation over the model chain {chain}")
    try:
        resp, used = generate_with_fallback(
            client, "Reply with the single word: ok", None,
            models=chain,
            max_attempts=settings.llm_max_attempts,
            backoff_seconds=settings.llm_backoff_seconds,
        )
        print(f"    OK - answered by {used!r}: {resp.text!r}")
        if used != settings.llm_model:
            print(f"    NOTE: primary {settings.llm_model!r} did not answer; a fallback did.")
    except LLMUnavailable as exc:
        print("    FAILED - every model in the chain was overloaded or unavailable.")
        print(f"    {exc}")
        print("    503 is transient. Wait a minute and re-run, or add another model")
        print("    to LLM_FALLBACK_MODELS in .env.")
        return 1
    except Exception:
        print("    FAILED - the key or the request is the problem")
        traceback.print_exc()
        return 1

    step(4, "Structured output against the CriteriaEvaluation schema")
    try:
        resp, used = generate_with_fallback(
            client,
            f"POLICY:\n{EXCERPT}\n\nCLINICAL NOTE:\n{REQUEST.clinical_note}",
            types.GenerateContentConfig(
                system_instruction=(
                    "List each coverage criterion in the policy and judge it against the "
                    "note as met, not_met, or unknown. Do not decide coverage."
                ),
                response_mime_type="application/json",
                response_schema=CriteriaEvaluation,
                temperature=0.0,
            ),
            models=chain,
            max_attempts=settings.llm_max_attempts,
            backoff_seconds=settings.llm_backoff_seconds,
        )
    except LLMUnavailable as exc:
        print("    INCONCLUSIVE - every model was overloaded, so the schema was")
        print("    never actually evaluated. This is not a schema problem.")
        print(f"    {exc}")
        return 1
    except Exception as exc:
        status = getattr(exc, "code", None)
        if status == 400:
            print("    FAILED - the API rejected the response_schema itself (400).")
            print("    Gemini structured output does not accept every JSON Schema;")
            print("    default values and some field metadata are common causes.")
        else:
            print(f"    FAILED - unexpected error (status {status})")
        traceback.print_exc()
        return 1
    print(f"    OK - answered by {used!r}")
    print(f"    raw: {resp.text}")

    step(5, "Running the real app.llm.evaluate_criteria")
    from app.llm import evaluate_criteria

    try:
        evaluation = evaluate_criteria(REQUEST, [EXCERPT])
    except Exception:
        print("    FAILED")
        traceback.print_exc()
        return 1
    print(f"    OK - policy_applies={evaluation.policy_applies}")
    for criterion in evaluation.criteria:
        print(f"      [{criterion.verdict:8s}] {criterion.criterion}")

    print("\nAll steps passed. The API should now return approve for this case.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
