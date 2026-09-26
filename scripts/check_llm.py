"""Diagnose the Gemini integration one layer at a time.

The service deliberately swallows upstream errors and escalates, which is right
for production and unhelpful for debugging. This walks the same path in
isolation and reports exactly which step fails.
"""
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import ConfigError, require_api_key, settings  # noqa: E402
from app.schemas import CriteriaEvaluation, PriorAuthRequest   # noqa: E402
from app.upstream import UpstreamUnavailable, run_with_fallback  # noqa: E402

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


def step(n: int, label: str) -> None:
    print(f"\n[{n}] {label}")


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
        print("    FAILED"); traceback.print_exc(); return 1
    names = [m.name for m in models]
    noise = ("tts", "image", "aqa", "gemma", "lyria", "robotics", "transcribe")
    for n in [x for x in names if not any(w in x for w in noise)]:
        mark = ""
        if n.endswith(settings.llm_model): mark = "  <-- LLM_MODEL"
        if n.endswith(settings.embedding_model): mark = "  <-- EMBEDDING_MODEL"
        print(f"      {n}{mark}")
    print("    NOTE: being listed does not prove your key can call it.")

    chain = settings.model_chain
    step(3, f"Text generation over the model chain ({len(chain)} models)")
    try:
        resp, used = run_with_fallback(
            lambda model: client.models.generate_content(
                model=model, contents="Reply with the single word: ok"),
            models=chain,
            max_attempts=settings.llm_max_attempts,
            backoff_seconds=settings.llm_backoff_seconds,
        )
        print(f"    OK - answered by {used!r}: {resp.text!r}")
    except UpstreamUnavailable as exc:
        print(f"    FAILED - every model was overloaded.\n    {exc}")
        print("    503 is transient. Wait a minute and re-run.")
        return 1
    except Exception:
        print("    FAILED - the key or the request is the problem")
        traceback.print_exc(); return 1

    step(4, f"Embeddings over the chain {settings.embedding_model_chain}")
    try:
        from app.embeddings import DOCUMENT, embed

        vectors = embed(["coverage criteria for bariatric surgery", "unrelated text"], DOCUMENT)
        print(f"    OK - shape {vectors.shape}, dtype {vectors.dtype}")
        if vectors.shape[0] != 2:
            print("    WARNING: expected 2 vectors. The aggregation fallback may be firing.")
    except UpstreamUnavailable as exc:
        print(f"    FAILED - no embedding model answered.\n    {exc}"); return 1
    except Exception as exc:
        code = getattr(exc, "code", None)
        if code == 404:
            print(f"    FAILED - {settings.embedding_model!r} is not available to this key.")
            print("    Pick one from the step 2 list and set EMBEDDING_MODEL in .env.")
        else:
            print(f"    FAILED (status {code})")
        traceback.print_exc(); return 1

    step(5, "Structured output against the CriteriaEvaluation schema")
    try:
        resp, used = run_with_fallback(
            lambda model: client.models.generate_content(
                model=model,
                contents=f"POLICY:\n{EXCERPT}\n\nCLINICAL NOTE:\n{REQUEST.clinical_note}",
                config=types.GenerateContentConfig(
                    system_instruction=(
                        "List each coverage criterion in the policy and judge it against "
                        "the note as met, not_met, or unknown. Do not decide coverage."
                    ),
                    response_mime_type="application/json",
                    response_schema=CriteriaEvaluation,
                    temperature=0.0,
                ),
            ),
            models=chain,
            max_attempts=settings.llm_max_attempts,
            backoff_seconds=settings.llm_backoff_seconds,
        )
    except UpstreamUnavailable as exc:
        print(f"    INCONCLUSIVE - every model overloaded, schema never evaluated.\n    {exc}")
        return 1
    except Exception as exc:
        if getattr(exc, "code", None) == 400:
            print("    FAILED - the API rejected the response_schema itself (400).")
        traceback.print_exc(); return 1
    print(f"    OK - answered by {used!r}\n    raw: {resp.text}")

    step(6, "Running the real app.llm.evaluate_criteria")
    from app.llm import evaluate_criteria

    try:
        evaluation = evaluate_criteria(REQUEST, [EXCERPT])
    except Exception:
        print("    FAILED"); traceback.print_exc(); return 1
    print(f"    OK - policy_applies={evaluation.policy_applies}")
    for c in evaluation.criteria:
        print(f"      [{c.verdict:8s}] {c.criterion}")

    print("\nAll steps passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
