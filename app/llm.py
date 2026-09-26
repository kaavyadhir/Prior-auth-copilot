"""Gemini client, constrained to structured output.

The model is never asked for a decision. It is asked only to (a) list the
criteria the retrieved policy imposes and (b) say, per criterion, whether the
clinical note satisfies it - with `unknown` available. The decision itself is
made by deterministic code in `app.decision`.
"""
from __future__ import annotations

import json
import logging

from app.config import require_api_key, settings
from app.schemas import CriteriaEvaluation, PriorAuthRequest
from app.upstream import run_with_fallback

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a clinical policy analyst supporting a health-plan prior-authorization \
review. You do NOT decide whether to approve or deny.

Your job:
1. From the POLICY EXCERPTS, list each distinct ELIGIBILITY CRITERION that this \
patient or request must satisfy for the procedure to be covered.
2. For each criterion, decide whether the CLINICAL NOTE satisfies it.

What counts as an eligibility criterion:
- A condition about the patient, their history, or the requested service that a \
clinical note could reasonably be expected to document.
- Examples: a threshold the patient must meet, a required diagnosis, a required \
prior treatment, a required test or its qualifying result.

What is NOT a criterion - never list these:
- Definitions of terms or clinical concepts.
- Descriptions of how a measurement or index is calculated, or rules that apply \
only when a measurement was taken in some unusual way.
- Effective dates, administrative rules, facility-certification history, or \
statements about which body decides coverage.
- Descriptions of procedures, or lists of what the policy does NOT cover.
- Anything the provider could not address by writing more in the clinical note.
- CONDITIONAL rules that only bite in a circumstance the note does not describe. \
If a sentence reads "if the measurement was taken in manner X, then Y must hold", and \
nothing in the note suggests manner X occurred, that rule does not apply to this request \
- do not list it as a criterion. Only list it if the note indicates the triggering \
circumstance is present.

Rules you must follow:
- Use ONLY the policy excerpts provided. Never rely on outside medical knowledge.
- Where the policy offers alternative qualifying paths (for example "A, OR B with \
additional findings"), treat that as ONE criterion satisfied by ANY path, not as \
several separate criteria.
- Mark a criterion "unknown" whenever the clinical note does not clearly address \
it. Do not infer, assume, or fill gaps with what is typical. "unknown" is the \
correct and expected answer for anything the note is silent on.
- "met" requires explicit support in the note. Quote that support in `evidence`.
- Set policy_applies=false if the excerpts do not actually govern this procedure.
"""


def _build_prompt(request: PriorAuthRequest, excerpts: list[str]) -> str:
    joined = "\n\n---\n\n".join(excerpts)
    return (
        f"POLICY EXCERPTS:\n{joined}\n\n"
        f"REQUESTED PROCEDURE: {request.procedure_code}\n"
        f"DIAGNOSIS CODES: {', '.join(request.diagnosis_codes) or 'none provided'}\n"
        f"REQUESTED UNITS: {request.requested_units}\n\n"
        f"CLINICAL NOTE:\n{request.clinical_note}"
    )


def evaluate_criteria(
    request: PriorAuthRequest, excerpts: list[str]
) -> CriteriaEvaluation:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=require_api_key())
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        response_mime_type="application/json",
        response_schema=CriteriaEvaluation,
        temperature=0.0,
    )
    prompt = _build_prompt(request, excerpts)

    response, model_used = run_with_fallback(
        lambda model: client.models.generate_content(
            model=model, contents=prompt, config=config
        ),
        models=settings.model_chain,
        max_attempts=settings.llm_max_attempts,
        backoff_seconds=settings.llm_backoff_seconds,
    )
    logger.info("criteria evaluated by %s", model_used)

    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, CriteriaEvaluation):
        return parsed
    return CriteriaEvaluation.model_validate(json.loads(response.text))
