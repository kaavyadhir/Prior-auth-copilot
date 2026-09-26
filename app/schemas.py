"""Request/response contracts for the prior-authorization service."""
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class Outcome(StrEnum):
    APPROVE = "approve"
    DENY = "deny"
    ROUTE_TO_HUMAN = "route_to_human"


class PriorAuthRequest(BaseModel):
    """What a provider submits when asking the payer to pre-approve a service."""

    member_id: str = Field(..., examples=["M10042"])
    procedure_code: str = Field(..., description="CPT/HCPCS code", examples=["70553"])
    diagnosis_codes: list[str] = Field(default_factory=list, examples=[["G43.909"]])
    clinical_note: str = Field(
        ...,
        description="Free-text clinical justification written by the provider.",
        examples=["38F with chronic migraine, failed 3 months of topiramate..."],
    )
    requested_units: int = 1


class Citation(BaseModel):
    """A verbatim passage from a published payer policy."""

    source_document: str
    page: int | None = None
    text: str
    similarity: float


class CriterionVerdict(BaseModel):
    """One coverage criterion the policy requires, and whether the note satisfies it.

    `unknown` is a first-class outcome: the model must say so when the clinical
    note simply does not address the criterion, rather than inferring an answer.
    """

    criterion: str
    verdict: Literal["met", "not_met", "unknown"]
    evidence: str = Field(
        "", description="Quote from the clinical note supporting the verdict."
    )


class CriteriaEvaluation(BaseModel):
    """Structured output we require back from the LLM."""

    criteria: list[CriterionVerdict]
    policy_applies: bool = Field(
        True,
        description="False when the retrieved policy does not govern this procedure.",
    )


class Decision(BaseModel):
    outcome: Outcome
    reason: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    criteria: list[CriterionVerdict] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    escalation_reason: str | None = None
