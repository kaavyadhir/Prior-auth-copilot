"""Unit tests for the decision rules.

These run without an API key and without a model: the rule engine is a pure
function, which is exactly why it was separated from the LLM call.
"""
import pytest

from app.decision import decide
from app.schemas import (
    Citation,
    CriteriaEvaluation,
    CriterionVerdict,
    Outcome,
    PriorAuthRequest,
)

MIN_SIM = 0.35


def req() -> PriorAuthRequest:
    return PriorAuthRequest(
        member_id="M1",
        procedure_code="70553",
        diagnosis_codes=["G43.909"],
        clinical_note="Chronic migraine, failed topiramate and propranolol.",
    )


def cite(sim: float) -> Citation:
    return Citation(source_document="policy.pdf", page=3, text="...", similarity=sim)


def ev(*verdicts, applies=True) -> CriteriaEvaluation:
    return CriteriaEvaluation(
        policy_applies=applies,
        criteria=[
            CriterionVerdict(criterion=f"c{i}", verdict=v)
            for i, v in enumerate(verdicts)
        ],
    )


def test_all_criteria_met_approves():
    d = decide(req(), [cite(0.8)], ev("met", "met"), MIN_SIM)
    assert d.outcome is Outcome.APPROVE
    assert d.escalation_reason is None


def test_any_unmet_criterion_denies():
    d = decide(req(), [cite(0.8)], ev("met", "not_met"), MIN_SIM)
    assert d.outcome is Outcome.DENY


def test_unknown_criterion_escalates_instead_of_guessing():
    d = decide(req(), [cite(0.8)], ev("met", "unknown"), MIN_SIM)
    assert d.outcome is Outcome.ROUTE_TO_HUMAN
    assert d.escalation_reason == "incomplete_documentation"


def test_documented_failure_outranks_missing_information():
    """An explicit not_met is a denial even when other criteria are unknown."""
    d = decide(req(), [cite(0.8)], ev("not_met", "unknown"), MIN_SIM)
    assert d.outcome is Outcome.DENY


def test_weak_retrieval_never_reaches_a_decision():
    d = decide(req(), [cite(0.10)], ev("met"), MIN_SIM)
    assert d.outcome is Outcome.ROUTE_TO_HUMAN
    assert d.escalation_reason == "insufficient_policy_match"


def test_no_citations_escalates():
    d = decide(req(), [], None, MIN_SIM)
    assert d.outcome is Outcome.ROUTE_TO_HUMAN
    assert d.escalation_reason == "insufficient_policy_match"


def test_llm_failure_escalates_rather_than_defaulting():
    d = decide(req(), [cite(0.9)], None, MIN_SIM)
    assert d.outcome is Outcome.ROUTE_TO_HUMAN
    assert d.escalation_reason == "evaluation_unavailable"


def test_inapplicable_policy_escalates():
    d = decide(req(), [cite(0.9)], ev("met", applies=False), MIN_SIM)
    assert d.outcome is Outcome.ROUTE_TO_HUMAN
    assert d.escalation_reason == "policy_not_applicable"


def test_empty_criteria_escalates():
    d = decide(req(), [cite(0.9)], ev(), MIN_SIM)
    assert d.outcome is Outcome.ROUTE_TO_HUMAN
    assert d.escalation_reason == "no_criteria_extracted"


@pytest.mark.parametrize(
    "verdicts,expected",
    [(("met", "met"), 0.8), (("met", "unknown"), 0.4)],
)
def test_confidence_scales_with_documentation_coverage(verdicts, expected):
    """Confidence = retrieval strength x fraction of criteria actually resolved."""
    d = decide(req(), [cite(0.8)], ev(*verdicts), MIN_SIM)
    assert d.confidence == pytest.approx(expected, abs=1e-3)
