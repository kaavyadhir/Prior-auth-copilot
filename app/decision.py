"""The decision engine.

This module is the point of the project, so it is worth being explicit about
the design:

The language model does *judgement* (does this note satisfy this criterion?).
Deterministic code does *decision-making* (given those judgements, what happens
to this request?). Separating them buys three things:

1. Auditability - the rule that produced an outcome can be pointed at.
2. Testability  - the rules are pure functions over a small input, so they are
                  unit-tested without calling an LLM at all.
3. Safety       - no prompt, adversarial or accidental, can talk the system
                  into an approval. Approval requires every criterion to be
                  explicitly met.

The system abstains rather than guesses in four situations, all of which route
to a human reviewer.
"""
from __future__ import annotations

from app.schemas import (
    Citation,
    CriteriaEvaluation,
    Decision,
    Outcome,
    PriorAuthRequest,
)


def decide(
    request: PriorAuthRequest,
    citations: list[Citation],
    evaluation: CriteriaEvaluation | None,
    min_similarity: float,
) -> Decision:
    # (1) Nothing in the policy corpus is close enough to this request.
    best = max((c.similarity for c in citations), default=0.0)
    if not citations or best < min_similarity:
        return Decision(
            outcome=Outcome.ROUTE_TO_HUMAN,
            reason="No published policy passage matched this request closely enough to evaluate.",
            confidence=round(best, 3),
            citations=citations,
            escalation_reason="insufficient_policy_match",
        )

    # (2) Upstream evaluation failed - never fall through to a default answer.
    if evaluation is None:
        return Decision(
            outcome=Outcome.ROUTE_TO_HUMAN,
            reason="Criteria evaluation was unavailable for this request.",
            confidence=0.0,
            citations=citations,
            escalation_reason="evaluation_unavailable",
        )

    # (3) The retrieved policy does not govern this procedure.
    if not evaluation.policy_applies:
        return Decision(
            outcome=Outcome.ROUTE_TO_HUMAN,
            reason="The closest policy does not govern the requested procedure.",
            confidence=round(best, 3),
            criteria=evaluation.criteria,
            citations=citations,
            escalation_reason="policy_not_applicable",
        )

    criteria = evaluation.criteria
    if not criteria:
        return Decision(
            outcome=Outcome.ROUTE_TO_HUMAN,
            reason="No coverage criteria could be extracted from the matched policy.",
            confidence=round(best, 3),
            citations=citations,
            escalation_reason="no_criteria_extracted",
        )

    resolved = sum(1 for c in criteria if c.verdict != "unknown")
    coverage = resolved / len(criteria)
    confidence = round(best * coverage, 3)

    failed = [c for c in criteria if c.verdict == "not_met"]
    unknown = [c for c in criteria if c.verdict == "unknown"]

    # A clear, documented failure is a denial even if other criteria are unknown:
    # no amount of missing information rescues an explicitly unmet requirement.
    if failed:
        names = "; ".join(c.criterion for c in failed)
        return Decision(
            outcome=Outcome.DENY,
            reason=f"Documented criteria not satisfied: {names}",
            confidence=confidence,
            criteria=criteria,
            citations=citations,
        )

    # (4) Nothing failed, but the note is silent on something the policy requires.
    if unknown:
        names = "; ".join(c.criterion for c in unknown)
        return Decision(
            outcome=Outcome.ROUTE_TO_HUMAN,
            reason=f"Clinical documentation does not address: {names}",
            confidence=confidence,
            criteria=criteria,
            citations=citations,
            escalation_reason="incomplete_documentation",
        )

    return Decision(
        outcome=Outcome.APPROVE,
        reason="All documented coverage criteria are explicitly satisfied.",
        confidence=confidence,
        criteria=criteria,
        citations=citations,
    )
