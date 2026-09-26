"""Find the policy passages that govern a given request."""
from __future__ import annotations

from app.embeddings import QUERY, embed
from app.schemas import Citation, PriorAuthRequest
from app.store import VectorStore


def build_query(request: PriorAuthRequest) -> str:
    """The query is built from codes *and* clinical text.

    Codes alone match the procedure list but miss the criteria prose; the note
    alone matches symptom language but drifts to the wrong procedure.
    """
    diagnoses = ", ".join(request.diagnosis_codes)
    return (
        f"Procedure code {request.procedure_code}. "
        f"Diagnosis codes {diagnoses}. "
        f"Coverage criteria, medical necessity, prior authorization requirements. "
        f"{request.clinical_note}"
    )


def retrieve(request: PriorAuthRequest, store: VectorStore, k: int) -> list[Citation]:
    hits = store.search(embed([build_query(request)], QUERY)[0], k)
    return [
        Citation(
            source_document=p.source_document,
            page=p.page,
            text=p.text,
            similarity=round(score, 4),
        )
        for p, score in hits
    ]
