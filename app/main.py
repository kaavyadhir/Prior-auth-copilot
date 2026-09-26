"""FastAPI surface for the prior-authorization decision support service."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from app import retrieval
from app.config import settings
from app.decision import decide
from app.llm import evaluate_criteria
from app.schemas import Decision, PriorAuthRequest
from app.store import NumpyStore

logger = logging.getLogger("prior_auth")
STATIC_DIR = Path(__file__).parent / "static"
store: NumpyStore | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the index and the embedding model once, at startup.

    Doing this per-request is the classic way to turn a 200ms endpoint into a
    30s one.
    """
    global store
    store = NumpyStore(settings.index_dir)
    from app.embeddings import get_model

    get_model()
    logger.info("index loaded: %s passages", store.count())
    yield


app = FastAPI(
    title="Prior Authorization Decision Support",
    description=(
        "Evaluates prior-authorization requests against published payer coverage "
        "policies. Returns approve, deny, or an explicit escalation to a human "
        "reviewer, with the governing policy passage cited."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> str:
    """Serve the demo UI from the API itself.

    One page, no build step, no second deployment, no CORS. A separate frontend
    would add three moving parts to show one JSON response.
    """
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "indexed_passages": store.count() if store else 0}


@app.post("/v1/prior-auth/evaluate", response_model=Decision)
def evaluate(request: PriorAuthRequest) -> Decision:
    if store is None or store.count() == 0:
        raise HTTPException(
            status_code=503,
            detail="Policy index is empty. Run scripts/ingest_policies.py first.",
        )

    citations = retrieval.retrieve(request, store, settings.top_k)

    evaluation = None
    best = max((c.similarity for c in citations), default=0.0)
    # Only spend an LLM call when retrieval actually found something relevant.
    if citations and best >= settings.min_retrieval_similarity:
        try:
            evaluation = evaluate_criteria(request, [c.text for c in citations])
        except Exception:
            logger.exception("criteria evaluation failed")

    return decide(request, citations, evaluation, settings.min_retrieval_similarity)
