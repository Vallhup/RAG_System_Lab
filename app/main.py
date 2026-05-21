import logging
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

if __package__:
    from .logging_config import configure_logging, log_retrieval_event
    from .scifact import (
        DEFAULT_TOP_K,
        MAX_TOP_K,
        peek_scifact_retriever,
        scifact_status,
        start_scifact_retriever_initialization,
    )
else:
    from logging_config import configure_logging, log_retrieval_event
    from scifact import (
        DEFAULT_TOP_K,
        MAX_TOP_K,
        peek_scifact_retriever,
        scifact_status,
        start_scifact_retriever_initialization,
    )


app = FastAPI(title="SciFact Retrieval API", version="2.0.0")
logger = logging.getLogger("rag.app")


class RetrieveRequest(BaseModel):
    query_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)


class Context(BaseModel):
    doc_id: str
    chunk_id: str
    score: float
    text: str


class RetrieveResponse(BaseModel):
    query_id: str
    contexts: list[Context]


@app.on_event("startup")
def startup():
    configure_logging()
    logger.info("Starting SciFact Retrieval API.")
    start_scifact_retriever_initialization()


@app.get("/health")
def health():
    start_scifact_retriever_initialization()
    return scifact_status()


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(request: RetrieveRequest):
    started = time.perf_counter()
    start_scifact_retriever_initialization()
    retriever = peek_scifact_retriever()
    if retriever is None or not retriever.ready:
        log_retrieval_event(
            {
                "query_id": request.query_id,
                "top_k": request.top_k,
                "mode": "not_ready",
                "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                "status": "error",
                "error": (retriever.error if retriever else None)
                or "SciFact retriever is still initializing.",
            }
        )
        raise HTTPException(
            status_code=503,
            detail=(retriever.error if retriever else None) or "SciFact retriever is still initializing.",
        )

    try:
        contexts = retriever.retrieve(request.question, request.top_k)
    except Exception as exc:
        logger.exception("Retrieval failed for query_id=%s", request.query_id)
        log_retrieval_event(
            {
                "query_id": request.query_id,
                "top_k": request.top_k,
                "mode": getattr(retriever, "mode", "unknown"),
                "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                "status": "error",
                "error": str(exc),
            }
        )
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    latency_ms = round((time.perf_counter() - started) * 1000, 3)
    log_retrieval_event(
        {
            "query_id": request.query_id,
            "top_k": request.top_k,
            "mode": getattr(retriever, "mode", "unknown"),
            "latency_ms": latency_ms,
            "status": "ok" if contexts else "empty",
            "top_doc_ids": [context["doc_id"] for context in contexts],
            "scores": [context["score"] for context in contexts],
        }
    )
    if not contexts:
        raise HTTPException(status_code=503, detail="No relevant SciFact contexts found.")

    return {"query_id": request.query_id, "contexts": contexts}
