import logging
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

if __package__:
    from .logging_config import (
        configure_logging,
        log_failure_event,
        log_retrieval_event,
        log_security_event,
    )
    from .observability import (
        RequestContext,
        config_snapshot_event,
        failure_event,
        retrieval_trace_event,
        security_audit_event,
    )
    from .scifact import (
        DEFAULT_TOP_K,
        MAX_TOP_K,
        peek_scifact_retriever,
        scifact_status,
        start_scifact_retriever_initialization,
    )
else:
    from logging_config import (
        configure_logging,
        log_failure_event,
        log_retrieval_event,
        log_security_event,
    )
    from observability import (
        RequestContext,
        config_snapshot_event,
        failure_event,
        retrieval_trace_event,
        security_audit_event,
    )
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
    log_security_event(config_snapshot_event())
    logger.info("Starting SciFact Retrieval API.")
    start_scifact_retriever_initialization()


@app.get("/health")
def health():
    start_scifact_retriever_initialization()
    return scifact_status()


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(request: RetrieveRequest):
    request_context = RequestContext.start(request.query_id)
    start_scifact_retriever_initialization()
    retriever = peek_scifact_retriever()
    if retriever is None or not retriever.ready:
        error = (retriever.error if retriever else None) or "SciFact retriever is still initializing."
        _log_retrieval_failure(
            request_context,
            request,
            event_type="retrieval.not_ready",
            failure_type="failure.retriever_not_ready",
            message="SciFact retriever is not ready",
            failure_message="Retriever was unavailable during request",
            mode="not_ready",
            error=error,
        )
        raise HTTPException(status_code=503, detail=error)

    try:
        contexts = retriever.retrieve(request.question, request.top_k)
    except Exception as exc:
        logger.exception("Retrieval failed for query_id=%s", request.query_id)
        _log_retrieval_failure(
            request_context,
            request,
            event_type="retrieval.exception",
            failure_type="failure.exception",
            message="Retrieval failed",
            failure_message="Retrieval raised an exception",
            mode=_retriever_mode(retriever),
            error=str(exc),
        )
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    _log_retrieval_success(request_context, request, retriever, contexts)
    if not contexts:
        _log_empty_result(request_context, retriever)
        raise HTTPException(status_code=503, detail="No relevant SciFact contexts found.")

    _log_security_allow(request_context)

    return {"query_id": request.query_id, "contexts": contexts}


def _log_retrieval_failure(
    request_context: RequestContext,
    request: RetrieveRequest,
    *,
    event_type: str,
    failure_type: str,
    message: str,
    failure_message: str,
    mode: str,
    error: str,
) -> None:
    log_retrieval_event(
        retrieval_trace_event(
            request_context,
            event_type=event_type,
            message=message,
            top_k=request.top_k,
            mode=mode,
            status="error",
            query_text=request.question,
            error=error,
        )
    )
    log_failure_event(
        failure_event(
            request_context,
            event_type=failure_type,
            message=failure_message,
            error=error,
            mode=mode,
        )
    )


def _log_retrieval_success(
    request_context: RequestContext,
    request: RetrieveRequest,
    retriever: Any,
    contexts: list[dict[str, Any]],
) -> None:
    log_retrieval_event(
        retrieval_trace_event(
            request_context,
            event_type="retrieval.context_selected" if contexts else "retrieval.empty_result",
            message="Retrieval completed" if contexts else "Retrieval returned no contexts",
            top_k=request.top_k,
            mode=_retriever_mode(retriever),
            status="ok" if contexts else "empty",
            query_text=request.question,
            contexts=contexts,
        )
    )


def _log_empty_result(request_context: RequestContext, retriever: Any) -> None:
    log_failure_event(
        failure_event(
            request_context,
            event_type="failure.empty_result",
            message="Retrieval returned no contexts",
            error="No relevant SciFact contexts found.",
            mode=_retriever_mode(retriever),
        )
    )


def _log_security_allow(request_context: RequestContext) -> None:
    if not request_context.security_policy.audit_enabled:
        return

    log_security_event(
        security_audit_event(
            request_context,
            event_type="security.policy_applied",
            message="Security policy applied to retrieval request",
            decision="allow",
            reason="SciFact corpus uses public retrieval policy",
        )
    )


def _retriever_mode(retriever: Any) -> str:
    return getattr(retriever, "mode", "unknown")
