import time
import uuid
from dataclasses import dataclass
from typing import Any

if __package__:
    from .runtime_config import RuntimeConfig, SecurityPolicy, redact_value
else:
    from runtime_config import RuntimeConfig, SecurityPolicy, redact_value


SERVICE_NAME = "scifact-retrieval-api"
SERVICE_VERSION = "2.0.0"
EVENT_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class RequestContext:
    query_id: str
    trace_id: str
    started_at: float
    runtime_config: RuntimeConfig
    security_policy: SecurityPolicy

    @classmethod
    def start(cls, query_id: str) -> "RequestContext":
        return cls(
            query_id=query_id,
            trace_id=uuid.uuid4().hex,
            started_at=time.perf_counter(),
            runtime_config=RuntimeConfig.from_env(),
            security_policy=SecurityPolicy.from_env(),
        )

    @property
    def latency_ms(self) -> float:
        return round((time.perf_counter() - self.started_at) * 1000, 3)

    def base_event(self, event_type: str, message: str) -> dict[str, Any]:
        return {
            "schema.version": EVENT_SCHEMA_VERSION,
            "event_type": event_type,
            "message": message,
            "service.name": SERVICE_NAME,
            "service.version": SERVICE_VERSION,
            "trace_id": self.trace_id,
            "query_id": self.query_id,
            "config.hash": self.runtime_config.hash,
            "policy.hash": self.security_policy.hash,
        }


def retrieval_trace_event(
    context: RequestContext,
    *,
    event_type: str,
    message: str,
    top_k: int,
    mode: str,
    status: str,
    query_text: str | None = None,
    contexts: list[dict[str, Any]] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    payload = context.base_event(event_type, message)
    candidate_multiplier = context.runtime_config.retrieval_candidate_multiplier
    top_doc_ids = [item["doc_id"] for item in contexts or []]
    scores = [item["score"] for item in contexts or []]

    payload.update(
        {
            "top_k": top_k,
            "mode": mode,
            "status": status,
            "latency_ms": context.latency_ms,
            "retrieval.mode": mode,
            "retrieval.top_k": top_k,
            "retrieval.candidate_count": max(top_k, 1) * candidate_multiplier,
            "retrieval.top_doc_ids": top_doc_ids,
            "retrieval.scores": scores,
            "security.privacy_max": context.security_policy.allowed_privacy_max,
            "security.redaction_applied": context.security_policy.require_redaction,
            "security.external_api_used": False,
        }
    )

    if context.security_policy.allow_raw_query_logging and query_text is not None:
        payload["query_text"] = query_text
    elif query_text is not None:
        payload["query_hash"] = _text_hash(query_text)

    if error:
        payload["error.message"] = error
        payload["error"] = error

    return redact_value(payload, context.security_policy)


def security_audit_event(
    context: RequestContext,
    *,
    event_type: str,
    message: str,
    decision: str,
    reason: str,
) -> dict[str, Any]:
    payload = context.base_event(event_type, message)
    payload.update(
        {
            "security.policy_decision": decision,
            "security.reason": reason,
            "security.privacy_max": context.security_policy.allowed_privacy_max,
            "security.redaction_applied": context.security_policy.require_redaction,
        }
    )
    return redact_value(payload, context.security_policy)


def failure_event(
    context: RequestContext,
    *,
    event_type: str,
    message: str,
    error: str,
    mode: str,
) -> dict[str, Any]:
    payload = context.base_event(event_type, message)
    payload.update(
        {
            "mode": mode,
            "latency_ms": context.latency_ms,
            "error.message": error,
            "error": error,
        }
    )
    return redact_value(payload, context.security_policy)


def config_snapshot_event() -> dict[str, Any]:
    runtime_config = RuntimeConfig.from_env()
    policy = SecurityPolicy.from_env()
    return {
        "schema.version": EVENT_SCHEMA_VERSION,
        "event_type": "runtime.config_loaded",
        "message": "Runtime config and security policy loaded",
        "service.name": SERVICE_NAME,
        "service.version": SERVICE_VERSION,
        "config.hash": runtime_config.hash,
        "policy.hash": policy.hash,
        "config": runtime_config.to_dict(),
        "policy": policy.to_dict(),
    }


def _text_hash(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
