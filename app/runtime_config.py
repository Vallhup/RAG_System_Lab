import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Any


SECRET_PATTERN = re.compile(
    r"(?i)(api[_-]?key|token|secret|password|authorization)\s*[:=]\s*([^\s,;]+)"
)
WINDOWS_USER_PATH_PATTERN = re.compile(r"C:\\Users\\[^\\\s]+")


@dataclass(frozen=True)
class RuntimeConfig:
    retrieval_top_k: int
    retrieval_candidate_multiplier: int
    bm25_weight: float
    vector_weight: float
    rrf_k: int
    llama_chunk_size: int
    llama_chunk_overlap: int
    bm25_chunk_words: int
    bm25_chunk_overlap: int
    embedding_model: str

    @classmethod
    def from_env(cls) -> "RuntimeConfig":
        return cls(
            retrieval_top_k=_env_int("RAG_TOP_K", 10),
            retrieval_candidate_multiplier=_env_int("RAG_HYBRID_CANDIDATE_MULTIPLIER", 5),
            bm25_weight=_env_float("RAG_BM25_WEIGHT", 0.8),
            vector_weight=_env_float("RAG_VECTOR_WEIGHT", 1.0),
            rrf_k=_env_int("RAG_RRF_K", 60),
            llama_chunk_size=_env_int("RAG_LLAMA_CHUNK_SIZE", 512),
            llama_chunk_overlap=_env_int("RAG_LLAMA_CHUNK_OVERLAP", 80),
            bm25_chunk_words=_env_int("RAG_BM25_CHUNK_WORDS", 220),
            bm25_chunk_overlap=_env_int("RAG_BM25_CHUNK_OVERLAP", 50),
            embedding_model=os.getenv("RAG_EMBED_MODEL", "BAAI/bge-small-en-v1.5"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class SecurityPolicy:
    allowed_privacy_max: str
    default_privacy: str
    allow_external_llm: bool
    allow_external_embedding: bool
    require_redaction: bool
    allow_raw_query_logging: bool
    allow_raw_context_logging: bool
    redact_local_paths: bool
    redact_secrets: bool
    audit_enabled: bool

    @classmethod
    def from_env(cls) -> "SecurityPolicy":
        return cls(
            allowed_privacy_max=os.getenv("RAG_PRIVACY_ALLOWED_MAX", "public"),
            default_privacy=os.getenv("RAG_PRIVACY_DEFAULT", "public"),
            allow_external_llm=_env_bool("RAG_ALLOW_EXTERNAL_LLM", False),
            allow_external_embedding=_env_bool("RAG_ALLOW_EXTERNAL_EMBEDDING", True),
            require_redaction=_env_bool("RAG_REQUIRE_REDACTION", True),
            allow_raw_query_logging=_env_bool("RAG_LOG_RAW_QUERY", True),
            allow_raw_context_logging=_env_bool("RAG_LOG_RAW_CONTEXT", False),
            redact_local_paths=_env_bool("RAG_REDACT_LOCAL_PATHS", True),
            redact_secrets=_env_bool("RAG_REDACT_SECRETS", True),
            audit_enabled=_env_bool("RAG_AUDIT_ENABLED", True),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def hash(self) -> str:
        return _stable_hash(self.to_dict())


def redact_value(value: Any, policy: SecurityPolicy) -> Any:
    if isinstance(value, str):
        redacted = value
        if policy.redact_secrets:
            redacted = SECRET_PATTERN.sub(r"\1=[REDACTED]", redacted)
        if policy.redact_local_paths:
            redacted = WINDOWS_USER_PATH_PATTERN.sub(r"C:\\Users\\[REDACTED]", redacted)
        return redacted
    if isinstance(value, list):
        return [redact_value(item, policy) for item in value]
    if isinstance(value, tuple):
        return [redact_value(item, policy) for item in value]
    if isinstance(value, dict):
        return {key: redact_value(item, policy) for key, item in value.items()}
    return value


def _stable_hash(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default
