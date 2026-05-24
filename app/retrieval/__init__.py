"""Tier-aware, corpus-agnostic retrieval runtime.

This package provides the architectural backbone shared by both the SciFact
evaluation endpoint and the future personal Vault corpus (see PARA-Aware spec
§7~§10, §26.4). It encodes the spec decisions:

* a corpus-agnostic ``Retriever`` Protocol so that BM25, vector, hybrid, and
  graph-augmented backends can be composed and swapped without touching
  ``app/main.py`` or the FastAPI surface;
* an explicit ``Tier`` (0–4) cost model with auto-promotion, separated from
  ``EvaluationLevel`` (§10.4);
* a ``WorkMode`` enum (§9.1) that dispatches a query to the right initial Tier
  policy — today only ``RetrievalOnly`` is wired up (SciFact), the others act
  as named placeholders so personal-Vault work can plug in without breaking
  the FastAPI contract;
* a ``RetrievalRequest`` / ``RetrievalResponse`` pair that carries
  ``corpus_id`` end-to-end (§3.2, §23 item 16) and exposes the metadata the
  observability layer needs (`trace_id`, `policy_hash`, `config_hash`).

This module deliberately has no dependency on FastAPI, LlamaIndex, or any
specific vector store. Concrete backends live under ``app/retrieval/...``
submodules and conform to the Protocols defined here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Tier / WorkMode enums (spec §9.1, §10.1)
# ---------------------------------------------------------------------------


class Tier(int, Enum):
    """Retrieval cost tier. Higher tier = more candidates + more compute."""

    NO_RETRIEVAL = 0
    FAST = 1            # BM25 only OR vector only
    HYBRID = 2          # BM25 + Vector + RRF (current SciFact baseline)
    GRAPH_AUGMENTED = 3  # Hybrid + small graph expansion + reranker
    FULL_EVAL = 4       # Tier 3 + LLM verification, contradiction check

    @classmethod
    def parse(cls, value: int | str | "Tier") -> "Tier":
        if isinstance(value, cls):
            return value
        if isinstance(value, int):
            return cls(value)
        normalized = str(value).strip().upper().replace("-", "_").replace(" ", "_")
        if normalized.isdigit():
            return cls(int(normalized))
        # accept short names like "FAST" or "HYBRID"
        return cls[normalized]


class WorkMode(str, Enum):
    """Why is the user asking?  Spec §9.1 — drives initial Tier policy."""

    RETRIEVAL_ONLY = "retrieval_only"  # SciFact /retrieve (current real use)
    RECALL = "recall"
    RESEARCH = "research"
    REFLECTION = "reflection"
    DECISION = "decision"
    CREATION = "creation"
    EXPLORATION = "exploration"
    MAINTENANCE = "maintenance"

    @classmethod
    def parse(cls, value: str | "WorkMode") -> "WorkMode":
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        return cls(normalized)


# ---------------------------------------------------------------------------
# Budgets and policies
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TierBudget:
    """Concrete cost budget for one tier execution.

    Auto-promotion (spec §10.5) inspects the resulting metrics and may
    construct a *higher* TierBudget for the next attempt within the same
    query.  The dispatcher decides this — the layer modules should treat the
    budget as immutable per attempt.
    """

    tier: Tier
    top_k: int = 10
    candidate_count: int = 50          # how many docs to fetch before final selection
    allow_reranker: bool = False
    allow_graph_expansion: bool = False
    allow_external_llm: bool = False
    rerank_window: int = 30            # cross-encoder/ColBERT input window
    notes: str = ""


# ---------------------------------------------------------------------------
# Request / Response dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetrievalRequest:
    """Input to a Retriever.  Carries corpus / mode / budget end-to-end."""

    query: str
    corpus_id: str
    work_mode: WorkMode = WorkMode.RETRIEVAL_ONLY
    initial_tier: Tier = Tier.HYBRID
    top_k: int = 10
    # Optional metadata propagated from the FastAPI layer for trace correlation.
    query_id: str | None = None
    trace_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Context:
    """One retrieved chunk.

    Mirrors the SciFact `/retrieve` response shape so the FastAPI layer can
    serialize it directly without renaming.  Personal Vault retrievers fill
    `source_uri` to point back at the originating Obsidian note (spec §15.3,
    §27.6).
    """

    doc_id: str
    chunk_id: str
    score: float
    text: str
    source_uri: str | None = None
    layer_origins: tuple[str, ...] = ()   # ("vector", "bm25", "rerank")
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalResponse:
    """Output of a Retriever.

    `score_breakdown` allows the evaluation harness and observability layer
    to record per-stage signals (spec §15.5, §27.7).
    """

    contexts: list[Context]
    final_tier: Tier
    layers_executed: tuple[str, ...]
    candidate_count: int
    latency_ms: float
    reranked: bool = False
    score_breakdown: dict[str, list[float]] = field(default_factory=dict)
    notes: str = ""


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class RetrievalLayer(Protocol):
    """One composable step inside the Tier Runtime.

    BM25, Vector, RRF fusion, reranker, graph-expand are all `RetrievalLayer`
    instances.  Each layer transforms a `list[Context]` (possibly empty)
    given the current `TierBudget`.  Layers are stateless w.r.t. the query
    and free to be reused across requests.
    """

    name: str

    @property
    def ready(self) -> bool: ...

    def execute(
        self,
        request: RetrievalRequest,
        budget: TierBudget,
        carry: list[Context],
    ) -> list[Context]: ...


@runtime_checkable
class Retriever(Protocol):
    """Corpus-bound entry point.

    SciFact and personal_vault each get their own Retriever implementation,
    but both expose the same surface so FastAPI / evaluation.py can swap them
    by registering a different builder.
    """

    corpus_id: str

    @property
    def ready(self) -> bool: ...

    @property
    def mode(self) -> str: ...

    def retrieve(self, request: RetrievalRequest) -> RetrievalResponse: ...

    def index_metadata(self) -> dict[str, Any]: ...


# ---------------------------------------------------------------------------
# Quality signals (used by the auto-promote policy)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QualitySignal:
    """Cheap signals computed from a layer's output without ground truth.

    Spec §10.5: auto-promote starts cheap, then escalates if these signals
    look weak. We keep them numeric and side-effect free so they can be
    written into the trace event verbatim.
    """

    candidate_count: int
    score_top: float
    score_gap_top1_top5: float
    distinct_doc_count: int

    @classmethod
    def from_contexts(cls, contexts: list[Context]) -> "QualitySignal":
        if not contexts:
            return cls(0, 0.0, 0.0, 0)
        scores = [ctx.score for ctx in contexts]
        top = scores[0]
        # if we have <5 candidates the "gap" is just top - last available
        anchor_index = min(4, len(scores) - 1)
        gap = top - scores[anchor_index]
        distinct = len({ctx.doc_id for ctx in contexts})
        return cls(
            candidate_count=len(contexts),
            score_top=float(top),
            score_gap_top1_top5=float(gap),
            distinct_doc_count=distinct,
        )


__all__ = [
    "Context",
    "QualitySignal",
    "RetrievalLayer",
    "RetrievalRequest",
    "RetrievalResponse",
    "Retriever",
    "Tier",
    "TierBudget",
    "WorkMode",
]
