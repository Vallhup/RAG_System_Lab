"""Tier Runtime.

Composes retrieval layers (BM25, Vector, RRF fusion, reranker, graph
expand, ...) into a tier-aware pipeline.  Encodes the spec rules:

* Tier 0/1/2/3/4 distinction (§10.1).
* Auto-promotion based on cheap quality signals (§10.5, §12.3).
* Each layer is independent and Protocol-conforming, so adding a graph
  expansion layer later is a list-append, not a refactor.
* WorkMode chooses the *initial* TierBudget (§9.1); the runtime may then
  escalate on its own.

This module is corpus-agnostic.  ``SciFactRetriever`` (defined in
``app/retrieval/runtime_factory.py``) wires the right layers + chunks for
the SciFact corpus, but the same TierRuntime instance type will be used by
the personal Vault retriever later (spec §30).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Iterable

if __package__:
    from . import (
        Context,
        QualitySignal,
        RetrievalLayer,
        RetrievalRequest,
        RetrievalResponse,
        Tier,
        TierBudget,
        WorkMode,
    )
else:
    from app.retrieval import (  # type: ignore[no-redef]
        Context,
        QualitySignal,
        RetrievalLayer,
        RetrievalRequest,
        RetrievalResponse,
        Tier,
        TierBudget,
        WorkMode,
    )


logger = logging.getLogger("rag.tier")


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TierStage:
    """One stage in the tier pipeline.

    ``layers`` execute sequentially; their outputs accumulate in ``carry``
    until a fusion layer collapses them.  ``required_ready`` flags any
    layer that *must* be ready (a vector layer that failed to build will
    cause the runtime to skip this stage instead of producing degraded
    output silently).
    """

    tier: Tier
    layers: tuple[RetrievalLayer, ...]
    requires_ready: tuple[str, ...] = ()


@dataclass(frozen=True)
class PromotionPolicy:
    """How aggressively to escalate after each tier.

    Numbers are intentionally conservative to start (spec §10.5: "Classifier
    is not a decider").  Tune via the regression harness once we have data.
    """

    min_candidates: int = 5
    min_score_gap_top1_top5: float = 0.0      # 0.0 means any positive gap is fine
    min_distinct_docs: int = 3
    promote_on_empty: bool = True

    def should_promote(self, signal: QualitySignal) -> bool:
        if signal.candidate_count == 0:
            return self.promote_on_empty
        if signal.candidate_count < self.min_candidates:
            return True
        if signal.distinct_doc_count < self.min_distinct_docs:
            return True
        if signal.score_gap_top1_top5 < self.min_score_gap_top1_top5:
            return True
        return False


# ---------------------------------------------------------------------------
# Tier Runtime
# ---------------------------------------------------------------------------


@dataclass
class TierRuntime:
    """Sequential, tier-aware retrieval pipeline.

    Construction is corpus-bound: each TierRuntime owns the layers for one
    corpus.  The same FastAPI process can hold multiple TierRuntime
    instances (SciFact + personal_vault) keyed by ``corpus_id``.
    """

    corpus_id: str
    stages: list[TierStage]
    promotion_policy: PromotionPolicy = field(default_factory=PromotionPolicy)
    max_tier: Tier = Tier.GRAPH_AUGMENTED

    def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        started = time.perf_counter()
        carry: list[Context] = []
        layers_executed: list[str] = []
        score_breakdown: dict[str, list[float]] = {}
        last_signal = QualitySignal.from_contexts([])
        final_tier = request.initial_tier
        reranked = False
        notes_parts: list[str] = []

        # find starting stage index by tier
        start_index = self._find_start_index(request.initial_tier)
        if start_index is None:
            notes_parts.append(f"No stage for initial tier {request.initial_tier.name}; defaulting to first.")
            start_index = 0

        for stage_index in range(start_index, len(self.stages)):
            stage = self.stages[stage_index]
            if stage.tier > self.max_tier:
                notes_parts.append(f"Skipping {stage.tier.name} (above max_tier={self.max_tier.name}).")
                break
            budget = self._budget_for(request, stage)
            if not self._stage_ready(stage):
                logger.warning(
                    "Tier %s skipped: layers not ready (%s)", stage.tier.name, stage.requires_ready
                )
                notes_parts.append(f"{stage.tier.name}: required layers not ready.")
                continue

            for layer in stage.layers:
                layer_start = time.perf_counter()
                carry = layer.execute(request, budget, carry)
                layer_latency = (time.perf_counter() - layer_start) * 1000
                layers_executed.append(layer.name)
                score_breakdown.setdefault(layer.name, []).extend(ctx.score for ctx in carry)
                logger.debug(
                    "Layer %s produced %d ctx in %.1fms",
                    layer.name,
                    len(carry),
                    layer_latency,
                )
                if layer.name == "rerank":
                    reranked = True

            last_signal = QualitySignal.from_contexts(carry)
            final_tier = stage.tier

            if not self._should_promote(stage_index, last_signal):
                notes_parts.append(f"Stopped at {stage.tier.name}: signal sufficient.")
                break
            notes_parts.append(f"Promoted past {stage.tier.name}: signal weak.")

        contexts = self._finalize(carry, request.top_k)
        return RetrievalResponse(
            contexts=contexts,
            final_tier=final_tier,
            layers_executed=tuple(layers_executed),
            candidate_count=last_signal.candidate_count,
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
            reranked=reranked,
            score_breakdown=score_breakdown,
            notes="; ".join(notes_parts),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_start_index(self, tier: Tier) -> int | None:
        for index, stage in enumerate(self.stages):
            if stage.tier >= tier:
                return index
        return None

    def _budget_for(self, request: RetrievalRequest, stage: TierStage) -> TierBudget:
        return TierBudget(
            tier=stage.tier,
            top_k=request.top_k,
            candidate_count=_candidate_count_for_tier(stage.tier, request.top_k),
            allow_reranker=stage.tier >= Tier.GRAPH_AUGMENTED,
            allow_graph_expansion=stage.tier >= Tier.GRAPH_AUGMENTED,
            allow_external_llm=stage.tier >= Tier.FULL_EVAL,
            rerank_window=_rerank_window_for_tier(stage.tier, request.top_k),
        )

    def _stage_ready(self, stage: TierStage) -> bool:
        if not stage.requires_ready:
            return True
        names_to_layers = {layer.name: layer for layer in stage.layers}
        for required in stage.requires_ready:
            layer = names_to_layers.get(required)
            if layer is None or not layer.ready:
                return False
        return True

    def _should_promote(self, stage_index: int, signal: QualitySignal) -> bool:
        # Last stage — nothing to promote to.
        if stage_index >= len(self.stages) - 1:
            return False
        return self.promotion_policy.should_promote(signal)

    def _finalize(self, carry: list[Context], top_k: int) -> list[Context]:
        # dedupe by doc_id, preserve first occurrence (spec §29.1).
        seen: set[str] = set()
        out: list[Context] = []
        for ctx in carry:
            if not ctx.doc_id or ctx.doc_id in seen:
                continue
            seen.add(ctx.doc_id)
            out.append(ctx)
            if len(out) >= top_k:
                break
        return out


# ---------------------------------------------------------------------------
# WorkMode dispatch
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkModePolicy:
    work_mode: WorkMode
    initial_tier: Tier
    max_tier: Tier
    description: str = ""


WORK_MODE_POLICIES: dict[WorkMode, WorkModePolicy] = {
    # SciFact /retrieve: start Hybrid, allow up to Graph-Augmented (reranker).
    WorkMode.RETRIEVAL_ONLY: WorkModePolicy(
        work_mode=WorkMode.RETRIEVAL_ONLY,
        initial_tier=Tier.HYBRID,
        max_tier=Tier.GRAPH_AUGMENTED,
        description="SciFact /retrieve: hybrid + optional reranker.",
    ),
    # Personal Vault placeholders — wired today, used once personal_vault loader lands.
    WorkMode.RECALL: WorkModePolicy(WorkMode.RECALL, Tier.FAST, Tier.HYBRID, "Personal: cheap recall."),
    WorkMode.RESEARCH: WorkModePolicy(WorkMode.RESEARCH, Tier.HYBRID, Tier.FULL_EVAL, "Personal: source-grounded."),
    WorkMode.REFLECTION: WorkModePolicy(WorkMode.REFLECTION, Tier.HYBRID, Tier.GRAPH_AUGMENTED, "Personal: thought patterns."),
    WorkMode.DECISION: WorkModePolicy(WorkMode.DECISION, Tier.GRAPH_AUGMENTED, Tier.FULL_EVAL, "Personal: decisions."),
    WorkMode.CREATION: WorkModePolicy(WorkMode.CREATION, Tier.HYBRID, Tier.GRAPH_AUGMENTED, "Personal: drafting."),
    WorkMode.EXPLORATION: WorkModePolicy(WorkMode.EXPLORATION, Tier.HYBRID, Tier.HYBRID, "Personal: hobby/exploration."),
    WorkMode.MAINTENANCE: WorkModePolicy(WorkMode.MAINTENANCE, Tier.NO_RETRIEVAL, Tier.FAST, "Personal: metadata scan."),
}


def policy_for(work_mode: WorkMode) -> WorkModePolicy:
    return WORK_MODE_POLICIES[work_mode]


# ---------------------------------------------------------------------------
# Tier-specific candidate-count helpers
# ---------------------------------------------------------------------------


def _candidate_count_for_tier(tier: Tier, top_k: int) -> int:
    if tier == Tier.NO_RETRIEVAL:
        return 0
    multiplier = {
        Tier.FAST: 3,
        Tier.HYBRID: 5,
        Tier.GRAPH_AUGMENTED: 8,
        Tier.FULL_EVAL: 10,
    }.get(tier, 5)
    return max(top_k * multiplier, 10)


def _rerank_window_for_tier(tier: Tier, top_k: int) -> int:
    if tier < Tier.GRAPH_AUGMENTED:
        return 0
    return max(top_k * 3, 30)


__all__ = [
    "PromotionPolicy",
    "TierRuntime",
    "TierStage",
    "WORK_MODE_POLICIES",
    "WorkModePolicy",
    "policy_for",
]
