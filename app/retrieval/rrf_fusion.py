"""Reciprocal Rank Fusion (RRF) layer.

Spec §28.3 prefers RRF over raw weighted score fusion because BM25 and
vector retrievers return scores on incomparable scales.  RRF uses the
*rank* of a doc within each list, so weight scale issues vanish — matching
LlamaIndex's reciprocal rerank fusion behaviour (spec §31, LlamaIndex docs).

This layer expects ``carry`` to contain the union of upstream layer
outputs.  Each context's ``layer_origins`` field tells the fusion which
sub-list to attribute the doc to.  If a doc appears in multiple sub-lists
its RRF score adds across them, which is the conventional formulation.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

if __package__:
    from . import Context, RetrievalRequest, TierBudget
else:
    from app.retrieval import Context, RetrievalRequest, TierBudget  # type: ignore[no-redef]


DEFAULT_RRF_K = 60


@dataclass(frozen=True)
class LayerWeight:
    layer: str
    weight: float = 1.0


class RRFFusionLayer:
    """Fuse contexts from multiple layers by reciprocal rank.

    Construction takes an explicit list of ``LayerWeight`` so the same
    fusion can mix BM25 + Vector (current SciFact baseline) or
    Vector + ColBERT + BM25 (future).  Unknown layer origins are ignored,
    not fused — that keeps a buggy upstream layer from silently dominating.
    """

    name = "rrf"

    def __init__(
        self,
        layer_weights: Iterable[LayerWeight] | None = None,
        rrf_k: int = DEFAULT_RRF_K,
        dedupe_by: str = "doc_id",
    ) -> None:
        self.layer_weights = {lw.layer: lw.weight for lw in (layer_weights or [])}
        self.rrf_k = rrf_k
        self.dedupe_by = dedupe_by

    @property
    def ready(self) -> bool:
        return bool(self.layer_weights)

    def execute(
        self,
        request: RetrievalRequest,
        budget: TierBudget,
        carry: list[Context],
    ) -> list[Context]:
        if not carry:
            return carry
        per_layer_ranks: dict[str, list[Context]] = defaultdict(list)
        for ctx in carry:
            origins = ctx.layer_origins or ("unknown",)
            for origin in origins:
                if origin in self.layer_weights:
                    per_layer_ranks[origin].append(ctx)
        if not per_layer_ranks:
            return carry

        scores: dict[str, float] = defaultdict(float)
        best_context: dict[str, Context] = {}
        for layer, ctxs in per_layer_ranks.items():
            weight = self.layer_weights.get(layer, 1.0)
            ctxs.sort(key=lambda ctx: ctx.score, reverse=True)
            for rank, ctx in enumerate(ctxs, start=1):
                key = ctx.doc_id if self.dedupe_by == "doc_id" else ctx.chunk_id
                if not key:
                    continue
                scores[key] += weight / (self.rrf_k + rank)
                # keep the context with the strongest raw score per key as the
                # representative (for `text`, `chunk_id`).
                existing = best_context.get(key)
                if existing is None or ctx.score > existing.score:
                    best_context[key] = ctx

        fused: list[Context] = []
        for key, rrf_score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
            base = best_context[key]
            origins = tuple(sorted({o for o in base.layer_origins} | {"rrf"}))
            fused.append(
                Context(
                    doc_id=base.doc_id,
                    chunk_id=base.chunk_id,
                    score=round(rrf_score, 6),
                    text=base.text,
                    source_uri=base.source_uri,
                    layer_origins=origins,
                    metadata={**base.metadata, "fusion_layer": self.name},
                )
            )
        return fused[: max(1, budget.top_k * 3)]


def env_rrf_k(default: int = DEFAULT_RRF_K) -> int:
    import os

    try:
        return int(os.getenv("RAG_RRF_K", str(default)))
    except ValueError:
        return default


__all__ = ["DEFAULT_RRF_K", "LayerWeight", "RRFFusionLayer", "env_rrf_k"]
