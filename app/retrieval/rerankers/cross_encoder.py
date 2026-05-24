"""Cross-encoder reranker layer.

Implements spec §28.4: a Tier-3 component that takes the top
``budget.rerank_window`` candidates and re-orders them with a
cross-encoder model.  The model is loaded lazily so an unavailable Tier 3
never breaks Tier 2 — the layer reports ``ready = False`` and the tier
runtime simply skips it.

Two implementations live here:

* ``CrossEncoderReranker`` — wraps a ``sentence-transformers`` cross
  encoder (default ``ms-marco-MiniLM-L-6-v2`` — small, fast, suitable for
  CPU-only Docker images).
* ``NullReranker`` — used when ``RAG_ENABLE_RERANKER`` is unset or the
  model fails to load.  Returns the carry unchanged.
"""

from __future__ import annotations

import logging
import os
from typing import Any

if __package__:
    from .. import Context, RetrievalRequest, TierBudget
else:
    from app.retrieval import Context, RetrievalRequest, TierBudget  # type: ignore[no-redef]


logger = logging.getLogger("rag.rerank")
DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class NullReranker:
    """No-op reranker used as a safe default.

    Keeps the tier pipeline composable even when the optional cross-encoder
    dependency isn't installed.  Always reports ``ready = True`` so it can
    be inserted unconditionally; it just does nothing.
    """

    name = "rerank"  # same name as the real reranker so callers don't branch

    @property
    def ready(self) -> bool:
        return True

    def execute(
        self,
        request: RetrievalRequest,
        budget: TierBudget,
        carry: list[Context],
    ) -> list[Context]:
        return carry


class CrossEncoderReranker:
    """Lazy-loaded cross-encoder reranker.

    The model is loaded the first time ``execute`` is called.  This keeps
    process startup fast and avoids importing ``sentence-transformers`` at
    module import time (it pulls in PyTorch which is heavy for unit
    tests).  Loading failure marks the layer permanently unready.
    """

    name = "rerank"

    def __init__(self, model_name: str | None = None, batch_size: int = 32) -> None:
        self.model_name = model_name or os.getenv("RAG_RERANKER_MODEL", DEFAULT_MODEL)
        self.batch_size = batch_size
        self._model: Any | None = None
        self._load_attempted = False
        self._load_error: str | None = None

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _ensure_model(self) -> bool:
        if self._model is not None:
            return True
        if self._load_attempted:
            return False
        self._load_attempted = True
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
            logger.info("Loaded cross-encoder reranker: %s", self.model_name)
            return True
        except Exception as exc:  # pragma: no cover - exercised via failure_event path
            self._load_error = str(exc)
            logger.warning("Cross-encoder reranker unavailable: %s", exc)
            return False

    @property
    def ready(self) -> bool:
        if self._model is not None:
            return True
        if not self._load_attempted:
            # treat as "might be ready"; first execute() will resolve it
            return True
        return False

    @property
    def error(self) -> str | None:
        return self._load_error

    # ------------------------------------------------------------------
    # RetrievalLayer surface
    # ------------------------------------------------------------------

    def execute(
        self,
        request: RetrievalRequest,
        budget: TierBudget,
        carry: list[Context],
    ) -> list[Context]:
        if not budget.allow_reranker:
            return carry
        if not carry:
            return carry
        if not self._ensure_model():
            return carry

        window = max(budget.rerank_window or 0, budget.top_k)
        head = carry[:window]
        tail = carry[window:]

        pairs = [(request.query, ctx.text) for ctx in head]
        try:
            assert self._model is not None
            scores = self._model.predict(pairs, batch_size=self.batch_size)
        except Exception as exc:  # pragma: no cover
            logger.warning("Cross-encoder predict failed: %s", exc)
            return carry

        scored = [
            (
                float(score),
                Context(
                    doc_id=ctx.doc_id,
                    chunk_id=ctx.chunk_id,
                    score=float(score),  # cross-encoder rank score replaces RRF
                    text=ctx.text,
                    source_uri=ctx.source_uri,
                    layer_origins=tuple(sorted(set(ctx.layer_origins) | {"rerank"})),
                    metadata={**ctx.metadata, "rerank_score": float(score)},
                ),
            )
            for ctx, score in zip(head, scores)
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        reordered = [ctx for _, ctx in scored]
        # tail keeps RRF order but ends up below reranked candidates.
        return reordered + tail


def build_default_reranker(enabled: bool | None = None) -> CrossEncoderReranker | NullReranker:
    """Factory honoring the ``RAG_ENABLE_RERANKER`` feature flag.

    ``enabled=None`` reads the env var; explicit ``True/False`` overrides.
    """
    if enabled is None:
        flag = os.getenv("RAG_ENABLE_RERANKER", "false").strip().lower()
        enabled = flag in {"1", "true", "yes", "on"}
    if not enabled:
        return NullReranker()
    return CrossEncoderReranker()


__all__ = ["CrossEncoderReranker", "NullReranker", "build_default_reranker", "DEFAULT_MODEL"]
