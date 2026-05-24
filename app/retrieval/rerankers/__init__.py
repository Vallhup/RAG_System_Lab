"""Reranker layers (Tier 3 components).

Reranker layers re-order existing candidates rather than producing new
ones.  Per spec §28.4 we only activate them when (a) the tier runtime has
escalated to Tier 3, and (b) the feature flag ``RAG_ENABLE_RERANKER`` is
set.  Cross-encoder scores are not comparable across queries, so the
fusion layer (RRF) uses only the *rank* output.
"""

from __future__ import annotations

from .cross_encoder import CrossEncoderReranker, NullReranker, build_default_reranker

__all__ = ["CrossEncoderReranker", "NullReranker", "build_default_reranker"]
