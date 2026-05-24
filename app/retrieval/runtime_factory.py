"""Factories that assemble a ready-to-use TierRuntime for a specific corpus.

Today only SciFact is wired up; once the personal Vault loader lands, a
sibling ``build_personal_vault_runtime`` will share the same Tier 1/2/3
layer types but plug a different loader + chunker + storage path.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__:
    from . import (
        Context,
        Retriever,
        RetrievalRequest,
        RetrievalResponse,
        Tier,
        WorkMode,
    )
    from .bm25_layer import BM25Layer
    from .chunkers import Chunk, Chunker, get_chunker
    from .corpora.scifact_loader import load_scifact_corpus
    from .rerankers import build_default_reranker
    from .rrf_fusion import LayerWeight, RRFFusionLayer, env_rrf_k
    from .tier_runtime import (
        PromotionPolicy,
        TierRuntime,
        TierStage,
        WORK_MODE_POLICIES,
        policy_for,
    )
    from .vector_layer import VectorIndexMetadata, VectorLayer, env_embedding_model
else:  # pragma: no cover - Docker flat-layout fallback
    from app.retrieval import (  # type: ignore[no-redef]
        Context,
        Retriever,
        RetrievalRequest,
        RetrievalResponse,
        Tier,
        WorkMode,
    )
    from app.retrieval.bm25_layer import BM25Layer  # type: ignore[no-redef]
    from app.retrieval.chunkers import Chunk, Chunker, get_chunker  # type: ignore[no-redef]
    from app.retrieval.corpora.scifact_loader import load_scifact_corpus  # type: ignore[no-redef]
    from app.retrieval.rerankers import build_default_reranker  # type: ignore[no-redef]
    from app.retrieval.rrf_fusion import LayerWeight, RRFFusionLayer, env_rrf_k  # type: ignore[no-redef]
    from app.retrieval.tier_runtime import (  # type: ignore[no-redef]
        PromotionPolicy,
        TierRuntime,
        TierStage,
        WORK_MODE_POLICIES,
        policy_for,
    )
    from app.retrieval.vector_layer import VectorIndexMetadata, VectorLayer, env_embedding_model  # type: ignore[no-redef]


logger = logging.getLogger("rag.factory")


@dataclass
class CorpusRuntime:
    """A TierRuntime instance bound to one corpus.

    Wraps the lower-level TierRuntime with the policy + metadata that
    ``main.py`` and ``evaluation.py`` need.  Implements the ``Retriever``
    Protocol via duck typing.
    """

    corpus_id: str
    runtime: TierRuntime
    bm25: BM25Layer
    vector: VectorLayer | None
    metadata: VectorIndexMetadata
    work_mode: WorkMode = WorkMode.RETRIEVAL_ONLY

    @property
    def ready(self) -> bool:
        return self.bm25.ready

    @property
    def mode(self) -> str:
        if self.vector is not None and self.vector.ready:
            return "hybrid"
        return "bm25_fallback"

    def retrieve(self, request_or_query: Any, top_k: int | None = None) -> Any:
        """Polymorphic entry: accept either a RetrievalRequest or a plain string.

        FastAPI today still passes (question, top_k).  The harness passes a
        full RetrievalRequest.  Both work without breaking the existing
        /retrieve contract.
        """
        if isinstance(request_or_query, RetrievalRequest):
            request = request_or_query
        else:
            question = str(request_or_query)
            policy = policy_for(self.work_mode)
            request = RetrievalRequest(
                query=question,
                corpus_id=self.corpus_id,
                work_mode=self.work_mode,
                initial_tier=policy.initial_tier,
                top_k=int(top_k or 10),
            )
        response = self.runtime.retrieve(request)
        # Legacy callers (FastAPI) expect list[dict]; new callers receive RetrievalResponse.
        if isinstance(request_or_query, RetrievalRequest):
            return response
        return [
            {
                "doc_id": ctx.doc_id,
                "chunk_id": ctx.chunk_id,
                "score": ctx.score,
                "text": ctx.text,
            }
            for ctx in response.contexts
        ]

    def index_metadata(self) -> dict[str, Any]:
        return self.metadata.to_dict()


# ---------------------------------------------------------------------------
# SciFact assembly
# ---------------------------------------------------------------------------


def build_scifact_runtime(
    corpus_path: Path,
    storage_dir: Path,
    *,
    chunker_name: str | None = None,
    work_mode: WorkMode = WorkMode.RETRIEVAL_ONLY,
    enable_reranker: bool | None = None,
    max_tier: Tier | None = None,
) -> CorpusRuntime:
    """Assemble a SciFact TierRuntime.

    ``chunker_name`` defaults to env ``RAG_CHUNKER`` (``sentence_window`` /
    ``abstract_one_node``).  Vector and reranker layers are constructed
    even if disabled — they simply report ``ready = False`` and the tier
    runtime skips them.
    """

    chunker_name = (chunker_name or os.getenv("RAG_CHUNKER", "sentence_window"))
    chunker = get_chunker(chunker_name, **_chunker_kwargs(chunker_name))

    load_result = load_scifact_corpus(corpus_path)
    chunks = chunker.split_many(load_result.documents)
    logger.info(
        "SciFact: corpus=%s docs=%d chunks=%d chunker=%s",
        corpus_path,
        load_result.document_count,
        len(chunks),
        chunker_name,
    )

    bm25 = BM25Layer(chunks)
    embedding_model = env_embedding_model()
    from ..runtime_config import RuntimeConfig, SecurityPolicy

    runtime_config = RuntimeConfig.from_env()
    security_policy = SecurityPolicy.from_env()

    metadata = VectorIndexMetadata(
        corpus_id=os.getenv("RAG_CORPUS_ID", "scifact"),
        corpus_sha256=load_result.corpus_sha256,
        chunker_name=chunker.name,
        chunker_version=chunker.version,
        embedding_model=embedding_model,
        chunk_count=len(chunks),
        config_hash=runtime_config.hash,
        policy_hash=security_policy.hash,
    )

    vector = VectorLayer(chunks=chunks, storage_dir=storage_dir, metadata=metadata)
    rrf = RRFFusionLayer(
        layer_weights=[
            LayerWeight("bm25", weight=_env_float("RAG_BM25_WEIGHT", 0.8)),
            LayerWeight("vector", weight=_env_float("RAG_VECTOR_WEIGHT", 1.0)),
        ],
        rrf_k=env_rrf_k(),
    )
    reranker = build_default_reranker(enable_reranker)

    stages = [
        TierStage(tier=Tier.FAST, layers=(bm25,)),
        TierStage(tier=Tier.HYBRID, layers=(bm25, vector, rrf)),
        TierStage(tier=Tier.GRAPH_AUGMENTED, layers=(bm25, vector, rrf, reranker)),
    ]

    work_mode_policy = policy_for(work_mode)
    runtime = TierRuntime(
        corpus_id=metadata.corpus_id,
        stages=stages,
        promotion_policy=PromotionPolicy(),
        max_tier=max_tier or work_mode_policy.max_tier,
    )

    return CorpusRuntime(
        corpus_id=metadata.corpus_id,
        runtime=runtime,
        bm25=bm25,
        vector=vector,
        metadata=metadata,
        work_mode=work_mode,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunker_kwargs(name: str) -> dict[str, Any]:
    if name in {"sentence_window", "sentence", "default"}:
        return {
            "chunk_words": _env_int("RAG_BM25_CHUNK_WORDS", 220),
            "overlap": _env_int("RAG_BM25_CHUNK_OVERLAP", 50),
        }
    return {}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


__all__ = ["CorpusRuntime", "build_scifact_runtime"]
