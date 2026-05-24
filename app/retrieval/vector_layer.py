"""Vector retrieval layer (LlamaIndex VectorStoreIndex adapter).

The layer is a thin adapter so the rest of the pipeline never imports
LlamaIndex directly.  When the optional dependency is unavailable, the layer
reports ``ready = False`` and the Tier Runtime falls back to BM25 alone
(spec §21.1, §29.5).

Indexing is intentionally deterministic given a fixed corpus + chunker
combo: the chunk ids of incoming ``Chunk`` records are written verbatim
into the LlamaIndex node ids so the evaluation harness sees stable
``chunk_id`` values across runs (spec §17.4 — per_query top_doc_ids).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__:
    from . import Context, RetrievalRequest, TierBudget
    from .chunkers import Chunk
else:
    from app.retrieval import Context, RetrievalRequest, TierBudget  # type: ignore[no-redef]
    from app.retrieval.chunkers import Chunk  # type: ignore[no-redef]


logger = logging.getLogger("rag.vector")


@dataclass
class VectorIndexMetadata:
    corpus_id: str
    corpus_sha256: str
    chunker_name: str
    chunker_version: str
    embedding_model: str
    chunk_count: int
    config_hash: str
    policy_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "corpus_id": self.corpus_id,
            "corpus_sha256": self.corpus_sha256,
            "chunker_name": self.chunker_name,
            "chunker_version": self.chunker_version,
            "embedding_model": self.embedding_model,
            "chunk_count": self.chunk_count,
            "config_hash": self.config_hash,
            "policy_hash": self.policy_hash,
        }


class VectorLayer:
    """LlamaIndex-backed vector retrieval layer.

    The layer is *lazy*: index building only happens when ``build`` is
    explicitly invoked.  The Tier Runtime calls ``build`` once during
    initialisation; if it raises (e.g. LlamaIndex missing or HF model
    download fails), the layer remains unready and the runtime continues
    with BM25 only.
    """

    name = "vector"

    def __init__(
        self,
        chunks: list[Chunk],
        storage_dir: Path,
        metadata: VectorIndexMetadata,
    ) -> None:
        self.chunks = chunks
        self.storage_dir = storage_dir
        self.metadata = metadata
        self._index: Any | None = None
        self._error: str | None = None

    @property
    def ready(self) -> bool:
        return self._index is not None

    @property
    def error(self) -> str | None:
        return self._error

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def build(self) -> None:
        try:
            self._configure_llama()
            metadata_path = self.storage_dir / "metadata.json"
            if self._can_load_existing(metadata_path):
                self._index = self._load_existing()
                logger.info("Loaded existing vector index from %s", self.storage_dir)
                return
            self._index = self._build_new()
            self._persist(metadata_path)
            logger.info("Built new vector index in %s", self.storage_dir)
        except Exception as exc:  # pragma: no cover - exercised via failure_event path
            self._index = None
            self._error = f"Vector layer unavailable: {exc}"
            logger.exception("Vector layer build failed: %s", exc)

    # ------------------------------------------------------------------
    # Retrieval Layer surface
    # ------------------------------------------------------------------

    def execute(
        self,
        request: RetrievalRequest,
        budget: TierBudget,
        carry: list[Context],
    ) -> list[Context]:
        if not self.ready:
            return carry
        try:
            hits = self._retrieve(request.query, budget.candidate_count)
        except Exception as exc:  # pragma: no cover - logged, fall through
            logger.exception("Vector retrieval failed: %s", exc)
            self._error = f"Vector retrieval failed: {exc}"
            return carry
        return [
            Context(
                doc_id=hit["doc_id"],
                chunk_id=hit["chunk_id"],
                score=hit["score"],
                text=hit["text"],
                source_uri=hit.get("source_uri"),
                layer_origins=("vector",),
                metadata={"layer": self.name},
            )
            for hit in hits
        ]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _configure_llama(self) -> None:
        from llama_index.core import Settings
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        Settings.embed_model = HuggingFaceEmbedding(model_name=self.metadata.embedding_model)
        Settings.llm = None  # we never call an LLM from the retrieval layer

    def _can_load_existing(self, metadata_path: Path) -> bool:
        if not self.storage_dir.exists() or not metadata_path.exists():
            return False
        try:
            stored = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False
        return stored == self.metadata.to_dict()

    def _load_existing(self) -> Any:
        from llama_index.core import StorageContext, load_index_from_storage

        storage_context = StorageContext.from_defaults(persist_dir=str(self.storage_dir))
        return load_index_from_storage(storage_context)

    def _build_new(self) -> Any:
        from llama_index.core import VectorStoreIndex
        from llama_index.core.schema import Document

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        documents = [
            Document(
                text=chunk.text,
                doc_id=chunk.chunk_id,
                metadata={
                    "doc_id": chunk.doc_id,
                    "chunk_id": chunk.chunk_id,
                    "title": chunk.title,
                    "parent_chunk_id": chunk.parent_chunk_id,
                    **(chunk.metadata or {}),
                },
            )
            for chunk in self.chunks
        ]
        # We already chunked upstream; LlamaIndex should embed each Document
        # as a single node with its chunk_id preserved verbatim.
        from llama_index.core.node_parser import SentenceSplitter

        parser = SentenceSplitter(chunk_size=8192, chunk_overlap=0)
        nodes = parser.get_nodes_from_documents(documents)
        for node, chunk in zip(nodes, self.chunks):
            node.id_ = chunk.chunk_id
            # ensure metadata round-trip
            node.metadata.setdefault("doc_id", chunk.doc_id)
            node.metadata.setdefault("chunk_id", chunk.chunk_id)
        return VectorStoreIndex(nodes)

    def _persist(self, metadata_path: Path) -> None:
        if self._index is None:
            return
        self._index.storage_context.persist(persist_dir=str(self.storage_dir))
        metadata_path.write_text(
            json.dumps(self.metadata.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _retrieve(self, query: str, top_k: int) -> list[dict[str, Any]]:
        if self._index is None:
            return []
        retriever = self._index.as_retriever(similarity_top_k=min(max(top_k, 1), 100))
        nodes = retriever.retrieve(query)
        hits: list[dict[str, Any]] = []
        for node_with_score in nodes:
            node = node_with_score.node
            doc_id = str(node.metadata.get("doc_id", "")).strip()
            chunk_id = str(node.metadata.get("chunk_id", "") or node.node_id).strip()
            if not doc_id:
                continue
            score = node_with_score.score if node_with_score.score is not None else 0.0
            hits.append(
                {
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "score": round(float(score), 6),
                    "text": _shorten(node.get_content().strip()),
                    "source_uri": node.metadata.get("source_uri"),
                }
            )
        return hits


def _shorten(text: str, limit: int = 1500) -> str:
    return text[:limit] + "..." if len(text) > limit else text


def env_embedding_model(default: str = "BAAI/bge-small-en-v1.5") -> str:
    return os.getenv("RAG_EMBED_MODEL", default)


__all__ = ["VectorIndexMetadata", "VectorLayer", "env_embedding_model"]
