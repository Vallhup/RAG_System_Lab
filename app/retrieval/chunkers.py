"""Chunker strategies.

Spec §28.1 says chunking is a corpus-dependent decision: SciFact has very
short abstracts (favor "abstract one node"), while a personal Vault note may
need parent-child or hierarchical splitting.  This module defines a
strategy-pattern surface so the right chunker can be picked per corpus, and
new strategies can be A/B'd via the regression harness without touching the
retrieval layers.

Each chunker accepts a tiny `RawDocument` (id + title + text + metadata) and
returns a list of `Chunk` records — never raw strings, so origin information
is preserved through the pipeline (spec §29.5, §18.1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol, runtime_checkable


@dataclass(frozen=True)
class RawDocument:
    """Document handed to a Chunker before splitting."""

    doc_id: str
    title: str
    text: str
    source_uri: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_combined_text(self) -> str:
        if self.title and self.text:
            return f"{self.title}\n\n{self.text}"
        return self.text or self.title


@dataclass(frozen=True)
class Chunk:
    """One chunk produced by a Chunker."""

    doc_id: str
    chunk_id: str
    title: str
    text: str
    parent_chunk_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Chunker(Protocol):
    name: str
    version: str

    def split(self, document: RawDocument) -> list[Chunk]: ...

    def split_many(self, documents: Iterable[RawDocument]) -> list[Chunk]:
        ...


# ---------------------------------------------------------------------------
# Strategy: sentence / word-window chunker — current SciFact default
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SentenceChunker:
    """Word-window chunker matching the current BM25 fallback behaviour.

    For SciFact abstracts most documents collapse to 1 chunk anyway, but
    keeping a window-based default lets us A/B against tighter windows
    (256/40, 384/64) without rewriting downstream layers.
    """

    name: str = "sentence_window"
    version: str = "1.0"
    chunk_words: int = 220
    overlap: int = 50
    keep_title: bool = True

    def split(self, document: RawDocument) -> list[Chunk]:
        words = (document.text or "").split()
        if not words:
            # still surface the title as a single chunk to preserve recall
            title = document.title.strip()
            if not title:
                return []
            return [
                Chunk(
                    doc_id=document.doc_id,
                    chunk_id=f"{document.doc_id}::{self.name}_000",
                    title=document.title,
                    text=title,
                    metadata=dict(document.metadata),
                )
            ]
        window = max(1, self.chunk_words)
        step = max(1, window - max(0, min(self.overlap, window - 1)))
        chunks: list[Chunk] = []
        for index, start in enumerate(range(0, len(words), step)):
            piece = " ".join(words[start : start + window]).strip()
            if not piece:
                continue
            text = f"{document.title}\n{piece}" if self.keep_title and document.title else piece
            chunks.append(
                Chunk(
                    doc_id=document.doc_id,
                    chunk_id=f"{document.doc_id}::{self.name}_{index:03d}",
                    title=document.title,
                    text=text,
                    metadata=dict(document.metadata),
                )
            )
            if start + window >= len(words):
                break
        return chunks

    def split_many(self, documents: Iterable[RawDocument]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for document in documents:
            chunks.extend(self.split(document))
        return chunks


# ---------------------------------------------------------------------------
# Strategy: abstract-one-node — favored by SciFact / BEIR style corpora
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AbstractOneNodeChunker:
    """Each document becomes a single chunk (title + abstract).

    SciFact abstracts are short; embedding-level recall is often best when
    the whole abstract is one node so the embedding represents the document
    as a whole instead of being averaged across overlapping windows
    (spec §28.1).  This is the natural BEIR baseline.
    """

    name: str = "abstract_one_node"
    version: str = "1.0"

    def split(self, document: RawDocument) -> list[Chunk]:
        body = document.as_combined_text().strip()
        if not body:
            return []
        return [
            Chunk(
                doc_id=document.doc_id,
                chunk_id=f"{document.doc_id}::abstract",
                title=document.title,
                text=body,
                metadata=dict(document.metadata),
            )
        ]

    def split_many(self, documents: Iterable[RawDocument]) -> list[Chunk]:
        return [
            chunk
            for document in documents
            for chunk in self.split(document)
        ]


# ---------------------------------------------------------------------------
# Strategy slot: parent-child / hierarchical
# ---------------------------------------------------------------------------


class ParentChildChunker:
    """Parent-child / Auto-merge skeleton (spec §28.2).

    Personal Vault notes (especially long Markdown pages) benefit from
    retrieving leaf chunks and surfacing the parent on context build. This
    class is intentionally a placeholder so the retriever interface already
    accepts it — the heuristic logic will be filled in once the personal
    Vault loader exists.
    """

    name = "parent_child"
    version = "0.1-skeleton"

    def __init__(self, leaf_words: int = 120, parent_words: int = 400, overlap: int = 30) -> None:
        self.leaf_words = leaf_words
        self.parent_words = parent_words
        self.overlap = overlap

    def split(self, document: RawDocument) -> list[Chunk]:  # pragma: no cover - placeholder
        raise NotImplementedError(
            "ParentChildChunker is a placeholder. Implement when personal_vault loader lands."
        )

    def split_many(self, documents: Iterable[RawDocument]) -> list[Chunk]:  # pragma: no cover
        raise NotImplementedError(
            "ParentChildChunker is a placeholder. Implement when personal_vault loader lands."
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def get_chunker(name: str, **kwargs: Any) -> Chunker:
    """Return a chunker by name. Used by env-driven factories."""
    normalized = name.strip().lower()
    if normalized in {"sentence_window", "sentence", "default"}:
        return SentenceChunker(**kwargs)
    if normalized in {"abstract_one_node", "abstract", "scifact_default"}:
        return AbstractOneNodeChunker()
    if normalized in {"parent_child", "auto_merge"}:
        return ParentChildChunker(**kwargs)
    raise ValueError(f"Unknown chunker: {name}")


__all__ = [
    "AbstractOneNodeChunker",
    "Chunk",
    "Chunker",
    "ParentChildChunker",
    "RawDocument",
    "SentenceChunker",
    "get_chunker",
]
