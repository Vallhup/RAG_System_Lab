"""SciFact corpus loader.

The SciFact corpus is a JSON Lines file where each line is::

    {"_id": "31715818", "title": "...", "text": "..."}

This module reads that file into ``RawDocument`` records that the rest of
the retrieval pipeline can consume.  Keeping the corpus shape isolated here
means the BM25 / Vector / RRF layers stay corpus-agnostic (spec §29).
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

if __package__:
    from ..chunkers import RawDocument
else:  # script-style invocation under Docker (`python -m ...`)
    from app.retrieval.chunkers import RawDocument  # type: ignore[no-redef]


logger = logging.getLogger("rag.index")


@dataclass(frozen=True)
class SciFactLoadResult:
    documents: list[RawDocument]
    corpus_path: Path
    corpus_sha256: str

    @property
    def document_count(self) -> int:
        return len(self.documents)


class SciFactJsonlLoader:
    """Loader for the SciFact ``corpus.jsonl`` file.

    The loader is intentionally pure: no embedding, no chunking, no
    side-effects beyond reading the file.  Chunking is performed by a
    ``Chunker`` instance downstream so we can swap strategies via
    ``RAG_CHUNKER`` without rebuilding the loader (spec §28.1).
    """

    def __init__(self, corpus_path: Path) -> None:
        self.corpus_path = corpus_path

    def load(self) -> SciFactLoadResult:
        if not self.corpus_path.exists():
            raise FileNotFoundError(f"SciFact corpus not found: {self.corpus_path}")
        logger.info("Loading SciFact corpus from %s", self.corpus_path)
        documents: list[RawDocument] = []
        with self.corpus_path.open("r", encoding="utf-8") as handle:
            for line_index, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "Skipping malformed JSON at %s:%d (%s)",
                        self.corpus_path,
                        line_index,
                        exc,
                    )
                    continue
                doc_id = str(record.get("_id") or record.get("doc_id") or "").strip()
                if not doc_id:
                    continue
                documents.append(
                    RawDocument(
                        doc_id=doc_id,
                        title=str(record.get("title", "") or "").strip(),
                        text=str(record.get("text", "") or "").strip(),
                        source_uri=None,
                        metadata={"corpus": "scifact"},
                    )
                )
        logger.info("Loaded %s SciFact documents.", len(documents))
        return SciFactLoadResult(
            documents=documents,
            corpus_path=self.corpus_path,
            corpus_sha256=_sha256_file(self.corpus_path),
        )


def load_scifact_corpus(corpus_path: Path) -> SciFactLoadResult:
    """Convenience wrapper used by the SciFact retriever adapter."""
    return SciFactJsonlLoader(corpus_path).load()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# Convenience for callers that need an iterator view rather than a list.
def iter_scifact_documents(corpus_path: Path) -> Iterable[RawDocument]:
    return iter(load_scifact_corpus(corpus_path).documents)


__all__ = [
    "SciFactJsonlLoader",
    "SciFactLoadResult",
    "iter_scifact_documents",
    "load_scifact_corpus",
]
