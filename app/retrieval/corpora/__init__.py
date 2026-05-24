"""Corpus loaders.

Each loader produces a list of :class:`app.retrieval.chunkers.RawDocument`
records.  The retrieval layers (BM25, Vector, RRF, reranker) consume the
resulting chunks without ever touching the corpus-specific JSON shape, so
adding a personal Vault loader later is a pure file addition (spec §3.2,
§30.2).
"""

from __future__ import annotations

from .scifact_loader import SciFactJsonlLoader, load_scifact_corpus

__all__ = ["SciFactJsonlLoader", "load_scifact_corpus"]
