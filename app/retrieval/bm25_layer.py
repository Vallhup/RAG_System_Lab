"""Pure-Python BM25 retrieval layer.

This module is the corpus-agnostic version of the BM25 fallback that
previously lived inside ``app/scifact.py``.  It accepts any iterable of
``Chunk`` records and exposes the ``RetrievalLayer`` Protocol so it can be
plugged into the Tier Runtime as either a standalone Tier-1 retriever or as
one half of a hybrid Tier-2 retriever (spec §28.3).

The intent is to remain dependency-free (no rank-bm25, no LlamaIndex) so it
keeps working as a fallback even when the optional vector backend is
unavailable (spec §21.1).
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable

if __package__:
    from . import Context, RetrievalLayer, RetrievalRequest, TierBudget
    from .chunkers import Chunk
else:
    from app.retrieval import Context, RetrievalLayer, RetrievalRequest, TierBudget  # type: ignore[no-redef]
    from app.retrieval.chunkers import Chunk  # type: ignore[no-redef]


logger = logging.getLogger("rag.bm25")
TOKEN_PATTERN = re.compile(r"[a-z0-9][a-z0-9\-]+")
DEFAULT_K1 = 1.5
DEFAULT_B = 0.75
TITLE_BOOST_WEIGHT = 0.35

# Conservative English stopword list — same as the previous SciFact fallback.
STOPWORDS = frozenset({
    "a", "about", "after", "also", "an", "among", "and", "are", "as", "at",
    "because", "been", "being", "between", "both", "by", "can", "could",
    "did", "do", "does", "for", "from", "has", "have", "having", "in", "is",
    "it", "into", "may", "more", "no", "not", "of", "on", "or", "our",
    "than", "that", "the", "their", "there", "these", "this", "those",
    "through", "to", "using", "was", "we", "were", "when", "where", "which",
    "while", "with",
})


def tokenize(text: str) -> list[str]:
    return [tok for tok in TOKEN_PATTERN.findall(text.lower()) if tok not in STOPWORDS]


@dataclass(frozen=True)
class BM25IndexEntry:
    chunk: Chunk
    tokens: tuple[str, ...]
    token_counts: Counter[str]
    length: int


@dataclass
class BM25Statistics:
    idf: dict[str, float] = field(default_factory=dict)
    avgdl: float = 0.0


class BM25Layer:
    """Stateful BM25 layer indexed over a fixed chunk set.

    Construction is O(N) over corpus tokens and only happens once per corpus
    snapshot.  Each query is O(K * V) where K is candidate breadth and V is
    vocabulary intersection — fast enough that we can run it on every
    request as either Tier 1 or part of Tier 2.
    """

    name = "bm25"

    def __init__(self, chunks: Iterable[Chunk]) -> None:
        self.entries: list[BM25IndexEntry] = []
        for chunk in chunks:
            tokens = tokenize(f"{chunk.title}\n{chunk.text}" if chunk.title else chunk.text)
            if not tokens:
                continue
            self.entries.append(
                BM25IndexEntry(
                    chunk=chunk,
                    tokens=tuple(tokens),
                    token_counts=Counter(tokens),
                    length=len(tokens),
                )
            )
        self.stats = self._build_statistics()

    @property
    def ready(self) -> bool:
        return bool(self.entries)

    def execute(
        self,
        request: RetrievalRequest,
        budget: TierBudget,
        carry: list[Context],
    ) -> list[Context]:
        """``RetrievalLayer`` entry point.

        ``carry`` is the running set of candidates produced by earlier
        layers; BM25 produces its own ranking and the fusion layer above
        merges them.  When used as Tier 1 the carry is empty and we return
        the BM25 ranking directly.
        """
        if not self.ready:
            return carry
        ranked = self.search(request.query, top_k=budget.candidate_count)
        return [
            Context(
                doc_id=hit.chunk.doc_id,
                chunk_id=hit.chunk.chunk_id,
                score=hit.score,
                text=hit.chunk.text,
                source_uri=getattr(hit.chunk, "source_uri", None),
                layer_origins=("bm25",),
                metadata={"layer": self.name},
            )
            for hit in ranked
        ]

    # ------------------------------------------------------------------
    # Search core
    # ------------------------------------------------------------------

    @dataclass(frozen=True)
    class Hit:
        chunk: Chunk
        score: float

    def search(self, query: str, top_k: int) -> list["BM25Layer.Hit"]:
        query_tokens = tokenize(query)
        if not query_tokens or not self.entries:
            return []
        query_counter = Counter(query_tokens)
        scored: list[BM25Layer.Hit] = []
        for entry in self.entries:
            score = self._bm25_score(query_counter, entry)
            if score <= 0:
                continue
            score += self._title_boost(query_counter, entry.chunk.title)
            scored.append(BM25Layer.Hit(chunk=entry.chunk, score=round(float(score), 6)))
        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[:top_k]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_statistics(self) -> BM25Statistics:
        stats = BM25Statistics()
        if not self.entries:
            return stats
        document_frequency: Counter[str] = Counter()
        total_length = 0
        for entry in self.entries:
            document_frequency.update(set(entry.tokens))
            total_length += entry.length
        chunk_count = len(self.entries)
        stats.avgdl = total_length / chunk_count if chunk_count else 0.0
        stats.idf = {
            token: math.log(1 + (chunk_count - freq + 0.5) / (freq + 0.5))
            for token, freq in document_frequency.items()
        }
        return stats

    def _bm25_score(self, query_counter: Counter[str], entry: BM25IndexEntry) -> float:
        if self.stats.avgdl <= 0:
            return 0.0
        score = 0.0
        for token, query_weight in query_counter.items():
            tf = entry.token_counts.get(token, 0)
            if tf == 0:
                continue
            idf = self.stats.idf.get(token, 0.0)
            denominator = tf + DEFAULT_K1 * (1 - DEFAULT_B + DEFAULT_B * entry.length / self.stats.avgdl)
            score += query_weight * idf * (tf * (DEFAULT_K1 + 1)) / denominator
        return score

    def _title_boost(self, query_counter: Counter[str], title: str) -> float:
        title_tokens = set(tokenize(title))
        if not title_tokens:
            return 0.0
        overlap = title_tokens & set(query_counter)
        return sum(self.stats.idf.get(token, 0.0) for token in overlap) * TITLE_BOOST_WEIGHT


__all__ = ["BM25Layer", "BM25IndexEntry", "BM25Statistics", "tokenize"]
