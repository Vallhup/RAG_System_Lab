import hashlib
import json
import logging
import math
import os
import re
import threading
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_CORPUS_PATH = Path("data/scifact/corpus.jsonl")
DEFAULT_STORAGE_DIR = Path("storage/scifact")
DEFAULT_TOP_K = 10
MAX_TOP_K = 10
CONTEXT_LIMIT = 1500
HYBRID_CANDIDATE_MULTIPLIER = 5
RRF_K = 60
LLAMA_CHUNK_SIZE = 512
LLAMA_CHUNK_OVERLAP = 80
BM25_CHUNK_WORDS = 220
BM25_CHUNK_OVERLAP = 50
TOKEN_PATTERN = re.compile(r"[a-z0-9][a-z0-9\-]+")
index_logger = logging.getLogger("rag.index")
STOPWORDS = {
    "a",
    "about",
    "after",
    "also",
    "an",
    "among",
    "and",
    "are",
    "as",
    "at",
    "because",
    "been",
    "being",
    "between",
    "both",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "has",
    "have",
    "having",
    "in",
    "is",
    "it",
    "into",
    "may",
    "more",
    "no",
    "not",
    "of",
    "on",
    "or",
    "our",
    "than",
    "that",
    "the",
    "their",
    "there",
    "these",
    "this",
    "those",
    "through",
    "to",
    "using",
    "was",
    "we",
    "were",
    "when",
    "where",
    "which",
    "while",
    "with",
}


@dataclass(frozen=True)
class CorpusDocument:
    doc_id: str
    title: str
    text: str


@dataclass(frozen=True)
class BM25Chunk:
    doc_id: str
    chunk_id: str
    title: str
    text: str
    tokens: tuple[str, ...]
    token_counts: Counter[str]


class BM25FallbackRetriever:
    def __init__(self, documents: list[CorpusDocument]) -> None:
        self.documents = documents
        self.chunks = self._build_chunks()
        self._idf: dict[str, float] = {}
        self._avgdl = 0.0
        self._build_statistics()

    @property
    def ready(self) -> bool:
        return bool(self.chunks)

    def retrieve(self, question: str, top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
        query_tokens = _tokens(question)
        if not query_tokens:
            return []

        query_counter = Counter(query_tokens)
        ranked: list[tuple[float, BM25Chunk]] = []
        for chunk in self.chunks:
            score = self._bm25_score(query_counter, chunk)
            if score <= 0:
                continue
            score += self._title_boost(query_counter, chunk.title)
            ranked.append((score, chunk))

        ranked.sort(key=lambda item: item[0], reverse=True)
        return _dedupe_contexts(
            [
                {
                    "doc_id": chunk.doc_id,
                    "chunk_id": chunk.chunk_id,
                    "score": round(float(score), 6),
                    "text": _shorten(chunk.text),
                }
                for score, chunk in ranked
            ],
            top_k,
        )

    def _build_chunks(self) -> list[BM25Chunk]:
        chunks: list[BM25Chunk] = []
        chunk_words = int(os.getenv("RAG_BM25_CHUNK_WORDS", str(BM25_CHUNK_WORDS)))
        overlap = int(os.getenv("RAG_BM25_CHUNK_OVERLAP", str(BM25_CHUNK_OVERLAP)))
        for document in self.documents:
            for chunk_index, chunk_text in enumerate(
                _chunk_text(document.title, document.text, chunk_words, overlap)
            ):
                token_list = _tokens(f"{document.title} {chunk_text}")
                if not token_list:
                    continue
                chunks.append(
                    BM25Chunk(
                        doc_id=document.doc_id,
                        chunk_id=f"{document.doc_id}::bm25_chunk_{chunk_index:03d}",
                        title=document.title,
                        text=chunk_text,
                        tokens=tuple(token_list),
                        token_counts=Counter(token_list),
                    )
                )
        return chunks

    def _build_statistics(self) -> None:
        if not self.chunks:
            return

        document_frequency: Counter[str] = Counter()
        total_length = 0
        for chunk in self.chunks:
            document_frequency.update(set(chunk.tokens))
            total_length += len(chunk.tokens)

        chunk_count = len(self.chunks)
        self._avgdl = total_length / chunk_count
        self._idf = {
            token: math.log(1 + (chunk_count - freq + 0.5) / (freq + 0.5))
            for token, freq in document_frequency.items()
        }

    def _bm25_score(self, query_counter: Counter[str], chunk: BM25Chunk) -> float:
        k1 = 1.5
        b = 0.75
        chunk_length = len(chunk.tokens)
        score = 0.0
        for token, query_weight in query_counter.items():
            term_frequency = chunk.token_counts.get(token, 0)
            if term_frequency == 0:
                continue
            idf = self._idf.get(token, 0.0)
            denominator = term_frequency + k1 * (1 - b + b * chunk_length / self._avgdl)
            score += query_weight * idf * (term_frequency * (k1 + 1)) / denominator
        return score

    def _title_boost(self, query_counter: Counter[str], title: str) -> float:
        title_tokens = set(_tokens(title))
        if not title_tokens:
            return 0.0
        overlap = title_tokens & set(query_counter)
        return sum(self._idf.get(token, 0.0) for token in overlap) * 0.35


class SciFactLlamaRetriever:
    def __init__(self, corpus_path: Path, storage_dir: Path) -> None:
        self.corpus_path = corpus_path
        self.storage_dir = storage_dir
        self.documents: list[CorpusDocument] = []
        self.fallback: BM25FallbackRetriever | None = None
        self.index: Any | None = None
        self.ready = False
        self.mode = "uninitialized"
        self.error: str | None = None
        self.reload_fallback()

    def reload_fallback(self) -> None:
        try:
            index_logger.info("Loading SciFact corpus from %s", self.corpus_path)
            self.documents = load_corpus(self.corpus_path)
            index_logger.info("Loaded %s SciFact documents.", len(self.documents))
            self.fallback = BM25FallbackRetriever(self.documents)
            index_logger.info("BM25 fallback prepared with %s chunks.", len(self.fallback.chunks))
        except Exception as exc:
            self.ready = False
            self.mode = "error"
            self.error = str(exc)
            index_logger.exception("Failed to prepare BM25 fallback.")
            return

        self.ready = bool(self.fallback and self.fallback.ready)
        self.mode = "bm25_fallback"
        self.error = "LlamaIndex index is not ready yet; using BM25 fallback."

    def build_llama_index(self) -> None:
        try:
            index_logger.info("Building or loading LlamaIndex index from %s", self.storage_dir)
            self.index = self._build_or_load_index()
            self.ready = True
            self.mode = "llama_index"
            self.error = None
            index_logger.info("LlamaIndex index is ready.")
        except Exception as exc:
            self.index = None
            self.ready = bool(self.fallback and self.fallback.ready)
            self.mode = "bm25_fallback" if self.ready else "error"
            self.error = f"LlamaIndex unavailable; using BM25 fallback. Cause: {exc}"
            index_logger.exception("LlamaIndex unavailable; BM25 fallback remains active.")

    def retrieve(self, question: str, top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
        if self.index is not None:
            try:
                return self._retrieve_hybrid(question, top_k)
            except Exception as exc:
                self.mode = "bm25_fallback"
                self.error = f"LlamaIndex retrieval failed; using BM25 fallback. Cause: {exc}"

        if not self.fallback or not self.fallback.ready:
            raise RuntimeError(self.error or "SciFact retriever is not ready.")
        return self.fallback.retrieve(question, top_k)

    def _build_or_load_index(self):
        _configure_llama_index()
        metadata_path = self.storage_dir / "metadata.json"
        expected_metadata = self._index_metadata()

        if self._can_load_existing_index(metadata_path, expected_metadata):
            from llama_index.core import StorageContext, load_index_from_storage

            index_logger.info("Loading existing LlamaIndex index from %s", self.storage_dir)
            storage_context = StorageContext.from_defaults(persist_dir=str(self.storage_dir))
            return load_index_from_storage(storage_context)

        from llama_index.core import VectorStoreIndex
        from llama_index.core.node_parser import SentenceSplitter
        from llama_index.core.schema import Document

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        index_logger.info("Creating LlamaIndex documents for %s SciFact records.", len(self.documents))
        llama_documents = [
            Document(
                text=f"{document.title}\n\n{document.text}" if document.title else document.text,
                metadata={"doc_id": document.doc_id, "title": document.title},
            )
            for document in self.documents
        ]

        parser = SentenceSplitter(
            chunk_size=int(os.getenv("RAG_LLAMA_CHUNK_SIZE", str(LLAMA_CHUNK_SIZE))),
            chunk_overlap=int(os.getenv("RAG_LLAMA_CHUNK_OVERLAP", str(LLAMA_CHUNK_OVERLAP))),
        )
        nodes = parser.get_nodes_from_documents(llama_documents)
        index_logger.info("Created %s LlamaIndex nodes.", len(nodes))
        chunk_counts: Counter[str] = Counter()
        for node in nodes:
            doc_id = str(node.metadata.get("doc_id", ""))
            chunk_index = chunk_counts[doc_id]
            chunk_counts[doc_id] += 1
            node.id_ = f"{doc_id}::chunk_{chunk_index:03d}"

        index = VectorStoreIndex(nodes)
        index.storage_context.persist(persist_dir=str(self.storage_dir))
        metadata_path.write_text(
            json.dumps(expected_metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        index_logger.info("Persisted LlamaIndex index to %s", self.storage_dir)
        return index

    def _retrieve_with_llama(self, question: str, similarity_top_k: int) -> list[dict[str, Any]]:
        retriever = self.index.as_retriever(similarity_top_k=min(max(similarity_top_k, 1), 100))
        nodes = retriever.retrieve(question)
        contexts: list[dict[str, Any]] = []
        for node_with_score in nodes:
            node = node_with_score.node
            doc_id = str(node.metadata.get("doc_id", "")).strip()
            if not doc_id:
                continue
            score = node_with_score.score if node_with_score.score is not None else 0.0
            contexts.append(
                {
                    "doc_id": doc_id,
                    "chunk_id": str(node.node_id),
                    "score": round(float(score), 6),
                    "text": _shorten(node.get_content().strip()),
                }
            )
        return contexts

    def _retrieve_hybrid(self, question: str, top_k: int) -> list[dict[str, Any]]:
        candidate_count = _candidate_count(top_k)
        vector_contexts = self._retrieve_with_llama(question, candidate_count)
        bm25_contexts = self.fallback.retrieve(question, candidate_count) if self.fallback else []
        fused = _rrf_fuse(
            ranked_lists=[
                ("vector", vector_contexts, 1.0),
                ("bm25", bm25_contexts, 0.8),
            ],
            top_k=top_k,
        )
        return fused

    def _can_load_existing_index(self, metadata_path: Path, expected_metadata: dict[str, Any]) -> bool:
        if not self.storage_dir.exists() or not metadata_path.exists():
            return False
        try:
            stored_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False
        return stored_metadata == expected_metadata

    def _index_metadata(self) -> dict[str, Any]:
        return {
            "corpus_path": str(self.corpus_path),
            "corpus_sha256": _sha256_file(self.corpus_path),
            "document_count": len(self.documents),
            "chunk_size": int(os.getenv("RAG_LLAMA_CHUNK_SIZE", str(LLAMA_CHUNK_SIZE))),
            "chunk_overlap": int(os.getenv("RAG_LLAMA_CHUNK_OVERLAP", str(LLAMA_CHUNK_OVERLAP))),
            "embedding_model": os.getenv("RAG_EMBED_MODEL", "BAAI/bge-small-en-v1.5"),
        }


_retriever: SciFactLlamaRetriever | None = None
_retriever_initializing = False
_retriever_lock = threading.Lock()
_configured_llama = False


def get_scifact_retriever() -> SciFactLlamaRetriever:
    global _retriever
    if _retriever is None:
        corpus_path = Path(os.getenv("RAG_CORPUS_PATH", str(DEFAULT_CORPUS_PATH)))
        storage_dir = Path(os.getenv("RAG_STORAGE_DIR", str(DEFAULT_STORAGE_DIR)))
        _retriever = SciFactLlamaRetriever(corpus_path, storage_dir)
    return _retriever


def start_scifact_retriever_initialization() -> None:
    global _retriever_initializing
    with _retriever_lock:
        if _retriever is not None or _retriever_initializing:
            return
        _retriever_initializing = True

    thread = threading.Thread(target=_initialize_retriever, daemon=True)
    thread.start()


def peek_scifact_retriever() -> SciFactLlamaRetriever | None:
    with _retriever_lock:
        return _retriever


def scifact_status() -> dict[str, bool | str]:
    with _retriever_lock:
        retriever = _retriever
        initializing = _retriever_initializing

    if retriever is not None:
        return {"status": "ok" if retriever.ready else "error", "ready": retriever.ready}
    if initializing:
        return {"status": "ok", "ready": False}
    return {"status": "ok", "ready": False}


def _initialize_retriever() -> None:
    global _retriever, _retriever_initializing
    corpus_path = Path(os.getenv("RAG_CORPUS_PATH", str(DEFAULT_CORPUS_PATH)))
    storage_dir = Path(os.getenv("RAG_STORAGE_DIR", str(DEFAULT_STORAGE_DIR)))
    retriever = SciFactLlamaRetriever(corpus_path, storage_dir)

    with _retriever_lock:
        _retriever = retriever

    retriever.build_llama_index()

    with _retriever_lock:
        _retriever = retriever
        _retriever_initializing = False


def load_corpus(corpus_path: Path) -> list[CorpusDocument]:
    if not corpus_path.exists():
        raise FileNotFoundError(f"SciFact corpus not found: {corpus_path}")

    documents: list[CorpusDocument] = []
    with corpus_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}: {exc}") from exc

            doc_id = str(item.get("_id", "")).strip()
            title = str(item.get("title", "")).strip()
            text = str(item.get("text", "")).strip()
            if not doc_id or not text:
                continue
            documents.append(CorpusDocument(doc_id=doc_id, title=title, text=text))

    if not documents:
        raise ValueError(f"No valid SciFact documents found in {corpus_path}")
    return documents


def _configure_llama_index() -> None:
    global _configured_llama
    if _configured_llama:
        return

    if __package__:
        from .core import configure_settings
    else:
        from core import configure_settings

    configure_settings()
    _configured_llama = True


def _dedupe_contexts(contexts: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen_doc_ids: set[str] = set()
    for context in sorted(contexts, key=lambda item: item["score"], reverse=True):
        if context["doc_id"] in seen_doc_ids:
            continue
        seen_doc_ids.add(context["doc_id"])
        deduped.append(context)
        if len(deduped) >= min(top_k, MAX_TOP_K):
            break
    return deduped


def _rrf_fuse(
    ranked_lists: list[tuple[str, list[dict[str, Any]], float]],
    top_k: int,
) -> list[dict[str, Any]]:
    fused_scores: dict[str, float] = {}
    representatives: dict[str, dict[str, Any]] = {}
    representative_scores: dict[str, float] = {}

    for source, contexts, weight in ranked_lists:
        seen_in_list: set[str] = set()
        for rank, context in enumerate(contexts, start=1):
            doc_id = context["doc_id"]
            if doc_id in seen_in_list:
                continue
            seen_in_list.add(doc_id)

            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + weight / (RRF_K + rank)

            source_score = _representative_score(source, context)
            if doc_id not in representatives or source_score > representative_scores[doc_id]:
                representative = dict(context)
                representative["retrieval_source"] = source
                representatives[doc_id] = representative
                representative_scores[doc_id] = source_score

    ranked_doc_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)
    contexts: list[dict[str, Any]] = []
    for doc_id in ranked_doc_ids[: min(top_k, MAX_TOP_K)]:
        context = dict(representatives[doc_id])
        context["score"] = round(float(fused_scores[doc_id]), 6)
        context.pop("retrieval_source", None)
        contexts.append(context)
    return contexts


def _representative_score(source: str, context: dict[str, Any]) -> float:
    score = float(context.get("score", 0.0))
    if source == "vector":
        return score * 100.0
    return score


def _candidate_count(top_k: int) -> int:
    requested = max(top_k, 1) * int(os.getenv("RAG_HYBRID_CANDIDATE_MULTIPLIER", str(HYBRID_CANDIDATE_MULTIPLIER)))
    return max(min(requested, 100), min(top_k, MAX_TOP_K))


def _tokens(text: str) -> list[str]:
    return [
        token
        for token in TOKEN_PATTERN.findall(text.lower())
        if token not in STOPWORDS and len(token) > 1
    ]


def _chunk_text(title: str, text: str, chunk_words: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    if chunk_words <= 0:
        chunk_words = BM25_CHUNK_WORDS
    overlap = max(0, min(overlap, chunk_words - 1))
    step = chunk_words - overlap

    chunks: list[str] = []
    for start in range(0, len(words), step):
        part = " ".join(words[start : start + chunk_words]).strip()
        if not part:
            continue
        chunks.append(f"{title}\n{part}" if title else part)
        if start + chunk_words >= len(words):
            break
    return chunks


def _shorten(text: str, limit: int = CONTEXT_LIMIT) -> str:
    return text[:limit] + "..." if len(text) > limit else text


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
