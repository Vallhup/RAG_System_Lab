"""SciFact retriever adapter.

Historically this module contained ~580 lines of BM25 / LlamaIndex / RRF
logic.  After the Tier Runtime refactor (spec §10, §28) the implementation
lives under ``app/retrieval/`` as a corpus-agnostic pipeline.  This file
keeps the module name and public symbols that ``app/main.py`` depended on
so the FastAPI surface is untouched and the SciFact ``/retrieve`` contract
stays identical.

Responsibilities:

* hold the singleton ``CorpusRuntime`` for the SciFact corpus
* expose ``ready``/``mode``/``retrieve``/``error`` in the shape the
  observability layer expects
* perform the lazy LlamaIndex build on a background thread (so ``/health``
  returns fast on first hit)
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any

if __package__:
    from .retrieval import RetrievalRequest, Tier, WorkMode
    from .retrieval.runtime_factory import CorpusRuntime, build_scifact_runtime
    from .retrieval.tier_runtime import policy_for
else:  # pragma: no cover - Docker flat layout
    from retrieval import RetrievalRequest, Tier, WorkMode  # type: ignore[no-redef]
    from retrieval.runtime_factory import CorpusRuntime, build_scifact_runtime  # type: ignore[no-redef]
    from retrieval.tier_runtime import policy_for  # type: ignore[no-redef]


index_logger = logging.getLogger("rag.index")


DEFAULT_CORPUS_ID = "scifact"
DEFAULT_CORPUS_PATH = Path("data/scifact/corpus.jsonl")
DEFAULT_STORAGE_DIR = Path("storage/scifact")
DEFAULT_TOP_K = 10
MAX_TOP_K = 10


class SciFactRetriever:
    """Thin wrapper around :class:`CorpusRuntime` that ``main.py`` uses."""

    def __init__(self, corpus_path: Path, storage_dir: Path) -> None:
        self.corpus_path = corpus_path
        self.storage_dir = storage_dir
        self._runtime: CorpusRuntime | None = None
        self._error: str | None = None
        self.reload_fallback()

    @property
    def ready(self) -> bool:
        return self._runtime is not None and self._runtime.ready

    @property
    def mode(self) -> str:
        if self._runtime is None:
            return "error"
        return self._runtime.mode

    @property
    def error(self) -> str | None:
        if self._runtime is None:
            return self._error
        vec = self._runtime.vector
        if vec is not None and vec.error and not vec.ready:
            return vec.error
        return None

    @property
    def index(self) -> Any:
        if self._runtime is None or self._runtime.vector is None:
            return None
        return self._runtime.vector._index  # type: ignore[attr-defined]

    @index.setter
    def index(self, value: Any) -> None:
        # used by ``evaluation.py --bm25-only`` to force BM25-only retrieval
        if self._runtime is not None and self._runtime.vector is not None:
            self._runtime.vector._index = value  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reload_fallback(self) -> None:
        """Build the corpus runtime up to Tier 2 readiness (BM25 + Vector slot).

        The vector layer is *constructed* here but not built.  That keeps
        the call cheap so the API can answer ``/health`` immediately.
        """
        try:
            index_logger.info("Building SciFact runtime from %s", self.corpus_path)
            enable_reranker_env = os.getenv("RAG_ENABLE_RERANKER")
            enable_reranker = (
                None if enable_reranker_env is None
                else enable_reranker_env.strip().lower() in {"1", "true", "yes", "on"}
            )
            self._runtime = build_scifact_runtime(
                corpus_path=self.corpus_path,
                storage_dir=self.storage_dir,
                enable_reranker=enable_reranker,
            )
            self._error = (
                None if self._runtime.vector and self._runtime.vector.ready
                else "LlamaIndex index is not ready yet; using BM25 fallback."
            )
            index_logger.info("SciFact runtime ready (mode=%s).", self._runtime.mode)
        except Exception as exc:
            self._runtime = None
            self._error = str(exc)
            index_logger.exception("Failed to build SciFact runtime.")

    def build_llama_index(self) -> None:
        """Trigger Vector index build (LlamaIndex). May download the HF model."""
        if self._runtime is None or self._runtime.vector is None:
            return
        try:
            index_logger.info("Building LlamaIndex vector index in %s", self.storage_dir)
            self._runtime.vector.build()
            if self._runtime.vector.ready:
                self._error = None
            else:
                self._error = self._runtime.vector.error or self._error
        except Exception as exc:  # pragma: no cover - logged downstream
            self._error = f"LlamaIndex unavailable; using BM25 fallback. Cause: {exc}"
            index_logger.exception("LlamaIndex build failed.")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, question: str, top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
        if self._runtime is None:
            raise RuntimeError(self._error or "SciFact retriever is not ready.")
        return self._runtime.retrieve(question, top_k=top_k)

    def retrieve_full(
        self,
        question: str,
        *,
        top_k: int = DEFAULT_TOP_K,
        tier: Tier | None = None,
        work_mode: WorkMode = WorkMode.RETRIEVAL_ONLY,
    ) -> Any:
        """Used by the evaluation harness to access the rich RetrievalResponse."""
        if self._runtime is None:
            raise RuntimeError(self._error or "SciFact retriever is not ready.")
        policy = policy_for(work_mode)
        request = RetrievalRequest(
            query=question,
            corpus_id=self._runtime.corpus_id,
            work_mode=work_mode,
            initial_tier=tier or policy.initial_tier,
            top_k=int(top_k),
        )
        return self._runtime.retrieve(request)

    def _index_metadata(self) -> dict[str, Any]:
        if self._runtime is None:
            return {}
        return self._runtime.index_metadata()


# ---------------------------------------------------------------------------
# Singleton management
# ---------------------------------------------------------------------------


_retriever: SciFactRetriever | None = None
_retriever_initializing = False
_retriever_lock = threading.Lock()


def get_scifact_retriever() -> SciFactRetriever:
    global _retriever
    if _retriever is None:
        corpus_path = Path(os.getenv("RAG_CORPUS_PATH", str(DEFAULT_CORPUS_PATH)))
        storage_dir = Path(os.getenv("RAG_STORAGE_DIR", str(DEFAULT_STORAGE_DIR)))
        _retriever = SciFactRetriever(corpus_path, storage_dir)
    return _retriever


def start_scifact_retriever_initialization() -> None:
    """Spin up the LlamaIndex build on a background thread.

    Called from FastAPI startup and from ``/health`` so the first incoming
    request triggers the build without blocking.  Thread-safe.
    """
    global _retriever_initializing
    with _retriever_lock:
        if _retriever is not None and (
            _retriever.mode == "hybrid"
            or (_retriever.mode == "bm25_fallback" and _retriever_initializing)
        ):
            return
        if _retriever is None:
            get_scifact_retriever()
        if _retriever_initializing:
            return
        _retriever_initializing = True

    thread = threading.Thread(target=_initialize_retriever, daemon=True)
    thread.start()


def peek_scifact_retriever() -> SciFactRetriever | None:
    with _retriever_lock:
        return _retriever


def scifact_status() -> dict[str, bool | str]:
    retriever = peek_scifact_retriever()
    if retriever is None:
        return {"status": "starting", "ready": False}
    return {
        "status": "ok" if retriever.ready else "starting",
        "ready": bool(retriever.ready),
        "mode": retriever.mode,
        "error": retriever.error or "",
    }


def _initialize_retriever() -> None:
    global _retriever_initializing
    try:
        retriever = get_scifact_retriever()
        retriever.build_llama_index()
    finally:
        with _retriever_lock:
            _retriever_initializing = False


__all__ = [
    "DEFAULT_TOP_K",
    "MAX_TOP_K",
    "SciFactRetriever",
    "get_scifact_retriever",
    "peek_scifact_retriever",
    "scifact_status",
    "start_scifact_retriever_initialization",
]
