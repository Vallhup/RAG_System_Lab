"""SciFact / Personal-Vault regression evaluation harness.

This module reads a golden query JSONL, runs each query through the configured
retriever, and produces a single EvaluationRun record on
``logs/evaluation/runs.jsonl``.

It is the Foundation Step 5 + 6 implementation described in the spec
``documents/개선 예정/para_aware_personal_graph_rag_runtime_spec.md`` §27 and
``documents/구현 스펙/logging_observability_runtime_spec.md``.

The harness supports two modes per query:

* label-aware — when the golden record has ``expected_doc_ids``, we compute
  hit@k / nDCG@k / MRR using a binary or graded relevance signal.
* label-free — when the golden record has no expected ids (the default for the
  personal vault where labels do not exist), we record only ``top_doc_ids``,
  score distribution stats, and latency.

In both cases the run-level summary is compared against the previous run in the
same JSONL using rank-biased overlap (RBO) and top-k Jaccard so that silent
regressions are still visible without labels. This follows the spec rule that
``EvaluationRun`` should remain useful "정답 label이 없는 경우에도".
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

if __package__:
    from .logging_config import configure_logging, log_evaluation_event
    from .retrieval import Tier, WorkMode
    from .runtime_config import RuntimeConfig, SecurityPolicy
    from .scifact import (
        DEFAULT_TOP_K,
        MAX_TOP_K,
        SciFactRetriever,
        get_scifact_retriever,
    )
else:  # script-style invocation: ``python app/evaluation.py``
    from logging_config import configure_logging, log_evaluation_event
    from retrieval import Tier, WorkMode  # type: ignore[no-redef]
    from runtime_config import RuntimeConfig, SecurityPolicy
    from scifact import (  # type: ignore[no-redef]
        DEFAULT_TOP_K,
        MAX_TOP_K,
        SciFactRetriever,
        get_scifact_retriever,
    )


DEFAULT_GOLDEN_SET = Path("tests/golden_queries/scifact.sample.jsonl")
DEFAULT_RUNS_LOG = Path(os.getenv("RAG_LOG_DIR", "logs")) / "evaluation" / "runs.jsonl"
RBO_PERSISTENCE = 0.9  # standard RBO p; emphasises early ranks


@dataclass(frozen=True)
class GoldenQuery:
    query_id: str
    corpus_id: str
    question: str
    expected_doc_ids: list[str]
    relevance: dict[str, int]
    notes: str
    tags: list[str]

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "GoldenQuery":
        expected = list(record.get("expected_doc_ids") or [])
        relevance = dict(record.get("relevance") or {})
        # default relevance = 1 for any expected doc without an explicit grade.
        for doc_id in expected:
            relevance.setdefault(doc_id, 1)
        return cls(
            query_id=str(record["query_id"]),
            corpus_id=str(record.get("corpus_id", "unknown")),
            question=str(record["question"]),
            expected_doc_ids=expected,
            relevance=relevance,
            notes=str(record.get("notes", "")),
            tags=list(record.get("tags") or []),
        )

    @property
    def is_labeled(self) -> bool:
        return bool(self.expected_doc_ids)


def load_golden_set(path: Path) -> list[GoldenQuery]:
    if not path.exists():
        raise FileNotFoundError(f"Golden set not found: {path}")
    queries: list[GoldenQuery] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{index + 1}: {exc}") from exc
        queries.append(GoldenQuery.from_record(record))
    return queries


def unique_doc_ids(contexts: Iterable[dict[str, Any]]) -> list[str]:
    """Spec §29.1: 같은 문서에서 여러 chunk가 검색되어도 평가는 문서 단위 rank."""
    seen: list[str] = []
    seen_set: set[str] = set()
    for ctx in contexts:
        doc_id = str(ctx.get("doc_id", "")).strip()
        if not doc_id or doc_id in seen_set:
            continue
        seen.append(doc_id)
        seen_set.add(doc_id)
    return seen


def hit_at_k(predicted: list[str], expected: set[str], k: int) -> float:
    if not expected:
        return 0.0
    top_k = predicted[:k]
    return 1.0 if any(d in expected for d in top_k) else 0.0


def reciprocal_rank(predicted: list[str], expected: set[str]) -> float:
    for index, doc_id in enumerate(predicted, start=1):
        if doc_id in expected:
            return 1.0 / index
    return 0.0


def ndcg_at_k(predicted: list[str], relevance: dict[str, int], k: int) -> float:
    if not relevance:
        return 0.0
    dcg = 0.0
    for index, doc_id in enumerate(predicted[:k], start=1):
        gain = relevance.get(doc_id, 0)
        if gain <= 0:
            continue
        dcg += (2 ** gain - 1) / math.log2(index + 1)
    ideal_grades = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 1) for i, g in enumerate(ideal_grades, start=1) if g > 0)
    return dcg / idcg if idcg > 0 else 0.0


def rank_biased_overlap(a: list[str], b: list[str], p: float = RBO_PERSISTENCE) -> float:
    """Webber, Moffat & Zobel (2010) — non-conjoint RBO with finite lists.

    Used to compare two runs' top_doc_ids without ground truth labels.
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    max_depth = max(len(a), len(b))
    set_a: set[str] = set()
    set_b: set[str] = set()
    rbo = 0.0
    for depth in range(1, max_depth + 1):
        if depth <= len(a):
            set_a.add(a[depth - 1])
        if depth <= len(b):
            set_b.add(b[depth - 1])
        overlap = len(set_a & set_b)
        agreement = overlap / depth
        rbo += (p ** (depth - 1)) * agreement
    return (1.0 - p) * rbo


def top_k_jaccard(a: list[str], b: list[str], k: int) -> float:
    set_a = set(a[:k])
    set_b = set(b[:k])
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else 0.0


def score_query(
    query: GoldenQuery,
    retriever: SciFactRetriever,
    top_k: int,
    *,
    tier: Tier | None = None,
    work_mode: WorkMode = WorkMode.RETRIEVAL_ONLY,
) -> dict[str, Any]:
    started = time.perf_counter()
    final_tier: str = ""
    layers_executed: list[str] = []
    reranked = False
    try:
        response = retriever.retrieve_full(
            query.question, top_k=top_k, tier=tier, work_mode=work_mode
        )
        contexts = [
            {"doc_id": ctx.doc_id, "chunk_id": ctx.chunk_id, "score": ctx.score, "text": ctx.text}
            for ctx in response.contexts
        ]
        final_tier = response.final_tier.name
        layers_executed = list(response.layers_executed)
        reranked = response.reranked
        error = None
    except Exception as exc:  # pragma: no cover - exercised by failure_event path
        contexts = []
        error = str(exc)
    latency_ms = round((time.perf_counter() - started) * 1000, 3)

    doc_ranking = unique_doc_ids(contexts)
    scores = [float(ctx.get("score", 0.0)) for ctx in contexts]
    expected_set = set(query.expected_doc_ids)

    summary: dict[str, Any] = {
        "query_id": query.query_id,
        "corpus_id": query.corpus_id,
        "top_doc_ids": doc_ranking[:top_k],
        "chunk_count": len(contexts),
        "doc_count": len(doc_ranking),
        "latency_ms": latency_ms,
        "score_stats": _score_stats(scores),
        "is_labeled": query.is_labeled,
        "final_tier": final_tier,
        "layers_executed": layers_executed,
        "reranked": reranked,
        "error": error,
    }
    if query.is_labeled:
        summary["metrics"] = {
            "hit@1": hit_at_k(doc_ranking, expected_set, 1),
            "hit@3": hit_at_k(doc_ranking, expected_set, 3),
            "hit@5": hit_at_k(doc_ranking, expected_set, 5),
            "hit@10": hit_at_k(doc_ranking, expected_set, 10),
            "ndcg@10": ndcg_at_k(doc_ranking, query.relevance, 10),
            "mrr": reciprocal_rank(doc_ranking, expected_set),
        }
    return summary


def _score_stats(scores: list[float]) -> dict[str, float]:
    if not scores:
        return {"count": 0}
    return {
        "count": len(scores),
        "max": round(max(scores), 6),
        "min": round(min(scores), 6),
        "median": round(statistics.median(scores), 6),
        "mean": round(statistics.fmean(scores), 6),
    }


def aggregate(per_query: list[dict[str, Any]]) -> dict[str, Any]:
    labeled = [q for q in per_query if q["is_labeled"] and not q["error"]]
    label_free = [q for q in per_query if not q["is_labeled"] and not q["error"]]
    failed = [q["query_id"] for q in per_query if q["error"]]
    metrics: dict[str, Any] = {
        "labeled_count": len(labeled),
        "label_free_count": len(label_free),
        "average_latency_ms": (
            round(statistics.fmean(q["latency_ms"] for q in per_query), 3)
            if per_query else 0.0
        ),
        "empty_result_count": sum(1 for q in per_query if q["chunk_count"] == 0),
    }
    if labeled:
        for key in ("hit@1", "hit@3", "hit@5", "hit@10", "ndcg@10", "mrr"):
            metrics[key] = round(
                statistics.fmean(q["metrics"][key] for q in labeled), 6
            )
    return {"metrics": metrics, "failed_query_ids": failed}


def load_previous_run(runs_path: Path, corpus_id: str, dataset_name: str) -> dict[str, Any] | None:
    if not runs_path.exists():
        return None
    last: dict[str, Any] | None = None
    for line in runs_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("corpus_id") == corpus_id and record.get("dataset_name") == dataset_name:
            last = record
    return last


def compare_runs(current_per_query: list[dict[str, Any]], previous: dict[str, Any]) -> dict[str, Any]:
    previous_map = {q["query_id"]: q for q in previous.get("per_query", [])}
    pairs: list[dict[str, Any]] = []
    rbo_scores: list[float] = []
    jaccard_scores: list[float] = []
    for current in current_per_query:
        prev = previous_map.get(current["query_id"])
        if prev is None:
            continue
        rbo = rank_biased_overlap(current["top_doc_ids"], prev["top_doc_ids"])
        jaccard = top_k_jaccard(current["top_doc_ids"], prev["top_doc_ids"], k=10)
        rbo_scores.append(rbo)
        jaccard_scores.append(jaccard)
        pairs.append(
            {
                "query_id": current["query_id"],
                "rbo": round(rbo, 6),
                "top10_jaccard": round(jaccard, 6),
            }
        )
    if not pairs:
        return {"compared_run_id": previous.get("run_id"), "pairs": []}
    return {
        "compared_run_id": previous.get("run_id"),
        "average_rbo": round(statistics.fmean(rbo_scores), 6),
        "average_top10_jaccard": round(statistics.fmean(jaccard_scores), 6),
        "min_rbo": round(min(rbo_scores), 6),
        "pairs": pairs,
    }


def build_run(
    golden_set_path: Path,
    queries: list[GoldenQuery],
    retriever: SciFactRetriever,
    top_k: int,
    dataset_name: str,
    code_version: str | None,
    *,
    tier: Tier | None = None,
    work_mode: WorkMode = WorkMode.RETRIEVAL_ONLY,
    variant_label: str = "",
) -> dict[str, Any]:
    runtime_config = RuntimeConfig.from_env()
    policy = SecurityPolicy.from_env()
    corpus_id = queries[0].corpus_id if queries else "unknown"

    per_query = [
        score_query(query, retriever, top_k, tier=tier, work_mode=work_mode)
        for query in queries
    ]
    aggregated = aggregate(per_query)

    variant_info = {
        "label": variant_label,
        "tier_max": (tier.name if tier else None),
        "work_mode": work_mode.value,
        "reranker_enabled": os.getenv("RAG_ENABLE_RERANKER", "false").strip().lower() in {"1", "true", "yes", "on"},
        "chunker": os.getenv("RAG_CHUNKER", "sentence_window"),
    }

    run_record: dict[str, Any] = {
        "schema.version": "1.1",
        "event_type": "evaluation.run_completed",
        "message": "Evaluation run completed",
        "run_id": uuid.uuid4().hex,
        "dataset_name": dataset_name,
        "dataset_path": str(golden_set_path),
        "corpus_id": corpus_id,
        "corpus_hash": _corpus_hash_for(retriever),
        "config_hash": runtime_config.hash,
        "policy_hash": policy.hash,
        "config": runtime_config.to_dict(),
        "policy": policy.to_dict(),
        "code_version": code_version or os.getenv("RAG_CODE_VERSION", "dev"),
        "retriever_mode": getattr(retriever, "mode", "unknown"),
        "variant": variant_info,
        "top_k": top_k,
        "query_count": len(queries),
        "metrics": aggregated["metrics"],
        "failed_query_ids": aggregated["failed_query_ids"],
        "per_query": per_query,
    }

    previous = load_previous_run(DEFAULT_RUNS_LOG, corpus_id, dataset_name)
    if previous is not None:
        run_record["diff_vs_previous"] = compare_runs(per_query, previous)
    return run_record


def _corpus_hash_for(retriever: SciFactRetriever) -> str:
    metadata = getattr(retriever, "_index_metadata", None)
    if callable(metadata):
        return str(metadata().get("corpus_sha256", ""))
    return ""


def run_cli(args: argparse.Namespace) -> int:
    configure_logging()
    golden_path = Path(args.golden)
    queries = load_golden_set(golden_path)
    if not queries:
        print(f"No queries loaded from {golden_path}", file=sys.stderr)
        return 1

    if args.matrix:
        return run_matrix(golden_path, queries, args)

    retriever = _prepare_retriever(args)
    if retriever is None:
        return 2

    tier = Tier.parse(args.tier_max) if args.tier_max else None
    work_mode = WorkMode.parse(args.work_mode) if args.work_mode else WorkMode.RETRIEVAL_ONLY

    run = build_run(
        golden_set_path=golden_path,
        queries=queries,
        retriever=retriever,
        top_k=args.top_k,
        dataset_name=args.dataset_name or golden_path.stem,
        code_version=args.code_version,
        tier=tier,
        work_mode=work_mode,
        variant_label=args.variant_label or "",
    )

    log_evaluation_event(run)
    _print_summary(run)
    return 0


def _prepare_retriever(args: argparse.Namespace) -> SciFactRetriever | None:
    retriever = get_scifact_retriever()
    if args.bm25_only:
        retriever.index = None
    elif args.require_hybrid:
        retriever.build_llama_index()
        if retriever.mode != "hybrid":
            print(
                "--require-hybrid requested but LlamaIndex retriever is not available",
                file=sys.stderr,
            )
            return None
    else:
        retriever.build_llama_index()
    return retriever


def run_matrix(
    golden_path: Path,
    queries: list[GoldenQuery],
    args: argparse.Namespace,
) -> int:
    """Run the five-variant baseline matrix described in the plan:

    1. Tier 1 BM25 only
    2. Tier 2 hybrid (RRF) — current SciFact baseline
    3. Tier 2 hybrid + abstract_one_node chunker
    4. Tier 3 hybrid + cross-encoder reranker
    5. Tier 3 + abstract_one_node chunker (combo)

    Each variant gets its own EvaluationRun record so the user can compare
    nDCG@10 / hit@5 / RBO directly inside ``logs/evaluation/runs.jsonl``.
    """
    variants = [
        {"label": "tier1_bm25_only", "tier_max": "FAST", "bm25_only": True, "chunker": None, "reranker": "false"},
        {"label": "tier2_hybrid_baseline", "tier_max": "HYBRID", "bm25_only": False, "chunker": "sentence_window", "reranker": "false"},
        {"label": "tier2_hybrid_abstract", "tier_max": "HYBRID", "bm25_only": False, "chunker": "abstract_one_node", "reranker": "false"},
        {"label": "tier3_reranker", "tier_max": "GRAPH_AUGMENTED", "bm25_only": False, "chunker": "sentence_window", "reranker": "true"},
        {"label": "tier3_reranker_abstract", "tier_max": "GRAPH_AUGMENTED", "bm25_only": False, "chunker": "abstract_one_node", "reranker": "true"},
    ]
    summaries: list[dict[str, Any]] = []
    for variant in variants:
        print(f"\n========= variant: {variant['label']} =========")
        # mutate env so a freshly-built retriever picks the right knobs.
        if variant["chunker"] is not None:
            os.environ["RAG_CHUNKER"] = variant["chunker"]
        os.environ["RAG_ENABLE_RERANKER"] = variant["reranker"]
        # force singleton rebuild by clearing the cached retriever
        from . import scifact as scifact_module  # type: ignore[import-not-found]
        scifact_module._retriever = None
        retriever = get_scifact_retriever()
        if variant["bm25_only"]:
            retriever.index = None
        else:
            retriever.build_llama_index()
        run = build_run(
            golden_set_path=golden_path,
            queries=queries,
            retriever=retriever,
            top_k=args.top_k,
            dataset_name=(args.dataset_name or golden_path.stem),
            code_version=args.code_version,
            tier=Tier.parse(variant["tier_max"]),
            work_mode=WorkMode.RETRIEVAL_ONLY,
            variant_label=variant["label"],
        )
        log_evaluation_event(run)
        _print_summary(run)
        summaries.append(
            {
                "label": variant["label"],
                "metrics": run["metrics"],
                "retriever_mode": run["retriever_mode"],
            }
        )
    _print_matrix_summary(summaries)
    return 0


def _print_matrix_summary(summaries: list[dict[str, Any]]) -> None:
    print("\n========= matrix summary =========")
    headers = ["variant", "mode", "hit@1", "hit@5", "ndcg@10", "mrr", "latency_ms"]
    print(" | ".join(f"{h:>22s}" for h in headers))
    for entry in summaries:
        metrics = entry["metrics"]
        row = [
            entry["label"],
            entry["retriever_mode"],
            f"{metrics.get('hit@1', '-')}",
            f"{metrics.get('hit@5', '-')}",
            f"{metrics.get('ndcg@10', '-')}",
            f"{metrics.get('mrr', '-')}",
            f"{metrics.get('average_latency_ms', '-')}",
        ]
        print(" | ".join(f"{c:>22s}" for c in row))


def _print_summary(run: dict[str, Any]) -> None:
    metrics = run["metrics"]
    diff = run.get("diff_vs_previous") or {}
    print()
    print(f"run_id          : {run['run_id']}")
    print(f"corpus_id       : {run['corpus_id']}")
    print(f"dataset_name    : {run['dataset_name']}")
    print(f"retriever_mode  : {run['retriever_mode']}")
    print(f"config_hash     : {run['config_hash']}")
    print(f"policy_hash     : {run['policy_hash']}")
    print(f"corpus_hash     : {run['corpus_hash']}")
    print(f"query_count     : {run['query_count']}  (failed: {len(run['failed_query_ids'])})")
    print()
    print("--- aggregated metrics ---")
    for key, value in metrics.items():
        print(f"  {key:>22s} : {value}")
    if diff:
        print()
        print("--- diff vs previous run ---")
        print(f"  compared_run_id      : {diff.get('compared_run_id')}")
        print(f"  average_rbo          : {diff.get('average_rbo')}")
        print(f"  average_top10_jaccard: {diff.get('average_top10_jaccard')}")
        print(f"  min_rbo              : {diff.get('min_rbo')}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regression evaluation harness for SciFact / Personal Vault.")
    parser.add_argument("--golden", default=str(DEFAULT_GOLDEN_SET), help="Path to golden query JSONL.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Top-K passed to retriever.")
    parser.add_argument("--dataset-name", default=None, help="Override dataset name (defaults to golden file stem).")
    parser.add_argument("--code-version", default=None, help="Optional code version label for the run record.")
    parser.add_argument("--bm25-only", action="store_true", help="Force BM25 fallback (skip LlamaIndex).")
    parser.add_argument("--require-hybrid", action="store_true", help="Fail if LlamaIndex hybrid retriever isn't available.")
    parser.add_argument("--tier-max", default=None, help="Cap tier escalation (FAST/HYBRID/GRAPH_AUGMENTED/FULL_EVAL).")
    parser.add_argument("--work-mode", default=None, help="WorkMode (retrieval_only/recall/research/...).")
    parser.add_argument("--variant-label", default=None, help="Free-form label for this run record.")
    parser.add_argument("--matrix", action="store_true", help="Run the 5-variant baseline comparison matrix.")
    args = parser.parse_args(argv)
    if args.top_k < 1 or args.top_k > MAX_TOP_K:
        parser.error(f"--top-k must be between 1 and {MAX_TOP_K}")
    if args.bm25_only and args.require_hybrid:
        parser.error("--bm25-only and --require-hybrid are mutually exclusive")
    return args


def main(argv: list[str] | None = None) -> int:
    return run_cli(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
