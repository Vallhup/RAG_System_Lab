# Log Layout

Runtime logs are split by purpose.

```text
logs/app/app.log
logs/app/error.log
logs/indexing/index.log
logs/retrieval/retrieve.jsonl
logs/security/audit.jsonl
logs/errors/failures.jsonl
logs/evaluation/runs.jsonl
```

## Files

- `app/app.log`: server lifecycle and general application events
- `app/error.log`: error-level logs and stack traces
- `indexing/index.log`: corpus loading, BM25 preparation, LlamaIndex build/load events
- `retrieval/retrieve.jsonl`: structured query trace events for retrieval requests
- `security/audit.jsonl`: policy/config snapshot and security audit events
- `errors/failures.jsonl`: retriever-not-ready, exception, empty-result, and fallback-style failure events
- `evaluation/runs.jsonl`: reserved for evaluation run summaries

`retrieval/retrieve.jsonl` is intended for tuning and debugging. It records fields such as `query_id`, `trace_id`, `config.hash`, `policy.hash`, `mode`, `latency_ms`, `retrieval.top_doc_ids`, and `retrieval.scores`.

Raw retrieved context is not logged by default. Query text can be disabled with `RAG_LOG_RAW_QUERY=false`, in which case only `query_hash` is logged.

Log output files are ignored by git.
