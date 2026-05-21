# Log Layout

Runtime logs are split by purpose.

```text
logs/app/app.log
logs/app/error.log
logs/indexing/index.log
logs/retrieval/retrieve.jsonl
```

## Files

- `app/app.log`: server lifecycle and general application events
- `app/error.log`: error-level logs and stack traces
- `indexing/index.log`: corpus loading, BM25 preparation, LlamaIndex build/load events
- `retrieval/retrieve.jsonl`: one JSON object per retrieval request

`retrieval/retrieve.jsonl` is intended for tuning and debugging. It records fields such as `query_id`, `mode`, `latency_ms`, `top_doc_ids`, and `scores`.

Log output files are ignored by git.
