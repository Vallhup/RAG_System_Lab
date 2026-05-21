# Storage Layout

Generated retrieval indexes live under `storage/`.

```text
storage/scifact/
```

`storage/scifact/` is the active LlamaIndex persistence directory for the SciFact corpus.

The root-level legacy index files from earlier labs are intentionally removed. The current application does not load indexes from `storage/` directly; it uses `RAG_STORAGE_DIR`, which defaults to `storage/scifact`.

Generated index files are ignored by git. They can be rebuilt from `data/scifact/corpus.jsonl`.
