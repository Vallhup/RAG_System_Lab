# Data Layout

## Active Corpus

`data/scifact/corpus.jsonl` is the active corpus for the SciFact retrieval API and the live evaluation endpoint.

Each line must be a JSON object with:

- `_id`: original document ID used as `doc_id`
- `title`: document title
- `text`: document text

## Practice Data

`data/practice/` contains files from earlier labs. They are kept for reference and are not used by the current SciFact `/retrieve` endpoint.

## Notes

- Do not mix generated indexes into `data/`.
- If `data/scifact/corpus.jsonl` changes, the SciFact index metadata changes and the index will be rebuilt on the next run.
