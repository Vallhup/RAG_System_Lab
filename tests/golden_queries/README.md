# Golden Query Sets

본 디렉터리는 RAG 회귀 비교용 **자체 query 세트**를 저장합니다.

## 목적

`app/evaluation.py` 회귀 하니스가 검색 설정·코드 변경 전후를 비교할 때 사용합니다.

## 실습 7 명세와의 관계

실습 7 평가 서버가 보내는 query와 정답 qrels는 비공개입니다. 따라서 본 디렉터리의 query는 평가 서버의 query를 모방하지 않으며, 절대 `/retrieve` endpoint 응답을 사전 계산해 하드코딩하지 않습니다.

여기 있는 `expected_doc_ids`는 corpus를 직접 살펴서 **본인이 회귀 감지용으로 라벨링한 값**이며 회귀 하니스(`app/evaluation.py`) 안에서만 사용됩니다. 평가 서버는 절대 이 파일을 보지 않습니다.

## 파일 명명 규칙

```text
{corpus_id}.{set_name}.jsonl
```

- `scifact.sample.jsonl` — SciFact corpus 회귀용 mini gold (현재)
- `personal_vault.smoke.jsonl` — 개인 Vault namespace 추가 후 사용 (향후)

## Schema

각 줄은 다음 키를 가지는 JSON object입니다.

| key | type | required | description |
|---|---|---|---|
| `query_id` | str | yes | 고유 ID. 회귀 비교 키로 사용 |
| `corpus_id` | str | yes | `scifact`, `personal_vault` 등 namespace |
| `question` | str | yes | retrieval에 보낼 문장 |
| `expected_doc_ids` | list[str] | optional | 라벨링된 정답 doc_id (label-aware 모드용). 없으면 label-free 평가만 수행 |
| `relevance` | dict[str, int] | optional | doc_id → relevance 등급 (nDCG 계산용). 없으면 expected_doc_ids는 등급 1로 처리 |
| `notes` | str | optional | 왜 이 doc을 expected로 둔지 인간이 읽는 메모 |
| `tags` | list[str] | optional | 도메인/유형 태그 |
| `work_mode` | str | optional | "Research" 등 본 spec 9장 work mode hint (향후 personal vault 용) |

## Label-free 모드

`expected_doc_ids`가 없으면 하니스는 다음만 계산합니다.

- `top_doc_ids` snapshot
- `score` 분포 stats (min/max/median)
- `latency_ms`
- 이전 run 대비 `rank_biased_overlap` (RBO) / `top_k_jaccard`

개인 Vault 처럼 라벨이 없는 corpus는 이 모드로 회귀를 잡습니다.
