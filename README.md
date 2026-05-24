# SciFact Retrieval API

LlamaIndex 기반으로 SciFact `corpus.jsonl` 문서를 검색해 평가 서버가 요구하는 `doc_id` ranked list를 반환하는 FastAPI 서버입니다. 답변 생성은 하지 않고, `POST /retrieve`에서 검색 context만 반환합니다.

## 실행 방식

Docker 기준 실행:

```bash
docker compose -p ragapi up --build
```

로컬 실행:

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Cloudflare Quick Tunnel:

```bash
cloudflared tunnel --url http://127.0.0.1:8000
```

`rag_endpoint.json`의 `api_base_url`에는 Cloudflare base URL만 넣고 `/health`, `/retrieve`는 붙이지 않습니다.

## 데이터

평가 대상 corpus는 다음 경로에 있어야 합니다.

```text
data/scifact/corpus.jsonl
```

Docker 이미지에서는 `WORKDIR /app` 기준으로 `/app/data/scifact/corpus.jsonl`을 읽습니다. 경로를 바꾸려면 `RAG_CORPUS_PATH` 환경변수를 설정합니다. LlamaIndex 인덱스는 기본적으로 `/app/storage/scifact`에 생성됩니다.

이전 실습 데이터는 `data/practice/`에 보관하며 현재 SciFact retrieval endpoint에서는 사용하지 않습니다.

## 로그와 저장소

런타임 로그는 목적별로 분리됩니다.

```text
logs/app/app.log
logs/app/error.log
logs/indexing/index.log
logs/retrieval/retrieve.jsonl
logs/security/audit.jsonl
logs/errors/failures.jsonl
logs/evaluation/runs.jsonl
```

`logs/retrieval/retrieve.jsonl`에는 `query_id`, `trace_id`, `config.hash`, `policy.hash`, retrieval `mode`, latency, top `doc_id`, score가 JSON Lines 형식으로 기록됩니다. Hybrid retrieval 튜닝과 장애 분석에 사용합니다.

`logs/security/audit.jsonl`에는 런타임 설정/보안 정책 snapshot과 policy decision이 기록됩니다. `logs/errors/failures.jsonl`에는 retriever 초기화 지연, 예외, empty result 같은 실패 이벤트가 분리되어 저장됩니다. raw retrieved context는 기본적으로 로그에 저장하지 않습니다.

기본 설정 예시는 다음 파일에 있습니다.

```text
config/runtime.default.yaml
config/security.default.yaml
```

현재 인덱스 저장소는 `storage/scifact/`만 사용합니다. 이전 실습에서 생성된 `storage/` 루트의 레거시 인덱스는 삭제했습니다.

`storage/scifact/metadata.json`은 corpus sha256과 함께 `corpus_id`, `config_hash`, `policy_hash`도 포함합니다. 설정 환경변수 하나만 바꿔도 cache가 자동으로 무효화되므로 회귀 실험 시 stale index가 끼어드는 것을 막습니다.

## 회귀 평가 (regression evaluation)

`app/evaluation.py`는 본 spec §27 Foundation Step 5 + 6 구현입니다. mini gold query 세트(`tests/golden_queries/scifact.sample.jsonl`)를 retriever에 흘려 보내고 한 줄짜리 `EvaluationRun` record를 `logs/evaluation/runs.jsonl`에 추가합니다.

Docker 환경에서 한 번 실행 (Tier 2 hybrid baseline):

```bash
docker compose -p ragapi --profile eval run --rm eval
```

5-variant 비교 매트릭스(BM25 단독 / Hybrid baseline / Hybrid + abstract chunker / Tier 3 reranker / Tier 3 reranker + abstract):

```bash
docker compose -p ragapi --profile eval run --rm eval-matrix
```

Reranker만 켠 단일 run (점수 비교용):

```bash
docker compose -p ragapi --profile eval run --rm eval-reranker
```

LlamaIndex 빌드 없이 BM25 fallback만으로 빠르게 확인:

```bash
docker compose -p ragapi run --rm rag python -m evaluation --bm25-only --top-k 10
```

`evaluation.py`의 주요 옵션:

```text
--tier-max FAST | HYBRID | GRAPH_AUGMENTED | FULL_EVAL
--work-mode retrieval_only | recall | research | ...
--variant-label <label>      # run record에 라벨 부착
--matrix                     # 5 variant 자동 비교
--bm25-only / --require-hybrid
```

산출되는 record는 다음을 포함합니다.

- `run_id`, `dataset_name`, `corpus_id`, `corpus_hash`, `config_hash`, `policy_hash`, `retriever_mode`
- `metrics`: `hit@1/3/5/10`, `ndcg@10`, `mrr`, `average_latency_ms`, `empty_result_count`
- `per_query`: 각 query의 `top_doc_ids`, `score_stats`, `latency_ms`, (라벨이 있으면) per-query metric
- `diff_vs_previous`: 같은 corpus + dataset의 직전 run과의 RBO / top-10 Jaccard 비교

라벨이 없는 query(개인 Vault 같은 경우)는 자동으로 label-free 모드가 되어 `top_doc_ids` snapshot과 분포 통계만 기록합니다. golden 파일 schema는 `tests/golden_queries/README.md`를 참고하세요.

평가 서버가 보내는 비공개 query를 모방하지 않으며, 본 회귀 세트는 본인이 corpus를 보고 직접 만든 query 만을 사용합니다(실습 7 명세의 "정답 하드코딩 금지" 조항 준수).

## API

### `GET /health`

SciFact corpus를 읽고 chunk index를 만들 수 있으면 다음 형태로 응답합니다.

```json
{
  "status": "ok",
  "ready": true
}
```

### `POST /retrieve`

요청:

```json
{
  "query_id": "selfcheck_001",
  "question": "scientific claim text for testing",
  "top_k": 10
}
```

응답:

```json
{
  "query_id": "selfcheck_001",
  "contexts": [
    {
      "doc_id": "31715818",
      "chunk_id": "31715818::chunk_000",
      "score": 12.345678,
      "text": "retrieved chunk text"
    }
  ]
}
```

# SciFact Retrieval Report

## 실행 방식
- 제출 코드 위치: `app/main.py`, `app/scifact.py`, `app/core.py`
- RAG 서버 실행 방식: Docker Compose
- 서버 실행 명령: `docker compose -p ragapi up --build`
- Cloudflare Quick Tunnel URL: `rag_endpoint.json`의 `api_base_url` 값
- 사용한 retriever/index: LlamaIndex `VectorStoreIndex` + BM25 hybrid retriever, LlamaIndex 실패 시 BM25 fallback

## 데이터 처리
- corpus ingest 방식: `data/scifact/corpus.jsonl`을 JSON Lines로 읽고 각 줄의 `_id`, `title`, `text`를 파싱
- title/text 사용 방식: LlamaIndex `Document` text에 `title`과 `text`를 함께 포함
- chunk size / overlap: LlamaIndex 기본 512 / 80, BM25 fallback 기본 220 words / 50 words overlap
- doc_id 보존 방식: 각 LlamaIndex `Document`와 chunk node metadata에 corpus `_id`를 `doc_id`로 그대로 저장

## 인덱스와 검색
- embedding 모델: `BAAI/bge-small-en-v1.5`
- vector store/index: LlamaIndex `VectorStoreIndex`, `storage/scifact`에 persist
- score 계산 방식: LlamaIndex와 BM25 후보를 Reciprocal Rank Fusion으로 결합, fallback 시 BM25 점수 + title token overlap boost
- top_k 설정: 요청값 사용, 최대 10

## 성능 개선 방법
- baseline에서 바꾼 점: 기존 파일 기반 RAG와 stale `storage/` 의존을 제거하고 SciFact JSONL 전용 hybrid retriever로 전환
- 성능을 높이기 위해 시도한 방법: title을 embedding 대상 text에 포함하고, LlamaIndex vector 검색과 BM25 검색을 RRF로 결합
- 효과가 있었던 방법: 평가가 문서 단위 `doc_id` rank를 보므로 중복 문서 제거 후 top_k를 구성

## Self-check
- /health 결과: `data/scifact/corpus.jsonl` 존재 시 `{"status":"ok","ready":true}`
- 자체 테스트 질문: `scientific claim text for testing`
- 자체 테스트 검색 결과 top doc_id: `POST /retrieve`로 확인
- 자체 테스트 검색 결과 top score: `POST /retrieve`로 확인
- 실패한 점 / 개선할 점: 첫 Docker 실행 시 embedding model 다운로드와 LlamaIndex index build 시간이 걸릴 수 있음
