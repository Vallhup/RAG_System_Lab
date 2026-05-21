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
