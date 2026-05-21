# Project Specification

## 개요

이 프로젝트는 LlamaIndex 기반 RAG 검색 시스템을 Docker 환경에서 실행하고, 실습 7 평가 서버가 요구하는 SciFact retrieval API를 제공한다.

평가 목표는 LLM 답변 생성이 아니라 SciFact 문서 검색이다. `/retrieve`는 질문에 대한 답변을 생성하지 않고, 관련 문서의 원본 `doc_id`가 포함된 context 목록을 score 내림차순으로 반환한다.

## 평가 계약

필수 endpoint:

- `GET /health`
- `POST /retrieve`

`POST /retrieve` 요청:

```json
{
  "query_id": "eval_001",
  "question": "scientific claim text from the evaluator",
  "top_k": 10
}
```

`POST /retrieve` 응답:

```json
{
  "query_id": "eval_001",
  "contexts": [
    {
      "doc_id": "31715818",
      "chunk_id": "31715818::chunk_000",
      "score": 0.91,
      "text": "retrieved chunk text"
    }
  ]
}
```

중요한 제약:

- `doc_id`는 `data/scifact/corpus.jsonl`의 `_id`와 정확히 같아야 한다.
- 같은 문서에서 여러 chunk가 검색되어도 평가는 문서 단위 rank를 사용한다.
- API는 `top_k <= 10`을 보장한다.

## 현재 구현 사항

현재 구현은 LlamaIndex `VectorStoreIndex`와 BM25 lexical retriever를 함께 사용하는 hybrid retriever를 기본으로 한다.

- corpus 경로: `data/scifact/corpus.jsonl`
- Docker 내부 corpus 경로: `/app/data/scifact/corpus.jsonl`
- index 저장소: `storage/scifact`
- 이전 실습 데이터: `data/practice`
- embedding model: `BAAI/bge-small-en-v1.5`
- chunking: LlamaIndex `SentenceSplitter`
- hybrid fusion: LlamaIndex vector 후보와 BM25 후보를 Reciprocal Rank Fusion으로 결합
- fallback: LlamaIndex index build/load/retrieve 실패 시 BM25 기반 lexical retriever 단독 사용

BM25는 hybrid retrieval의 한 축이면서, LlamaIndex가 준비되지 않았을 때 평가 API를 계속 살리기 위한 fallback이다.

## Python 아키텍처

```text
app/
  main.py       FastAPI endpoint와 request/response schema
  scifact.py    SciFact corpus ingest, LlamaIndex index build/load, fallback retrieval
  core.py       LlamaIndex Settings, embedding model, LLM 설정
  engine.py     이전 실습의 범용 파일 기반 index helper
  readers.py    이전 실습의 파일 reader helper
```

실습 7 평가 경로에서 직접 사용하는 파일은 다음이다.

```text
POST /retrieve
  -> app/main.py
  -> get_scifact_retriever()
  -> app/scifact.py SciFactLlamaRetriever
  -> LlamaIndex VectorStoreIndex + BM25FallbackRetriever
  -> RRF fusion
  -> doc_id/chunk_id/score/text contexts
```

`app/engine.py`, `app/readers.py`는 이전 실습의 일반 문서 RAG용 코드이며, SciFact 평가 endpoint에서는 직접 사용하지 않는다.

## 인덱스 정책

`storage/scifact/metadata.json`에 다음 값을 저장한다.

- corpus path
- corpus sha256
- document count
- chunk size
- chunk overlap
- embedding model

저장된 metadata가 현재 설정과 같으면 기존 LlamaIndex index를 로드한다. 다르면 `corpus.jsonl`을 다시 ingest해서 index를 재생성한다.

이 정책은 기존 `storage/`에 남아 있는 다른 과제 문서 기반 index를 SciFact 평가에 잘못 사용하는 일을 막기 위한 것이다.

현재 애플리케이션은 `storage/scifact`만 active index directory로 사용한다. 이전 실습에서 생성된 `storage/` 루트의 레거시 index 파일은 삭제 대상이며, 재사용하지 않는다.

## 로그 정책

로그는 목적별로 분리한다.

```text
logs/app/app.log
logs/app/error.log
logs/indexing/index.log
logs/retrieval/retrieve.jsonl
```

- `app.log`: 서버 시작, 설정, 일반 application event
- `error.log`: error level 이상과 stack trace
- `index.log`: corpus load, fallback 준비, LlamaIndex build/load 상태
- `retrieve.jsonl`: retrieval request 단위 구조화 로그

`retrieve.jsonl`은 hybrid retrieval 튜닝에 사용한다. 각 줄은 JSON object이며 `query_id`, `mode`, `latency_ms`, `top_doc_ids`, `scores`를 포함한다.

## Docker 실행 기준

기본 실행 명령:

```powershell
docker compose -p ragapi up --build
```

현재 작업 경로에 공백과 한글이 포함되어 있으므로 `-p ragapi`를 붙여 Compose project name을 명시한다.

Docker Compose 환경 변수:

```yaml
RAG_CORPUS_PATH: data/scifact/corpus.jsonl
RAG_STORAGE_DIR: storage/scifact
RAG_LLAMA_CHUNK_SIZE: "512"
RAG_LLAMA_CHUNK_OVERLAP: "80"
RAG_HYBRID_CANDIDATE_MULTIPLIER: "5"
RAG_BM25_CHUNK_WORDS: "220"
RAG_BM25_CHUNK_OVERLAP: "50"
```

## 실행 확인

서버 health check:

```powershell
curl.exe http://127.0.0.1:8000/health
```

검색 self-check:

```powershell
curl.exe -X POST http://127.0.0.1:8000/retrieve `
  -H "Content-Type: application/json" `
  -d '{"query_id":"selfcheck_001","question":"scientific evidence about cancer and immune response","top_k":3}'
```

외부 평가용 tunnel:

```powershell
.\cloudflared.exe tunnel --url http://127.0.0.1:8000
```

Cloudflare URL이 바뀌면 `rag_endpoint.json`의 `api_base_url`만 갱신한다. `/health`나 `/retrieve`는 붙이지 않는다.

## 설계 원칙

- SciFact 평가는 `doc_id` ranking이 핵심이므로 모든 chunk metadata에 원본 `_id`를 보존한다.
- LlamaIndex vector retrieval과 BM25 lexical retrieval을 함께 사용해 의미 유사성과 정확한 용어 매칭을 모두 반영한다.
- LlamaIndex가 준비되지 않은 경우에도 BM25 fallback으로 평가 API 가용성을 유지한다.
- 기존 과제 문서용 `storage/`와 SciFact index를 섞지 않는다.
- Docker 내부 경로 기준으로 동작해야 하며, 로컬 절대 경로에 의존하지 않는다.

## 향후 Knowledge Base 확장

장기 목표는 개인 지식 베이스를 RAG corpus로 연결하는 것이다. 현재 후보 경로는 다음과 같다.

```text
C:\Users\Hadenpel\Desktop\Hadenpel\Knowledge Base
```

현재 이 Knowledge Base는 Obsidian/PARA 스타일 구조다.

```text
Knowledge Base/
  .obsidian/
  00. Inbox/
  10. Projects/
  20. Areas/
  30. Resources/
  90. Archives/
  Template/
```

현재 SciFact 평가 endpoint에는 연결하지 않는다. RAG 시스템이 충분히 안정화된 뒤 별도 corpus source로 추가한다.

권장 연결 방식:

```text
Host Knowledge Base
  -> Docker read-only volume mount
  -> ingest policy
  -> metadata extraction
  -> dedicated index namespace
```

예상 Docker mount:

```yaml
volumes:
  - "C:/Users/Hadenpel/Desktop/Hadenpel/Knowledge Base:/kb:ro"
```

예상 환경변수:

```text
RAG_KB_ROOT=/kb
```

중요한 원칙:

- SciFact 평가 index와 개인 Knowledge Base index를 섞지 않는다.
- Knowledge Base는 별도 storage namespace를 사용한다.
- `.obsidian/`, `Template/`은 기본 ingest 제외 대상이다.
- `90. Archives/`는 기본 검색에서 제외하거나 낮은 우선순위로 처리한다.
- `00. Inbox/`는 낮은 신뢰도 또는 정리 전 상태로 metadata를 부여한다.
- HWP 원본은 가능하면 Markdown/PDF 변환본을 우선 사용한다.
