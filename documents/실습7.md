# RAG Retrieval Live Evaluation

이번 실습부터는 여러분의 `web-` repo에 제출된 RAG endpoint를 교수자 평가 서버가 10분마다 호출하고, GitHub Issue 리더보드를 갱신합니다.

목표는 LLM 답변 생성이 아닙니다. **SciFact 문서를 검색해서 관련 문서의 `doc_id`를 상위 rank에 올리는 retrieval 성능**을 평가합니다.

## 평가 대상

- 대상 repo: 본인의 `web-` prefix GitHub Classroom repo
- 대상 branch: `main`
- 제출 파일: repo root의 `rag_endpoint.json`
- 제출 코드: RAG 서버, ingest, index 생성, retrieval 구현 코드
- 평가 endpoint: `GET /health`, `POST /retrieve`
- 평가 metric: `nDCG@10`
- 평가 주기: 약 10분마다 교수자 Windows 평가 서버가 실행
- 공개 결과: 이 repo의 GitHub Issue 리더보드

## 데이터

ingest 대상 데이터는 이 `3-Practice` repo에 포함되어 있습니다. 학생은 이 데이터를 받아서 본인 RAG 시스템에 넣어야 합니다.

```text
data/scifact/corpus.jsonl
```

학생 RAG 시스템에는 **`data/scifact/corpus.jsonl` 전체**를 넣으세요.

`corpus.jsonl`은 JSON Lines 형식이며, 한 줄이 문서 하나입니다.

```json
{
  "_id": "31715818",
  "title": "...",
  "text": "..."
}
```

여기서 `_id`가 평가에 쓰는 원본 `doc_id`입니다.

평가 query와 정답 qrels는 공개하지 않습니다. 학생에게 공개되는 데이터는 검색 대상 문서인 `corpus.jsonl`뿐입니다.

평가 서버는 비공개 train/test qrels를 병합한 뒤, 그중 어떤 query를 보낼지 평가 시점에 선택합니다. 학생은 어떤 query가 들어올지 모른다고 가정해야 합니다. 특정 query_id에 대한 정답 `doc_id`를 하드코딩해서 반환하는 방식은 실습 취지에 맞지 않습니다.

## 필수 구현

### `GET /health`

RAG 서버와 index가 준비되었는지 확인합니다.

응답 예:

```json
{
  "status": "ok",
  "ready": true
}
```

### `POST /retrieve`

요청 예:

```json
{
  "query_id": "eval_001",
  "question": "scientific claim text from the evaluator",
  "top_k": 10
}
```

응답 예:

```json
{
  "query_id": "eval_001",
  "contexts": [
    {
      "doc_id": "31715818",
      "chunk_id": "31715818::chunk_003",
      "score": 0.91,
      "text": "retrieved chunk text"
    }
  ]
}
```

필수 조건:

- `contexts`는 JSON list여야 합니다.
- 각 context에는 원본 문서 ID인 `doc_id`가 있어야 합니다.
- `doc_id`는 `corpus.jsonl`의 `_id`와 정확히 같아야 합니다.
- 각 context에는 numeric `score`가 있어야 합니다.
- `score`는 높을수록 더 관련 있는 문서라는 의미여야 합니다.
- `text`는 디버깅을 위해 포함하세요.
- `chunk_id`는 권장 필드입니다.

chunking은 자유입니다. 문서를 원하는 크기로 자르고 overlap을 넣어도 됩니다. 단, 모든 chunk metadata에 원본 `doc_id`를 반드시 보존해야 합니다.

평가 서버는 `score` 기준 내림차순으로 context rank를 만든 뒤, 같은 `doc_id`는 첫 등장만 유지하고, 문서 단위 ranked list로 바꿔 `nDCG@10`을 계산합니다.

## `rag_endpoint.json` 제출

본인 `web-` repo root에 아래 파일을 만들고 commit/push하세요.

```json
{
  "api_base_url": "https://YOUR-TUNNEL.trycloudflare.com"
}
```

주의:

- 파일명은 정확히 `rag_endpoint.json`이어야 합니다.
- `main` branch에 push해야 합니다.
- `api_base_url` 끝에 `/health`나 `/retrieve`를 붙이지 마세요.
- Cloudflare URL이 바뀌면 이 파일을 수정해서 다시 push하세요.

## Cloudflare Quick Tunnel

RAG API 서버는 Docker 또는 로컬에서 실행해도 됩니다. 서버는 외부에서 접근 가능해야 하며, Docker 안에서는 `0.0.0.0`에 bind해야 합니다.

예:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

host PC에서 Cloudflare tunnel을 실행합니다.

```bash
cloudflared tunnel --url http://127.0.0.1:8000
```

Windows PowerShell:

```powershell
.\cloudflared.exe tunnel --url http://127.0.0.1:8000
```

Cloudflare URL은 `cloudflared` 프로세스가 살아 있는 동안 유지됩니다. `cloudflared`를 끄거나 PC를 재부팅하면 URL이 바뀝니다.

## 직접 확인

`localhost`가 아니라 Cloudflare public URL로 확인하세요.

```bash
curl https://YOUR-TUNNEL.trycloudflare.com/health
```

```bash
curl -X POST https://YOUR-TUNNEL.trycloudflare.com/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query_id":"selfcheck_001","question":"scientific claim text for testing","top_k":10}'
```

Windows PowerShell에서는 `curl.exe`를 쓰세요.

## 리더보드

교수자 Windows 평가 서버가 약 10분마다 다음 작업을 수행합니다.

1. `web-` repo 목록 조회
2. 각 repo의 `rag_endpoint.json` 읽기
3. `/health` 확인
4. SciFact query subset으로 `/retrieve` 호출
5. `contexts[*].score` 기준으로 검색 결과 정렬
6. `contexts[*].doc_id`를 문서 rank로 변환
7. `nDCG@10` 계산
8. GitHub Issue 리더보드 갱신

서버가 꺼져 있거나 URL이 바뀌면 `latest_status`가 실패 상태로 표시됩니다. 이전에 성공한 최고 점수는 `best_score_so_far`로 보존됩니다.

## 코드 제출

본인 `web-` repo에는 평가 endpoint만 올리는 것이 아니라, 재현 가능한 구현 코드도 함께 올려야 합니다.

- RAG API 서버 코드
- SciFact ingest 코드 또는 ingest 절차
- index 생성 코드 또는 index 로딩 코드
- retrieval 코드
- 실행에 필요한 dependency 파일
- 실행 방법을 적은 `README.md`

API key, token, 개인 경로 같은 secret은 repo에 올리지 마세요.

## 보고서

본인 `web-` repo의 `README.md` 하단에 아래 내용을 간단히 작성하세요.

```markdown
# SciFact Retrieval Report

## 실행 방식
- 제출 코드 위치:
- RAG 서버 실행 방식:
- 서버 실행 명령:
- Cloudflare Quick Tunnel URL:
- 사용한 retriever/index:

## 데이터 처리
- corpus ingest 방식:
- title/text 사용 방식:
- chunk size / overlap:
- doc_id 보존 방식:

## 인덱스와 검색
- embedding 모델:
- vector store/index:
- score 계산 방식:
- top_k 설정:

## 성능 개선 방법
- baseline에서 바꾼 점:
- 성능을 높이기 위해 시도한 방법:
- 효과가 있었던 방법:

## Self-check
- /health 결과:
- 자체 테스트 질문:
- 자체 테스트 검색 결과 top doc_id:
- 자체 테스트 검색 결과 top score:
- 실패한 점 / 개선할 점:
```
