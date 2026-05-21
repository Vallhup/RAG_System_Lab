> **평가 안내**
> - **대상 리파지토리**: 본인의 `web-` prefix 리파지토리 (GitHub Classroom으로 생성된 본인 repo)
> - **대상 브랜치**: `main`
> - **제출 파일**: `rag_endpoint.json`, `README.md`
> - **자동평가 대상**: `GET /health`, `POST /retrieve`
> - **인정 기준**: **12:20까지 `main` 브랜치에 commit된 내용만 인정**
> - **자동 평가 시각**: **11:20**, **12:20**
>
> 11:20과 12:20에 교수자 평가 서버가 여러분이 제출한 `api_base_url`에 접속해 자동 평가합니다. LLM 생성 답변은 이번 자동평가 대상이 아닙니다. RAG 시스템의 retrieve 결과(`contexts`)만 평가합니다.

# 이번 주 실습: Retrieval API 공개 + 리더보드 Baseline

이번 주 목표는 본인이 만든 RAG 검색기를 외부 평가 서버가 호출할 수 있게 만드는 것입니다.

```
내 RAG 서버
  -> Cloudflare Quick Tunnel
  -> public HTTPS URL
  -> rag_endpoint.json에 URL 제출
  -> 평가 서버가 /health, /retrieve 호출
  -> 리더보드 점수 계산
```

핵심은 **답변 생성이 아니라 retrieval**입니다. 평가 서버는 질문을 보내고, 여러분의 서버가 반환한 `contexts` 안에 필요한 근거 정보가 들어 있는지 확인합니다.

---

## Mission 1 — RAG API 서버 만들기

본인 RAG 시스템에 아래 두 endpoint를 구현하세요.

### `GET /health`

서버와 인덱스가 평가 가능한 상태인지 확인합니다.

응답 예시:

```json
{
  "status": "ok",
  "ready": true
}
```

### `POST /retrieve`

질문을 받아 관련 context를 반환합니다.

요청 예시:

```json
{
  "question": "출장비 초과 시 어떤 승인이 필요한가?",
  "top_k": 5
}
```

응답 예시:

```json
{
  "contexts": [
    {
      "text": "국내출장 한도는 50만원이며 초과 시 부서장 승인 필요...",
      "source": "Policy.pdf",
      "score": 0.82
    }
  ]
}
```

필수 조건:

- `contexts`는 JSON list여야 합니다.
- 각 context는 최소 `text` 필드를 가져야 합니다.
- `source`, `score`는 권장 필드입니다.
- context 하나의 `text`는 너무 길게 보내지 마세요. 1,000~2,000자 정도를 권장합니다.

> **Mock RAG 부분 점수**: 실제 RAG 연결이 어려우면, 고정/하드코딩된 `contexts`를 반환하는 Mock 서버로 제출해도 부분 점수 인정. 단 endpoint 스펙(`/health`, `/retrieve`)과 응답 JSON 형식은 동일해야 합니다.

---

## Mission 2 — Cloudflare Quick Tunnel로 외부 공개

Cloudflare Quick Tunnel을 사용하면 공유기 포트포워딩 없이 public HTTPS URL을 만들 수 있습니다.

이번 실습은 Docker Compose 실행을 기준으로 합니다. 컨테이너 안 웹서버는 반드시 `0.0.0.0`에 bind해야 합니다.

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Docker Compose 예:

```yaml
services:
  rag:
    build: .
    ports:
      - "8000:8000"
    command: uvicorn app:app --host 0.0.0.0 --port 8000
```

호스트에서 tunnel을 실행합니다.

```bash
cloudflared tunnel --url http://127.0.0.1:8000
```

Windows PowerShell 예:

```powershell
.\cloudflared.exe tunnel --url http://127.0.0.1:8000
```

구조:

```
Docker container
  RAG API: 0.0.0.0:8000
        |
        | ports: 8000:8000
        v
Host PC
  http://127.0.0.1:8000
        |
        | cloudflared tunnel
        v
https://*.trycloudflare.com
```

---

## Mission 3 — Public URL로 직접 확인

`localhost`가 아니라 Cloudflare가 발급한 public URL로 확인해야 합니다.

```bash
curl https://YOUR-TUNNEL.trycloudflare.com/health
```

```bash
curl -X POST https://YOUR-TUNNEL.trycloudflare.com/retrieve \
  -H "Content-Type: application/json" \
  -d '{"question":"출장비 초과 시 어떤 승인이 필요한가?","top_k":5}'
```

Windows PowerShell에서는 `curl` 대신 `curl.exe`를 쓰면 혼동이 적습니다.

```powershell
curl.exe https://YOUR-TUNNEL.trycloudflare.com/health
```

---

## Mission 4 — `rag_endpoint.json` 제출

리파지토리 루트에 `rag_endpoint.json` 파일을 만들고 commit/push하세요.

```json
{
  "api_base_url": "https://YOUR-TUNNEL.trycloudflare.com"
}
```

주의:

- 파일명은 정확히 `rag_endpoint.json`이어야 합니다.
- `main` 브랜치에 push해야 합니다.
- `api_base_url` 끝에는 `/health`나 `/retrieve`를 붙이지 마세요.
- Cloudflare URL이 바뀌면 `rag_endpoint.json`을 수정해서 다시 commit/push하세요.

---

## Quick Tunnel URL 유지 규칙

Cloudflare Quick Tunnel URL은 `cloudflared` 프로세스가 살아 있는 동안 유지됩니다.

유지되는 경우:

```text
cloudflared 프로세스는 계속 실행
+ RAG 웹서버/Docker 컨테이너만 재시작
-> URL 유지
```

바뀌는 경우:

```text
cloudflared 종료
PC 재부팅
cloudflared 재실행
-> 새 URL 발급
```

따라서 평가 시간에는 `cloudflared`를 끄지 마세요. RAG 서버나 Docker 컨테이너는 재시작해도 됩니다.

---

## 이번 주 보고서

본인 repo의 `README.md` 하단에 아래 내용을 작성하세요.

```markdown
# Retrieval API Baseline Report

## 1. 실행 방식
- RAG 서버 실행 방식: 로컬 / Docker / 기타
- Cloudflare Quick Tunnel URL:
- 사용한 데이터:
- 사용한 index/retriever:

## 2. Public URL self-check
- /health 결과:
- /retrieve 테스트 질문:
- /retrieve 반환 contexts 수:

## 3. Baseline 검색 결과
| 질문 | 기대 정보 | 검색된 context에 포함 여부 | 실패 원인 |
|---|---|---|---|
| | | | |

## 4. 다음 개선 계획
- chunking:
- metadata:
- query rewrite:
- reranker/top_k:
```

---

## 자동 평가 안내

- 11:20과 12:20에 교수자 평가 서버가 `rag_endpoint.json`의 `api_base_url`을 읽어 자동 평가합니다.
- 12:20까지 `main` 브랜치에 commit된 내용만 인정합니다.
- 자동평가는 RAG 시스템의 retrieve 결과만 평가합니다.
- LLM 생성 답변은 자동평가 대상이 아닙니다.
