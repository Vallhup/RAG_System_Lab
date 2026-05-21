> **⏰ 평가 안내**
> - **대상 리파지토리**: 새로 생성하는 **`mcphttp-` prefix 리파지토리** (GitHub Classroom 초대 링크로 생성)
> - **대상 브랜치**: `main`
> - **마감**: **2026-04-30(목) 12:20 이전 commit** 만 채점
>
> ⚠️ 지난주 `mcp-*` 리파지토리가 아닌 **새로 생성한 `mcphttp-*` 리파지토리에 commit 한 것만** 평가 대상입니다.
> ⚠️ `main` 외의 브랜치 push 는 반영되지 않습니다.

# 이번 주 실습: Streamable HTTP MCP + RAG 한계 발견

두 미션은 **하나의 VectorStoreIndex 위에서** 진행. Mission 2 가 만든 RAG 를 Mission 1 이 그대로 MCP 서버로 노출하는 구조.

```
[ VectorStoreIndex ] ──── Mission 2: Top-10 실패 케이스 발굴
        ↑
        │ 같은 index
        ↓
[ FastMCP @tool ]    ──── Mission 1: Streamable HTTP 서버로 노출
```

## 🎯 Mission 1 — Streamable HTTP MCP 서버

Mission 2에서 만든 VectorStoreIndex 기반 RAG QueryEngine 을 **Streamable HTTP** 트랜스포트로 노출.

- [ ] `mcp.run(transport="http", host="0.0.0.0", port=...)` 로 서버 기동
- [ ] 별도 클라이언트 스크립트로 도구 호출 성공
- [ ] 서버 기동 + 클라이언트 호출이 한 화면에 보이는 스크린샷 1장 첨부

> 힌트: 5주차 강의자료 §4 (트랜스포트), §4.6 (코드 비교) 참조

## 🚩 Mission 2 — RAG Top-10 실패 케이스 찾기

**VectorStoreIndex** 기반 RAG를 만들고, **Top-10 검색에 정답 근거 노드가 들어오지 않는 데이터-질문 페어**를 직접 발굴.

- [ ] 본인이 준비한 자료로 VectorStoreIndex 구축 (**총 10MB 이하**)
- [ ] 인덱스를 `storage/` 폴더에 영속화하여 **repo 에 함께 push** (`index.storage_context.persist("./storage")`)
- [ ] 데이터 원본도 `data/` 폴더에 함께 push
- [ ] 실패 질문에 대한 Top-10 검색 결과를 **`mission2_log.md`** 에 저장 (score · 텍스트 앞 200자 · metadata 포함)
- [ ] `run_mission2.py` 한 줄 실행으로 `mission2_log.md` 가 재현되도록 작성 (평가자가 직접 돌려봄)
- [ ] 실패 원인을 본인 언어로 분석해 `README.md` 에 2~3 문단 정리

## 평가 기준

- ✅ Mission 1 정상 동작 + 서버 기동 스크린샷
- ✅ Mission 2 — `storage/`·`data/`·`mission2_log.md`·`run_mission2.py` 모두 push, 재현 시 동일한 실패 결과 확인
- ✅ git Discussion 참여도

## 참고

- 지난주 가이드: [`README_W4_MCP_Gemini_RAG.md`](./README_W4_MCP_Gemini_RAG.md)
- 3주차 RAG 기본기: [`README_llamaindex_basic.md`](./README_llamaindex_basic.md)
