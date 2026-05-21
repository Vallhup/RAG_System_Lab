# Knowledge Base Extension Plan

## 목적

현재 프로젝트는 SciFact retrieval 평가를 우선한다. 장기적으로는 개인 Knowledge Base를 RAG corpus로 연결해 실제 LLM 사용에 활용한다.

이 문서는 향후 연결할 Knowledge Base 구조와 ingest 정책을 기록한다. 현재 구현에는 바로 연결하지 않는다.

## 후보 Knowledge Base 경로

```text
C:\Users\Hadenpel\Desktop\Hadenpel\Knowledge Base
```

현재 구조:

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

파일 타입:

- Markdown notes: primary target
- HWP: optional, prefer converted Markdown/PDF
- Obsidian JSON settings: exclude

## Ingest 대상

기본 포함:

```text
00. Inbox/**/*.md
10. Projects/**/*.md
20. Areas/**/*.md
30. Resources/**/*.md
90. Archives/**/*.md
```

기본 제외:

```text
.obsidian/**
Template/**
**/*.json
**/*.tmp
**/~*
```

선택적 제외 또는 낮은 우선순위:

```text
90. Archives/**
```

## Metadata 정책

각 note는 경로 기반 metadata를 가진다.

예:

```json
{
  "source_type": "knowledge_base",
  "source_path": "30. Resources/논문 분석/example.md",
  "collection": "Resources",
  "topic": "논문 분석",
  "status": "active",
  "extension": ".md"
}
```

폴더별 권장 status:

- `00. Inbox`: `inbox`, lower confidence
- `10. Projects`: `project`, active
- `20. Areas`: `area`, active
- `30. Resources`: `resource`, reference
- `90. Archives`: `archive`, lower priority
- `Template`: excluded
- `.obsidian`: excluded

## Docker 연결 방식

Knowledge Base는 Docker image에 복사하지 않는다. read-only volume으로 mount한다.

예상 Compose 설정:

```yaml
services:
  rag:
    volumes:
      - "C:/Users/Hadenpel/Desktop/Hadenpel/Knowledge Base:/kb:ro"
    environment:
      RAG_KB_ROOT: /kb
```

이 방식의 장점:

- Knowledge Base 수정 시 Docker image rebuild가 필요 없다.
- 개인 파일을 Docker image에 포함하지 않는다.
- ingest/index build 정책만 별도로 관리하면 된다.

## Storage Namespace

SciFact와 Knowledge Base index는 분리한다.

```text
storage/
  scifact/
  knowledge_base/
```

Knowledge Base metadata에는 다음 fingerprint를 저장한다.

- included file paths
- each file size
- each file modified time
- parser version
- chunking config
- embedding model

파일 변경 감지는 전체 hash보다 파일 단위 fingerprint를 우선한다. 향후 증분 indexing을 가능하게 하기 위해서다.

## Retrieval 정책

초기 검색 정책:

```text
Vector retrieval
  + BM25 lexical retrieval
  + RRF fusion
  + metadata filters
```

예상 filter:

- exclude archive by default
- include archive if explicitly requested
- boost Projects and Areas for current work questions
- boost Resources for research questions
- lower Inbox confidence

## Note 작성 가이드

RAG 품질을 위해 note에는 frontmatter를 권장한다.

```markdown
---
type: paper-review
status: active
tags: [ecs, taskgraph, game-server]
created: 2026-05-21
---

# 문서 제목
```

권장:

- 제목을 명확히 쓴다.
- `무제 1.md` 같은 파일명은 피한다.
- 하나의 note는 하나의 주제에 집중한다.
- 출처 URL, 논문명, 날짜를 명시한다.

## 구현 시점

현재 단계에서는 연결하지 않는다.

먼저 진행할 작업:

1. SciFact hybrid retrieval 안정화
2. retrieval logging 기반 튜닝 루프 구축
3. `/ask` endpoint와 LLM provider 연결
4. Knowledge Base ingest 추가
5. metadata filter와 namespace routing 추가
