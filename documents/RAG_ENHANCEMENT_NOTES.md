# RAG Enhancement Notes

## 목적

이 문서는 현재 SciFact retrieval API를 넘어 RAG 시스템 자체를 고도화할 때 검토할 수 있는 방법론과 우선순위를 정리한다.

현재 프로젝트의 기준 구현은 LlamaIndex `VectorStoreIndex` 기반 retrieval이며, BM25는 fallback이다. 실습 7 평가는 LLM 답변 생성이 아니라 `doc_id` ranking 품질, 특히 `nDCG@10`을 본다.

## 주요 방법론

### Chunking 고도화

문서를 어떻게 자를지 결정하는 방법이다.

- fixed-size chunk
- sentence-based chunk
- semantic chunk
- sentence-window retrieval
- parent-child chunk
- auto-merging retrieval

장점:

- 구현 비용 대비 효과가 크다.
- 검색 정밀도와 context 보존에 직접 영향을 준다.

단점:

- 너무 작으면 문맥이 깨진다.
- 너무 크면 embedding이 여러 주제를 평균내며 흐려질 수 있다.
- SciFact처럼 abstract 중심 데이터에서는 과도한 chunking보다 title과 abstract 보존이 더 중요할 수 있다.

### Parent-child / Auto-merging Retrieval

작은 leaf chunk로 검색하고, 반환 context는 더 큰 parent chunk로 확장하는 방식이다.

장점:

- 작은 chunk의 검색 정밀도와 큰 chunk의 문맥 보존을 함께 얻을 수 있다.

단점:

- 구현과 디버깅이 복잡하다.
- 현재 평가는 생성 답변 품질보다 `doc_id` rank를 보므로 효과가 제한적일 수 있다.

### Hybrid Retrieval

Dense vector retrieval과 BM25 lexical retrieval을 병렬로 실행한 뒤 rank를 합친다.

대표 fusion 방식:

- weighted score fusion
- Reciprocal Rank Fusion, RRF

장점:

- vector retrieval은 의미 유사성에 강하다.
- BM25는 정확한 용어, 약어, 고유명사, 수치에 강하다.
- SciFact처럼 scientific term이 중요한 corpus에 적합하다.

단점:

- score scale이 달라 normalization이나 rank fusion이 필요하다.
- weight를 잘못 잡으면 vector-only보다 나빠질 수 있다.

### Reranking

1차 retriever가 top 30-100개 후보를 가져오고, cross-encoder나 LLM reranker가 top 10을 재정렬한다.

장점:

- `nDCG@10` 같은 ranking metric에 직접적이다.
- 후보 recall만 충분하면 최종 순위를 크게 개선할 수 있다.

단점:

- latency와 모델 크기 부담이 있다.
- Docker 평가 환경에서 모델 다운로드, 메모리, 실행 시간을 확인해야 한다.

### Query Rewriting / Query Expansion

사용자 질문을 검색에 유리한 형태로 바꾸거나 여러 query로 확장한다.

예:

- Rewrite-Retrieve-Read
- HyDE
- multi-query retrieval

장점:

- 질문과 문서의 표현 차이를 줄일 수 있다.
- paraphrase나 동의어가 많은 도메인에서 recall이 좋아질 수 있다.

단점:

- LLM rewrite가 원래 claim의 의미를 바꿀 수 있다.
- SciFact 평가에서는 query 의미 보존이 중요하므로 과한 rewrite는 위험하다.

### GraphRAG

문서에서 entity, relation, claim을 추출해 graph를 만들고 graph neighborhood나 community summary를 검색한다.

장점:

- corpus 전체의 테마, 관계, multi-hop 질문에 강하다.
- local chunk retrieval만으로 어려운 global question answering에 유리하다.

단점:

- entity extraction, relation extraction, graph normalization, community summary 비용이 크다.
- 현재 SciFact `doc_id` retrieval 평가에는 과할 수 있다.

### RAPTOR / Hierarchical Summary Retrieval

chunk를 clustering하고 cluster summary를 계층적으로 만들어 tree 형태로 검색한다.

장점:

- 긴 문서나 여러 chunk를 종합해야 하는 질문에 강하다.
- 문서 전체 구조를 계층적으로 반영할 수 있다.

단점:

- 요약 생성 비용이 크다.
- 요약이 원문 `doc_id` ranking을 흐릴 수 있다.
- SciFact abstract 단위 corpus에는 우선순위가 낮다.

### Corrective / Self-reflective RAG

검색 결과가 충분한지 평가하고, 부족하면 재검색하거나 외부 검색을 붙이는 방식이다.

예:

- CRAG
- Self-RAG
- FLARE

장점:

- 생성형 QA에서 hallucination을 줄이는 데 유리하다.
- 검색 실패를 감지하고 회복할 수 있다.

단점:

- 현재 평가는 generation이 아니라 retrieval ranking이다.
- 추가 LLM 호출이 필요할 수 있다.

### Long-context 관리

많은 context를 한꺼번에 넣는다고 항상 좋은 것은 아니다. 관련 정보가 긴 context 중간에 있을 때 모델이 활용하지 못하는 문제가 보고되어 있다.

장점:

- top-k 관리와 reranking의 필요성을 설명한다.
- 생성형 RAG로 확장할 때 중요하다.

단점:

- 현재 실습 7의 핵심 metric은 retrieval `nDCG@10`이므로 직접 우선순위는 낮다.

## 현재 프로젝트에 대한 권장 우선순위

### 1. Hybrid Retrieval

현재 BM25 fallback이 이미 있으므로, 이를 fallback에만 두지 말고 LlamaIndex vector 결과와 병렬로 합치는 방식을 검토한다.

권장 구조:

```text
vector top_n
BM25 top_n
  -> RRF fusion
  -> doc_id 중복 제거
  -> top_k 반환
```

### 2. Reranker

Hybrid retrieval로 후보를 넓게 모은 뒤 reranker로 top 10을 재정렬한다.

검토 기준:

- 모델 크기
- Docker 메모리
- latency
- nDCG@10 개선 여부

### 3. Chunking 실험

현재 baseline:

```text
LlamaIndex chunk_size = 512
chunk_overlap = 80
```

실험 후보:

```text
256 / 40
384 / 64
512 / 80
768 / 100
title + abstract one-node
```

SciFact는 abstract가 짧은 편이므로, 문서 단위 보존 실험도 필요하다.

### 4. Query Expansion

LLM rewrite보다는 의미 보존이 강한 방식부터 검토한다.

- deterministic query normalization
- scientific abbreviation handling
- biomedical synonym expansion

### 5. GraphRAG / RAPTOR

현재 실습 7 평가에는 후순위다. 문서 전체 요약, multi-hop reasoning, corpus-wide question answering이 요구될 때 검토한다.

## 결론

현재 과제의 평가 기준에는 다음 순서가 가장 현실적이다.

```text
LlamaIndex VectorStoreIndex
  + BM25 병렬 검색
  + RRF fusion
  + optional reranker
  + doc_id 중복 제거
  + nDCG@10 기준 self-eval
```

GraphRAG, RAPTOR, Self-RAG 계열은 중요한 고도화 방향이지만, 현재 SciFact `doc_id` retrieval 과제에서는 비용 대비 우선순위가 낮다.
