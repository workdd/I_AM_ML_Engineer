# GraphRAG·지식그래프·온톨로지 읽을 논문

[GraphRAG 지식 지도](graphrag_지식지도.md)에서 빈칸으로 남은 영역을 채우기 위한 읽기 대기열입니다. 하나씩 정리하면서 상태를 바꿉니다.

- **작성**: 2026-09-28
- **수집 방법**: arXiv·GitHub 웹 검색 5회. 이미 정리한 LogicRAG, ROGRAG, LLM on Graphs 서베이는 뺐습니다
- **주의**: 요지 칸은 검색 결과의 초록 요약을 옮긴 것이고 본문은 확인하지 않았습니다. 정리할 때 원문으로 다시 확인합니다

## 진행 방법

1. 아래 표에서 `대기` 중 우선순위가 가장 높은 논문을 고릅니다
2. `readings/papers/[YYYYMMDD] 제목.md`로 README의 정리 형식을 따라 정리합니다
3. 이 표의 상태를 `정리 완료`로 바꾸고 제목에 노트 링크를 겁니다
4. README의 "그래프와 지식 표현" 표와 [지식 지도](graphrag_지식지도.md)에 반영합니다
5. 읽어 보니 정리할 가치가 없으면 상태를 `보류`로 두고 이유를 한 줄 적습니다

상태 값은 `대기`, `읽는 중`, `정리 완료`, `보류` 네 가지입니다.

## 추천 순서

지금 저장소는 **GraphRAG를 어떻게 하는가**에 치우쳐 있습니다. 그래서 **써야 하는가**(A)를 먼저 읽고, 그다음 비어 있는 **그래프를 만드는 쪽**(D)으로 넘어갑니다.

```
A1 → A2 → D1 → D4 → B1 → 나머지
```

- A1과 A2는 [LogicRAG](../papers/%5B20260702%5D%20LogicRAG_Adaptive_Reasoning_Structures.md)의 "그래프를 미리 만들 필요 없다"는 주장과 함께 읽으면 좋습니다
- D1은 기업 데이터에서 온톨로지를 만드는 문제라 실무와 가장 가깝습니다
- D4 서베이를 D 그룹 초반에 읽어 두면 나머지 구축 논문의 위치를 잡기 쉽습니다

## A. GraphRAG가 정말 필요한가

| ID | 우선 | 논문 | arXiv | 요지 | 상태 |
|---|---|---|---|---|---|
| A1 | 1 | [Do We Still Need GraphRAG? Benchmarking RAG and GraphRAG for Agentic Search Systems](../papers/%5B20260928%5D%20RAGSearch_Do_We_Still_Need_GraphRAG.md) | [2604.09666](https://arxiv.org/abs/2604.09666) | 에이전트형 검색에서 일반 RAG와 GraphRAG를 같은 조건으로 비교 (RAGSearch) | 정리 완료 |
| A2 | 1 | [When to use Graphs in RAG (ICLR 2026)](../papers/%5B20260929%5D%20GraphRAG-Bench_When_to_use_Graphs_in_RAG.md) | [2506.05690](https://arxiv.org/abs/2506.05690) | GraphRAG-Bench. 그래프가 이기는 조건과 지는 조건. [코드](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark) | 정리 완료 |
| A3 | 2 | WildGraphBench | [2602.02053](https://arxiv.org/abs/2602.02053) | 다듬은 짧은 지문이 아니라 길고 이질적인 실제 문서로 평가 | 대기 |
| A4 | 3 | RAG vs. GraphRAG: A Systematic Evaluation and Key Insights | [2502.11371](https://arxiv.org/abs/2502.11371) | RAG와 GraphRAG의 초기 체계적 비교 | 대기 |
| A5 | 3 | GraphRAG-Bench: Domain-Specific Reasoning | [2506.02404](https://arxiv.org/abs/2506.02404) | 16개 분야 교재 기반 대학 수준 다단계 추론 벤치마크 | 대기 |

## B. 비용을 줄인 GraphRAG

| ID | 우선 | 논문 | arXiv | 요지 | 상태 |
|---|---|---|---|---|---|
| B1 | 2 | LiteRAG: Cost-Efficient Graph-Based RAG | [2609.10239](https://arxiv.org/abs/2609.10239) | 그래프 기반 RAG의 구축·질의 비용 절감. 이 목록에서 가장 최신 | 대기 |
| B2 | 2 | Pruning Minimal Reasoning Graphs for Efficient RAG | [2602.04926](https://arxiv.org/abs/2602.04926) | 추론에 필요한 최소 그래프만 남기는 가지치기 | 대기 |
| B3 | 3 | Efficient RAG via Token Co-occurrence Graphs | [2606.30093](https://arxiv.org/abs/2606.30093) | LLM 추출 없이 토큰 동시 출현으로 그래프 구성 | 대기 |
| B4 | 3 | EraRAG: Efficient and Incremental RAG for Growing Corpora | [2506.20963](https://arxiv.org/abs/2506.20963) | 코퍼스가 늘 때 전체 재구축 없이 증분 갱신 | 대기 |
| B5 | - | [PathRAG: Pruning Graph-based RAG with Relational Paths](../papers/%5B20260928%5D%20PathRAG_Pruning_Graph_RAG_with_Relational_Paths.md) | [2502.14902](https://arxiv.org/abs/2502.14902) | 노드 쌍 사이 핵심 경로만 흐름 전파로 골라 토큰 절감. 대기열 밖에서 추가 | 정리 완료 |

## C. 에이전트 결합과 확장

| ID | 우선 | 논문 | arXiv | 요지 | 상태 |
|---|---|---|---|---|---|
| C1 | 2 | A-RAG: Agentic RAG via Hierarchical Retrieval Interfaces | [2602.03442](https://arxiv.org/abs/2602.03442) | 고정된 검색 알고리즘 대신 모델이 계층형 검색 인터페이스를 골라 씀 | 대기 |
| C2 | 3 | LivingRAG: Augmenting Graph RAG with Experience | [2608.25960](https://arxiv.org/abs/2608.25960) | 지난 질의응답 경험을 그래프에 누적 | 대기 |
| C3 | 3 | HVM-GraphRAG: Holistic-View Multimodal GraphRAG | [2607.24861](https://arxiv.org/abs/2607.24861) | 복잡한 문서 대상 멀티모달 GraphRAG | 대기 |
| C4 | 3 | FinReflectKG HalluBench | [2603.20252](https://arxiv.org/abs/2603.20252) | 지식그래프 기반 금융 질의응답의 환각 탐지 벤치마크 | 대기 |
| C5 | 3 | LegalGraphRAG | [2605.28120](https://arxiv.org/abs/2605.28120) | 법률 추론용 멀티 에이전트 GraphRAG | 대기 |

## D. 온톨로지와 지식그래프 구축

저장소에 아직 한 건도 없는 영역입니다.

| ID | 우선 | 논문 | arXiv | 요지 | 상태 |
|---|---|---|---|---|---|
| D1 | 1 | OntoEKG: LLM-Driven Ontology Construction for Enterprise KGs | [2602.01276](https://arxiv.org/abs/2602.01276) | 기업 비정형 데이터에서 클래스·속성 추출, 계층화, RDF 직렬화 | 대기 |
| D2 | 2 | OntoKG: Ontology-Oriented KG Construction with Intrinsic-Relational Routing | [2604.02618](https://arxiv.org/abs/2604.02618) | 속성을 스키마 모듈로 분류해 보냄. Wikidata 2026-01 덤프로 구현 | 대기 |
| D3 | 2 | Ontology Generation using Large Language Models | [2503.05388](https://arxiv.org/abs/2503.05388) | LLM 온톨로지 생성. 예전에 한 번 검토했지만 노트는 없음 | 대기 |
| D4 | 2 | LLM-empowered Knowledge Graph Construction: A Survey | [2510.20345](https://arxiv.org/abs/2510.20345) | 지식그래프 구축 쪽 전체 지형을 정리한 서베이 | 대기 |
| D5 | 3 | Towards Automated Ontology Generation from Unstructured Text: A Multi-Agent LLM Approach | [2604.23090](https://arxiv.org/abs/2604.23090) | 여러 LLM 에이전트가 비정형 텍스트에서 온톨로지 생성 | 대기 |
| D6 | 3 | Automatic Ontology Construction Using LLMs as an External Layer of Memory, Verification, and Planning | [2604.20795](https://arxiv.org/abs/2604.20795) | RDF/OWL 그래프를 LLM의 외부 기억·검증 계층으로 사용 | 대기 |

## 참고 서베이와 목록

정리 대상은 아니고 위치를 잡을 때 참고합니다.

- [2501.00309 Retrieval-Augmented Generation with Graphs](https://arxiv.org/abs/2501.00309): GraphRAG 구성 요소를 질의 처리기, 검색기, 정리기, 생성기, 데이터 소스로 나눈 서베이
- [Awesome-GraphRAG](https://github.com/DEEP-PolyU/Awesome-GraphRAG): 새 논문이 계속 추가되는 목록. 이 대기열을 갱신할 때 확인합니다
