# GraphRAG·지식그래프·온톨로지 읽을 논문

[GraphRAG 지식 지도](graphrag_지식지도.md)에서 빈칸으로 남은 영역을 채우기 위한 읽기 대기열입니다. 하나씩 정리하면서 상태를 바꿉니다.

- **작성**: 2026-09-28 · **갱신**: 2026-10-02 (E 그룹 추가)
- **수집 방법**: A~D는 arXiv·GitHub 웹 검색 5회. E는 arXiv API 검색어 5개로 2026년 제출분 247건을 모아 관련 235건 중 골랐습니다. 이미 정리했거나 대기열에 있는 논문은 뺐습니다
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
- E 그룹은 우선 1부터 읽습니다. 순서는 E2(SCAIR) → E1(post-graph-rag) → E30(k-core 계층) → E4(CacheRAG) → E11 · E12 → E16 → 나머지 우선 1입니다

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

## E. 2026년 최신 논문

2026년 arXiv 제출분에서 기법·벤치마크·보안 논문만 골랐습니다. 도메인 적용 사례는 뺐습니다. 우선 1은 **스키마가 정해진 정형 데이터를 PostgreSQL 위 그래프로 운영하는 경우**에 바로 옮길 수 있는 것입니다.

### 기업 그래프 운영과 거버넌스

| ID | 우선 | 논문 | arXiv | 요지 | 상태 |
|---|---|---|---|---|---|
| E1 | 1 | post-graph-rag: PostgreSQL-Native Bi-Temporal Graph RAG | [2608.24921](https://arxiv.org/abs/2608.24921) | 벡터·그래프·커뮤니티 요약을 PostgreSQL 하나에 둔다. 적재 전 추출 검증, 정규 엔티티 하나로 병합, 관계의 유효 기간을 기록해 대체된 사실을 구분 | 대기 |
| E2 | 1 | SCAIR: Schema-Conditioned Agentic Iterative Reasoning for Enterprise KGs (ACL 2026 Industry) | [2607.22571](https://arxiv.org/abs/2607.22571) | 실제 CMDB로 만든 기업 그래프 벤치마크. 스키마를 사전 지식으로 주입하고 스키마를 따르는 순회만 허용. 학습 없음 | 대기 |
| E3 | 1 | MAGG: Multi-Agent KG Construction with Domain-Expert Review | [2608.28642](https://arxiv.org/abs/2608.28642) | 트리플마다 소유자·채택 근거·감사 메타데이터를 남기는 거버넌스 그래프. 질의도 소유 도메인별 전문가로 라우팅 | 대기 |
| E4 | 1 | CacheRAG: Semantic Caching for KGQA | [2604.26176](https://arxiv.org/abs/2604.26176) | 질의 계획을 캐시해 상태 없는 플래너를 계속 배우게 한다. 중간 의미 표현으로 스키마 환각을 줄이고 깊이·폭이 제한된 결정적 부분그래프 확장 | 대기 |
| E5 | 3 | EvoRAG: Feedback-driven Evolving KG-RAG | [2604.15676](https://arxiv.org/abs/2604.15676) | 응답 피드백을 역전파해 그래프 의존 관계를 과제에 맞게 고친다 | 대기 |

### 그래프가 필요한가와 비용

| ID | 우선 | 논문 | arXiv | 요지 | 상태 |
|---|---|---|---|---|---|
| E6 | 1 | When Is Graph Structure Worth Its Cost? (EffiRAG) | [2609.18099](https://arxiv.org/abs/2609.18099) | 그래프는 관련 단락을 찾는 데만 쓰고 답은 원문에서 생성. LightRAG 대비 120문항 중 93건 선호, 전체 비용 57% 절감 | 대기 |
| E7 | 2 | UnWeaving the knots of GraphRAG: VectorRAG is almost enough | [2603.29875](https://arxiv.org/abs/2603.29875) | 청크를 독립 벡터로 다루는 문제를 풀면 벡터 RAG로 거의 충분하다는 주장 | 대기 |
| E8 | 2 | Is GraphRAG Needed? (ACL 2026 GEM) | [2606.25656](https://arxiv.org/abs/2606.25656) | 반정형 지식베이스에서 RAG·GraphRAG·모듈형·에이전트형 9개 시나리오를 표준 구현으로 비교 | 대기 |
| E9 | 2 | Use Graph When It Needs | [2602.03578](https://arxiv.org/abs/2602.03578) | 필요한 질의에만 그래프를 쓰는 적응형 통합 | 대기 |
| E10 | 2 | GraphRAG-Router | [2604.16401](https://arxiv.org/abs/2604.16401) | 질의마다 GraphRAG와 생성 LLM을 강화학습으로 고른다. 큰 LLM 과사용 약 30% 감소 | 대기 |

### 질의별 탐색 제어와 에이전트

| ID | 우선 | 논문 | arXiv | 요지 | 상태 |
|---|---|---|---|---|---|
| E11 | 1 | MOSAIC: Query-Aware Exploration Policy Adaptation | [2609.11065](https://arxiv.org/abs/2609.11065) | LLM이 질의마다 시작점·순회·중단·근거 선택 정책을 정한다. 고정 정책 대비 경로 평가 81.9% 감소. 학습 없음 | 대기 |
| E12 | 1 | A2RAG: Adaptive Agentic Graph Retrieval | [2601.21162](https://arxiv.org/abs/2601.21162) | 근거 충분성을 확인하고 부족할 때만 검색을 키운다. 그래프 신호를 원문 근거로 되돌려 추출 손실에 대응. 토큰·지연 약 50% 감소 | 대기 |
| E13 | 2 | PathRouter | [2606.16409](https://arxiv.org/abs/2606.16409) | 결과 보상만 쓰면 지름길 정답도 보상받는 문제를 고쳐 보상을 검색 품질에 맞춘다 | 대기 |
| E14 | 3 | GraphScout | [2603.01410](https://arxiv.org/abs/2603.01410) | LLM이 스스로 그래프를 탐색하는 능력을 학습 | 대기 |
| E15 | 3 | MemGraphRAG (KDD 2026) | [2606.00610](https://arxiv.org/abs/2606.00610) | 기억 장치를 둔 다중 에이전트 GraphRAG | 대기 |

### 불완전한 그래프와 근거

| ID | 우선 | 논문 | arXiv | 요지 | 상태 |
|---|---|---|---|---|---|
| E16 | 1 | Toward Robust GraphRAG: Retrieval Drift from Imperfect KGs (CS-RAG) | [2603.14828](https://arxiv.org/abs/2603.14828) | 그래프 잡음은 검색 표류를, 누락은 없는 구조를 지어내게 만든다. 질의를 원자 제약 순서로 계획하고 근거가 부족하면 원문으로 복구 | 대기 |
| E17 | 2 | Relink (AAAI 2026) | [2601.07192](https://arxiv.org/abs/2601.07192) | 질의 시점에 근거 그래프를 만든다. 원문에서 뽑은 잠재 관계로 끊긴 경로를 즉석 복구 | 대기 |
| E18 | 2 | EvLink (EMNLP 2026) | [2609.29695](https://arxiv.org/abs/2609.29695) | 그래프에서 이어진다고 근거가 되는 것은 아니다. 단락 단위로 근거 연결을 만든다 | 대기 |
| E19 | 3 | LineageRAG | [2608.16004](https://arxiv.org/abs/2608.16004) | 근거마다 원문 구간을 붙여 근거 계보를 만든다 | 대기 |
| E20 | 2 | When to Trust (WWW 2026) | [2601.09241](https://arxiv.org/abs/2601.09241) | 불완전한 부분그래프에서도 확신하는 과신을 인과 관점으로 보정 | 대기 |
| E21 | 2 | Why Neighborhoods Matter: Provenance in Agentic GraphRAG | [2605.15109](https://arxiv.org/abs/2605.15109) | 인용만으로는 충분하지 않다. 방문했지만 인용하지 않은 엔티티도 답에 영향을 주므로 출처를 탐색 궤적 단위로 봐야 한다 | 대기 |

### 검색 알고리즘

| ID | 우선 | 논문 | arXiv | 요지 | 상태 |
|---|---|---|---|---|---|
| E22 | 2 | Query-Aware Flow Diffusion (ICLR 2026) | [2605.18775](https://arxiv.org/abs/2605.18775) | 질의 반영 흐름 확산. 부분그래프 품질에 이론적 보장 | 대기 |
| E23 | 3 | Corpus-Guided Dual-Path Propagation | [2609.37661](https://arxiv.org/abs/2609.37661) | 질의 유사도가 낮은 연결 근거를 살리고 무관한 엔티티 활성화를 억제 | 대기 |
| E24 | 3 | Minimal and Sufficient Reasoning Subgraphs with GFMs | [2603.07179](https://arxiv.org/abs/2603.07179) | 그래프 기반 모델로 최소·충분 추론 부분그래프. 데이터가 적은 도메인 겨냥 | 대기 |
| E25 | 3 | DotRAG: Retrieval-Time Reasoning Along Paths | [2605.18760](https://arxiv.org/abs/2605.18760) | 학습 없이 경로를 따라 검색 단계에서 추론 | 대기 |
| E26 | 3 | HELP: HyperNode Expansion and Logical Path-Guided Localization | [2602.20926](https://arxiv.org/abs/2602.20926) | 하이퍼노드 확장과 논리 경로로 근거 위치 탐색 | 대기 |
| E27 | 3 | PAGE-RAG: Provenance-Aware Graph Evidence Promotion | [2608.29753](https://arxiv.org/abs/2608.29753) | 정해진 예산 안에서 출처를 따져 근거를 앞으로 올린다 | 대기 |
| E28 | 3 | FlowRAG | [2606.17856](https://arxiv.org/abs/2606.17856) | 빈도 인식 다중 단위 그래프 흐름 | 대기 |
| E29 | 3 | Query-Aware Spreading Activation | [2606.30133](https://arxiv.org/abs/2606.30133) | 시작 노드 이후에도 질의를 반영해 전파 | 대기 |

### 그래프 구축과 커뮤니티

| ID | 우선 | 논문 | arXiv | 요지 | 상태 |
|---|---|---|---|---|---|
| E30 | 1 | Core-based Hierarchies for Efficient GraphRAG (KDD 2026) | [2603.05207](https://arxiv.org/abs/2603.05207) | 희소 그래프에서 Leiden 커뮤니티는 재현되지 않음을 증명하고 선형 시간·결정적인 k-core 계층으로 대체. 토큰 예산 기반 요약 샘플링 | 대기 |
| E31 | 2 | Cross-Chunk Graph Augmentation | [2605.28004](https://arxiv.org/abs/2605.28004) | 청크 안에서만 관계를 뽑아 놓치는 청크 간 관계를 복원 | 대기 |
| E32 | 3 | AtomicRAG: Atom-Entity Graphs | [2604.20844](https://arxiv.org/abs/2604.20844) | 청크 대신 원자 사실을 기본 단위로 | 대기 |
| E33 | 3 | EHRAG (ACL 2026 Findings) | [2604.17458](https://arxiv.org/abs/2604.17458) | NER 기반 경량 그래프에 하이퍼그래프로 의미 연결 추가 | 대기 |
| E34 | 3 | LiteSemRAG | [2604.16350](https://arxiv.org/abs/2604.16350) | 색인과 질의 모두 LLM을 쓰지 않는 그래프 검색 | 대기 |
| E35 | 3 | OMD-GraphRAG | [2603.25152](https://arxiv.org/abs/2603.25152) | 온톨로지 기반 추출, 다차원 군집 | 대기 |
| E36 | 2 | Ontology-grounded Post-extraction Correction | [2605.29168](https://arxiv.org/abs/2605.29168) | 먼저 추출하고 온톨로지로 나중에 교정하는 신경-기호 구축 | 대기 |
| E37 | 3 | Agentic Crawling and Graph Construction in Enterprise Documents | [2604.14220](https://arxiv.org/abs/2604.14220) | 기업 문서를 재귀 탐색해 대체 관계와 다단계 참조를 따라간다 | 대기 |

### 보안

| ID | 우선 | 논문 | arXiv | 요지 | 상태 |
|---|---|---|---|---|---|
| E38 | 1 | AGEA: Agentic Graph Extraction Attacks (ACL 2026) | [2601.14662](https://arxiv.org/abs/2601.14662) | 질의 예산 제한 블랙박스 조건에서 엔티티·관계를 최대 90% 복원 | 대기 |
| E39 | 2 | Graphs Don't Stay Secret | [2602.06495](https://arxiv.org/abs/2602.06495) | 방어 장치가 있어도 부분그래프 복원 가능 | 대기 |
| E40 | 2 | LogicPoison | [2604.02954](https://arxiv.org/abs/2604.02954) | 텍스트 오염에는 강한 GraphRAG를 논리 구조로 공격 | 대기 |
| E41 | 3 | KEPo (WWW 2026) | [2603.11501](https://arxiv.org/abs/2603.11501) | 지식 변화로 위장한 그래프 오염 | 대기 |
| E42 | 3 | Adulteration-Based Protection of Proprietary KGs | [2601.00274](https://arxiv.org/abs/2601.00274) | 가짜 정보를 섞어 훔친 그래프를 무용하게 만드는 방어 | 대기 |

### 평가와 벤치마크

| ID | 우선 | 논문 | arXiv | 요지 | 상태 |
|---|---|---|---|---|---|
| E43 | 1 | MissDiag | [2608.18489](https://arxiv.org/abs/2608.18489) | 누락 유형별로 강건성을 나눠 잰다. 정답에 인접한 근거 손실이 가장 크고, 원문 맥락 제거는 영향이 없거나 오히려 도움이 되기도 함 | 대기 |
| E44 | 2 | Structural Analysis of Graph-Augmented Retrieval for Industrial KGs | [2606.06003](https://arxiv.org/abs/2606.06003) | 항공우주 공급망 그래프(46노드)에서 검색 구조 8종을 질의 의도 10종으로 비교 | 대기 |
| E45 | 2 | The Commercial Tax | [2608.16096](https://arxiv.org/abs/2608.16096) | 멀티홉 벤치마크 상위 시스템을 상용 라이선스·구축비로 점검 | 대기 |
| E46 | 3 | MKG-RAG-Bench (KDD 2026) | [2606.26458](https://arxiv.org/abs/2606.26458) | 멀티모달 지식그래프 RAG 검색 벤치마크 | 대기 |
| E47 | 3 | Signal or Noise? Multimodal GraphRAG | [2609.35304](https://arxiv.org/abs/2609.35304) | 모달리티를 더 넣을수록 좋다는 가정 검증 | 대기 |
| E48 | 3 | TrioRAG: Graph-free Multimodal RAG | [2609.19417](https://arxiv.org/abs/2609.19417) | 그래프 없이 세 검색 신호를 늦게 결합 | 대기 |

## 참고 서베이와 목록

정리 대상은 아니고 위치를 잡을 때 참고합니다.

- [2501.00309 Retrieval-Augmented Generation with Graphs](https://arxiv.org/abs/2501.00309): GraphRAG 구성 요소를 질의 처리기, 검색기, 정리기, 생성기, 데이터 소스로 나눈 서베이
- [Awesome-GraphRAG](https://github.com/DEEP-PolyU/Awesome-GraphRAG): 새 논문이 계속 추가되는 목록. 이 대기열을 갱신할 때 확인합니다
