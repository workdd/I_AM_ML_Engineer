# GraphRAG 오픈소스 지형

GitHub에서 GraphRAG 관련 저장소를 별 순으로 훑고, 고정 스키마의 정형 데이터(문서가 아닌 자원 · 권한 같은 레코드)를 Postgres 위 속성 그래프(Apache AGE)로 다루는 경우에 비춰 어디를 볼지 고른 목록이다.

- **조사일**: 2026-10-27. 별 수 · 최근 푸시 · 라이선스는 이날 GitHub API 값이다
- **방법**: `graphrag` · `graph rag` · `knowledge graph rag` · `graph retrieval augmented generation` 검색 상위 160건을 합쳐 중복을 지우고, 주요 저장소는 README와 일부 코드(AGE 지원 여부)를 확인했다
- **한계**: 어느 저장소도 직접 실행하지 않았다. 성능 주장은 각 README의 자기 보고이고 검증하지 않았다
- 이미 정리한 노트: [PathRAG](../papers/%5B20260928%5D%20PathRAG_Pruning_Graph_RAG_with_Relational_Paths.md), [GraphRAG-Bench](../papers/%5B20260929%5D%20GraphRAG-Bench_When_to_use_Graphs_in_RAG.md), [Open Ontologies](../repos/%5B20260309%5D%20fabio-rovai_open-ontologies_%EC%98%A8%ED%86%A8%EB%A1%9C%EC%A7%80_%EB%B3%80%EA%B2%BD%EA%B3%84%ED%9A%8D%EA%B3%BC_%EA%B2%80%EC%A6%9D%EA%B0%80%EB%8A%A5%ED%95%9C_%EC%B6%94%EB%A1%A0.md), [OpenCrab](../repos/%5B20260322%5D%20AlexAI-MCP_OpenCrab_MCP%EB%A1%9C_%EB%B6%99%EC%9D%B4%EB%8A%94_%EC%98%A8%ED%86%A8%EB%A1%9C%EC%A7%80_%EA%B3%B5%EC%9E%A5.md), [Omnigraph](../repos/%5B20260410%5D%20ModernRelay_omnigraph_%EB%B8%8C%EB%9E%9C%EC%B9%98%EB%90%98%EB%8A%94_%ED%83%80%EC%9E%85_%EA%B7%B8%EB%9E%98%ED%94%84DB.md), [Graphicle](../repos/%5B20261014%5D%20Graphicle-AI_graphicle_%EC%BD%94%EB%94%A9%EC%97%90%EC%9D%B4%EC%A0%84%ED%8A%B8%EC%9A%A9_%EC%BD%94%EB%93%9C%EA%B7%B8%EB%9E%98%ED%94%84_%EB%A9%94%EB%AA%A8%EB%A6%AC.md)

## 먼저 볼 것

고정 스키마의 정형 데이터를 Postgres 위 속성 그래프로 다루는 경우와 맞닿는 정도로 줄 세웠다.

| 순위 | 저장소 | 볼 지점 | 이유 |
|---|---|---|---|
| 1 | [HKUDS/LightRAG](https://github.com/HKUDS/LightRAG) | `lightrag/kg/postgres_impl.py`의 AGE 그래프 저장소와 AGE 버전 검사 테스트 | 별 4만 규모 프로젝트 중 **Apache AGE를 그래프 백엔드로 지원**하는 드문 사례. AGE 위에서 노드·엣지 일괄 조회, 차수 계산을 어떻게 짰는지 볼 수 있다 |
| 2 | [raphaelmansuy/edgequake](https://github.com/raphaelmansuy/edgequake) | `specs/106-kg-persist-bug/` | LightRAG를 Rust로 옮긴 프로젝트로 역시 AGE를 쓴다. `ag_catalog.graphid = ag_catalog.graphid` 연산자가 없어 관계 병합 조회가 실패한 사례와 `::text` 캐스팅 우회를 기록해 두었다. AGE 운영 함정 목록으로 읽을 만하다 |
| 3 | [trustgraph-ai/trustgraph](https://github.com/trustgraph-ai/trustgraph) | 정책 런타임, PROV-O 근거 기록 | "에이전트가 행동을 제안하면 그래프의 의미 · 권한 · 근거로 실행 전에 검사"한다는 구조. 권한 · 역할 · 소속 그래프를 에이전트 행동의 사람 승인(HITL) 근거로 쓰는 문제와 같은 축이다. RDF/OWL/SHACL 기반 |
| 4 | [getzep/graphiti](https://github.com/getzep/graphiti) · [deeplethe/utopia](https://github.com/deeplethe/utopia) | 시간 축이 있는 엣지(언제부터 언제까지 참이었나) | 적재 시점 스냅샷 그래프는 "지난주에 있던 VM", "소속이 바뀐 자원"을 답하지 못한다. Graphiti는 Neo4j · FalkorDB · Kuzu · Neptune, Utopia는 Rust 바이너리 + Postgres 하나로 쌍시간(bitemporal) 지식 기반을 만든다 |
| 5 | [TencentCloudADP/youtu-graphrag](https://github.com/TencentCloudADP/youtu-graphrag) | 시드 스키마로 추출을 묶는 방식, 4단 계층(속성 · 관계 · 키워드 · 커뮤니티) | 스키마를 먼저 주고 추출을 그 안에 가두는 발상이 고정 스키마와 맞는다. 다만 정형 데이터는 LLM 추출이 필요 없어서, 볼 것은 계층 요약 쪽이다. ICLR 2026 |

## 범용 GraphRAG 프레임워크

문서를 넣으면 엔티티·관계를 뽑아 그래프를 만들고 검색하는 쪽이다. 정형 데이터를 그대로 그래프로 올리는 경우에는 추출 단계가 필요 없어서, 검색과 요약 부분만 참고 대상이다.

| 저장소 | 별 | 최근 푸시 | 라이선스 | 특징 |
|---|---|---|---|---|
| [microsoft/graphrag](https://github.com/microsoft/graphrag) | 36,454 | 2026-10-26 | MIT | Local / Global Search의 원형. 최신 릴리즈 v3.3.0 (2026-10-08) |
| [HKUDS/LightRAG](https://github.com/HKUDS/LightRAG) | 40,352 | 2026-10-22 | MIT | 저수준(엔티티) · 고수준(주제) 키워드 이중 검색. 백엔드가 가장 다양하다(Neo4j, Memgraph, Postgres+AGE, Mongo, Redis, Milvus, Qdrant 등) |
| [infiniflow/ragflow](https://github.com/infiniflow/ragflow) | 92,922 | 2026-10-27 | Apache-2.0 | 문서 파싱 중심 RAG 엔진. GraphRAG는 기능 중 하나 |
| [OpenSPG/KAG](https://github.com/OpenSPG/KAG) | 9,125 | 2026-01-28 | Apache-2.0 | 논리식(logical form)으로 질의를 분해해 그래프 추론. 9개월째 푸시가 없다 |
| [raphaelmansuy/edgequake](https://github.com/raphaelmansuy/edgequake) | 2,286 | 2026-10-26 | Apache-2.0 | LightRAG의 Rust 재구현, Postgres + AGE |
| [Zleap-AI/SAG](https://github.com/Zleap-AI/SAG) | 2,637 | 2026-10-24 | MIT | 사건 · 엔티티 색인과 질의 시점 동적 하이퍼엣지. SQLite + LanceDB로 시작하는 단일 사용자 앱 |
| [pingcap/autoflow](https://github.com/pingcap/autoflow) | 2,978 | 2026-10-23 | Apache-2.0 | TiDB 위의 GraphRAG 지식 베이스 |
| [apecloud/ApeRAG](https://github.com/apecloud/ApeRAG) | 1,320 | 2026-09-12 | - | 멀티모달 색인 + MCP |
| [gusye1234/nano-graphrag](https://github.com/gusye1234/nano-graphrag) | 4,005 | 2026-01-27 | MIT | 고치기 쉬운 최소 구현. 읽기용 |
| [circlemind-ai/fast-graphrag](https://github.com/circlemind-ai/fast-graphrag) | 3,965 | 2025-11-01 | MIT | PageRank 기반 탐색. 1년 가까이 멈춤 |

## 에이전트 메모리와 시간 그래프

대화 · 사건이 계속 들어오면서 그래프가 바뀌는 쪽이다. 사실이 언제 참이었는지를 엣지에 남긴다.

| 저장소 | 별 | 최근 푸시 | 라이선스 | 특징 |
|---|---|---|---|---|
| [topoteretes/cognee](https://github.com/topoteretes/cognee) | 32,786 | 2026-10-27 | Apache-2.0 | 에이전트 메모리 플랫폼. 문서 → 그래프 + 벡터 |
| [getzep/graphiti](https://github.com/getzep/graphiti) | 32,250 | 2026-10-26 | Apache-2.0 | 시간 지식 그래프. 지금 참인 것과 예전에 참이던 것을 함께 추적 |
| [deeplethe/utopia](https://github.com/deeplethe/utopia) | 8,556 | 2026-10-27 | Apache-2.0 | 쌍시간 지식 기반. 계약 개정 이력, "그때 누가 담당이었나" 같은 질의. 의심스러운 건은 사람 검토로 넘김 |

## 그래프 구축 SDK

| 저장소 | 별 | 최근 푸시 | 특징 |
|---|---|---|---|
| [neo4j-labs/llm-graph-builder](https://github.com/neo4j-labs/llm-graph-builder) | 5,284 | 2026-10-23 | 비정형 문서 → Neo4j 그래프 |
| [neo4j/neo4j-graphrag-python](https://github.com/neo4j/neo4j-graphrag-python) | 1,320 | 2026-10-26 | Neo4j 공식 GraphRAG 패키지. Text2Cypher 검색기 포함 |
| [FalkorDB/GraphRAG-SDK](https://github.com/FalkorDB/GraphRAG-SDK) | 1,027 | 2026-10-26 | FalkorDB용 |

## 온톨로지와 거버넌스

| 저장소 | 별 | 최근 푸시 | 특징 |
|---|---|---|---|
| [trustgraph-ai/trustgraph](https://github.com/trustgraph-ai/trustgraph) | 2,836 | 2026-10-27 | 온톨로지 기반 운영 그래프 + 에이전트 행동 정책 런타임. 결정마다 근거와 정책 버전을 PROV-O로 남김 |
| [fabio-rovai/open-ontologies](https://github.com/fabio-rovai/open-ontologies) | 1,499 | 2026-10-26 | 온톨로지 변경 계획과 추론 증명서. 정리 완료 |

## 연구 구현

논문 재현 코드다. 대부분 문서 QA 벤치마크(HotpotQA, 2Wiki, MuSiQue) 기준이라 정형 데이터에 그대로 옮겨지지는 않는다. 검색 전략 아이디어만 본다.

| 저장소 | 별 | 학회 | 한 줄 |
|---|---|---|---|
| [OSU-NLP-Group/HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG) | 4,079 | NeurIPS 2024 | 개인화 PageRank로 장기 기억처럼 다중 홉 검색 |
| [JayLZhou/GraphRAG](https://github.com/JayLZhou/GraphRAG) | 1,541 | - | 여러 GraphRAG 방법을 한 틀에서 비교하는 통합 구현. 2025-07 이후 멈춤 |
| [DEEP-PolyU/LinearRAG](https://github.com/DEEP-PolyU/LinearRAG) | 565 | ICLR 2026 | 관계를 뽑지 않고 엔티티 인식과 의미 연결만으로 그래프 구성. LLM 토큰 비용 0. GPL-3.0 |
| [hhy-huang/HiRAG](https://github.com/hhy-huang/HiRAG) | 555 | EMNLP 2025 Findings | 계층 지식 그래프 |
| [LHRLAB/HyperGraphRAG](https://github.com/LHRLAB/HyperGraphRAG) | 459 | NeurIPS 2025 | 다항 관계를 하이퍼엣지로 |
| [zilliztech/vector-graph-rag](https://github.com/zilliztech/vector-graph-rag) | 262 | - | 그래프 DB 없이 엔티티·관계를 Milvus 벡터로 저장해 다중 홉 검색 |
| [BUPT-GAMMA/PathRAG](https://github.com/BUPT-GAMMA/PathRAG) | 378 | - | 경로 단위 검색. 정리 완료 |
| [GraphRAG-Bench/GraphRAG-Benchmark](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark) | 515 | ICLR 2026 | 그래프가 언제 필요한가. 정리 완료 |
| [XMUDeepLIT/MemGraphRAG](https://github.com/XMUDeepLIT/MemGraphRAG) | 231 | KDD 2026 | 메모리 기반 다중 에이전트 GraphRAG |
| [DataArcTech/GraphSearch](https://github.com/DataArcTech/GraphSearch) | 99 | - | 에이전트식 심층 검색 워크플로 |

목록 정리: [DEEP-PolyU/Awesome-GraphRAG](https://github.com/DEEP-PolyU/Awesome-GraphRAG) (2,681, 2026-06 이후 멈춤).

## 이름은 GraphRAG지만 다른 문제

검색 상위에 많이 뜨지만 **코드베이스 이해용 그래프**다. 정형 데이터 그래프 검색과는 문제가 다르다.

| 저장소 | 별 | 한 줄 |
|---|---|---|
| [Graphify-Labs/graphify](https://github.com/Graphify-Labs/graphify) | 136,206 | 코드 · 문서 · PDF를 지식 그래프로. 엣지마다 `EXTRACTED`(원문에 있음) / `INFERRED`(추론함) 표시 |
| [tirth8205/code-review-graph](https://github.com/tirth8205/code-review-graph) | 32,412 | 코드 그래프 MCP |
| [vitali87/code-graph-rag](https://github.com/vitali87/code-graph-rag) | 5,286 | 모노레포 질의·편집 |
| [zzet/gortex](https://github.com/zzet/gortex) | 1,921 | Go 코드 인텔리전스 엔진 |

Graphify의 `EXTRACTED` / `INFERRED` 구분은 엣지 신뢰도(권위 있는 원천으로 이은 것과 추측으로 이은 것)를 표기하는 문제와 같아 표기 방식만 참고할 만하다.

## 그래프 DB

| 저장소 | 별 | 최근 푸시 | 비고 |
|---|---|---|---|
| [apache/age](https://github.com/apache/age) | 4,916 | 2026-10-13 | PostgreSQL 확장으로 동작하는 속성 그래프. openCypher 질의 |
| [FalkorDB/FalkorDB](https://github.com/FalkorDB/FalkorDB) | 10,495 | 2026-10-26 | GraphBLAS 희소 행렬 기반. Graphiti 기본 백엔드 중 하나 |
| [memgraph/memgraph](https://github.com/memgraph/memgraph) | 4,630 | 2026-10-26 | 인메모리, GraphRAG를 주력 용도로 내세움 |
| [vesoft-inc/nebula](https://github.com/vesoft-inc/nebula) | 12,452 | 2026-09-08 | 분산 그래프 DB |

## 다음 정리 후보

1. LightRAG의 AGE 저장소 구현과 EdgeQuake의 AGE 함정 기록을 묶어 "AGE 위의 GraphRAG 운영"으로 정리
2. Graphiti · Utopia의 시간 엣지 모델을 이력 질의 관점에서 정리
