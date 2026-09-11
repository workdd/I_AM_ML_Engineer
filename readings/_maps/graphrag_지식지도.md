# GraphRAG 지식 지도

저장소에 쌓인 GraphRAG 계열 정리 7건이 서로 어디에 놓이는지 그린 지도입니다. 개별 노트는 각자 완결돼 있지만, **어느 것이 어느 것을 반박하고 보완하는지는 본문 링크로만 흩어져 있어** 따로 모았습니다.

- **기준 좌표계**: [LLM on Graphs 서베이](../papers/%5B20260910%5D%20LLM_on_Graphs_Comprehensive_Survey.md) (arXiv:2312.02783)
- **갱신**: 2026-09-11

## 범례

| 표시 | 뜻 |
| --- | --- |
| 초록 테두리 | 정리가 있습니다 |
| 노랑 테두리 | 다른 노트에서 스쳐 지나갔고 전용 정리는 없습니다 |
| 분홍 바탕 | 아직 다루지 않았습니다 |

관계도의 모양과 선은 다음을 뜻합니다.

| 요소 | 뜻 |
| --- | --- |
| 문서 모양 | 논문 정리 |
| 둥근 사각형 | 블로그 정리 |
| 원통 | 저장소 정리 |
| 굵은 초록 테두리 | 이 계열의 기준선 |
| 빨간 점선 | 반박 관계 |
| 초록 실선 | 개선 관계 |
| 회색 실선 | 계보나 공급 관계 |

## 좌표계: 어디를 읽었나

서베이가 제시한 격자입니다. **무엇이 최종 예측을 내놓느냐**로 역할이 갈립니다.

```mermaid
block-beta
  columns 4
  space h1["LLM as Predictor<br/>LLM이 최종 출력"] h2["LLM as Encoder<br/>GNN이 최종 출력"] h3["LLM as Aligner<br/>둘을 정렬"]
  r1["Pure Graphs<br/>텍스트 없는 그래프"] a1["읽음 · 5건<br/>Graph as Sequence"] a2["빈칸"] a3["빈칸"]
  r2["Text-Attributed<br/>노드·엣지에 텍스트"] b1["읽음 · 2건<br/>Graph as Sequence"] b2["빈칸"] b3["빈칸"]
  r3["Text-Paired<br/>그래프 전체에 설명"] c1["부분<br/>서베이에서만"] c2["빈칸"] c3["빈칸"]

  classDef hdr fill:#f0ede7,stroke:#b6b0a6,color:#5b554e
  classDef row fill:#f0ede7,stroke:#b6b0a6,color:#1e1c1a
  classDef read fill:#eef3ee,stroke:#3d6b45,stroke-width:2px,color:#1e1c1a
  classDef part fill:#f9f4e8,stroke:#b8a86a,color:#1e1c1a
  classDef gap  fill:#fdf6f4,stroke:#d9b3a8,color:#a8503c
  class h1,h2,h3 hdr
  class r1,r2,r3 row
  class a1,b1 read
  class c1 part
  class a2,a3,b2,b3,c2,c3 gap
```

**읽은 것이 한 열에 몰려 있습니다.** 일곱 건 전부 `LLM as Predictor` 안의 `Graph as Sequence`, 곧 그래프를 텍스트로 풀어 컨텍스트에 넣는 방식입니다. GNN 을 학습시키는 나머지 여섯 칸은 손대지 않았습니다.

## 계열 내부 관계

```mermaid
flowchart TB
  SUR@{ shape: doc, label: "**LLM on Graphs 서베이**<br/>9칸 격자로 전체 지형을 정의" }
  LAND@{ shape: rounded, label: "**Graph RAG의 모든 것**<br/>패턴 5종 · 구현체 3종" }
  MS@{ shape: rounded, label: "**Microsoft GraphRAG**<br/>커뮤니티 요약 · Local · Global · DRIFT" }
  LOGIC@{ shape: doc, label: "**LogicRAG**<br/>사전 그래프를 만들지 말고<br/>질의 시점에 DAG 를 세운다" }
  ROG@{ shape: doc, label: "**ROGRAG**<br/>그래프는 두고 검색을<br/>logic form 다음 dual-level 로" }
  OKF@{ shape: rounded, label: "**Open Knowledge Format**<br/>포맷 표준 제안" }
  CRAB@{ shape: lin-cyl, label: "**OpenCrab**<br/>온톨로지 공장 · MCP 30종" }

  SUR ==>|"한 칸으로 좁히면"| LAND
  LAND ==>|"대표 구현"| MS
  MS -.->|"반박 · 전처리가 비싸다"| LOGIC
  MS -->|"개선 · 검색을 여러 겹으로"| ROG
  OKF --> CRAB
  CRAB -.->|"그래프를 공급"| MS

  linkStyle 0,1 stroke:#5b554e,stroke-width:2.5px
  linkStyle 2 stroke:#a8503c,stroke-width:2.5px
  linkStyle 3 stroke:#3d6b45,stroke-width:2.5px
  linkStyle 4 stroke:#8a837a,stroke-width:1.5px
  linkStyle 5 stroke:#b6b0a6,stroke-width:1.5px

  classDef paper fill:#ffffff,stroke:#3d6b45,stroke-width:2px,color:#1e1c1a
  classDef blog  fill:#f7f5f0,stroke:#8a837a,stroke-width:1.5px,color:#1e1c1a
  classDef repo  fill:#f2f0ea,stroke:#5b554e,stroke-width:1.5px,color:#1e1c1a
  classDef base  fill:#eef3ee,stroke:#3d6b45,stroke-width:3px,color:#1e1c1a
  class SUR,LOGIC,ROG paper
  class LAND,OKF blog
  class CRAB repo
  class MS base
```

가운데 **Microsoft GraphRAG가 기준선**입니다. LogicRAG는 그 전처리 비용을 문제 삼아 그래프를 아예 없애자는 쪽이고, ROGRAG는 그래프를 유지한 채 검색 단계를 쌓는 쪽입니다. **같은 대상을 두고 정반대 방향으로 갈라집니다.**

## 노트 목록

| 노트 | 위치 | 역할 |
| --- | --- | --- |
| [LLM on Graphs 서베이](../papers/%5B20260910%5D%20LLM_on_Graphs_Comprehensive_Survey.md) | 최상위 좌표계 | 9칸 격자와 미해결 목록 |
| [Graph RAG의 모든 것](../blogs/%5B20260910%5D%20devto_Graph_RAG%EC%9D%98_%EB%AA%A8%EB%93%A0_%EA%B2%83.md) | 계열 지형도 | 디자인 패턴 5종, RAPTOR, MS GraphRAG, AWS Toolkit |
| [Microsoft GraphRAG 해부](../blogs/%5B20260908%5D%20TowardsAI_Microsoft_GraphRAG_%EB%8F%99%EC%9E%91%EC%9B%90%EB%A6%AC_%EB%8B%A8%EA%B3%84%EB%B3%84.md) | 기준선 각론 | 인덱싱 8단계, 토큰 예산 배분, 정렬 기준 |
| [LogicRAG](../papers/%5B20260702%5D%20LogicRAG_Adaptive_Reasoning_Structures.md) | 반박 | 질의 시점 DAG, rolling memory |
| [ROGRAG](../papers/%5B20260831%5D%20ROGRAG_Robustly_Optimized_GraphRAG.md) | 개선 | dual-level과 logic form 다단계, ablation |
| [Open Knowledge Format](../blogs/%5B20260614%5D%20PyTorchKR_Open_Knowledge_Format_OKF.md) | 표현 계층 | 지식 포맷 표준 제안 |
| [OpenCrab](../repos/%5B20260322%5D%20AlexAI-MCP_OpenCrab_MCP%EB%A1%9C_%EB%B6%99%EC%9D%B4%EB%8A%94_%EC%98%A8%ED%86%A8%EB%A1%9C%EC%A7%80_%EA%B3%B5%EC%9E%A5.md) | 구축 도구 | 온톨로지 공장, MCP 도구 30종 |

## 읽는 순서

처음 보는 사람에게 권하는 순서입니다.

```mermaid
flowchart LR
    A["1. Graph RAG의 모든 것<br/>전체 판을 먼저 잡는다"]
    B["2. Microsoft GraphRAG 해부<br/>기준선이 실제로 어떻게 도는지 본다"]
    C["3. LogicRAG · ROGRAG<br/>두 갈래 반응을 대조한다"]
    D["4. LLM on Graphs 서베이<br/>지금 읽은 것이 전체의 어디인지 확인한다"]
    A --> B --> C --> D
```

서베이를 마지막에 두는 이유가 있습니다. **격자만 먼저 보면 추상적이라 안 남습니다.** 구체적인 구현을 먼저 읽고 나서 좌표를 확인해야 "우리가 한 열에만 있었구나"가 체감됩니다.

## 비어 있는 곳

지도를 그리고 나서 드러난 빈칸입니다.

- **LLM as Encoder 계열 전체.** LLM을 텍스트 인코더로만 쓰고 GNN이 최종 예측을 내는 방식입니다. 사내 문서망이나 이슈 트래커처럼 노드에 텍스트가 붙은 데이터에는 이쪽이 더 맞을 수 있는데 한 건도 없습니다.
- **LLM as Aligner 계열 전체.** 대조 학습으로 LLM 임베딩과 GNN 임베딩을 같은 공간에 맞추는 방식입니다.
- **Text-Paired Graphs.** 분자처럼 그래프 하나가 개체이고 캡션이 붙은 경우입니다. 서베이에서 개념만 읽었고 전용 정리는 없습니다.
- **LightRAG, HippoRAG, KAG.** 지형도 노트가 다루지 않았고 ROGRAG가 참고 구현으로 언급만 합니다.
- **비용 비교.** 일곱 건 중 어느 것도 인덱싱 LLM 호출 비용을 정량으로 다루지 않습니다. GraphRAG 계열 전체의 공통 공백입니다.

마지막 항목이 실무에서 가장 아픕니다. **구조는 일곱 건이나 읽었는데 "얼마나 드는가"에 답할 자료가 없습니다.**
