# LogicRAG: 사전 구축 그래프 없이 질의 논리 구조로 RAG 검색을 조율하기

- **출처**: [GitHub - chensyCN/LogicRAG](https://github.com/chensyCN/LogicRAG)
- **논문**: [You Don't Need Pre-built Graphs for RAG: Retrieval Augmented Generation with Adaptive Reasoning Structures](https://arxiv.org/abs/2508.06105)
- **저자**: Shengyuan Chen, Chuang Zhou, Zheng Yuan, Qinggang Zhang, Zeyang Cui, Hao Chen, Yilin Xiao, Jiannong Cao, Xiao Huang
- **발표**: AAAI 2026
- **arXiv**: 2025-08-08 제출, 2025-11-17 v2
- **읽은 날짜**: 2026-07-02
- **태그**: #RAG #GraphRAG #MultiHopQA #QueryDecomposition #DAG #Retrieval

## 한 줄 요약

코퍼스 전체를 미리 지식 그래프로 만들지 않고, **질문을 하위 문제 DAG로 분해한 뒤 그 논리 의존 순서에 맞춰 검색·요약·답변을 진행하는 GraphRAG 대안**.

## 문제의식 — GraphRAG의 그래프를 꼭 미리 만들어야 하나?

기존 GraphRAG 계열은 문서 간 관계를 미리 그래프로 만들어 복잡한 multi-hop 질문을 잘 풀려고 한다. 문제는 이 전처리 그래프가 실무 환경에서 꽤 무겁다는 점이다.

- **구축 비용**: 대규모 코퍼스를 엔티티/관계/커뮤니티 그래프로 바꾸는 과정이 토큰·시간을 많이 쓴다.
- **업데이트 지연**: 지식베이스가 자주 바뀌면 그래프 재구축 또는 증분 업데이트가 병목이 된다.
- **질문-그래프 불일치**: 실제 질문마다 필요한 논리 구조가 다른데, 사전 구축 그래프가 그 구조와 맞는다는 보장이 없다.

LogicRAG의 관점은 반대다. 문서 그래프를 먼저 만들지 말고, **질문이 요구하는 추론 구조를 test-time에 만들자**는 접근이다.

## 핵심 아이디어 — Query Logic Dependency Graph

LogicRAG는 질문을 하위 문제들로 나누고, 하위 문제 간 선후관계를 DAG로 표현한다.

예를 들어 "프랑스 수도의 시장은 누구인가?"는 다음 의존성을 가진다.

```text
1. 프랑스의 수도는 무엇인가?
2. 그 수도의 시장은 누구인가?

2는 1의 답에 의존한다.
```

이 구조를 검색 계획으로 쓰면, 모든 검색을 원 질문 하나에 밀어 넣는 대신 **먼저 풀어야 하는 정보부터 검색하고, 그 결과를 다음 검색의 조건으로 넘길 수 있다**.

## 방법론 흐름

### 1. 질의 분해와 DAG 구성

LLM이 원 질문을 하위 문제로 분해한다. 이후 각 하위 문제가 어떤 하위 문제의 답에 의존하는지 추론해 방향성 있는 그래프를 만든다. 논문에서는 DAG를 전제로 하며, 위상 정렬 가능해야 한다.

### 2. Graph Reasoning Linearization

DAG 자체는 구조적이지만, RAG 실행은 결국 검색과 생성의 순차 과정이다. LogicRAG는 DAG를 위상 정렬해 실행 순서를 만든다.

```text
질문 -> 하위 문제 DAG -> 위상 정렬 -> 순차 검색/요약/답변
```

이렇게 하면 재귀적으로 얽힌 추론을 한 번의 forward pass에 가까운 순차 절차로 바꿀 수 있다. 중요한 점은 각 단계의 검색이 이전 단계의 요약과 중간 답변을 조건으로 삼는다는 것이다.

### 3. Context Pruning: Rolling Memory

multi-hop RAG는 라운드가 늘수록 누적 context가 커지고 노이즈도 증가한다. LogicRAG는 매 라운드 새로 검색한 chunk를 원 질문 기준으로 요약하고, 기존 요약과 합쳐 **rolling memory** 하나로 압축한다.

이 방식은 긴 원문 chunk를 계속 들고 다니는 대신, 답변에 필요한 핵심 사실만 누적하려는 전략이다. 비용은 줄지만, 요약 단계에서 중요한 세부사항이 빠질 수 있다는 리스크도 있다.

### 4. Graph Pruning: 같은 rank의 하위 문제 병합

논문은 같은 topological rank에 있는 하위 문제들을 하나의 unified query로 합쳐 검색 횟수를 줄이는 graph pruning을 제안한다. 서로 느슨하게 연결된 sibling/leaf 문제를 따로 검색하지 않고 한 번에 검색해 중복 검색을 줄이는 방식이다.

### 5. Sampling Without Replacement

agentic RAG는 불확실한 상태에서 비슷한 subquery를 반복 생성하며 제자리걸음할 수 있다. 논문은 이미 처리한 하위 문제를 제거하며 앞으로 진행하는 **sampling without replacement**를 기본 전략으로 선택한다. 답 품질을 크게 희생하지 않으면서 반복 subquery와 토큰 비용을 줄이려는 장치다.

## 실험 결과

평가는 HotpotQA, 2WikiMultiHopQA, MuSiQue의 multi-hop QA 벤치마크에서 수행했다. 생성·평가 LLM은 GPT-4o-mini, embedding은 `sentence-transformers/all-MiniLM-L6-v2`로 맞췄다.

| 모델 | HotpotQA Str-Acc | 2WikiMQA Str-Acc | MuSiQue Str-Acc |
|------|------------------|------------------|-----------------|
| VanillaRAG Top-5 | 44.1 | 46.7 | 21.0 |
| GraphRAG | 39.6 | 46.3 | 16.5 |
| LightRAG | 47.8 | 43.1 | 18.1 |
| HippoRAG2 | **56.7** | 50.0 | 27.0 |
| LogicRAG | 54.8 | **64.7** | **30.4** |

해석:

- HotpotQA string accuracy는 HippoRAG2가 약간 더 높지만, LogicRAG는 LLM-Acc에서 62.6으로 가장 높다.
- 2WikiMQA에서는 LogicRAG가 다음 최고 baseline보다 큰 폭으로 높다.
- MuSiQue처럼 hop이 늘어나는 데이터셋에서도 기존 graph-based RAG보다 우세하다.
- 2WikiMQA query-time 기준 LogicRAG는 평균 9.83초, 1777.9 토큰으로, GraphRAG/LightRAG/KGP보다 토큰 비용이 낮다. 단, VanillaRAG보다는 느리다.

## 저장소 구현 메모

현재 GitHub 코드의 핵심 클래스는 `src/models/logic_rag.py`의 `LogicRAG`다. 동작은 논문 전체 알고리즘의 단순 구현에 가깝다.

구현 흐름:

1. 원 질문으로 warm-up retrieval 수행
2. 검색 context를 원 질문 기준으로 요약
3. 현재 요약만으로 답 가능한지 LLM에게 JSON으로 판단시킴
4. 부족하면 dependencies 목록을 만들고 LLM으로 dependency pair를 생성
5. `_topological_sort()`로 하위 문제 순서를 정렬
6. 각 dependency를 순서대로 검색하고 rolling summary를 갱신
7. 답 가능하다고 판단되거나 `max_rounds`에 도달하면 최종 답변 생성

주의할 점:

- 논문이 설명하는 "같은 topological rank의 subproblem 병합"은 코드에서 명확한 rank-batch 형태로 보이지 않는다.
- dynamic DAG augmentation도 논문 설명만큼 일반적인 그래프 업데이트보다는, 라운드별 dependency 순회에 가깝다.
- `filter_repeats=True` 옵션으로 반복 chunk를 피할 수 있지만 기본값은 `False`다.
- LLM 출력 JSON 파싱에 많이 의존하므로 production 사용에는 schema validation, retry, structured output 강제가 필요하다.

## 장점

| 장점 | 의미 |
|------|------|
| 사전 그래프 불필요 | 대규모/동적 지식베이스에서 GraphRAG 전처리 병목을 피할 수 있음 |
| 질문별 구조 적응 | 코퍼스 구조가 아니라 질문의 논리 구조를 검색 계획으로 사용 |
| 해석 가능성 | 어떤 하위 문제를 어떤 순서로 풀었는지 추적 가능 |
| 토큰 비용 제어 | rolling memory와 graph pruning으로 context 증가를 억제 |

## 한계와 리스크

| 리스크 | 설명 |
|--------|------|
| LLM 분해 품질 의존 | 질문 분해나 dependency edge가 틀리면 전체 검색 계획이 흔들림 |
| 요약 손실 | rolling memory가 비용을 줄이는 대신 세부 근거를 지울 수 있음 |
| 조기 확신 | 논문도 긴 multi-hop에서 LLM이 부분 정보만으로 premature confidence를 보일 수 있다고 지적 |
| 실무 구현 격차 | 공개 코드가 논문 알고리즘의 모든 pruning/dynamic adaptation을 완전하게 담았는지는 확인 필요 |
| 단순 질문 overhead | 질문이 단순하면 vanilla RAG보다 느릴 수 있음 |

## 내 작업과의 연결

이 레포에서 고민 중인 RAG/환각/검증 흐름과 직접 연결된다.

1. **Agentic RAG 설계**

   기존 agentic RAG가 "다음에 무엇을 검색할지"를 매 라운드 즉흥적으로 정한다면, LogicRAG는 먼저 질문의 논리 DAG를 세워 검색 순서를 제한한다. 이 방식은 subquery 반복과 탐색 흔들림을 줄이는 데 유용하다.

2. **GraphRAG 대안**

   사전 지식 그래프를 만드는 방식은 데이터가 안정적이고 규모가 관리 가능할 때 좋다. 반대로 문서가 자주 바뀌거나 고객별 corpus가 동적으로 들어오는 환경에서는 LogicRAG처럼 query-time structure를 만드는 편이 운영상 단순할 수 있다.

3. **Confidence gating과 결합**

   LogicRAG의 약점은 "현재 요약으로 답 가능한가?"를 LLM 판단에 맡긴다는 점이다. 여기에 logprob confidence, self-consistency, future-context 검출 같은 별도 신호를 붙이면 조기 답변을 줄일 수 있다.

4. **구현 아이디어**

   실무형으로 바꾸려면 `dependencies`를 자유 텍스트 리스트가 아니라 구조화된 객체로 관리하는 편이 좋다.

```json
{
  "id": "dep_1",
  "question": "프랑스의 수도는 무엇인가?",
  "depends_on": [],
  "status": "resolved",
  "evidence_ids": ["chunk_12", "chunk_98"],
  "answer": "Paris",
  "confidence": 0.92
}
```

이렇게 하면 DAG 실행, evidence 추적, confidence gating, 재검색 조건을 한꺼번에 다룰 수 있다.

## 결론

LogicRAG의 핵심 가치는 "GraphRAG의 구조화 이점을 코퍼스 전처리가 아니라 질문 해석 시점으로 옮긴 것"이다. 모든 RAG에 필요한 만능 해법이라기보다는, **multi-hop 질문이 많고 지식베이스가 자주 바뀌며 사전 그래프 구축 비용이 부담스러운 환경**에서 특히 실용적인 선택지다.

다만 공개 코드는 논문 아이디어를 빠르게 실험하는 baseline에 가깝다. production으로 가져가려면 structured output, DAG 상태 저장, evidence attribution, confidence 기반 중단 조건, 요약 손실 방지 장치가 추가로 필요하다.
