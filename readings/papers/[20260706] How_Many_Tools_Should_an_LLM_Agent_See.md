# How Many Tools Should an LLM Agent See?: LLM Agent에게 몇 개의 도구를 보여줘야 하는가

- **논문**: [How Many Tools Should an LLM Agent See? A Chance-Corrected Answer](https://arxiv.org/html/2605.24660)
- **저자**: Vyzantinos Repantis, Ameya Gawde, Harshvardhan Singh, Joey Blackwell II
- **소속**: Meta Platforms
- **arXiv**: 2605.24660v2, 2026-06-07
- **읽은 날짜**: 2026-07-06
- **태그**: #LLMAgent #ToolSelection #Retrieval #MCP #BitsOverRandom #RL

## 한 줄 요약

LLM agent에게 도구를 몇 개 보여줄지 고정된 Top-K로 정하지 말고, **랜덤으로 맞힐 확률을 보정한 Bits-over-Random(BoR) 지표로 query별 shortlist 깊이를 평가하고 조절하자**는 논문.

## 문제의식 — Top-K는 왜 애매한가?

LLM agent나 MCP 기반 시스템을 만들면 결국 같은 질문을 만난다.

```text
사용자 요청이 들어왔을 때, 전체 도구 중 몇 개를 LLM에게 보여줄 것인가?
```

도구를 너무 적게 보여주면 정답 도구가 후보에 없을 수 있다. 반대로 너무 많이 보여주면 context token이 낭비되고, LLM이 비슷한 도구들 사이에서 잘못 고를 가능성이 커진다.

실무에서는 보통 `K=5`, `K=10`, 혹은 "가능한 도구 전부" 같은 고정값을 둔다. 문제는 query 난이도가 모두 다르다는 점이다.

- 쉬운 query는 top-1만 보여줘도 충분할 수 있다.
- 중간 난이도 query는 top-3~5 안에 정답이 있을 수 있다.
- 어려운 query는 top-20까지 봐야 정답 도구가 나올 수 있다.

그런데 단순 `Success@K`는 K가 커질수록 자연스럽게 올라간다. 도구 100개 중 50개를 보여주면 정답이 포함될 확률이 커지는 것은 retrieval이 똑똑해서가 아니라 랜덤으로도 쉬워졌기 때문이다. 이 논문은 바로 이 부분을 보정해야 한다고 본다.

## 핵심 아이디어 — Bits-over-Random

BoR는 "현재 깊이 K에서 시스템의 성공률이 랜덤 선택보다 얼마나 나은가?"를 bit 단위로 측정한다.

단일 정답 도구가 있는 경우, 전체 도구 수가 `N`, 보여준 도구 수가 `K`라면 랜덤으로 정답 도구를 포함할 확률은 단순히 다음과 같다.

```text
P_random = K / N
```

시스템의 실제 성공률을 `P_system`이라고 하면 BoR는 다음 로그 비율이다.

```text
BoR = log2(P_system / P_random)
```

해석은 직관적이다.

| BoR | 의미 |
|-----|------|
| 0 bit | 랜덤과 같은 수준 |
| 양수 | 랜덤보다 선택성이 좋음 |
| 음수 | 랜덤보다 못함 |
| 1 bit 증가 | 랜덤 대비 선택성이 2배 좋아짐 |

중요한 점은 K가 커질수록 `P_random`도 커진다는 것이다. 즉, 정답 도구를 찾더라도 너무 많은 도구를 보여주면 reward가 작아진다. 이 때문에 BoR는 별도의 강한 depth penalty를 설계하지 않아도 자연스럽게 짧은 shortlist를 선호한다.

## 방법론 흐름

### 1. Tool retrieval을 depth 선택 문제로 분리

논문은 tool-use 성능을 한꺼번에 보지 않고 다음 세 단계를 분리한다.

1. 정답 도구가 후보 목록에 들어왔는가?
2. LLM이 후보 중 정답 도구를 골랐는가?
3. 선택한 도구 실행이 실제로 성공했는가?

이 논문이 주로 다루는 것은 1번이다. 즉, LLM에게 보여줄 후보 목록의 길이 K를 어떻게 평가하고 조절할 것인가에 집중한다.

### 2. RL agent가 STOP / CONTINUE 결정

저자들은 query 하나를 하나의 episode로 보고, retrieval scorer가 정렬한 도구 목록을 앞에서부터 하나씩 확인하는 MDP로 만든다.

- **State**: 현재까지 본 score들, top score와 현재 score의 gap, score spread, 현재 K, registry size N, 현재 K에서의 BoR ceiling
- **Action**: `CONTINUE` 또는 `STOP`
- **Reward**: STOP했을 때 정답 도구가 포함되어 있으면 현재 K의 BoR reward, 아니면 0

RL agent 자체는 deliberately simple하게 둔다. 논문의 목적은 production용 agent 구조를 제안하는 것이 아니라, BoR reward가 depth 선택 신호로 쓸 만한지 확인하는 데 있다.

### 3. Self-pruning property

BoR의 가장 중요한 성질은 **K가 커질수록 성공의 가치가 자동으로 낮아진다**는 점이다.

예를 들어 도구 500개 중 1개를 맞히는 문제에서 K=1로 맞히는 것은 매우 선택적인 성공이다. 하지만 K=100을 보여주고 맞히는 것은 랜덤 기준으로도 훨씬 쉬운 일이다. 그래서 BoR reward는 K가 커질수록 줄어든다.

논문은 이를 self-pruning property라고 부른다. "많이 보여주면 penalty"를 사람이 따로 세게 넣는 것이 아니라, 랜덤 성공률 보정 때문에 수학적으로 자연스럽게 깊이가 억제된다.

## 실험 설정

논문은 세 가지 tool-selection benchmark에서 BoR를 평가한다.

| 벤치마크 | 도구 수 | 특징 |
|----------|---------|------|
| BFCL | 370 | Berkeley Function Calling Leaderboard simple category를 전체 registry 문제로 재구성 |
| MetaTool | 199 | tool-usage awareness 데이터, embedding/BM25 scorer 비교 |
| ToolBench | 3,251 | 대규모 tool registry, tool-level matching |

비교 대상은 크게 두 가지다.

- **Fixed-K**: 모든 query에 같은 K를 적용
- **F1 ablation**: 정답 포함 여부와 depth를 반영하지만, BoR처럼 랜덤 성공률을 보정하지 않는 reward

## 주요 결과

### 1. BFCL: BoR는 FK=50에 가까운 coverage를 훨씬 짧은 목록으로 달성

BFCL+BM25 조건에서 BoR agent는 평균 `K=7.4±2.5`만 보여주면서 `90.3±2.4%` found rate를 달성했다. 이는 `FK=50`의 `90.8%`에 거의 근접한다.

해석하면, 무조건 50개 도구를 보여주는 대신 평균 7개 정도만 보여줘도 비슷한 수준으로 정답 도구를 후보에 넣을 수 있었다는 뜻이다. 도구 설명 하나가 수백 token을 먹는 실제 agent 시스템에서는 이 차이가 꽤 크다.

embedding scorer를 쓰면 scorer가 더 강해져서 정답 도구가 앞쪽에 더 자주 온다. 이때 BoR agent는 평균 `K=1.4±0.1`까지 더 짧게 멈춘다. 같은 reward를 썼는데도 scorer 품질에 따라 policy가 달라진다는 점이 중요하다.

### 2. ToolBench: aggregate coverage만 보면 Fixed-K가 좋아 보이지만, hard query 복구는 BoR가 한다

ToolBench에서는 BoR agent가 `K=4.4±0.4`에서 `61.9±0.6%` found rate를 얻었다. `FK=5`는 `64.7%`로 aggregate coverage가 더 높고, `FK=20`은 `77.3%`까지 올라간다.

하지만 bucket별로 보면 그림이 달라진다.

| 난이도 | gold rank | BoR 동작 | 해석 |
|--------|-----------|----------|------|
| easy | 1위 | `K=2.5±0.2`, found 100% | 쉬운 query에서는 짧게 멈춤 |
| medium | 2~5위 | `K=4.8±0.5`, found `74.4±0.4%` | 필요할 때 조금 더 탐색 |
| hard | 6~20위 | `K=5.7±0.5`, found `16.7±4.3%` | FK=5/F1/FK=1이 0%인 영역에서 일부 복구 |
| very hard | 21위 이상 | `K=6.9±0.7`, found 0.2% | 완전한 복구는 어렵지만 더 깊게 시도 |

즉 `FK=5`는 쉬운/중간 query를 잘 커버해서 전체 평균은 높지만, 정답이 6위 이후에 있는 hard query에서는 아무것도 못 찾는다. BoR는 전체 평균에서 약간 손해를 보더라도 어려운 query에서 더 깊게 들어가 일부를 회수한다.

### 3. Scorer 품질이 나쁘면 BoR는 "깊게 봐야 한다"는 신호를 드러낸다

MetaTool에서 BM25는 found@1이 33%로 약했다. 이 조건에서 BoR agent는 평균 `K=80.7`까지 확장하며 `96.2%` found rate를 얻지만, selectivity는 낮다.

이건 성공이라기보다 진단 신호에 가깝다. scorer가 좋지 않으면 적은 K에서 멈출 근거가 없고, BoR는 그 문제를 "많이 봐야 겨우 찾는다"는 형태로 드러낸다. 반면 embedding scorer에서는 MiniLM이 `K=2.3`, BGE가 `K=2.4` 수준으로 멈춘다.

이 결과는 실무적으로 중요하다. 고정 K만 보면 retrieval scorer가 약한지, K 설정이 나쁜지 분리하기 어렵다. BoR는 scorer 품질과 depth 선택 문제를 더 잘 드러낸다.

### 4. Downstream tool choice에서도 짧은 adaptive list가 유리하다

논문은 BFCL에서 Claude Sonnet 4.6을 사용해 "후보에 정답 도구가 있을 때 LLM이 실제로 그 도구를 고르는가?"도 확인했다.

| Method | Presented% | Choice Acc% | End-to-End% | Avg K |
|--------|------------|-------------|-------------|-------|
| BoR | 76.9±0.4 | 93.1±0.5 | 71.7±0.0 | 2.2±0.4 |
| F1 | 72.8±2.7 | 94.3±1.7 | 68.6±1.7 | 1.7±0.3 |
| FK=5 | 84.2 | 87.1 | 73.3 | 5.0 |
| FK=1 | 65.0 | 100.0 | 65.0 | 1.0 |

여기서 핵심은 `Choice Acc%`다. 정답 도구가 후보 안에 있을 때, BoR의 선택 정확도는 `93.1±0.5%`이고 FK=5는 `87.1%`다. 후보를 많이 보여줄수록 정답이 포함될 확률은 올라가지만, LLM이 헷갈릴 distractor도 같이 늘어난다.

medium difficulty query에서는 차이가 더 명확하다. FK=5는 정답 도구를 항상 포함하지만 Claude가 정답을 고른 비율은 60.9%에 그쳤다. BoR는 정답 도구를 포함하는 비율은 `62.3±2.0%`로 낮지만, 포함된 경우 선택 정확도는 `76.8±2.5%`였다.

## 장점

| 장점 | 의미 |
|------|------|
| Chance correction | K가 커져서 쉬워진 성공을 보정해 retrieval 깊이를 더 공정하게 평가 |
| Token 비용 절감 | 평균 shortlist 길이를 줄여 tool description context 비용을 줄일 수 있음 |
| Distractor 감소 | LLM이 비슷한 도구들 사이에서 헷갈리는 문제를 줄임 |
| Scorer 진단 | 약한 scorer는 BoR policy가 깊게 탐색하는 형태로 드러남 |
| Query별 적응 | 쉬운 query는 짧게, 어려운 query는 더 깊게 탐색 |

## 한계와 리스크

| 리스크 | 설명 |
|--------|------|
| 실행 성공은 미평가 | 논문은 주로 정답 도구가 shortlist에 들어왔는지를 본다. 실제 tool execution correctness는 범위 밖이다. |
| benchmark 재구성 | 기존 BFCL/MetaTool/ToolBench는 search depth 평가용으로 설계된 데이터가 아니라, 논문이 candidate set을 재구성했다. |
| oracle reward 의존 | 학습에는 query별 gold tool 정보가 필요하다. 실제 운영에서는 로그, human feedback, synthetic labeling 등이 필요하다. |
| scorer가 너무 약하면 깊게 본다 | MetaTool+BM25처럼 scorer 품질이 낮으면 BoR agent가 거의 전체 도구를 보려 할 수 있다. |
| multi-tool query 일반화 | 실험은 대부분 query당 정답 도구 하나인 설정이다. 여러 도구 조합이 필요한 agent task에는 추가 설계가 필요하다. |

## 내 작업과의 연결

이 논문은 MCP/tool registry를 쓰는 agent 설계와 바로 연결된다.

1. **MCP tool filtering**

   MCP 서버나 agent framework에서 tool 목록이 커지면 "몇 개를 model에게 노출할 것인가"가 성능과 비용을 동시에 좌우한다. 단순 Top-K 대신 BoR 기반 평가를 붙이면 K=5가 정말 적절한지, 혹은 query별 adaptive K가 필요한지 판단할 수 있다.

2. **RAG의 context selection**

   문서 retrieval에서도 비슷한 문제가 있다. Top-20을 넣어 답이 맞았다고 해서 retriever가 좋은 것은 아니다. K가 커질수록 랜덤 포함 확률도 올라간다는 관점을 적용하면, RAG context depth를 평가할 때도 더 엄격한 기준을 둘 수 있다.

3. **confidence gating과 결합**

   현재 고민 중인 logprob confidence gating과도 잘 맞는다. 예를 들어 tool retrieval 단계에서는 BoR로 shortlist 깊이를 정하고, tool 선택 단계에서는 LLM의 logprob/confidence로 실행 여부를 gate할 수 있다.

```text
query
  -> tool scorer ranking
  -> BoR/adaptive-K로 shortlist 결정
  -> LLM tool choice
  -> logprob/confidence gate
  -> tool execution
```

4. **실무 구현 아이디어**

   production에서는 RL부터 도입하기보다, 먼저 offline evaluation metric으로 BoR를 붙이는 편이 현실적이다.

```json
{
  "query_id": "q_001",
  "registry_size": 370,
  "gold_tool": "calculate_triangle_area",
  "rank": 6,
  "k": 10,
  "success_at_k": true,
  "random_success_prob": 0.027,
  "bor_bits": 5.21
}
```

이 로그를 쌓으면 scorer별, domain별, query 난이도별로 "우리 시스템이 랜덤 대비 얼마나 선택적인가"를 볼 수 있다. 이후 충분한 데이터가 쌓이면 heuristic adaptive-K나 학습 기반 policy로 넘어갈 수 있다.

## 결론

이 논문의 핵심 가치는 "tool retrieval에서 K 자체를 평가 대상으로 삼았다"는 점이다. 기존에는 정답 도구가 후보에 들어왔는지만 봤다면, BoR는 **그 성공이 현재 K에서 랜덤보다 얼마나 의미 있는 성공인지**를 묻는다.

실무적으로는 당장 RL agent를 붙이는 것보다, 먼저 BoR를 offline metric으로 도입하는 편이 좋아 보인다. tool registry가 커지고 MCP/tool calling이 일반화될수록 "무조건 Top-K"는 점점 약한 기본값이 된다. BoR는 그 기본값을 query별, scorer별로 점검할 수 있는 깔끔한 기준이다.
