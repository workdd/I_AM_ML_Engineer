# Agentic AI: Single vs Multi-Agent Systems 정리

- **출처**: [Medium - Agentic AI: Single vs Multi-Agent Systems](https://medium.com/data-science-collective/agentic-ai-single-vs-multi-agent-systems-e5c8b0e3cb28)
- **작성자**: Ida Silfverskiöld
- **게시일**: 2025-10-28
- **읽은 날짜**: 2026-07-06
- **태그**: #AgenticAI #MultiAgent #SingleAgent #LangGraph #LangSmith #WorkflowDesign

## 한 줄 요약

Agentic AI에서 중요한 선택은 "에이전트를 몇 개 쓸 것인가" 자체가 아니라, **도구 접근 범위, 데이터 흐름, state, 역할 분리, orchestration을 얼마나 명확히 설계할 것인가**다.

## 글의 배경

이 글은 LangGraph와 LangSmith Studio를 사용해 tech trend research agent를 만든 뒤, 단일 에이전트 방식과 multi-agent 방식의 결과를 비교한다.

사용 사례는 다음과 같다.

```text
최근 하루/일주일 동안 tech 분야에서 무엇이 트렌드인지 수집하고,
사용자 persona에 맞춰 newsworthy한 내용을 요약한다.
```

핵심은 데이터 소스 자체가 아니라, 같은 structured data source를 두고 agentic workflow를 어떻게 설계하느냐에 따라 결과 품질과 제어 가능성이 어떻게 달라지는지 보는 것이다.

## Agentic AI에 대한 관점

글에서 LLM은 "모든 것을 알아서 처리하는 지능"이라기보다, structured system 위에 올라가는 자연어 기반 communication layer에 가깝다.

중요한 전제는 다음과 같다.

- LLM은 모호한 자연어를 해석하는 데 강하다.
- 하지만 clean data, tool, system boundary가 없으면 쉽게 부정확해진다.
- 숫자 계산, API 호출, 정형 데이터 처리처럼 명확한 작업은 전통적인 코드가 여전히 낫다.
- agentic system의 가치는 LLM과 programmatic logic을 적절히 조합하는 데 있다.

즉 "LLM에게 모든 것을 맡기기"가 아니라, LLM이 잘하는 부분과 코드가 잘하는 부분을 분리해야 한다.

## Single-Agent Workflow

단일 에이전트 방식은 하나의 LLM에 여러 도구와 하나의 system prompt를 제공하고, 사용자의 요청에 따라 에이전트가 알아서 도구를 선택하게 하는 구조다.

```text
user request
  -> single LLM agent
  -> tool A / tool B / tool C / ...
  -> final answer
```

장점은 단순하다.

| 장점 | 설명 |
|------|------|
| 빠른 구현 | 하나의 agent와 prompt만 관리하면 됨 |
| 빠른 실행 | orchestration 단계가 적어 latency가 낮음 |
| 디버깅 진입장벽 낮음 | 처음 agentic workflow를 실험하기 좋음 |
| Q&A에 적합 | 사람이 중간에 계속 질의하며 방향을 잡는 작업에 잘 맞음 |

하지만 핵심 문제는 control이다. 하나의 agent에게 너무 많은 도구와 선택지를 주면, 어떤 도구를 언제 써야 하는지 모델이 일관되게 따르지 못할 수 있다.

글의 단일 에이전트 예제는 trending keyword API 몇 개를 호출해 tech summary를 만든다. 결과는 빠르고 그럴듯하지만, 깊게 파고들지는 못한다. 특히 복잡한 research workflow에서는 agent가 일부 단계를 생략하거나 shortcut을 택할 수 있다.

## Multi-Agent Workflow

multi-agent 방식은 하나의 큰 agent에게 모든 일을 맡기지 않고, 역할과 도구를 나눈 여러 agent/team으로 workflow를 구성한다.

글의 예시는 크게 두 팀으로 나뉜다.

| 팀 | 역할 |
|----|------|
| Research team | 트렌드 키워드와 관련 데이터를 수집 |
| Editing team | 수집된 내용을 정리하고 최종 briefing으로 편집 |

각 agent는 제한된 도구와 명확한 지시를 가진다. 이 구조는 특히 작은 모델이나 빠른 모델을 쓸 때 유리하다. 모델 하나가 복잡한 전체 프로세스를 모두 기억하고 실행하는 대신, 각 agent가 좁은 역할에 집중하기 때문이다.

```text
user request
  -> lead / coordinator
  -> research team
      -> keyword agent
      -> source/context agent
      -> notes/research pad
  -> editing team
      -> editor/summarizer
  -> final report
```

글에서는 shared research pad를 도입한다. research team이 findings를 쓰고, editing team이 이를 읽어 최종 요약을 만든다. 대안으로 state 안의 scratchpad를 둘 수도 있지만, 어느 team/agent가 어떤 memory를 볼지 설계해야 한다.

## Single vs Multi-Agent 비교

| 항목 | Single Agent | Multi-Agent |
|------|--------------|-------------|
| 구현 난이도 | 낮음 | 높음 |
| 실행 속도 | 빠름 | 느림 |
| 도구 제어 | 약함 | agent별 scope 제한 가능 |
| 결과 깊이 | 얕아지기 쉬움 | 단계별 수집/편집으로 깊어질 수 있음 |
| 디버깅 | 단순하지만 내부 결정이 뭉침 | 복잡하지만 역할별 trace 가능 |
| 비용 | 낮은 편 | agent 수와 tool call 증가로 높음 |
| 적합한 작업 | 단순 질의, human-in-the-loop Q&A | research, report generation, multi-step workflow |

핵심은 multi-agent가 항상 더 좋다는 것이 아니다. multi-agent는 더 많은 control을 제공하지만, 그만큼 architecture를 먼저 설계해야 한다.

## LangGraph 관점

글은 LangGraph를 graph-based agent framework로 사용한다. 단일 에이전트 예제는 비교적 단순한 구조이고, multi-agent 예제는 dynamic routing과 team 단위 구성이 들어간다.

실무적으로 보면 LangGraph의 장점은 다음이다.

- node/edge로 workflow를 명시할 수 있다.
- LangSmith Studio에서 실행 trace를 볼 수 있다.
- agent, tool, state, route를 분리해 구성할 수 있다.
- multi-step workflow를 실험하기 좋다.

반대로 추상화가 많기 때문에, framework를 그대로 쓰더라도 내부에서 어떤 state가 어디로 흐르는지 이해하지 못하면 디버깅이 어려워질 수 있다.

## 좋은 데이터 소스가 먼저다

글에서 반복되는 메시지는 clean structured data의 중요성이다.

agent가 아무리 복잡해도 입력 데이터가 지저분하면 output도 불안정해진다. 반대로 API가 잘 정리되어 있고, 도구가 명확한 schema로 데이터를 주면 agent는 훨씬 안정적으로 동작한다.

따라서 agentic system 설계 순서는 다음에 가깝다.

```text
1. 신뢰할 수 있는 데이터 소스 확보
2. 도구/API boundary 정의
3. programmatic logic으로 처리 가능한 부분 분리
4. LLM이 판단해야 하는 부분만 agent에게 맡김
5. 필요한 경우 역할별 agent/team으로 분해
```

특히 "코드로 할 수 있으면 코드로 하라"는 메시지가 중요하다. trend keyword를 가져오고 필터링하는 deterministic step은 굳이 LLM에게 맡기지 않아도 된다. LLM은 해석, 선택, 요약, persona 맞춤 판단처럼 애매한 부분에서 쓰는 편이 낫다.

## State와 Memory 설계

multi-agent workflow에서 state는 성능과 비용에 직접 영향을 준다.

글의 예제는 모든 message를 state에 계속 넣고 여러 agent가 접근하게 하는 방식에 가깝다. 저자도 이 부분을 개선점으로 지적한다.

문제는 다음과 같다.

- 불필요한 context가 계속 쌓여 token 비용이 증가한다.
- agent별로 필요 없는 정보까지 보게 되어 판단이 흐려질 수 있다.
- research team과 editing team의 memory boundary가 불명확해진다.

더 나은 설계는 agent/team별 scratchpad를 분리하는 것이다.

```json
{
  "research_state": {
    "queries": [],
    "findings": [],
    "evidence": []
  },
  "editing_state": {
    "selected_findings": [],
    "draft": "",
    "open_questions": []
  },
  "user_profile": {
    "persona": "tech investor",
    "time_window": "last week"
  }
}
```

이렇게 하면 어떤 정보가 누구에게 필요한지 명확해지고, context window와 비용도 더 잘 제어할 수 있다.

## 한계와 개선점

글의 마지막에서 저자는 현재 workflow가 작동은 하지만 아직 개선할 부분이 많다고 본다.

| 개선점 | 설명 |
|--------|------|
| user query 구조화 | 사용자의 자연어 요청을 persona, time window, topic, output format 등으로 파싱 |
| guardrail | agent가 반드시 필요한 도구를 쓰도록 강제하거나 검증 |
| error handling | API 실패, 빈 결과, rate limit, parsing failure 처리 |
| state 분리 | 모든 agent가 모든 message를 보는 구조 개선 |
| summarization compression | research doc이 길어질수록 요약/압축 전략 필요 |
| long-term memory | 사용자의 반복 선호와 관심사를 장기적으로 반영 |

이 개선점들은 실제 production agent에서 거의 필수에 가깝다.

## 내 작업과의 연결

이 글은 최근 정리한 tool selection, OpenWiki, LogicRAG와 같은 흐름으로 볼 수 있다.

| 글 | 연결 지점 |
|----|----------|
| How Many Tools Should an LLM Agent See? | 단일 agent에 도구를 너무 많이 주면 distractor가 늘어나는 문제 |
| OpenWiki | agent가 필요한 structured context를 지속적으로 참조하게 하는 방식 |
| LogicRAG | 복잡한 질문을 하위 구조로 나누고 순서대로 처리하는 접근 |

실무 agent 설계 관점에서는 다음 기준이 유용하다.

1. **Single agent로 시작**

   작업이 단순하고, 실패해도 사람이 바로 보정할 수 있고, 도구 수가 적다면 single agent가 더 낫다.

2. **복잡도가 늘면 workflow로 먼저 분해**

   곧바로 multi-agent로 가기보다 deterministic step, retrieval step, summarization step을 먼저 workflow로 분리한다.

3. **역할과 state boundary가 명확할 때 multi-agent화**

   research, validation, editing처럼 책임이 분명하고 서로 다른 tool set이 필요할 때 agent를 나누는 편이 좋다.

4. **shared memory는 신중하게 설계**

   모든 agent가 모든 state를 보는 구조는 빠르게 비대해진다. team별 scratchpad, evidence store, final report draft를 분리해야 한다.

## 결론

이 글의 핵심은 multi-agent가 single-agent보다 항상 우월하다는 이야기가 아니다. 중요한 것은 **복잡한 작업일수록 agent에게 자유도를 더 주는 것이 아니라, 역할·도구·데이터 흐름을 더 명확히 설계해야 한다**는 점이다.

single agent는 빠르고 단순하지만 control이 약하다. multi-agent는 더 깊은 결과와 명확한 역할 분리를 줄 수 있지만, state, routing, error handling, tool boundary를 설계하지 않으면 복잡도만 늘어난다.

결국 agentic AI는 "LLM을 몇 개 붙일까"의 문제가 아니라, 소프트웨어 시스템 설계 문제다. LLM은 workflow 안에서 판단과 언어 처리를 담당하는 구성요소이고, 안정적인 결과는 clean data, programmatic logic, tool design, state management가 함께 맞아야 나온다.
