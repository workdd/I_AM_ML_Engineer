# RAG-MCP: RAG로 MCP 도구 선택의 Prompt Bloat 줄이기

- **논문**: [RAG-MCP: Mitigating Prompt Bloat in LLM Tool Selection via Retrieval-Augmented Generation](https://arxiv.org/abs/2505.03275)
- **PDF**: [arXiv PDF](https://arxiv.org/pdf/2505.03275)
- **저자**: Tiantian Gan, Qiyao Sun
- **소속**: Beijing University of Posts and Telecommunications, Queen Mary University of London
- **arXiv**: 2025-05-06 제출
- **읽은 날짜**: 2026-07-10
- **태그**: #MCP #RAG #ToolSelection #PromptBloat #FunctionCalling #Agent

## 한 줄 요약

MCP 서버/도구가 많아질수록 모든 tool schema를 프롬프트에 넣는 방식은 토큰과 선택 정확도 모두에서 무너진다. RAG-MCP는 **도구 설명을 외부 벡터 인덱스에 저장하고, 질의마다 관련 MCP만 검색해 LLM에 제공하는 tool retrieval 구조**다.

## 문제의식 - 도구가 많아질수록 LLM은 더 똑똑해지는가?

MCP는 LLM이 파일, DB, GitHub, Slack, 검색, 사내 시스템 같은 외부 도구를 표준 방식으로 호출하게 해준다. 문제는 사용 가능한 MCP 서버가 늘어날수록 LLM에게 제공해야 하는 schema, parameter, 사용법 설명도 같이 늘어난다는 점이다.

논문은 이 문제를 **prompt bloat**로 부른다.

- **토큰 비용 증가**: 모든 MCP 설명을 매번 넣으면 프롬프트가 커진다.
- **선택 복잡도 증가**: 비슷한 도구가 많으면 LLM이 어떤 도구를 써야 하는지 헷갈린다.
- **환각성 tool call 증가**: 존재하지 않는 API를 만들거나, 비슷하지만 틀린 도구를 고를 수 있다.
- **인프라 비용 증가**: 모든 MCP 서버를 사전에 활성화하거나 연결해두는 방식은 서버 수가 커질수록 비효율적이다.

핵심 관찰은 간단하다. 질문 하나를 처리하는 데 전체 도구 목록이 필요한 경우는 드물다. RAG에서 전체 문서를 넣지 않고 관련 chunk만 검색하듯이, tool calling에서도 전체 tool registry를 넣지 말고 관련 tool schema만 검색하자는 접근이다.

## 핵심 아이디어 - Tool Schema를 검색 대상으로 보기

RAG-MCP는 MCP 도구 선택을 generation 문제가 아니라 retrieval 문제로 앞단에서 분리한다.

```text
사용자 질의
  -> 질의 임베딩
  -> MCP schema / metadata 벡터 인덱스 검색
  -> top-k 후보 검증
  -> 가장 적절한 MCP schema만 LLM에 주입
  -> LLM이 해당 도구 호출 및 답변 수행
```

일반 RAG가 knowledge retrieval을 수행한다면, RAG-MCP는 **tool retrieval**을 수행한다.

## 방법론 흐름

### 1. MCP Stress Test

논문은 먼저 도구 수가 늘어날 때 LLM의 tool selection이 어떻게 무너지는지 확인하기 위해 stress test를 만든다.

설정은 Needle-in-a-Haystack 테스트와 비슷하다. 하나의 정답 WebSearch MCP를 여러 distractor MCP 사이에 섞어두고, LLM이 올바른 MCP를 선택하고 호출하는지 측정한다. 후보 MCP 수는 1개부터 11100개까지 늘린다.

이 테스트의 목적은 "긴 context 안에 정답 schema가 들어 있어도 LLM이 실제로 잘 찾는가?"를 확인하는 것이다.

### 2. MCP Schema Index

각 MCP 서버의 설명, parameter schema, usage metadata를 외부 인덱스에 저장한다. 논문에서는 Qwen 계열 retriever를 사용해 사용자 질의와 MCP metadata 사이의 semantic similarity를 계산한다.

중요한 점은 이 인덱스가 LLM prompt 바깥에 있다는 것이다. 새 MCP를 추가할 때 모델을 fine-tuning하지 않고 metadata만 index에 추가하면 된다.

### 3. Retrieval

사용자 질의가 들어오면 retriever가 전체 MCP registry에서 관련 후보를 top-k로 찾는다. 이 단계가 기존 MCP client의 "모든 tool schema를 prompt에 넣고 LLM이 고르게 하기"를 대체한다.

### 4. Validation

논문은 retrieved MCP에 대해 synthetic few-shot query를 생성하고 응답을 확인하는 sanity check 단계를 둘 수 있다고 설명한다. 즉, semantic similarity만으로 바로 호출하지 않고, 기본 호환성을 한 번 더 검증하는 구조다.

다만 실험과 설명을 보면 validation이 얼마나 엄밀하게 구현되어 있는지는 제한적으로 드러난다. production에서는 이 단계가 꽤 중요해진다.

### 5. Invocation

최종적으로 선택된 하나의 MCP schema와 parameter 설명만 LLM prompt 또는 function-calling API에 전달한다. LLM은 더 이상 수백 개 도구 중에서 고르는 일을 하지 않고, 선택된 도구를 어떻게 호출할지만 결정한다.

## 실험 결과

논문은 MCPBench의 WebSearch subset에서 세 가지 방식을 비교한다.

| 방식 | Accuracy | Avg Prompt Tokens | Avg Completion Tokens |
|------|----------|-------------------|-----------------------|
| Blank Conditioning | 13.62% | 2133.84 | 162.25 |
| Actual Match | 18.20% | 1646.00 | 23.60 |
| MCP-RAG / RAG-MCP | **43.13%** | **1084.00** | 78.14 |

해석:

- RAG-MCP는 모든 도구 설명을 넣는 Blank Conditioning보다 prompt token을 약 49% 줄인다.
- tool selection accuracy는 13.62%에서 43.13%로 3배 이상 오른다.
- keyword 기반 Actual Match보다도 정확도가 높다.
- completion token은 Actual Match보다 많지만, 논문은 더 많은 reasoning/verification이 정확도 개선과 연결된다고 해석한다.

Stress test 분석에서는 MCP 위치가 30 이하일 때는 성공률이 높지만, 후보 수가 커지고 position이 100을 넘어가면 실패가 지배적으로 늘어난다고 보고한다. 즉, context 안에 정답 도구가 들어 있어도 도구 수가 커지면 "그냥 넣어두면 알아서 찾겠지"가 잘 작동하지 않는다.

## 장점

| 장점 | 의미 |
|------|------|
| Prompt bloat 완화 | 전체 tool schema 대신 관련 schema만 넣어 토큰을 줄임 |
| Tool selection 정확도 개선 | LLM의 선택지를 좁혀 decision overhead를 낮춤 |
| 확장성 | 새 MCP를 index에 추가하면 되므로 모델 재학습 없이 도구 확장 가능 |
| 인프라 효율 | 모든 MCP 서버를 상시 활성화하지 않고 필요한 도구만 사용할 수 있음 |
| Multi-turn context 절약 | 대화가 길어져도 매 턴 전체 tool registry를 반복 주입하지 않아도 됨 |

## 한계와 리스크

| 리스크 | 설명 |
|--------|------|
| Retriever 품질 의존 | 관련 MCP를 retrieval 단계에서 놓치면 LLM은 정답 도구를 볼 수 없음 |
| Single-tool 가정 | 논문 실험은 주로 하나의 올바른 MCP 선택에 가깝고, 복수 도구 조합 workflow는 약함 |
| 평가 범위 제한 | WebSearch 중심의 MCPBench subset이라 실제 사내 도구 생태계와 차이가 있을 수 있음 |
| Validation 불명확 | retrieved MCP를 어떻게 안전하게 검증할지 production 관점의 설계가 더 필요함 |
| 낮은 절대 정확도 | 43.13%는 baseline 대비 크지만, 실서비스 자동 도구 선택 기준으로는 아직 낮음 |
| Metadata 품질 문제 | MCP 설명이 부실하거나 서로 비슷하면 semantic retrieval도 흔들릴 수 있음 |

## 내 작업과의 연결

이 논문은 Codex/Claude Code 스타일의 tool ecosystem, MCP, skill/plugin 구조와 직접 연결된다. 도구가 많아지는 환경에서는 모델에게 모든 도구 설명을 한 번에 보여주는 방식보다, **도구 탐색 자체를 별도 retrieval layer로 빼는 구조**가 운영상 자연스럽다.

특히 개인적으로 신경 쓸 만한 지점은 세 가지다.

1. **MCP 라우터 설계**

   여러 MCP 서버를 연결하는 agent를 만들 때, LLM prompt에 모든 tool schema를 넣는 대신 `tool_index -> candidate tool -> selected schema injection` 구조를 두는 편이 좋다. 이건 agent router, plugin marketplace, internal automation hub에 모두 적용 가능하다.

2. **Tool metadata schema 정리**

   RAG-MCP의 성능은 MCP metadata 품질에 크게 좌우된다. 단순 description만 넣기보다 다음 필드를 구조화해 index에 넣는 편이 실용적이다.

```json
{
  "tool_id": "github_create_issue",
  "server": "github",
  "capability": "Create a GitHub issue in a repository",
  "inputs": ["owner", "repo", "title", "body", "labels"],
  "good_for": ["bug report", "task tracking", "repository workflow"],
  "bad_for": ["reading local files", "deploying services"],
  "examples": [
    "Create an issue in workdd/I_AM_ML_Engineer for summarizing a paper"
  ]
}
```

3. **검증 레이어 필요**

   검색으로 후보를 좁혀도 바로 실행하면 위험하다. 실무에서는 다음 방어선이 필요하다.

   - retrieved tool이 사용자 의도와 맞는지 LLM 또는 rule로 재판정
   - destructive tool은 별도 confirmation 요구
   - tool input schema validation
   - 실행 전 dry-run 또는 capability check
   - 실패 시 후보 2, 3순위로 fallback

## 구현 아이디어

간단한 tool router는 다음 구조로 시작할 수 있다.

```text
MCP registry crawler
  -> tool metadata normalizer
  -> embedding index
  -> query-time retriever
  -> reranker / validator
  -> schema injector
  -> function call executor
  -> trace logger
```

실서비스에 가깝게 만들려면 retrieval 결과를 그대로 믿기보다, 후보별 score와 선택 이유를 남겨야 한다.

```json
{
  "query": "GitHub repo에 paper summary issue 만들어줘",
  "candidates": [
    {
      "tool_id": "github_create_issue",
      "score": 0.86,
      "reason": "Matches repository issue creation intent"
    },
    {
      "tool_id": "github_search_repositories",
      "score": 0.64,
      "reason": "Related to GitHub but not issue creation"
    }
  ],
  "selected_tool": "github_create_issue",
  "needs_user_confirmation": false
}
```

이렇게 trace를 남겨야 tool selection 실패를 나중에 디버깅할 수 있다.

## 결론

RAG-MCP의 핵심 가치는 "MCP 도구 선택을 LLM의 긴 프롬프트 내부 문제가 아니라 검색 문제로 분리한 것"이다. 아이디어 자체는 단순하지만, MCP 서버와 agent tool이 계속 늘어나는 환경에서는 꽤 실용적이다.

다만 논문 결과의 절대 정확도는 아직 낮고, 실험도 WebSearch 중심이라 바로 production-ready라고 보긴 어렵다. 이 논문은 완성된 해법이라기보다 **tool registry가 커질 때 필요한 아키텍처 방향성**, 즉 tool retrieval layer의 필요성을 보여주는 자료로 보는 편이 맞다.
