채팅 요청 인자와 프롬프트에 들어가는 컨텍스트 구성을 본다.

## 요청 인자: 모드, 히스토리, 검색 범위

코드 분석: 스트리밍 경로가 에이전트에 넘기는 요청을 본다. [chat-panel.tsx#L987-L1006](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/components/chat/chat-panel.tsx#L987-L1006)

```tsx
            await invoke<string>("agent_start_turn_stream", {
              projectId: project?.id ?? "current",
              llmConfig,
              request: {
                message: text,
                sessionId: convId,
                runId: backendRunId,
                mode: sendOptions.agentMode,
                retrievalMode: sendOptions.retrievalMode,
                stream: true,
                tools: {
                  wiki: true,
                  web: sendOptions.useWebSearch,
                  anytxt: sendOptions.useAnyTxtSearch,
                },
                topK: sendOptions.agentMode === "deep" ? 8 : 5,
                includeContent: sendOptions.agentMode === "deep",
                history: activeConvMessages,
                historyExplicit: true,
```

위키 검색 도구는 항상 켜고, 웹 검색과 AnyTXT 검색(6장)은 입력창의 토글을 따른다. 히스토리는 현재 대화의 user, assistant 메시지 중 마지막 `maxHistoryMessages`개이고 기본값은 10이다([chat-store.ts#L119-L123](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/stores/chat-store.ts#L119-L123)). 방금 추가한 질문 메시지는 히스토리에서 빼고 `message`로 따로 보낸다.

| 인자 | 값 | 의미 |
| --- | --- | --- |
| `mode` | `fast`, `standard`(기본), `deep`, `local_first` | 반복 예산과 검색 예산 |
| `retrievalMode` | `standard`(기본), `smart`, `faithful` | 근거를 모으는 방식과 근거로 인정하는 범위 |
| `topK` | deep이면 8, 그 밖에는 5 | `wiki.search` 한 번에 받는 결과 수 |
| `includeContent` | deep이면 `true` | 검색 결과에 페이지 본문 전체를 실을지 |
| `contextFiles` | 입력창에서 `@`로 고른 파일 | 사용자가 명시한 첨부 파일 |
| `skills`, `skillMode` | 선택한 스킬 또는 전체 | 스킬 지시문 (11장) |
| `images` | base64 이미지 | vision 입력 |

에이전트 루프의 반복 횟수와 검색 도구 호출 횟수에는 상한이 있다([runtime.rs#L2798-L2847](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/agent/runtime.rs#L2798-L2847)).

| `mode` | 반복 상한 (스킬 없음) | 검색 상한 `standard` | 검색 상한 `smart` | 검색 상한 `faithful` |
| --- | --- | --- | --- | --- |
| `fast` | 4 | 2 | 3 | 2 |
| `standard`, `local_first` | 8 | 4 | 4 | 3 |
| `deep` | 12 | 8 | 6 | 5 |

`standard` 반복 상한 8은 `MAX_AGENT_TOOL_ITERATIONS` 상수다([runtime.rs#L38](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/agent/runtime.rs#L38)). 스킬을 쓰면 반복 상한이 8, 16, 20으로 늘고, `standard` 검색 모드에서 명시 스킬이 있으면 검색 상한에 4를 더한다. 검색 상한에 닿으면 다음 반복은 도구 없이 최종 답만 요구하는 시스템 프롬프트로 바뀐다.

### 세 가지 검색 모드

`AgentRetrievalMode` 정의의 주석이 세 모드의 뜻을 한 줄씩 적는다([types.rs#L20-L29](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/agent/types.rs#L20-L29)).

| 모드 | 동작 | 근거로 쓰는 것 |
| --- | --- | --- |
| `standard` | 기존의 플래너 중심 검색 | 위키 페이지, 그래프 문맥, 켜져 있으면 웹과 AnyTXT |
| `smart` | 관찰마다 남은 근거 공백만 찾아 질의를 고쳐 다시 검색, 같은 질의 반복 금지 | `standard`와 같음 |
| `faithful` | 첫 반복 전에 `source.search`를 질문 그대로 실행 | 원본 문서 발췌와 명시 첨부 파일만. 위키 페이지, 그래프, 웹, AnyTXT는 근거가 아님 |

`faithful`은 이 장의 주제와 반대 방향의 선택지다. 위키는 LLM이 원본을 요약하고 해석한 2차 산출물이므로, 사용자가 원문 그대로의 근거를 원하면 위키를 건너뛰고 `raw/` 원본을 직접 본다. 이를 프롬프트 지시만으로 맡기지 않고, `overview.md`와 `schema.md`를 컨텍스트에서 아예 지운다.

코드 분석: faithful 모드가 프로젝트 문맥을 비우는 함수를 본다. [runtime.rs#L2783-L2796](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/agent/runtime.rs#L2783-L2796)

```rust
fn project_context_for_retrieval_mode(
    mut project: super::context::ProjectContext,
    retrieval_mode: AgentRetrievalMode,
) -> super::context::ProjectContext {
    if retrieval_mode == AgentRetrievalMode::Faithful {
        // Faithful-source mode must enforce its evidence boundary before prompt
        // construction. overview.md is generated knowledge, and schema.md can
        // contain domain descriptions; asking the model to ignore either would
        // be a soft guarantee rather than source-only context isolation.
        project.overview = None;
        project.schema = None;
    }
    project
}
```

주석의 "soft guarantee"라는 표현이 설계 의도를 보여 준다. "무시하라"는 지시는 모델이 따르지 않을 수 있으니, 근거 경계는 프롬프트를 만들기 전에 데이터 단계에서 강제한다. 같은 이유로 faithful 모드에서는 웹과 AnyTXT 도구를 아예 등록하지 않는다([runtime.rs#L225-L240](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/agent/runtime.rs#L225-L240)).

## 프롬프트 구성 요소

에이전트 컨텍스트는 `build_agent_context`가 시스템과 사용자 두 덩어리로 만든다([context.rs#L57-L62](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/agent/context.rs#L57-L62)). 각 조각의 글자 수 상한은 파일 맨 위에 있다([context.rs#L9-L18](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/agent/context.rs#L9-L18)).

| 위치 | 조각 | 상한 (문자) | 출처 |
| --- | --- | --- | --- |
| 시스템 | 에이전트 역할과 근거 원칙 | 고정 문구 | 코드 상수 |
| 시스템 | 도구 정책, 라우터 힌트, 생성 파일 정책 | 고정 문구 | `route_query` 결과 |
| 시스템 | Project overview | 8,000 | `overview.md` 또는 `wiki/overview.md` |
| 시스템 | Project schema | 6,000 | `schema.md` 또는 `wiki/schema.md` |
| 시스템 | 스킬 목록 또는 선택 스킬 본문 | 12,000 / 18,000 | 스킬 폴더 (11장) |
| 사용자 | 최근 대화 | 12,000 (최대 12개) | 요청의 `history` |
| 사용자 | 사용자가 고른 파일 | 전체 24,000, 파일당 8,000, 최대 8개 | `contextFiles` |
| 사용자 | Retrieved project context | 24,000 | 지금까지 쌓인 `references` |
| 사용자 | Retrieval summary | 8,000 | 루프에서는 빈 문자열 |
| 사용자 | Latest user request | 제한 없음 | `message` |

`load_project_context`가 읽는 파일은 `overview.md`와 `schema.md` 두 개뿐이다([context.rs#L27-L35](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/agent/context.rs#L27-L35)). README의 Phase 4 설명은 시스템 프롬프트에 "purpose.md, language rules, citation format, index.md"가 들어간다고 적지만([README.md#L226-L229](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/README.md#L226-L229)), 현재 Rust 에이전트 컨텍스트에는 `purpose.md`와 `index.md`가 없다. README의 "numbered pages", "[1], [2]" 인용 형식도 현재 시스템 프롬프트에서는 "mention the relevant page paths naturally"로 바뀌었다.

코드 분석: 검색 결과 하나가 프롬프트에 어떤 모양으로 들어가는지 본다. [context.rs#L223-L240](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/agent/context.rs#L223-L240)

```rust
        for (idx, reference) in input.references.iter().enumerate() {
            rendered.push_str(&format!(
                "{}. [{}] {} ({})\n",
                idx + 1,
                reference.kind,
                reference.title,
                reference.path
            ));
            if let Some(snippet) = reference.snippet.as_deref() {
                rendered.push_str(&format!("Snippet: {}\n", collapse_whitespace(snippet)));
            }
            if let Some(context) = reference.knowledge_context.as_ref() {
                if !context.related_to.is_empty() {
                    rendered.push_str(&format!(
                        "Graph neighbors of: {}\n",
                        context.related_to.join(", ")
                    ));
                }
```

참조 한 줄은 번호, 종류(`wiki` 또는 `graph`), 제목, 경로이고, 그 아래 snippet과 그래프 문맥이 붙는다. 그래프 문맥은 `run_wiki_search`가 검색 결과마다 덧붙인 태그, 나가는 링크, 백링크, 링크 수, 최근 편집 이력이다([tools.rs#L942-L997](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/agent/tools.rs#L942-L997)). 6장의 그래프 확장으로 들어온 결과는 종류가 `graph`로 바뀌고 "Graph neighbors of: ..." 줄을 받는다. 기본 모드에서는 페이지 본문이 아니라 snippet만 들어가므로, 모델이 본문이 필요하다고 판단하면 `wiki.read_page` 도구로 다시 읽는다.

마지막으로 `fit_context_to_model`이 전체 길이를 모델 설정의 `max_context_size`에 맞춘다. 이 값을 8,000..400,000 문자로 clamp하고, 넘치면 시스템을 먼저 줄이되 사용자 부분은 최소 4,000자를 남긴다([runtime.rs#L3845-L3867](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/agent/runtime.rs#L3845-L3867)). 단위가 토큰이 아니라 문자다. 잘리는 쪽은 문자열 끝이므로, 사용자 부분이 길면 맨 뒤의 최신 질문이 잘릴 수 있다(코드로 확인한 동작, 실측 미검증).

> **주의**: 모델이 검색을 한 번도 부르지 않고 바로 `final`을 고르면 프롬프트에는 위키 근거가 없다. "Retrieved project context" 절에는 "No matching wiki references were found."만 들어가고, 모델은 일반 지식으로 답한다. 시스템 프롬프트가 근거가 부족하면 그렇다고 말하라고 지시하지만, 검색을 강제하지는 않는다. 플래너를 쓸 수 없는 경우(CLI 공급자 등)에는 런타임이 일반 위키 질문에 한해 `wiki.search`를 대신 실행한다([runtime.rs#L303-L313](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/agent/runtime.rs#L303-L313)).

### CLI 공급자 경로의 프롬프트

CLI 공급자는 `agent_start_turn`(비스트리밍)으로 검색만 받아 온 뒤, 결과를 텍스트로 이어 붙여 사용자 메시지를 만든다.

코드 분석: 검색 결과가 있을 때와 없을 때 프롬프트를 다르게 만드는 부분을 본다. [chat-panel.tsx#L1129-L1143](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/components/chat/chat-panel.tsx#L1129-L1143)

```tsx
        const contextText = retrievedContext
          ? [
              "You have access to the current LLM Wiki project context below. Use it as retrieved evidence when it is relevant.",
              "",
              retrievedContext,
              "",
              `User request: ${text}`,
            ].join("\n")
          : [
              "No relevant LLM Wiki evidence was retrieved for this request.",
              "Answer the request directly with the selected CLI provider. Clearly distinguish general knowledge from project evidence.",
              "",
              `User request: ${text}`,
            ].join("\n")
```

근거가 없을 때 "general knowledge"와 "project evidence"를 구분하라고 명시한다. 추론 모델이 답 없이 사고 과정만 내보내면(`isReasoningOnlyResponseError`) 추론을 끄고 한 번 더 스트리밍한다.

## context-budget.ts 사용 현황

`context-budget.ts`는 원래 채팅 패널이 프롬프트를 채울 때 쓰던 예산 계산기다. 머리 주석은 아직 "used by chat-panel when packing the prompt"라고 적고 있고, `tryAddPage in chat-panel`도 언급한다([context-budget.ts#L1-L33](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/context-budget.ts#L1-L33)). 그러나 기준 커밋에서 이 모듈을 가져다 쓰는 곳은 `llm-providers.ts`와 `ingest.ts` 두 곳뿐이고, 채팅 프롬프트 조립은 Rust `fit_context_to_model`로 옮겨 갔다.

코드 분석: 예산 비율 상수와 계산을 본다. [context-budget.ts#L55-L91](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/context-budget.ts#L55-L91) 중 일부

```ts
const DEFAULT_MAX_CTX = 204_800
const RESPONSE_RESERVE_FRAC = 0.15
const INDEX_BUDGET_FRAC = 0.05
const PAGE_BUDGET_FRAC = 0.5
const PER_PAGE_FRAC = 0.3
const PER_PAGE_FLOOR = 5_000
```

| 예산 | 비율 | 기본 `maxCtx` 204,800자일 때 |
| --- | --- | --- |
| `responseReserve` | 15% | 30,720 |
| `indexBudget` | 5% | 10,240 |
| `pageBudget` | 50% | 102,400 |
| `maxPageSize` | `pageBudget`의 30%, 최소 5,000, 최대 `pageBudget` | 30,720 |

README는 "60% wiki pages, 20% chat history, 5% index, 15% system"으로 적지만([README.md#L221-L224](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/README.md#L221-L224)), 코드의 배분은 페이지 50%, 색인 5%, 응답 여유 15%이고 나머지 약 30%를 히스토리와 시스템 몫으로 남긴다. 현재 쓰임 중 하나는 Anthropic 계열 요청의 기본 `max_tokens`다. `deriveAnthropicMaxTokens`가 `responseReserve`를 3으로 나눠 토큰으로 근사하고 1..16,384로 자른다([llm-providers.ts#L924-L927](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/llm-providers.ts#L924-L927)). 단위가 문자라는 점은 주석이 "quirky unit"이라고 인정한다.
