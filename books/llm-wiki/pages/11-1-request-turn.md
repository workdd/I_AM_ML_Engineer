에이전트 턴은 요청 검증과 한도 확인을 거쳐 두 경로 중 하나로 시작된다.

## 백엔드 측 한도 관리

`runtime.rs` 머리에 있는 상수들은 비용과 안전 한도다. 주석은 API와 MCP 호출자가 React UI를 거치지 않으므로 한도를 UI가 아닌 백엔드에 둔다고 적는다.

코드 분석: [`src-tauri/src/agent/runtime.rs#L31-L45`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/agent/runtime.rs#L31-L45)

무엇을 보는지: 반복 횟수, 이미지 크기, 출력 토큰, 사용자 입력 폼의 상한.

```rust
// These limits are intentionally enforced in the backend Agent rather than the
// React UI. API and MCP callers bypass the UI, so safety and cost boundaries
// must live here.
const DEFAULT_CHAT_SEARCH_RESULTS: usize = 5;
const MAX_CHAT_SEARCH_RESULTS: usize = 10;
const MAX_IMAGES_PER_TURN: usize = 5;
const MAX_IMAGE_BASE64_BYTES: usize = 7 * 1024 * 1024;
const MAX_AGENT_TOOL_ITERATIONS: usize = 8;
const AGENT_STRUCTURED_MAX_TOKENS: u32 = 8192;
const AGENT_SKILL_STRUCTURED_MAX_TOKENS: u32 = 16384;
const MAX_SKILL_REFERENCE_BYTES: u64 = 256 * 1024;
const MAX_USER_INPUT_FIELDS: usize = 12;
const MAX_USER_INPUT_OPTIONS: usize = 8;
const MAX_USER_INPUT_TEXT_CHARS: usize = 400;
const SHELL_APPROVAL_REQUIRED_OBSERVATION: &str = "shell.exec.approval_required";
```

이미지는 한 턴에 5장, base64 기준 장당 7 MiB까지이고 형식은 png · jpeg · webp · gif만 받는다([`runtime.rs#L3891-L3913`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/agent/runtime.rs#L3891-L3913)). 스킬이 있는 턴은 구조화 출력 상한이 8,192에서 16,384 토큰으로 늘어난다.

## 요청 형식

한 턴의 입력은 `AgentChatRequest` 하나다. UI, API, MCP가 모두 이 구조를 JSON으로 보낸다.

| 필드 | 기본값 | 의미 |
|---|---|---|
| `message` | 필수 | 사용자 메시지. 비어 있으면 `message is required` |
| `sessionId`, `runId` | 없음 | 비어 있으면 UI는 `ui_`, API는 `api_` 접두사로 새로 만든다 |
| `mode` | `standard` | `fast` · `standard` · `deep` · `local_first`. 반복 · 검색 예산을 바꾼다 |
| `retrievalMode` | `standard` | `standard` · `smart` · `faithful`. 근거 수집 방식을 바꾼다 |
| `tools` | `{wiki: true, web: false, anytxt: false}` | 이번 턴에 켤 도구 묶음 |
| `history`, `historyExplicit` | 빈 배열, `false` | 대화 이력. `historyExplicit`가 거짓이고 이력이 비면 저장된 세션 최근 12개를 불러온다 |
| `skills`, `skillMode` | 빈 배열, `explicit` | 켤 스킬 이름과 주입 방식 |
| `contextFiles` | 빈 배열 | 사용자가 `@`로 붙인 프로젝트 상대 경로 |
| `approvedShellCommands`, `shellCommand` | 빈 배열, 없음 | 사용자가 승인한 셸 명령 원문 |
| `images` | 빈 배열 | 첫 모델 호출에만 붙는 이미지 |
| `allowEmptyRetrieval` | `false` | CLI 공급자용 검색 전용 사전 호출 표시 |
| `persistSession` | `true` | 턴을 세션 파일에 남길지 |

출처는 [`types.rs#L88-L170`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/agent/types.rs#L88-L170)이다. `approvedShellCommands` 위에는 "모델 출력, 저장된 대화, 스킬 지시에서 채우면 안 된다"는 주석이 붙어 있다. 이 목록은 사람이 승인한 명령만 담는다는 계약이다.

`AgentRetrievalMode`의 세 값은 주석이 뜻을 밝힌다([`types.rs#L18-L29`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/agent/types.rs#L18-L29)). `standard`는 기존 단일 패스 동작, `smart`는 예산과 무진전 감시 아래 근거 공백을 반복해서 메우는 방식, `faithful`은 raw 원본 발췌만 근거로 쓰는 방식이다. `faithful`은 사용자의 명시적 선택이며 문구로 추론하지 않는다.

## 턴의 진입점과 두 경로

`run_once_with_cancel_and_events`는 메시지와 이미지를 검사하고, 세션 id를 정하고, `AgentStart` · `TurnStart` 이벤트를 낸 뒤 라우터를 돌리고 스킬을 읽는다. 그다음 갈림길이 나온다.

코드 분석: [`src-tauri/src/agent/runtime.rs#L243-L265`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/agent/runtime.rs#L243-L265)

무엇을 보는지: HTTP로 부를 수 있는 LLM이 있으면 루프로, 없으면 아래쪽 검색 전용 경로로 간다.

```rust
        let tool_registry = tools::BuiltinToolRegistry::default();
        if self
            .llm_config
            .as_ref()
            .is_some_and(|cfg| cfg.is_usable_for_backend_http())
        {
            return self
                .run_agent_loop(
                    &request,
                    message,
                    // ... 세션 id, 라우터 결과, 스킬, 정책, 레지스트리, 이벤트 버퍼 등
                    cancellation.as_ref(),
                )
                .await;
        }
```

`is_usable_for_backend_http`는 공급자별로 모델 이름과 키 또는 엔드포인트가 있는지 본다. `claude-code`와 `codex-cli`는 항상 거짓을 돌려준다([`provider.rs#L66-L82`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/agent/provider.rs#L66-L82)). 그래서 CLI 공급자를 쓰면 런타임은 루프에 들어가지 않는다.

채팅 패널은 이 성질을 이용한다. 공급자가 CLI가 아니면 `agent_start_turn_stream`으로 루프를 돌리고, CLI이면 `agent_start_turn`에 `allowEmptyRetrieval: true`, `persistSession: false`를 실어 검색만 먼저 시킨 뒤 받은 참조를 컨텍스트로 붙여 TypeScript의 `streamChat`으로 CLI를 부른다([`chat-panel.tsx#L771-L773`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/components/chat/chat-panel.tsx#L771-L773), [`chat-panel.tsx#L1065-L1073`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/components/chat/chat-panel.tsx#L1065-L1073)). 검색 전용 경로는 모델 계획기를 쓰지 못하므로 `should_fallback_wiki_search`가 참이면 `wiki.search`를 대신 돌린다. 단, 스킬이 있는 턴은 이 대체 검색을 하지 않는다([`runtime.rs#L2907-L2913`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/agent/runtime.rs#L2907-L2913)).

| 공급자 | 런타임 경로 | 답을 쓰는 곳 |
|---|---|---|
| `openai`, `anthropic`, `google`, `azure`, `minimax`, `ollama`, `custom` | `run_agent_loop` | Rust `LlmClient` |
| `claude-code`, `codex-cli` (UI) | 검색 전용 경로 (`allowEmptyRetrieval`) | TypeScript CLI 전송 (12장) |
| LLM 미설정 (API) | 검색 전용 경로 | 참조가 있으면 참조 목록, 없으면 설정 오류 |

API는 `prepare_chat`에서 `allow_empty_retrieval`을 항상 거짓으로 덮어쓴다([`api_server.rs#L2003-L2005`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/api_server.rs#L2003-L2005)). 외부 호출자가 생성기 없이 빈 답을 받는 일을 막고, 설정을 고치라는 오류를 그대로 받게 한다.

## 라우터의 힌트 부착 방식

`route_query`는 메시지에 "web search", "latest", "graph", "write to wiki" 같은 영어 · 중국어 낱말이 있는지 보고 의도 라벨을 붙인다. 그러나 위키 검색을 켜는 결정은 하지 않는다.

코드 분석: [`src-tauri/src/agent/router.rs#L85-L90`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/agent/router.rs#L85-L90)

무엇을 보는지: 메시지 길이나 물음표 같은 겉모양으로 검색을 강제하지 않는다는 원칙.

```rust
    // This router is intentionally conservative. It may label obvious user
    // hints for the final prompt, but it must not infer retrieval from message
    // shape such as length or a question mark. Tool execution is decided by the
    // model planner so capability/meta questions can be answered from the
    // runtime context without an unnecessary wiki search.
    let should_search_wiki = false;
```

라우터 결과는 system 프롬프트의 `Router hint` 한 줄로만 들어간다. 테스트 `router_does_not_force_search_from_question_shape`는 "지금 쓸 수 있는 skill이 뭐가 있나" 같은 질문이 `Ambiguous`로 남고 검색을 켜지 않는지 확인한다([`router.rs#L145-L154`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/agent/router.rs#L145-L154)). 도구 목록을 묻는 질문에 위키를 뒤지는 실수를 막으려는 장치다.
