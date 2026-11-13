Rust 에이전트 쪽 공급자 구현과의 관계, 연결 시험, 설정 보조 유틸을 본다.

## Rust 쪽 공급자와의 관계

11장의 에이전트 루프는 TypeScript `streamChat`을 쓰지 않고 Rust `LlmClient`로 같은 공급자를 직접 부른다([`provider.rs#L151-L215`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/agent/provider.rs#L151-L215)). 같은 wire를 두 언어로 두 번 구현한 셈이다. Rust 쪽 상수도 TypeScript와 맞춰져 있다. Anthropic 버전 `2023-06-01`, Azure API 버전 `2024-10-21`이고, 요청 타임아웃은 180초다([`provider.rs#L13-L16`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/agent/provider.rs#L13-L16)).

| 호출 경로 | 구현 | CLI 공급자 |
|---|---|---|
| 수집 · 린트 · 딥 리서치 · 연결 시험 | TypeScript `streamChat` | 지원 |
| 채팅 (HTTP 공급자) | Rust `AgentRuntime` + `LlmClient` | 해당 없음 |
| 채팅 (CLI 공급자) | Rust 검색 전용 + TypeScript `streamChat` | 지원 |
| HTTP API · MCP의 `/chat` | Rust `AgentRuntime` | 미지원, 설정 오류 반환 |

프로젝트별 모델 덮어쓰기도 두 곳에 있다. TypeScript `resolveProjectLlmConfig`와 Rust `project_llm_config`가 같은 규칙(삭제된 사용자 프리셋은 전역 설정으로 되돌림)을 따르도록 서로를 가리키는 주석이 달려 있다([`llm-task-routing.ts#L14-L31`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/llm-task-routing.ts#L14-L31), [`api_server.rs#L2363-L2440`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/api_server.rs#L2363-L2440)). API 키와 사용자 헤더는 프로젝트 기록에 복사하지 않고 실행 시점에 공급자 자격 증명 기록에서 합친다.

> **주의**: 두 구현은 서로 다른 테스트로 지켜진다. 한쪽에만 들어간 공급자 보정(예: 특정 게이트웨이의 temperature 처리)이 다른 쪽에 없을 수 있으므로, 같은 모델이라도 수집과 채팅의 동작이 다를 가능성이 있다(미검증).

## 연결 시험

설정 화면의 "연결 시험"과 "기능 시험" 버튼은 `connection-tests.ts`를 부른다. 둘 다 실제 `streamChat`을 거치므로 CLI 공급자도 같은 버튼으로 확인한다.

코드 분석: [`src/lib/connection-tests.ts#L93-L123`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/connection-tests.ts#L93-L123) (발췌)

무엇을 보는지: 기능 시험은 정해진 토큰을 그대로 따라 말하는지 본다.

```ts
export async function testLlmFunction(cfg: LlmConfig): Promise<ProviderTestResult> {
  let content = ""
  let errorMessage: string | null = null

  await streamChat(
    cfg,
    [
      {
        role: "system",
        content: "You are a deterministic API test. Do not explain. Output only the requested token.",
      },
      { role: "user", content: "Output exactly this token and nothing else: LLM_WIKI_TEST_OK" },
    ],
```

| 시험 | 보내는 것 | 통과 조건 |
|---|---|---|
| `testLlmConnection` | "Reply with one short word." | 오류 없음, 빈 응답 아님. 걸린 ms와 응답 앞 80자 표시 |
| `testLlmFunction` | `LLM_WIKI_TEST_OK`를 그대로 출력하라는 지시 | 응답에 그 토큰이 들어 있음 |
| `testEmbeddingConnection` | 짧은 문장 하나 | 벡터가 돌아옴. 차원 수 표시 |
| `testEmbeddingFunction` | 같은 문장 두 번 | 두 벡터 차원이 같고, 값이 모두 유한하고, norm이 0보다 큼 |

LLM 시험은 `max_tokens: 512`, `reasoning: { mode: "auto" }`로 보낸다([`connection-tests.ts#L10`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/connection-tests.ts#L10)). 첫 시험은 연결 여부, 두 번째는 지시 준수 여부를 확인한다. 엔드포인트는 맞는데 모델이 추론만 하다 끝나거나 엉뚱한 모델이 걸린 경우를 두 번째 시험이 걸러 낸다.

## 설정 보조 유틸

| 모듈 | 내용 | 쓰는 곳 |
|---|---|---|
| `concurrency-limits.ts` | 사용자 입력 동시성 값을 정수로 내리고 1에서 64 사이로 묶는다 | 임베딩 · 멀티모달 동시성, 원본 감시 설정 |
| `api-token.ts` | Web Crypto로 32바이트 난수를 만들어 base64url(`+`, `/`, `=` 없음)로 인코딩한다 | API 설정 화면의 토큰 생성 버튼 |
| `api-server-constants.ts` | 포트 19828, `http://127.0.0.1:19828`, `/api/v1/health` | API 설정 화면의 상태 표시 |

출처는 [`concurrency-limits.ts#L1-L10`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/concurrency-limits.ts#L1-L10), [`api-token.ts#L1-L23`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/api-token.ts#L1-L23), [`api-server-constants.ts#L1-L3`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/api-server-constants.ts#L1-L3)이다. 토큰은 256비트이고, 셸 · 환경 변수 · 헤더 · 질의 문자열에 이스케이프 없이 붙여 넣을 수 있게 base64url을 골랐다고 주석이 적는다. 이 토큰이 서버에서 어떻게 검사되는지는 13장에서 다룬다.
