HTTP 공급자는 streamChat에서 공급자별 형식으로 요청을 만들고 스트림을 처리한다.

## streamChat 진입점

`streamChat`은 앱의 모든 TypeScript LLM 호출(수집, 린트, 질의, 연결 시험)이 쓰는 함수다. 콜백 계약은 `onToken`, `onReasoningToken`, `onDone`, `onError` 네 개다([`llm-client.ts#L10-L20`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/llm-client.ts#L10-L20)).

코드 분석: [`src/lib/llm-client.ts#L226-L243`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/llm-client.ts#L226-L243)

무엇을 보는지: CLI 공급자는 HTTP 설정을 만들기 전에 갈라 나간다.

```ts
  if (config.provider === "claude-code") {
    return streamViaClaudeCodeCli(
      config,
      messages,
      config.streamingEnabled === false ? bufferedStreamCallbacks(callbacks) : callbacks,
      signal,
      requestOverrides,
    )
  }

  if (config.provider === "codex-cli") {
    return streamViaCodexCli(
      config,
      messages,
      config.streamingEnabled === false ? bufferedStreamCallbacks(callbacks) : callbacks,
      signal,
      requestOverrides,
    )
  }
```

CLI 전송 모듈은 동적 `import()`로 불러온다. 주석은 Tauri 이벤트 바인딩이 테스트 번들에 섞이지 않게 하려는 것이라고 적는다([`llm-client.ts#L76-L96`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/llm-client.ts#L76-L96)). 스트리밍을 끈 설정이면 `bufferedStreamCallbacks`가 토큰을 모았다가 `onDone` 직전에 한 번에 넘긴다([`llm-client.ts#L22-L35`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/llm-client.ts#L22-L35)). 그래서 호출자는 스트리밍 여부와 관계없이 같은 콜백을 쓴다.

`getProviderConfig`에도 같은 방어가 있다. CLI 공급자로 이 함수에 들어오면 "배선이 잘못됐다"는 예외를 던진다([`llm-providers.ts#L1069-L1077`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/llm-providers.ts#L1069-L1077)).

## HTTP 전송

### 요청과 타임아웃

HTTP 공급자는 `getHttpFetch()`가 돌려주는 fetch로 요청한다. 이 fetch는 Tauri의 Rust 기반 `plugin-http`이므로 브라우저 CORS를 거치지 않는다. `tauri-fetch.ts` 머리 주석은 MiniMax의 `/anthropic`이 CORS 허용 헤더에 `x-api-key`를 빠뜨리고, 일부 중국 클라우드 게이트웨이가 `Authorization`을 빠뜨리는 사례를 이유로 든다([`tauri-fetch.ts#L1-L17`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/tauri-fetch.ts#L1-L17)).

요청 시간 상한은 `requestTimeoutMinutes`(기본 30분)를 1분에서 1,440분 사이로 묶은 값이다([`llm-client.ts#L253-L254`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/llm-client.ts#L253-L254)). 이 상한은 막힌 요청을 끊는 안전망이고, 몇 초 만에 뜨는 "Load failed"는 DNS · TLS · 404 같은 빠른 네트워크 실패라고 주석이 구분한다. 그래서 시간 초과가 실제로 일어났는지 `timeoutFired` 플래그로 따로 기록하고, 빠른 실패는 "엔드포인트 URL, API 키, 연결을 확인하라"는 문구로 바꾼다.

응답 처리에서 눈여겨볼 지점은 셋이다.

- `custom` 공급자가 temperature를 거부하는 400 · 422를 돌려주면, 오류 본문에 "temperature"와 "unsupported" 같은 낱말이 함께 있을 때 temperature를 빼고 한 번 더 보낸다([`llm-client.ts#L187-L204`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/llm-client.ts#L187-L204)).
- 스트림이 끝났는데 본문 글자가 0이고 finish reason이 `length` · `max_tokens` 류면 `truncated: true`로 끝낸다.
- 본문 글자가 0인데 추론(reasoning) 글자가 200자 이상이면 "추론만 하고 답을 내지 않았다"는 진단 오류를 낸다([`llm-client.ts#L474-L500`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/llm-client.ts#L474-L500)). 빈 문자열이 몇 단계 위에서 "분석 없음"으로 조용히 바뀌는 일을 막으려는 것이다.

### 헤더 병합

사용자는 공급자 프리셋마다 게이트웨이용 헤더를 추가할 수 있다. 이 헤더가 인증 헤더를 덮어쓰지 못하게 하는 함수가 `mergeLlmRequestHeaders`다.

코드 분석: [`src/lib/llm-providers.ts#L85-L101`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/llm-providers.ts#L85-L101)

무엇을 보는지: 사용자 헤더를 먼저 넣고 프로토콜 헤더를 나중에 넣어, 이름이 겹치면 프로토콜 쪽이 이긴다.

```ts
export function mergeLlmRequestHeaders(
  custom: Record<string, string> | undefined,
  required: Record<string, string>,
): Record<string, string> {
  const merged = new Map<string, [string, string]>()
  for (const [name, value] of Object.entries(custom ?? {})) {
    const trimmedName = name.trim()
    const trimmedValue = value.trim()
    if (!HTTP_HEADER_NAME_RE.test(trimmedName) || !trimmedValue || /[\r\n]/.test(trimmedValue)) continue
    merged.set(trimmedName.toLowerCase(), [trimmedName, trimmedValue])
  }
  for (const [name, value] of Object.entries(required)) {
    if (!value) continue
    merged.set(name.toLowerCase(), [name, value])
  }
  return Object.fromEntries(merged.values())
}
```

키를 소문자로 비교하므로 `authorization`처럼 대소문자만 바꾼 헤더로 인증 우선순위를 뒤집을 수 없다. 이름이 RFC 토큰 문자가 아니거나 값에 CR · LF가 있으면 버린다. 주석은 설정 UI 검증 이전에 저장된 값이 남아 있을 수 있어서 경계에서 다시 거른다고 적는다.

### 로컬 서버와 Origin 헤더

Ollama 분기는 항상 `Origin: http://localhost`를 붙인다. `localLlmOriginHeader` 위의 긴 주석이 이유를 설명한다([`llm-providers.ts#L105-L153`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/llm-providers.ts#L105-L153)). Tauri `plugin-http`가 웹뷰의 Origin(`tauri://localhost` 또는 Windows의 `http://tauri.localhost`)을 자동으로 넣는데, Ollama 기본 `OLLAMA_ORIGINS`는 후자를 허용하지 않아 Windows에서 403이 났다. 요청 대상의 실제 Origin을 넣으면 LAN의 다른 PC에 있는 Ollama가 거절한다. `http://localhost`는 Ollama 기본 허용 목록에 항상 있으므로 이 값으로 고정했다.

`custom` 공급자는 엔드포인트가 사설 주소일 때만 같은 헤더를 붙인다. `isLocalOrPrivateHttpEndpoint`는 `localhost`, `127.0.0.1`, `::1`, `10.x`, `192.168.x`, `172.16`부터 `172.31`까지를 사설로 본다([`llm-providers.ts#L155-L172`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/llm-providers.ts#L155-L172)). 공개 게이트웨이는 낯선 브라우저 Origin을 거절할 수 있기 때문이다.

### Anthropic wire의 URL과 인증

Anthropic 형식은 `anthropic`, `minimax`, `apiMode: "anthropic_messages"`인 `custom`이 함께 쓴다. 사용자가 붙여 넣는 주소 모양이 제각각이라 URL을 맞추는 함수가 따로 있다.

코드 분석: [`src/lib/llm-providers.ts#L801-L821`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/llm-providers.ts#L801-L821)

무엇을 보는지: `/v1/messages`가 이미 있으면 그대로, `/v1`로 끝나면 `/messages`만, 그 밖에는 `/v1/messages`를 붙인다.

```ts
export function buildAnthropicUrl(base: string): string {
  const trimmed = normalizeMiniMaxAnthropicBase(base).replace(/\/+$/, "")
  if (/\/v\d+\/messages$/i.test(trimmed)) return trimmed
  if (/\/v\d+$/i.test(trimmed)) return `${trimmed}/messages`
  return `${trimmed}/v1/messages`
}

function buildAnthropicHeaders(config: LlmConfig, url: string): Record<string, string> {
  const apiKey = config.apiKey
  const base: Record<string, string> = {
    "Content-Type": JSON_CONTENT_TYPE,
    "anthropic-version": "2023-06-01",
  }
  if (requiresBearerAuth(url)) {
    base.Authorization = `Bearer ${apiKey}`
  } else {
    base["x-api-key"] = apiKey
    base["anthropic-dangerous-direct-browser-access"] = "true"
  }
```

`requiresBearerAuth`는 Anthropic 형식을 흉내 내지만 `Authorization: Bearer`로 인증하는 호스트 목록이다. MiniMax(전역 · 중국), Alibaba Bailian Coding Plan, Xiaomi MiMo Token Plan, Kimi Coding, Moonshot이 들어 있다([`llm-providers.ts#L696-L719`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/llm-providers.ts#L696-L719)). system 프롬프트는 텍스트 블록 배열로 보내고 `cache_control: { type: "ephemeral" }`을 달아 prompt caching을 켠다([`llm-providers.ts#L591-L603`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/llm-providers.ts#L591-L603)). Anthropic wire의 기본 `max_tokens`는 고정값이 아니라 `maxContextSize`에서 응답 예비분을 구해 3으로 나누고 16,384로 자른 값이다([`llm-providers.ts#L924-L927`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/llm-providers.ts#L924-L927)).

$$
\text{max\_tokens} = \max\left(1,\ \min\left(16384,\ \left\lfloor \frac{\text{responseReserve}}{3} \right\rfloor\right)\right)
$$

여기서 3은 "토큰당 약 3글자"라는 근사다. 호출자가 `max_tokens`를 넘기면 그 값이 이긴다.

### OpenAI 호환 wire의 공급자별 보정

OpenAI 형식 본문은 `buildOpenAiCompatibleBody`가 만든 뒤 엔드포인트에 따라 고친다([`llm-providers.ts#L461-L546`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/llm-providers.ts#L461-L546)). 수집 단계가 결정성을 위해 `temperature: 0.1`을 보내는데, 이를 거부하는 모델이 많아서 생긴 분기가 대부분이다.

| 대상 판정 | 보정 | 근거 |
|---|---|---|
| `gpt-5` 이상, `o` 계열 (OpenAI · Azure) | `max_tokens`를 `max_completion_tokens`로, temperature · top_p · top_k 삭제 | [`llm-providers.ts#L391-L420`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/llm-providers.ts#L391-L420) |
| Moonshot · Kimi | temperature 삭제 | [`llm-providers.ts#L422-L430`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/llm-providers.ts#L422-L430) |
| Xiaomi MiMo | `max_completion_tokens`, 추론 off면 `thinking.type=disabled` | [`llm-providers.ts#L432-L459`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/llm-providers.ts#L432-L459) |
| OpenRouter | `reasoning.max_tokens` 또는 `reasoning.effort` | [`llm-providers.ts#L479-L486`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/llm-providers.ts#L479-L486) |
| DeepSeek V4 | `thinking.type` enabled · disabled, `reasoning_effort` | [`llm-providers.ts#L488-L504`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/llm-providers.ts#L488-L504) |
| Ollama | 추론 off면 `reasoning_effort: "none"`, `max`는 `high` | [`llm-providers.ts#L506-L532`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/llm-providers.ts#L506-L532) |
| `custom` (Anthropic 모드 제외) | `extraRequestBody` 병합, 단 `messages` · `model` · `stream` · `__proto__` 등은 무시 | [`llm-providers.ts#L359-L368`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/llm-providers.ts#L359-L368) |

Ollama 분기 주석은 이 보정이 왜 중요한지 구체적으로 적는다. thinking 모델이 토큰 예산을 모두 사고 과정에 쓰고 `content`가 빈 채로 끝나면, 수집 결과 페이지 하나가 사라진다. 그래서 `LlmConfig`에는 채팅용 `reasoning`과 별개로 수집용 `ingestReasoning`이 있고 기본값은 off다([`wiki-store.ts#L37-L46`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/stores/wiki-store.ts#L37-L46)).

이미지 입력 가능 여부도 공급자마다 다르다. `supportsImageInput`은 `codex-cli`를 불가로, MiniMax는 M3 모델만, Zhipu BigModel은 GLM 비전 모델만 가능으로 판정한다([`llm-providers.ts#L777-L786`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/llm-providers.ts#L777-L786)).

### Azure 엔드포인트

Azure는 리소스 주소, 배포 이름, API 버전 세 조각으로 URL을 만든다. `parseAzureOpenAiEndpoint`는 사용자가 호스트만, 배포 경로까지, 전체 채팅 URL까지 어느 모양으로 붙여 넣어도 세 조각으로 나눈다. 경로 안의 배포 이름이 설정의 모델 이름보다 우선하고, 질의 문자열의 `api-version`이 설정값보다 우선한다([`azure-openai.ts#L25-L62`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/azure-openai.ts#L25-L62)). 기본 API 버전은 `2024-10-21`이다([`azure-openai.ts#L1`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/azure-openai.ts#L1)). `custom` 공급자라도 호스트가 `.openai.azure.com`으로 끝나면 같은 Azure 규칙과 `api-key` 헤더를 쓴다.

## 엔드포인트 입력 보정

설정 화면의 엔드포인트 입력란은 포커스를 잃을 때 `normalizeEndpoint`로 주소를 고치고 경고를 보여 준다([`llm-provider-section.tsx#L989`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/components/settings/sections/llm-provider-section.tsx#L989)). 주석이 꼽는 흔한 실수는 둘이다. 요청 경로까지 붙여 넣어 `/chat/completions`가 두 번 붙는 경우, 버전 구간(`/v1`)을 빠뜨리는 경우다.

코드 분석: [`src/lib/endpoint-normalizer.ts#L36-L40`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/endpoint-normalizer.ts#L36-L40)

무엇을 보는지: 언제나 지워도 되는 꼬리와 모드에 따라 다른 꼬리를 나눈다.

```ts
const ALWAYS_WRONG_TAILS = /\/+(chat\/completions|embeddings)\/?$/i
// `/messages` is ambiguous: in anthropic_messages mode our dispatch uses
// it verbatim when present, so we must preserve it. Only strip when the
// configured mode is chat_completions.
const MESSAGES_TAIL = /\/+messages\/?$/i
```

| 입력 상황 | 처리 | 저장을 막는가 |
|---|---|---|
| `http(s)://` 없음 | 끝 슬래시만 지우고 경고 | 아니오 |
| `new URL`로 해석 불가 | 경고 ("호스트 · 포트 · 경로 오타 확인") | 아니오 |
| 숫자 점 표기인데 IPv4 4옥텟이 아님 | 경고 | 아니오 |
| Azure 호스트 | `/chat/completions`와 질의 문자열 제거, 배포 경로는 유지 | 아니오 |
| `/chat/completions`, `/embeddings` 꼬리 | 제거하고 알림 | 아니오 |
| chat 모드에서 `/messages` 꼬리 | 제거하고 알림 | 아니오 |
| chat 모드에서 `/v1`, `/api/paas/v4` 류 버전 구간 없음 | 경고만 | 아니오 |

경고는 저장을 막지 않는다. 주석은 일부 자체 호스팅 게이트웨이가 정말로 맨 호스트에 API를 둔다고 설명한다([`endpoint-normalizer.ts#L4-L19`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/endpoint-normalizer.ts#L4-L19)). 그리고 `getProviderConfig`의 `custom`과 `ollama` 분기가 같은 꼬리를 한 번 더 지운다. 정규화 기능이 생기기 전에 저장된 설정을 위한 이중 방어다([`llm-providers.ts#L1019-L1031`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/llm-providers.ts#L1019-L1031)).
