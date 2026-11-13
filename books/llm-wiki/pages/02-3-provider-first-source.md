프로젝트를 연 뒤에는 LLM 공급자를 설정하고 첫 원본을 넣어 수집을 시작한다.

## LLM 공급자 선택

프로젝트만 만들어서는 수집이 돌지 않는다. README 빠른 시작의 두 번째 단계가 "Settings 에서 LLM provider 설정"이다 ([README.md L428-L438](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/README.md#L428-L438)).

### 설정 화면 진입

설정은 사이드바 톱니 아이콘이나 `Cmd+,` (Windows · Linux 는 `Ctrl+,`) 로 연다. 단축키는 `App` 에서 전역으로 등록한다 ([App.tsx L146-L153](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/App.tsx#L146-L153)). 설정 화면은 왼쪽 범주 목록과 오른쪽 본문으로 나뉜다. 범주는 15개다 ([settings-view.tsx L83-L97](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/components/settings/settings-view.tsx#L83-L97)).

| 범주 id | 다루는 장 |
|---|---|
| `general`, `interface`, `network`, `maintenance`, `changelog`, `about` | 이 장과 3장 |
| `llm` | 12장 |
| `embedding` | 6장 |
| `multimodal`, `mineru` | 4장 |
| `web-search`, `source-watch`, `scheduled-import` | 10장 |
| `api-server` | 13장 |
| `output` | 4장, 8장 |

### 코드 분석: LLM 범주만 저장 방식이 다른 이유

LLM 범주가 공용 저장 버튼을 쓰지 않는 부분을 본다. [settings-view.tsx L660-L668](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/components/settings/settings-view.tsx#L660-L668)

```tsx
const body = useMemo(() => {
  switch (active) {
    case "general":
      return <GeneralSection draft={draft} setDraft={setDraft} />
    case "llm":
      // (주석 3줄 생략: LLM 범주는 자체 상태를 바로 저장한다)
      return <LlmProviderSection />
```

다른 범주는 draft 를 고친 뒤 화면 아래 저장 버튼을 눌러야 반영된다. LLM 범주는 입력할 때마다 바로 저장하므로 저장 바가 숨겨진다 ([L755-L758](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/components/settings/settings-view.tsx#L755-L758)). 처음 쓰는 사람이 "저장 버튼이 없다"고 헷갈리기 쉬운 지점이다.

### 프리셋 목록과 활성화

LLM 범주는 프리셋 행 목록이다. 프리셋은 [llm-presets.ts L62](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/components/settings/llm-presets.ts#L62) 의 `LLM_PRESETS` 배열에 23개가 정의돼 있다. Anthropic, Claude Code CLI, Codex CLI, OpenAI, Google, Azure OpenAI, DeepSeek, Groq, xAI, NVIDIA NIM, 여러 중국 클라우드, Ollama 로컬 · 클라우드, Custom 이 들어 있다. 사용자가 직접 만든 프리셋도 같은 목록에 붙는다.

한 번에 하나의 프리셋만 활성화된다. 행의 활성 토글은 다음 함수로 연결된다. [llm-provider-section.tsx L121-L126](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/components/settings/sections/llm-provider-section.tsx#L121-L126)

```ts
function toggleActive(id: string) {
  const state = useWikiStore.getState()
  const next = id === state.activePresetId ? null : id
  setActivePresetId(next)
  persist(state.providerConfigs, next).catch(() => {})
}
```

이미 활성인 프리셋을 다시 누르면 `null` 이 되어 LLM 이 꺼진다. `persist` 는 공급자별 설정, 활성 프리셋 id, 그리고 프리셋 기본값에 사용자 덮어쓰기를 합친 최종 `llmConfig` 를 모두 `app-state.json` 에 쓴다 ([L74-L100](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/components/settings/sections/llm-provider-section.tsx#L74-L100)). 모든 프리셋을 끄면 API 키가 비어 있는 상태로 바꿔서 수집 · 중복 제거 · 리뷰 큐가 즉시 멈추게 한다.

### LLM 준비 상태 판정

앱 곳곳의 큐는 같은 함수 하나로 LLM 사용 가능 여부를 판단한다. [src/lib/has-usable-llm.ts L42-L47](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/has-usable-llm.ts#L42-L47)

```ts
export function hasUsableLlm(
  cfg: Pick<LlmConfig, "provider" | "apiKey">,
): boolean {
  if (PROVIDERS_WITHOUT_KEY.has(cfg.provider)) return true
  return (cfg.apiKey ?? "").trim().length > 0
}
```

키가 필요 없는 공급자는 `ollama`, `custom`, `claude-code`, `codex-cli` 네 가지다 ([L21-L26](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/has-usable-llm.ts#L21-L26)). 나머지는 API 키가 비어 있지 않아야 한다. 기본 `llmConfig` 는 `provider: "openai"`, `apiKey: ""` 이므로 ([wiki-store.ts L504-L514](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/stores/wiki-store.ts#L504-L514)) 설치 직후에는 판정이 거짓이고, 원본을 넣어도 수집이 시작되지 않는다.

가장 빨리 시작하는 조합은 다음 셋이다.

| 상황 | 고를 프리셋 | 추가 작업 |
|---|---|---|
| 상용 API 키가 있다 | Anthropic, OpenAI, Google 등 | 키 입력, 모델 선택 |
| Claude Code 또는 Codex 구독이 있다 | Claude Code CLI (local), Codex CLI (local) | CLI 가 PATH 에 있고 로그인돼 있어야 한다. 행에 감지 상태가 표시된다 |
| 로컬 모델만 쓴다 | Ollama (Local) 또는 Custom | 기본 Ollama 주소는 `http://localhost:11434` |

연결 테스트, 작업별 모델 라우팅(채팅과 수집에 다른 모델 쓰기), 프로젝트별 덮어쓰기, CLI 전송 방식은 모두 12장에서 다룬다.

> **주의**: API 키는 앱 데이터 폴더의 `app-state.json` 에 평문 JSON 으로 저장된다. Rust 쪽 에이전트도 이 파일을 직접 읽는다 ([src-tauri/src/lib.rs L405-L434](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/lib.rs#L405-L434)). OS 키체인을 쓰는 코드는 찾지 못했다.

## 첫 원본 추가 절차

LLM 이 활성화되면 README 빠른 시작의 나머지 단계로 넘어간다.

1. Sources 화면에서 PDF, DOCX, Markdown 등을 가져온다. 파일은 `raw/sources/` 로 복사된다 (4장).
2. 오른쪽 Activity 패널에서 수집 진행을 본다. 완료되면 `wiki/` 아래에 페이지가 생기고 `index.md`, `log.md` 가 갱신된다 (4장, 5장).
3. Chat 에서 위키에 질문한다. 답을 위키 페이지로 저장할 수 있다 (8장).
4. Graph 화면에서 위키링크 연결을 본다 (7장).
5. Review 와 Lint 화면에서 사람이 판단할 항목과 위키 상태 문제를 확인한다 (9장).

사이드바의 화면 목록은 chat, wiki, sources, search, graph, lint, review 일곱 개다 ([icon-sidebar.tsx L20-L28](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/components/layout/icon-sidebar.tsx#L20-L28)).
