프론트엔드는 상태 저장소와 invoke 래퍼를 중심으로 구성되고, 핵심 로직 대부분은 src/lib에 있다.

## 프론트엔드 진입점

`src/main.tsx` 는 테마를 먼저 적용한 뒤 React 루트를 그린다 ([src/main.tsx L17-L28](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/main.tsx#L17-L28)). 렌더 전에 테마를 정하는 이유는 화면 깜박임을 막기 위해서다. 루트에는 `App` 과 전역 대화상자 호스트 `AppDialogHost` 두 개가 붙는다. 시작 중 예외가 나면 React 없이 오류 스택을 HTML 로 직접 그린다.

`App` 은 2장에서 본 대로 설정 복원과 프로젝트 열기를 맡는다. 프로젝트가 열리면 `AppLayout` 이 화면을 구성한다. 왼쪽 아이콘 사이드바, 파일 트리와 Activity 패널, 가운데 본문, 오른쪽 리서치 패널 순서다 ([app-layout.tsx L110-L164](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/components/layout/app-layout.tsx#L110-L164)). 본문은 `wiki-store` 의 `activeView` 값으로 바뀐다 ([content-area.tsx L43-L64](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/components/layout/content-area.tsx#L43-L64)). 라우터 라이브러리를 쓰지 않고 상태 값 하나로 화면을 고르는 방식이다.

| `activeView` | 컴포넌트 폴더 | 상세 |
|---|---|---|
| `wiki` | `components/editor/` | 5장 |
| `sources` | `components/sources/` | 4장 |
| `search` | `components/search/` | 6장 |
| `graph` | `components/graph/` | 7장 |
| `chat` | `components/chat/` | 8장 |
| `lint` | `components/lint/` | 9장 |
| `review` | `components/review/` | 9장 |
| `skills`, `settings` | `components/settings/` | 11장, 12장 |

Sources 화면은 한 번 열리면 다른 화면으로 가도 숨김 처리만 하고 언마운트하지 않는다 ([content-area.tsx L20-L36](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/components/layout/content-area.tsx#L20-L36)). 가져오기 진행 상태를 잃지 않으려는 처리로 보인다 (추정).

## 상태 저장소: src/stores

전역 상태는 Zustand 스토어 10개가 나눠 가진다. 모두 `create<State>()` 로 만든 단순 스토어이며 Zustand 의 `persist` 미들웨어는 쓰지 않는다. 디스크 저장은 별도 모듈이 맡는다.

| 스토어 | 줄 수 | 담는 것 | 디스크 저장 |
|---|---|---|---|
| `wiki-store` | 756 | 현재 프로젝트, 파일 트리, 선택 파일, `activeView`, LLM · 임베딩 · 검색 · 프록시 · API 등 모든 설정, 그래프 UI 상태, `dataVersion` | 설정은 각 설정 화면이 `project-store.ts` 로 `app-state.json` 에 저장 |
| `chat-store` | 398 | 대화 목록, 메시지, 스트리밍 상태, 에이전트 · 검색 모드, 선택한 스킬과 문맥 파일 | `auto-save.ts` 가 2초 지연 후 `.llm-wiki/` 에 저장 |
| `review-store` | 182 | 리뷰 항목 | 1초 지연 후 `.llm-wiki/review.json` |
| `lint-store` | 81 | 린트 결과 | 1초 지연 후 `.llm-wiki/lint.json` |
| `research-store` | 127 | 딥 리서치 작업 큐, 패널 열림, 동시 실행 수 | 저장 없음 |
| `activity-store` | 55 | 수집 · 린트 · 질의 진행 항목 | 저장 없음 |
| `file-sync-store` | 22 | 파일 변경 큐, 실행 여부, 마지막 오류 | Rust 쪽이 큐를 관리 |
| `update-store` | 87 | 업데이트 확인 결과, 숨긴 버전, 자동 확인 여부 | `project-store.ts` 의 업데이트 상태 저장 함수 |
| `zoom-store` | 29 | 화면 확대 비율 | `app-state.json` |
| `app-dialog-store` | 74 | 확인 · 경고 대화상자 요청 큐 | 저장 없음 |

### 코드 분석: wiki-store 의 두 LLM 설정

설정 상태 중 LLM 관련 필드를 본다. [src/stores/wiki-store.ts L430-L440](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/stores/wiki-store.ts#L430-L440)

```ts
activeView: "chat" | "wiki" | "sources" | "search" | "graph" | "lint" | "review" | "skills" | "settings"
llmConfig: LlmConfig
/** Persisted global/default config, kept separate while a project override is effective. */
globalLlmConfig: LlmConfig
/** Per-provider-preset stored overrides (API key, model, endpoint, …). */
providerConfigs: ProviderConfigs
customLlmPresets: CustomLlmPreset[]
/** Which preset is currently active. `null` = no LLM configured. */
activePresetId: string | null
taskModelRouting: TaskModelRoutingConfig
projectLlmOverride: ProjectLlmOverride
```

`llmConfig` 는 지금 실제로 쓰는 값이고 `globalLlmConfig` 는 저장된 전역 값이다. 프로젝트별 덮어쓰기가 켜져 있으면 둘이 달라진다. 프로젝트를 열 때 `resolveProjectLlmConfig` 가 전역 값, 공급자별 설정, 프로젝트 덮어쓰기를 합쳐 `llmConfig` 를 다시 계산한다 ([App.tsx L448-L457](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/App.tsx#L448-L457)). 해석 규칙은 12장에서 다룬다.

`dataVersion` 은 숫자 하나다. `bumpDataVersion` 이 1씩 올리고 ([wiki-store.ts L753](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/stores/wiki-store.ts#L753)), 그래프 같은 무거운 화면은 이 값이 바뀌면 캐시를 버리고 다시 계산한다. README 는 이를 "dataVersion signaling" 이라고 부른다 ([README.md L377](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/README.md#L377)).

### 코드 분석: 스토어 구독으로 자동 저장

스토어 변경을 디스크로 내보내는 방식을 본다. [src/lib/auto-save.ts L83-L94](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/auto-save.ts#L83-L94)

```ts
export function setupAutoSave(): void {
  // Auto-save review items (debounced 1s)
  useReviewStore.subscribe((state) => {
    if (suspended) return
    const projectPath = useWikiStore.getState().project?.path
    if (reviewTimer) clearTimeout(reviewTimer)
    reviewTimer = setTimeout(() => {
      if (projectPath) {
        saveReviewItems(projectPath, state.items).catch(() => {})
      }
    }, 1000)
  })
```

스토어마다 `subscribe` 로 변경을 받아 디바운스 후 `.llm-wiki/` 아래 JSON 으로 쓴다. 리뷰 · 린트는 1초, 채팅은 2초이며 채팅은 스트리밍 중이면 저장을 건너뛴다 ([L108-L129](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/auto-save.ts#L108-L129)). `suspended` 플래그는 프로젝트를 바꾸는 순간을 위한 것이다. 전환 중에는 스토어가 비워지는데, 그때 자동 저장이 돌면 빈 배열이 이전 프로젝트 파일을 덮어쓴다. 그래서 `App` 은 프로젝트를 열 때 `runWithSuspendedAutoSave` 안에서 이전 상태를 먼저 쓰고, 스토어를 비우고, 새 상태를 채운 뒤에 자동 저장을 다시 켠다 ([App.tsx L433-L445](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/App.tsx#L433-L445)).

스토어를 비우는 일은 `resetProjectState` 가 한 곳에서 한다. 채팅, 리뷰, 린트, Activity, 리서치 스토어를 초기값으로 돌리고, 모듈 수준 캐시도 정리한다 ([src/lib/reset-project-state.ts L18-L56](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/reset-project-state.ts#L18-L56)). 프로젝트 간 데이터가 섞이지 않게 하는 지점이다.

## 프론트엔드의 invoke 래퍼: src/commands

`src/commands/` 에는 파일 두 개만 있다.

| 파일 | invoke 호출 수 | 감싸는 command |
|---|---|---|
| `fs.ts` | 33 | `commands::fs` 전부, `file_history`, 프로젝트 생성 · 열기, 서버 상태 조회, MCP 서버 경로 |
| `file-sync.ts` | 7 | `commands::file_sync` 전부 |

### 코드 분석: 래퍼의 역할

래퍼가 단순 전달 이상을 하는 예를 본다. [src/commands/fs.ts L22-L35](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/commands/fs.ts#L22-L35)

```ts
export async function writeFile(path: string, contents: string): Promise<void> {
  assertAbsoluteFsPath("writeFile", path)
  return invoke<void>("write_file", { path, contents })
}

export async function writeFileBase64(path: string, base64: string): Promise<void> {
  assertAbsoluteFsPath("writeFileBase64", path)
  return invoke<void>("write_file_base64", { path, base64 })
}

export async function writeFileAtomic(path: string, contents: string): Promise<void> {
  assertAbsoluteFsPath("writeFileAtomic", path)
  return invoke<void>("write_file_atomic", { path, contents })
}
```

래퍼는 세 가지를 더한다. 첫째, 타입이다. `invoke<T>` 의 반환 타입과 인자 이름을 한 곳에 고정한다. Tauri 는 camelCase 인자를 Rust 의 snake_case 로 바꿔 주므로 `extractImages` 가 `extract_images` 로 들어간다. 둘째, 사전 검사다. 쓰기 계열은 절대 경로가 아니면 invoke 전에 예외를 던진다. 셋째, 보강이다. `listDirectory` 는 같은 인자의 동시 요청을 하나로 합치고 ([L48-L63](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/commands/fs.ts#L48-L63)), `createProject` 와 `openProject` 는 Rust 가 돌려준 값에 프로젝트 UUID 를 붙인다 (2장).

그런데 invoke 가 모두 이 폴더를 거치지는 않는다. 하위 시스템 전용 command 는 해당 `lib/` 모듈이나 컴포넌트가 직접 부른다.

| 직접 invoke 하는 곳 | 호출 수 | command |
|---|---|---|
| `lib/embedding.ts` | 10 | `embedding_fetch*`, `vector_*_chunks`, `vector_delete_page`, `vector_legacy_row_count`, `vector_drop_legacy` |
| `lib/claude-cli-transport.ts`, `codex-cli-transport.ts` | 4, 3 | `claude_cli_spawn` · `kill`, `codex_cli_spawn` · `kill` |
| `components/chat/chat-panel.tsx` | 4 | `agent_list_skills`, `agent_start_turn_stream`, `agent_start_turn`, `agent_cancel_turn` |
| `components/settings/settings-view.tsx` | 3 | `set_proxy_env`, `api_server_reload_config`, `set_close_behavior` |
| `components/settings/sections/maintenance-section.tsx` | 3 | `rebuild_wiki_index`, `export_project_archive`, `import_project_archive` |
| `lib/search.ts`, `web-search.ts`, `anytxt-search.ts` | 1씩 | `search_project`, `web_search`, `anytxt_search` |

> **주의**: command 이름 문자열이 여러 파일에 흩어져 있어서 Rust 쪽 이름을 바꾸면 컴파일러가 잡아 주지 않는다. 타입 생성 도구(예: tauri-specta)는 쓰지 않는다. `invoke` 를 검색해 호출처를 모두 찾아야 한다.

## src/lib 모듈 지도

`src/lib/` 에는 테스트를 뺀 소스 파일이 112개(JSON 1개 포함), 테스트 파일이 123개 있다. 앱 로직 대부분이 여기에 있다. 하위 시스템별로 묶으면 다음과 같다.

| 하위 시스템 | 주요 모듈 | 상세 |
|---|---|---|
| 수집 | `ingest.ts`, `ingest-queue.ts`, `ingest-cache.ts`, `ingest-commit-coordinator.ts`, `ingest-sanitize.ts`, `source-preprocess.ts`, `parsed-source-output.ts`, `mineru.ts`, `extract-source-images.ts`, `image-caption-pipeline.ts`, `vision-caption.ts`, `text-chunker.ts`, `detect-language.ts`, `output-language.ts` | 4장 |
| 위키 페이지와 원본 | `frontmatter.ts`, `wiki-filename.ts`, `wiki-page-types.ts`, `wiki-schema.ts`, `templates.ts`, `wikilink-transform.ts`, `enrich-wikilinks.ts`, `page-merge.ts`, `wiki-page-resolver.ts`, `wiki-page-delete.ts`, `wiki-cleanup.ts`, `source-lifecycle.ts`, `source-identity.ts`, `source-delete-decision.ts`, `sources-merge.ts`, `raw-source-resolver.ts`, `markdown-image-resolver.ts` | 5장 |
| 검색과 중복 제거 | `search.ts`, `embedding.ts`, `dedup.ts`, `dedup_embedding.ts`, `dedup-queue.ts`, `dedup-runner.ts`, `dedup-storage.ts`, `knowledge-source-filter.ts` | 6장 |
| 그래프 | `wiki-graph.ts`, `wiki-graph-analysis.ts` (+ worker), `graph-relevance.ts`, `graph-insights.ts`, `graph-search.ts`, `graph-filters.ts`, `graph-visibility.ts` | 7장 |
| 채팅과 질의 | `chat-save-to-wiki.ts`, `chat-agent-types.ts`, `chat-image-utils.ts`, `context-budget.ts`, `greeting-detector.ts`, `agent-file-activity.ts`, `persist.ts` | 8장 |
| 린트와 리뷰 | `lint.ts`, `lint-structural-core.ts` (+ worker), `lint-fixes.ts`, `lint-config.ts`, `review-utils.ts`, `review-create-page.ts`, `review-batch-research.ts`, `sweep-reviews.ts` | 9장 |
| 리서치와 가져오기 | `deep-research.ts`, `optimize-research-topic.ts`, `research-query-grounding.ts`, `web-search.ts`, `anytxt-search.ts`, `scheduled-import.ts`, `url-source-import.ts`, `clip-watcher.ts`, `project-file-sync.ts`, `source-watch-config.ts` | 10장 |
| LLM 공급자 | `llm-client.ts`, `llm-providers.ts`, `llm-task-routing.ts`, `has-usable-llm.ts`, `claude-cli-transport.ts`, `codex-cli-transport.ts`, `azure-openai.ts`, `endpoint-normalizer.ts`, `reasoning-*.ts`, `connection-tests.ts`, `tauri-fetch.ts`, `proxy-config.ts` | 12장 |
| 로컬 API | `api-token.ts`, `api-server-constants.ts` | 13장 |
| 프로젝트와 앱 기반 | `project-store.ts`, `project-identity.ts`, `project-mutex.ts`, `reset-project-state.ts`, `auto-save.ts`, `project-file-tree-refresh.ts`, `file-tree-order.ts`, `path-utils.ts`, `file-types.ts`, `concurrency-limits.ts`, `theme.ts`, `update-check.ts`, `changelog.ts`, `keyboard-utils.ts`, `utils.ts` | 이 장, 2장 |

이 표에서 읽을 수 있는 것은 수집, 중복 제거, 린트, 딥 리서치 같은 LLM 파이프라인이 TypeScript 쪽에 있다는 점이다. Rust 는 그 파이프라인이 쓰는 부품(파일, 파서, 벡터 저장소, 검색, CLI 프로세스)을 command 로 제공한다. 무거운 계산 중 일부는 Web Worker 로 뺐다. 그래프 분석과 구조 린트가 `*.worker.ts` 로 따로 있다.
