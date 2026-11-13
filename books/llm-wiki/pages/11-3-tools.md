에이전트가 쓰는 도구 목록과 권한, 쓰기 도구의 안전장치, 셸 실행 승인을 본다.

## 도구 목록과 권한

`builtin_tool_specs`는 도구 14개를 이름, 설명, 효과(`Read` · `Write` · `Network` · `Process`), JSON 스키마로 정의한다([`tools.rs#L433-L640`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/agent/tools.rs#L433-L640)). 실제 실행은 `BuiltinToolRegistry::execute`의 `match name` 한 곳에서 갈린다([`tools.rs#L103-L288`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/agent/tools.rs#L103-L288)).

| 도구 | 효과 | 켜지는 조건 | 필요 권한 |
|---|---|---|---|
| `wiki.search` | Read | `tools.wiki`, faithful 아님 | `SearchWiki` |
| `wiki.read_page` | Read | `tools.wiki`, faithful 아님 | `ReadProject` |
| `graph.search` | Read | `tools.wiki`, faithful 아님 | `ReadProject` |
| `source.search` | Read | `tools.wiki` | `ReadSource` |
| `wiki.write_page` | Write | `tools.wiki`, faithful 아님 | `WriteWiki` |
| `web.search` | Network | `tools.web`, faithful 아님 | `Network` |
| `anytxt.search` | Network, Read | `tools.anytxt`, faithful 아님 | `Network` |
| `skill.read_file` | Read | 스킬이 하나 이상 | `ReadProject` |
| `workspace.write_file`, `workspace.append_file` | Write | 스킬이 하나 이상 | `WriteWiki` |
| `shell.exec` | Read, Process | 스킬이 하나 이상, 승인 또는 작업 공간 한정 | `Process` |
| `user.ask` | 없음 | 루프 프로토콜에 항상 예시로 실린다. 도구 목록 안내는 스킬이 있을 때만 | 없음 |
| `deep_research.run` | Network, Read | 루프에서는 항상 거절 | 해당 없음 |
| `llm.generate`, `skills.load` | Network, Read | 이벤트 이름으로만 쓰인다 | 해당 없음 |

"켜지는 조건"은 `build_agent_loop_user`가 모델에게 보여 주는 목록([`runtime.rs#L3025-L3049`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/agent/runtime.rs#L3025-L3049))과 `require_tool_permission`의 검사를 합친 것이다. 모델이 목록에 없는 도구를 불러도 두 번째 검사에서 거절되어 관찰로 돌아간다.

### 권한 정책

`PermissionPolicy`는 능력(capability) 목록이다. 그런데 API 기본 정책은 `RunDeepResearch`를 뺀 모든 능력을 허용한다.

코드 분석: [`src-tauri/src/agent/permissions.rs#L23-L42`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/agent/permissions.rs#L23-L42)

무엇을 보는지: `Process`가 허용 목록에 있지만 별도 승인 없이는 쓰이지 않는다는 주석.

```rust
    pub fn api_default() -> Self {
        Self {
            allowed: vec![
                AgentCapability::ReadProject,
                AgentCapability::ReadSource,
                AgentCapability::SearchWiki,
                AgentCapability::SearchWeb,
                AgentCapability::SearchAnyTxt,
                AgentCapability::WriteWiki,
                AgentCapability::Network,
                // Process remains inert unless AgentChatRequest carries a
                // separately approved exact shell command. Do not populate that
                // approval list from model output or persisted conversation data.
                AgentCapability::Process,
            ],
        }
    }
```

런타임은 턴마다 `PermissionPolicy::api_default()`만 쓴다([`runtime.rs#L198`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/agent/runtime.rs#L198)). 그래서 실제로 도구를 가르는 것은 정책보다 요청의 `tools` 플래그, `retrievalMode`, 스킬 유무, 셸 승인 목록이다. 능력 체계는 정책을 바꿔 끼울 자리를 만들어 둔 구조로 보인다(추정).

코드 분석: [`src-tauri/src/agent/runtime.rs#L3519-L3533`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/agent/runtime.rs#L3519-L3533)

무엇을 보는지: faithful 모드의 근거 경계를 프롬프트가 아니라 실행 검사로 강제한다.

```rust
fn require_tool_permission(
    tool: &str,
    request: &AgentChatRequest,
    permission_policy: &PermissionPolicy,
) -> Result<(), String> {
    if request.retrieval_mode == AgentRetrievalMode::Faithful
        && matches!(
            tool,
            "wiki.search" | "wiki.read_page" | "graph.search" | "web.search" | "anytxt.search"
        )
    {
        return Err(format!(
            "{tool} is unavailable in faithful-source mode; use source.search"
        ));
    }
```

faithful 모드는 이 검사 말고도 두 겹이 더 있다. 루프 시작 전에 `source.search`를 사용자 메시지로 한 번 돌리고, `project_context_for_retrieval_mode`가 `overview.md`와 `schema.md`를 컨텍스트에서 지운다([`runtime.rs#L2783-L2796`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/agent/runtime.rs#L2783-L2796)). 주석은 overview가 LLM이 생성한 지식이므로 "무시하라"고 부탁하는 것은 약한 보장일 뿐이라고 적는다.

## 쓰기 도구의 안전장치

### wiki.write_page

위키 쓰기는 기본이 생성 전용이다. 경로는 `wiki/`로 시작하고 `.md`로 끝나야 하며 숨김 경로와 `..`를 거절한다([`tools.rs#L2723-L2748`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/agent/tools.rs#L2723-L2748)). 본문은 2 MiB까지이고, 전역 `WIKI_WRITE_LOCK`으로 동시 쓰기를 한 줄로 세운다.

코드 분석: [`src-tauri/src/agent/tools.rs#L711-L746`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/agent/tools.rs#L711-L746) (아래는 앞부분 발췌)

무엇을 보는지: 덮어쓰기 거절, 심볼릭 링크 거절, 쓰기 후 바이트 검증과 되돌리기.

```rust
    // Create-only by default. Prompt injection in retrieved context must not be
    // able to silently truncate an existing wiki page.
    if path.exists() && !allow_overwrite {
        return Err(
            "wiki.write_page refuses to overwrite an existing page without allowOverwrite=true"
                .to_string(),
        );
    }
    if path
        .symlink_metadata()
        .map(|metadata| metadata.file_type().is_symlink())
        .unwrap_or(false)
    {
        return Err("wiki.write_page refuses to overwrite a symlink".to_string());
    }
```

이어지는 코드는 쓰기 전 버전을 `file_history`에 `baseline`으로 남기고, 임시 파일에 쓴 뒤 바꿔치기(`atomic_replace_file`)하고, 다시 읽어 바이트가 다르면 이전 내용으로 되돌리거나 새 파일을 지운다. 성공하면 `agent` 작성자로 버전을 하나 더 남긴다. 검색된 컨텍스트에 숨은 프롬프트 인젝션이 기존 페이지를 지우는 경우를 주석이 직접 위협으로 든다. 루프는 성공 시 `FileChanged` 이벤트에 이전 내용(`previousContent`)을 실어 UI가 즉시 되돌리기를 보여 줄 수 있게 하는데, HTTP API로 나갈 때는 이 값을 지운다(13장).

### workspace.write_file과 append_file

스킬이 만드는 HTML, SVG, 스크립트 같은 산출물은 프로젝트 안의 `agent-workspace/` 아래에만 쓴다([`workspace.rs#L1-L17`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/agent/workspace.rs#L1-L17)). 경로가 `wiki/`나 `raw/`로 시작하거나 숨김 구간을 포함하면 거절한다([`tools.rs#L2750-L2774`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/agent/tools.rs#L2750-L2774)). 파일당 2 MiB이고, 되돌리기 스냅샷은 512 KiB 이하 파일만 만든다([`tools.rs#L31-L37`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/agent/tools.rs#L31-L37)). 폴더 이름을 숨김으로 두지 않은 이유는 사용자가 생성물을 앱 메타데이터 폴더를 뒤지지 않고 찾게 하려는 것이라고 주석이 적는다.

## shell.exec와 승인

셸 실행은 가장 경계가 많은 도구다. 단계별로 보면 다음과 같다.

1. 스킬이 하나도 없는 턴에서는 거절한다([`runtime.rs#L1900-L1909`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/agent/runtime.rs#L1900-L1909)).
2. 스킬 선호 설정 파일을 조회만 하는 명령(`test -f` 류)은 실행하지 않고 건너뛰었다고 관찰을 돌려준다.
3. 명령이 `approvedShellCommands`의 한 항목과 공백을 뺀 문자열로 정확히 같거나, 작업 공간 한정 명령이면 바로 실행한다.
4. 둘 다 아니면 `shell.exec.approval_required` 관찰을 만들고 그 자리에서 턴을 끝낸다. UI는 승인 버튼을 띄우고, 승인되면 `shellCommand`와 `approvedShellCommands`에 같은 명령을 넣어 다시 요청한다.

코드 분석: [`src-tauri/src/agent/runtime.rs#L3664-L3686`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/agent/runtime.rs#L3664-L3686)

무엇을 보는지: 승인 없이 실행해도 되는 "작업 공간 한정" 판정의 첫 단계.

```rust
fn is_shell_command_scoped_to_agent_workspace(command: &str, project_path: &str) -> bool {
    let command = command.trim();
    if command.is_empty() {
        return false;
    }
    let lower = command.to_ascii_lowercase();
    if lower.contains("http://")
        || lower.contains("https://")
        || lower.contains("ftp://")
        || lower.contains("sftp://")
        || lower.contains("curl ")
        || lower.starts_with("curl ")
        || lower.contains("wget ")
        || lower.starts_with("wget ")
        || lower.contains("scp ")
        || lower.starts_with("scp ")
```

URL, 네트워크 도구 이름, 명령 치환(`$(`와 백틱)이 보이면 승인 대상이다. 그다음 명령을 따옴표를 고려해 토큰으로 쪼개고, 토큰이 `~`, `$HOME`, `$TMP` 같은 변수, `..`, 작업 공간 밖 절대 경로를 가리키면 승인 대상이다([`runtime.rs#L3702-L3752`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/agent/runtime.rs#L3702-L3752)).

> **주의**: 이 판정은 명령 문자열의 토큰만 본다. 작업 공간 안의 스크립트를 실행하는 명령(예: 상대 경로 스크립트 실행)은 토큰에 외부 경로가 없으므로 자동 실행되고, 그 스크립트가 안에서 무엇을 하는지는 검사하지 않는다. `apply_sanitized_shell_env` 주석도 "이것은 환경 변수 최소화이지 OS 샌드박스가 아니다"라고 적는다([`tools.rs#L1197-L1245`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/agent/tools.rs#L1197-L1245)).

실행 쪽 제약은 다음 표와 같다.

| 항목 | 값 | 근거 |
|---|---|---|
| 셸 | Unix `/bin/sh -c`, Windows `ComSpec /C` | [`tools.rs#L1020-L1033`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/agent/tools.rs#L1020-L1033) |
| 작업 디렉터리 | `<프로젝트>/agent-workspace` | 같은 함수 |
| 시간 제한 | 기본 30초, 스키마상 최대 30초 | [`tools.rs#L43`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/agent/tools.rs#L43) |
| 명령 길이 | 4,000자 | [`tools.rs#L44`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/agent/tools.rs#L44) |
| stdout · stderr 상한 | 각 20,000자 | [`tools.rs#L45`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/agent/tools.rs#L45) |
| 생성 파일 보고 | 최대 50개 | [`tools.rs#L46`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/agent/tools.rs#L46) |
| 환경 변수 | `env_clear` 후 PATH · LANG · HOME 등만 남기고 `LLM_WIKI_PROJECT`, `LLM_WIKI_AGENT_WORKSPACE` 추가 | [`tools.rs#L1197-L1245`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/agent/tools.rs#L1197-L1245) |

실행 전후로 작업 공간 파일의 크기와 내용 해시를 찍어 바뀐 파일을 찾고, 그 파일들을 `workspace` 종류 참조로 답에 붙인다([`tools.rs#L1101-L1177`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/agent/tools.rs#L1101-L1177)). 크기가 같은 덮어쓰기도 해시로 잡는다.

> **주의**: `approvedShellCommands`는 HTTP API의 `/chat` 요청 본문에서도 그대로 역직렬화되며 `prepare_chat`이 이 필드를 비우지 않는다. 토큰이 있는 API 호출자는 승인 목록을 직접 실어 보낼 수 있다는 뜻이다. 다만 `shell.exec`는 스킬이 켜진 턴에서만 동작하고, `/chat`은 무인증 모드에서도 토큰을 요구한다(13장).
