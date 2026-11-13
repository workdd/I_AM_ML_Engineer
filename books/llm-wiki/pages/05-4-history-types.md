파일 변경 이력, 앱 변경 기록, 위키 타입 정의를 정리한다.

## 변경 이력: `file_history.rs`

LLM 이 위키를 고치는 앱에서는 "무엇이 바뀌었나" 를 되돌릴 방법이 필요하다. llm-wiki.md 는 "The wiki is just a git repo of markdown files" 라며 git 에 맡긴다([llm-wiki.md#L62](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/llm-wiki.md#L62)). LLM Wiki 는 git 을 요구하지 않고, 파일 쓰기 명령 안에 작은 버전 저장소를 넣었다.

코드 분석: Rust `write_file` 이 쓰기 전후로 버전을 남기는 부분이다. [src-tauri/src/commands/fs.rs#L1252-L1261](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/commands/fs.rs#L1252-L1261)

```rust
            file_sync::mark_app_write_path(p);
            crate::commands::file_history::record_file_version(
                p,
                "baseline",
                "before.ui.write_file",
            );
            fs::write(&path, contents)
                .map_err(|e| format!("Failed to write file '{}': {}", path, e))?;
            crate::commands::file_history::record_file_version(p, "human", "ui.write_file");
            file_sync::mark_app_write_path(p);
```

쓰기 직전 내용을 `baseline` 으로, 쓴 뒤 내용을 다시 기록한다. 직전 기록과 내용이 같으면 새 항목을 만들지 않으므로, 앱이 연속으로 쓴 파일은 버전이 하나씩만 늘어난다([file_history.rs#L234-L290](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/commands/file_history.rs#L234-L290)). 에이전트 도구(11장)는 `author: "agent"`, `tool: "wiki.write_page"` 처럼 다른 꼬리표로 같은 함수를 부른다.

| 항목 | 값 | 근거 |
|---|---|---|
| 기본 상태 | 꺼짐(`enabled: false`) | [#L42-L49](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/commands/file_history.rs#L42-L49) |
| 파일당 버전 수 | 기본 10, 최대 30 | [#L10-L11](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/commands/file_history.rs#L10-L11) |
| 기록 대상 파일 크기 | 512 KB 이하 | [#L9](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/commands/file_history.rs#L9) |
| 저장소 전체 | 128 MB, 파일 2,048개까지, 넘으면 오래된 것부터 삭제 | [#L12-L13](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/commands/file_history.rs#L12-L13), [#L189-L205](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/commands/file_history.rs#L189-L205) |
| 저장 위치 | `.llm-wiki/history/<경로의 FNV-1a 해시>.json` | [#L112-L122](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/commands/file_history.rs#L112-L122) |
| 설정 위치 | `.llm-wiki/history-settings.json` | [#L58-L60](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/commands/file_history.rs#L58-L60) |
| 제외 | `.llm-wiki/` 아래 파일 | [#L238-L240](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/commands/file_history.rs#L238-L240) |

프로젝트 루트는 경로를 거슬러 올라가며 `.llm-wiki` 폴더가 있는 첫 디렉터리로 정한다. 심볼릭 링크인 `.llm-wiki` 는 인정하지 않는다([#L98-L110](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/commands/file_history.rs#L98-L110)). 해시 키로 FNV-1a 를 직접 구현한 이유는 주석에 "Rust 툴체인이 바뀌어도 주소가 같게" 라고 적혀 있다. 표준 라이브러리 해시는 버전 간 안정성을 보장하지 않기 때문이다.

복원(`restore_file_history`)은 현재 내용이 고르려는 버전과 다르면 먼저 현재 내용을 `before.history.restore` 로 기록한 뒤 덮어쓴다([#L379-L434](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/commands/file_history.rs#L379-L434)). 복원 자체도 되돌릴 수 있다. 현재 파일이 512 KB 를 넘으면 백업을 남길 수 없어 복원을 거부한다.

> **주의**: 프론트의 `writeFile` 은 모두 Rust `write_file` 하나로 간다([commands/fs.ts#L22-L25](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/commands/fs.ts#L22-L25)). 그래서 수집 파이프라인이 LLM 출력으로 쓴 페이지도 이력에는 `author: "human"`, `tool: "ui.write_file"` 로 남는다. 이력의 author 칸만으로 사람 편집과 수집 결과를 구분할 수는 없다.

수집의 병합 대체 경로가 남기는 `.llm-wiki/page-history/<경로>-<시각>` 백업은 이 저장소와 별개다. 그쪽은 설정과 무관하게 병합이 실패할 때만 쓰이고, 개수나 크기 제한이 없다(코드상 정리 루틴을 찾지 못함, 미검증).

## 앱 변경 기록: `changelog.ts`

이름이 비슷하지만 `changelog.ts` 는 위키 페이지가 아니라 앱 자체의 릴리스 기록이다. 설정 화면의 Changelog 절이 그대로 렌더한다. 주석은 GitHub Releases 를 받아오지 않고 코드에 넣은 이유를 "오프라인에서도 보이고 코드와 함께 버전 관리되도록" 이라고 적는다([changelog.ts#L1-L16](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/changelog.ts#L1-L16)).

| 항목 | 내용 |
|---|---|
| 형식 | `{ version, date, highlights: { en, zh, it?, ru? } }` |
| 순서 | 최신 버전이 배열 맨 앞 |
| 범위 | 이 커밋 기준 41개 항목, 0.4.0 부터 0.6.13(2026-10-21)까지 |
| 규칙 | 사용자에게 보이는 변경만, 내부 리팩터링과 테스트는 커밋 메시지로 |

위키 쪽 `log.md` 와 비교하면 쓰임이 분명히 갈린다. `log.md` 는 위키에 무슨 일이 있었는지(수집 · 삭제)를 기록하고, `changelog.ts` 는 앱이 어떻게 바뀌었는지를 기록한다. 이 장과 관련된 항목도 보인다. 0.6.12 에는 "malformed generated wikilinks, page-merge scaffolding" 수정이 있고([#L78](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/changelog.ts#L78)), 더 이전 항목에는 원본을 앱 밖에서 지웠을 때 index · wikilink · `related` 를 일관되게 정리한다는 설명이 있다([#L638](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/changelog.ts#L638)). 이 장에서 본 정규화 함수와 삭제 연쇄가 어떤 순서로 들어왔는지 거꾸로 추적할 때 쓸 수 있다.

## 타입 정의: `src/types/wiki.ts`

앱 전역의 위키 타입은 20줄짜리 파일 하나다([types/wiki.ts#L1-L20](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/types/wiki.ts#L1-L20)).

| 타입 | 필드 | 메모 |
|---|---|---|
| `WikiProject` | `id`, `name`, `path` | `id` 는 `.llm-wiki/project.json` 의 UUID. 폴더를 옮겨도 유지 |
| `FileNode` | `name`, `path`, `is_dir`, `children?` | Rust `list_directory` 응답 모양 그대로(`is_dir` 이 snake_case) |
| `WikiPage` | `path`, `content`, `frontmatter` | `frontmatter` 는 `Record<string, unknown>` |

페이지 타입이나 frontmatter 필드가 타입 시스템에 없다는 점이 눈에 띈다. `type`, `sources`, `related` 는 모두 문자열로 다뤄지고, 계약은 이 장에서 본 런타임 함수들이 지킨다. schema 가 프로젝트마다 사용자 정의 타입을 허용하므로, 정적 타입으로 고정하기 어려운 설계와 맞물린 선택으로 보인다(추정).
