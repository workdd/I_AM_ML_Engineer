위키 페이지는 YAML frontmatter 와 Markdown 본문으로 된 평범한 `.md` 파일이고, 페이지끼리는 `[[wikilink]]` 로 이어진다. 이 장은 그 파일 형식을 코드가 어떻게 정의하고, LLM 이 망가뜨린 형식을 어떻게 고치고, 원본이 바뀌거나 지워질 때 페이지를 어떻게 따라 바꾸는지 본다. `index.md` · `log.md` · `overview.md` 같은 특수 파일과, 파일을 고치기 전 내용을 남기는 `file_history.rs` 도 함께 다룬다.

## 원본 아이디어의 세 층과 파일 배치

llm-wiki.md 는 시스템을 세 층으로 나눈다([llm-wiki.md#L25-L33](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/llm-wiki.md#L25-L33)). 변하지 않는 원본(raw sources), LLM 이 쓰는 위키(the wiki), 위키 구조와 규칙을 적은 스키마(the schema)다. 팁 절에서는 "If your LLM adds YAML frontmatter to wiki pages (tags, dates, source counts), Dataview can generate dynamic tables" 라고 frontmatter 를 선택 사항으로 언급한다([#L61](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/llm-wiki.md#L61)). LLM Wiki 는 이 선택 사항을 필수 계약으로 바꿨다. 수집 · 삭제 · 검색 · 그래프가 모두 frontmatter 의 `type` 과 `sources` 를 읽기 때문이다.

| 층 | 위치 | 누가 쓰나 | 코드가 보는 방식 |
|---|---|---|---|
| 원본 | `raw/sources/**` | 사용자(가져오기, 감시기) | 읽기 전용. 식별자는 `raw/sources/` 아래 상대 경로 |
| 위키 | `wiki/**/*.md` | LLM 과 코드 | frontmatter 로 타입 · 출처를 판정 |
| 스키마 | `schema.md`, `purpose.md` (프로젝트 루트) | 템플릿이 생성, 사용자가 수정 | 페이지 타입 표를 파싱해 폴더 라우팅에 사용 |
| 앱 상태 | `.llm-wiki/**` | 코드 | 캐시 · 큐 · 체크포인트 · 변경 이력. 위키 처리 대상에서 제외 |

프로젝트를 만들면 Rust `create_project` 가 기본 폴더(`raw/sources`, `raw/assets`, `wiki/entities`, `wiki/concepts`, `wiki/sources`, `wiki/queries`, `wiki/comparisons`, `wiki/synthesis`)와 `schema.md` · `purpose.md` · `wiki/index.md` · `wiki/log.md` · `wiki/overview.md` 를 만들고, Obsidian 설정 폴더도 함께 만든다([project.rs#L15-L197](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/commands/project.rs#L15-L197)). 이어서 생성 대화상자가 고른 템플릿의 schema 와 purpose 로 두 파일을 덮어쓰고 템플릿 전용 폴더를 만든다.

> **주의**: 대화형 수집 경로 `executeIngestWrites` 는 `wiki/schema.md` 를 읽지만 프로젝트는 루트에 `schema.md` 를 만든다. `ingest.ts` 안에는 두 위치를 차례로 시도하는 fallback 이 없다. 이 차이는 4장 끝에서 다뤘다.

## 정리

- 위키 페이지는 frontmatter 일곱 필드와 Markdown 본문으로 된 파일이며, `type` 과 `sources` 가 라우팅 · 병합 · 삭제 · 그래프의 기준이 된다. llm-wiki.md 에서 선택 사항이던 frontmatter 가 필수 계약이 됐다.
- 쓰기 쪽은 엄격하다. LLM 출력의 네 가지 frontmatter 손상을 고치고, 날짜와 `sources` 는 코드가 덮어쓰고, schema 표에서 읽은 타입 → 폴더 지도로 경로를 교정한다. 읽기 쪽은 관대하게 받아 오래된 파일을 살린다.
- 기존 페이지는 배열 합집합, `type` · `title` · `created` 고정, 본문 LLM 병합(70% 길이 하한)으로 갱신되고, 한 원본만 출처인 페이지는 재수집 때 본문을 교체한다.
- wikilink 는 Obsidian 문법을 쓰고 파일 이름 기준으로 풀린다. 생성 · 정규화 · 렌더 · 클릭 · 삭제 정리의 각 단계에 전용 함수가 있으며, LLM 보강 모듈은 이 커밋에서 호출되지 않는다.
- index 와 overview 는 LLM 이 아니라 코드가 상한 있는 최근 목록만 관리하고, log 는 llm-wiki.md 가 제안한 `## [날짜] 동작 | 대상` 접두로 덧붙인다. 다만 템플릿 schema 와 초기 log 는 다른 형식을 쓴다.
- 원본의 이동 · 수정 · 삭제는 `sources` 를 따라 위키에 반영되고, 선택 기능인 파일 이력은 모든 쓰기의 전후 내용을 파일당 최대 30개까지 남긴다.
