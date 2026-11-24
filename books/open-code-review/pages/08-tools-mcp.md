리뷰 에이전트가 쓸 수 있는 도구는 JSON 카탈로그 하나에 정의되고, 템플릿이 그중 일부를 이름으로 골라 모델에 보여 주며, 실제 실행은 Go 레지스트리에 등록된 provider가 맡는다. 이 장은 카탈로그와 레지스트리의 관계, 도구 하나가 실행되는 경로, 읽기 도구 네 개의 인자와 범위 제한, 그리고 외부 MCP 서버의 도구를 같은 레지스트리에 붙이는 방법을 코드로 따라간다.

## 도구 카탈로그 개요

도구 정의는 바이너리에 포함된 [`tools.json`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/config/toolsconfig/tools.json) 하나에 모여 있다. 항목은 여덟 개이고, 어느 템플릿이 어느 도구를 고르는지는 템플릿의 `tools` 배열이 정한다.

| 도구 | 필수 인자 | 선택 인자 | diff 리뷰 | 스캔 | 동작 영향 분석 |
|---|---|---|---|---|---|
| `task_done` | `state` (`DONE`, `FAILED`) | 없음 | 메인 | 메인 | 없음 |
| `code_comment` | `comments[]` (항목마다 `content`, `existing_code`, `category`, `severity`, `path`) | 항목의 `suggestion_code` | 메인 | 메인 | 없음 |
| `code_search` | `search_text` | `file_patterns`, `case_sensitive`, `use_perl_regexp` | 메인, plan 참고 | 메인 | 메인 |
| `file_read` | `file_path` | `start_line`, `end_line` | 메인, plan 참고 | 메인 | 메인 |
| `file_read_diff` | `path_array` | 없음 | 메인, plan 참고 | 없음 | 메인 |
| `file_find` | `query_name` | `case_sensitive` | 메인, plan 참고 | 메인 | 메인 |
| `context_read` | `kind`, `id` | `cursor`, `max_bytes` | 없음 | 없음 | 메인 |
| `submit_analysis` | `markdown` | `declared_completion` | 없음 | 없음 | 메인, 마지막 라운드 |

선택 목록의 출처는 [`task_template.json` L3-L10](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/config/template/task_template.json#L3-L10), [`scan_template.json` L3](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/config/template/scan_template.json#L3), [`behavior_template.json` L4-L17](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/config/template/behavior_template.json#L4-L17)이다. diff 리뷰의 plan 단계는 도구를 호출하지 않는 단일 요청이고, `PLAN_TASK.tool_references`에 적힌 도구는 프롬프트 안의 참고 설명으로만 들어간다(06장).

> **주의**: 원본 문서의 [Tools 페이지](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/pages/src/content/docs/en/tools.md)는 "six built-in tools"라고 적는다. 리뷰에서 보이는 도구 기준으로는 맞지만 카탈로그 자체에는 동작 영향 분석 전용 도구 두 개(`context_read`, `submit_analysis`)를 포함해 여덟 개가 있다. 이 두 도구는 12장에서 다룬다.

이 장에서는 diff 리뷰와 스캔이 쓰는 여섯 도구를 중심으로 본다. `code_comment`가 코멘트를 받아 줄 번호를 찾는 과정은 09장 담당이라 여기서는 실행 경로만 짚는다.

### 코드 분석: Resolve

카탈로그를 읽어 템플릿이 고른 이름만 `llm.ToolDef` 목록으로 바꾸는 함수다. 잘못된 카탈로그와 잘못된 참조를 모두 여기서 거른다.

파일: [`internal/config/toolsconfig/toolsconfig.go` L34-L69](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/config/toolsconfig/toolsconfig.go#L34-L69)

```go
func Resolve(entries []ToolConfigEntry, names []string) ([]llm.ToolDef, error) {
	catalog := make(map[string]llm.ToolDef, len(entries))
	for _, entry := range entries {
		// ... 빈 이름, 중복 정의, JSON 파싱 검사
		if function.Name != entry.Name {
			return nil, fmt.Errorf("tool %q has mismatched definition name %q", entry.Name, function.Name)
		}
		if strings.TrimSpace(function.Description) == "" || function.Parameters["type"] != "object" {
			return nil, fmt.Errorf("tool %q requires a description and object parameters", entry.Name)
		}
		function.RawDefinition = entry.Definition
		catalog[entry.Name] = llm.ToolDef{Type: "function", Function: function}
	}
	var definitions []llm.ToolDef
	seen := make(map[string]bool, len(names))
	for _, name := range names {
		if seen[name] {
			continue
		}
		definition, exists := catalog[name]
		if !exists {
			return nil, fmt.Errorf("unknown tool reference %q", name)
		}
```

- 카탈로그 검증은 다섯 가지다. 빈 이름, 중복 이름, JSON 파싱 실패, 바깥 `name`과 정의 안 `name` 불일치, 설명이 없거나 `parameters.type`이 `object`가 아닌 경우
- 참조 목록에서 같은 이름이 반복되면 처음 나온 위치만 남긴다. 목록이 비어 있으면 도구가 하나도 선택되지 않는다. 전체 카탈로그로 대체하지 않는다
- 원본 JSON을 `RawDefinition`에 보관한다. 뒤에서 볼 `side` 인자 주입이 이 원본을 고쳐 새 정의를 만든다

카탈로그는 [`//go:embed`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/config/toolsconfig/toolsconfig.go#L22-L32)로 바이너리에 들어간다. 원본 문서는 예전에 있던 `--tools` 플래그가 제거되어 `unknown flag: --tools`로 실패한다고 밝힌다. 사용자가 내장 도구의 정의나 설명을 바꿀 방법은 없고, 프로젝트별 지침은 리뷰 규칙(05장)으로, 외부 정보는 MCP로 넣는다.

## 정리

- 도구 정의는 바이너리에 포함된 `tools.json` 하나에 있고, 템플릿의 `tools` 배열이 이름으로 골라 쓴다. 카탈로그에는 리뷰용 여섯 개와 동작 영향 분석용 두 개, 모두 여덟 개가 있으며 사용자가 바꿀 방법은 없다
- 실행은 `Registry`의 provider가 맡고, `task_done`만 레지스트리 밖에서 이름으로 처리한다. 같은 하위 작업에서 같은 도구가 세 번 연속 실패하면 오류 내용 대신 "건너뛰었다"는 문구를 돌려줘 반복 재시도를 끊는다
- 읽기 도구 세 개는 `FileReader`의 모드에 따라 디스크 또는 `git show`, `git ls-tree`, `git grep`으로 동작한다. `file_read`는 500줄, 줄당 2048바이트, 결과 256KiB로, `code_search`와 `file_find`는 100개로 결과를 자르고, 작업 트리 경로는 심볼릭 링크까지 따라가 저장소 밖 접근을 막는다
- `side` 인자는 카탈로그가 아니라 리뷰 설정 단계에서 원본 정의를 복사해 덧붙인다. 스캔은 이 주입이 없어 `side`를 거부한다
- 원본 문서와 코드가 다른 곳이 여럿 있다. `code_search` 상한(문서는 파일당 100, 코드는 전체 100), `file_find`의 경로 검색 조건, `code_comment` 스키마의 `category`와 `severity`, `file_read`와 `code_search` 설명의 출력 예시 형식이 그렇다
- MCP 서버는 이름순으로 연결하고, 실패는 경고만 남긴다. 도구는 접두어 없이 같은 이름 공간에 등록되며 예약어와 먼저 등록된 이름을 건너뛴다. MCP는 `ocr review`에만 연결되고 `ocr scan`에는 없다
