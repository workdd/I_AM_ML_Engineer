프롬프트 템플릿 구조와 자리표시자 치환, effort 수준별 수치를 다룬다.

## 프롬프트 템플릿 구조

규칙이 정해지면 그것을 담을 prompt가 필요하다. prompt는 [`task_template.json`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/config/template/task_template.json#L1-L48) 하나와 `prompts/` 아래 Markdown 파일 17개로 구성된다. JSON은 각 작업의 메시지 목록을 `prompt_file` 참조로만 갖고, 본문은 Markdown 파일에 있다.

| 작업 키 | 메시지 파일 | 도구 설정 | 쓰는 곳 |
|---|---|---|---|
| `MAIN_TASK` | `main_task_system.md`, `main_task_user.md` | `tools` 6개 | 06장, 07장 |
| `PLAN_TASK` | `plan_task_system.md`, `plan_task_user.md` | `tool_references` 4개 | 06장 |
| `MEMORY_COMPRESSION_TASK` | `memory_compression_task_*.md` | 없음 | 07장 |
| `REVIEW_FILTER_TASK` | `review_filter_task_*.md` | 없음 (코드에 고정) | 09장 |
| `RE_LOCATION_TASK` | `re_location_task_*.md` | 없음 | 09장 |
| `GROUPING_TASK` | `grouping_task_*.md` | 없음 | 04장 |

`scan_template.json`, `behavior_template.json`은 같은 패키지에 있지만 각각 11장, 12장에서 다룬다.

### 코드 분석: resolveConversation

`internal/config/template/template.go`, [template.go L191-L205](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/config/template/template.go#L191-L205)

`prompt_file` 참조를 실제 본문으로 바꾸는 함수다.

```go
func resolveConversation(m manifestConversation) (LlmConversation, error) {
	conv := LlmConversation{Tools: m.Tools, ToolReferences: m.ToolReferences}
	conv.Messages = make([]ChatMessage, len(m.Messages))
	for i, mm := range m.Messages {
		data, err := templateFS.ReadFile("prompts/" + mm.PromptFile)
		if err != nil {
			return LlmConversation{}, fmt.Errorf("read prompt file %q: %w", mm.PromptFile, err)
		}
		conv.Messages[i] = ChatMessage{
			Role:    mm.Role,
			Content: strings.TrimRight(string(data), "\r\n"),
		}
	}
	return conv, nil
}
```

`templateFS`는 [`go:embed task_template.json behavior_template.json prompts/*`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/config/template/template.go#L157-L158)로 만든 내장 파일 시스템이다. 템플릿을 외부 파일로 바꿔 끼우는 CLI 옵션은 없고, 공식 문서도 prompt를 바꾸려면 JSON과 Markdown을 고친 뒤 다시 빌드하라고 안내한다. 사용자가 실행 시점에 바꿀 수 있는 것은 규칙 파일과 `--background` 문맥뿐이다.

### 코드 분석: Template.Validate와 도구 바인딩

`internal/config/template/template.go`, [template.go L362-L376](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/config/template/template.go#L362-L376)

어떤 작업이 도구를 가질 수 있는지 검사한다.

```go
func validateToolBindings(conversations map[string]*LlmConversation, referenceTask string) error {
	for _, name := range slices.Sorted(maps.Keys(conversations)) {
		conversation := conversations[name]
		if conversation == nil {
			continue
		}
		if name != "MAIN_TASK" && len(conversation.Tools) > 0 {
			return fmt.Errorf("%s.tools is not supported by this task", name)
		}
		if name != referenceTask && len(conversation.ToolReferences) > 0 {
			return fmt.Errorf("%s.tool_references is not supported by this task", name)
		}
	}
	return nil
}
```

| 필드 | 허용 작업 | 의미 |
|---|---|---|
| `tools` | `MAIN_TASK`만 | 실제로 호출 가능한 도구 정의를 요청에 실어 보냄 |
| `tool_references` | `PLAN_TASK`만 (리뷰 템플릿) | 도구 설명을 텍스트로만 prompt에 넣고 호출은 불가 |

`PLAN_TASK`는 도구를 호출하지 않는다. `plan_task_system.md`의 [`{{plan_tools}}`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/config/template/prompts/plan_task_system.md#L6-L7) 자리에 도구 설명이 글로 들어가고, 모델은 "어떤 도구를 어떤 인자로 부를지"를 계획에 적기만 한다. 실제 호출은 `MAIN_TASK`에서 일어난다. `REVIEW_FILTER_TASK`가 쓰는 `report_incorrect_comments`, `approve_all_comments` 두 도구는 템플릿이 아니라 Go 코드에 고정되어 있어 이 검사를 거치지 않는다(09장).

## 자리표시자 치환

"템플릿 엔진"의 실체는 `strings.ReplaceAll`의 연속이다. 조건문이나 반복문 문법은 없고, 빈 블록을 지우는 일만 정규식이 맡는다.

### 코드 분석: buildMainTaskMessages

`internal/agent/agent.go`, [agent.go L1347-L1368](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/agent/agent.go#L1347-L1368)

`MAIN_TASK` 메시지를 라운드마다 다시 만드는 함수다.

```go
func (a *Agent) buildMainTaskMessages(rule, changeFiles, diffs, planResult, confirmed string) []llm.Message {
	rawMsgs := a.args.Template.MainTask.Messages
	messages := make([]llm.Message, 0, len(rawMsgs))
	for _, m := range rawMsgs {
		content := m.Content
		content = strings.ReplaceAll(content, "{{current_system_date_time}}", a.currentDate)
		content = strings.ReplaceAll(content, "{{system_rule}}", rule)
		content = strings.ReplaceAll(content, "{{change_files}}", changeFiles)
		content = strings.ReplaceAll(content, "{{diffs}}", diffs)
		content = strings.ReplaceAll(content, "{{requirement_background}}", a.args.Background)
		if planResult == "" {
			content = stripEmptyPlanBlock(content)
		}
		content = strings.ReplaceAll(content, "{{plan_guidance}}", planResult)
		if confirmed == "" {
			content = stripEmptyConfirmedBlock(content)
		}
		content = strings.ReplaceAll(content, "{{confirmed_comments}}", confirmed)
		messages = append(messages, llm.NewTextMessage(m.Role, content))
	}
	return messages
}
```

치환 순서가 의미를 가진다. `{{diffs}}`가 `{{system_rule}}`보다 뒤에 치환되므로, 규칙 본문 안에 `{{diffs}}`라는 글자가 있으면 그 자리도 diff로 바뀐다. 반대로 diff 안에 있는 `{{plan_guidance}}` 같은 글자는 뒤 단계에서 치환된다. 사용자 입력(diff, 규칙, 배경 문맥)에 이런 문자열이 들어갈 가능성은 낮지만, 치환이 한 번에 일어나지 않는다는 점은 알아 둘 만하다(추정).

| 자리표시자 | 들어가는 값 | 사용 prompt |
|---|---|---|
| `{{system_rule}}` | `resolveGroupSystemRule` 결과 | `main_task_user.md`, `plan_task_user.md` |
| `{{diffs}}` | 묶음 diff, 파일마다 `<file path="...">` | 위 두 파일 |
| `{{change_files}}` | 묶음 밖 변경 파일 목록, `MODIFIED   path (+N/-M)` 형식 | 위 두 파일 |
| `{{requirement_background}}` | `--background` 또는 `--background-file` 내용 | 위 두 파일 |
| `{{current_system_date_time}}` | `2006-01-02 15:04` 형식 현재 시각 | 위 두 파일 |
| `{{plan_guidance}}` | 계획 단계 출력, 라운드 2부터 빈 값 | `main_task_user.md` |
| `{{confirmed_comments}}` | 이전 라운드 확정 코멘트 요약 | `main_task_user.md` |
| `{{plan_tools}}` | 계획용 도구 설명 텍스트 | `plan_task_system.md` |

### 코드 분석: 빈 블록 제거

`internal/agent/util.go`, [util.go L23-L31](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/agent/util.go#L23-L31)

계획이 없을 때 제목 줄까지 함께 지우는 정규식이다.

```go
var planBlockPattern = regexp.MustCompile(
	`(?m)^### [^\n]*Review Plan[^\n]*\n\{\{plan_guidance\}\}\n\n?`)

// stripEmptyPlanBlock removes the "### Review Directive\n{{plan_guidance}}\n\n"
// wrapper from a MAIN_TASK user message when the plan phase produced no
// guidance. Strip is a no-op when the wrapper is absent.
func stripEmptyPlanBlock(content string) string {
	return planBlockPattern.ReplaceAllString(content, "")
}
```

[`main_task_user.md`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/config/template/prompts/main_task_user.md#L16-L23)는 `### Review Plan` 아래 `{{plan_guidance}}`, `### Previously Confirmed Findings` 아래 `{{confirmed_comments}}`를 둔다. 값이 비면 제목 줄까지 지워서 모델이 빈 절을 보지 않게 한다. 정규식은 제목에 `Review Plan`이 들어 있고 다음 줄이 정확히 자리표시자일 때만 일치한다. prompt 파일에서 제목을 바꾸거나 자리표시자와 제목 사이에 줄을 넣으면 제거가 조용히 실패한다.

> **주의**: 함수 주석은 지울 대상을 `### Review Directive`라고 적지만, 정규식과 실제 prompt 파일의 제목은 `### Review Plan`이다. 주석이 이전 제목을 남긴 것으로 보인다(추정).

### MAIN_TASK prompt의 커버리지 지시

`main_task_system.md`의 끝부분은 묶음 안의 모든 파일을 따로 보라고 반복한다.

```markdown
## Strict Focus Rules
- Review every file listed in <review_files> individually.
- Cross-file observations within <review_files> are encouraged — look for inconsistencies, missing updates, and broken contracts across related files.
- Context tools are for gathering background information only. Your comments must address code within <review_files> — never produce comments targeting files outside it.

## Reply limit
- Before calling `task_done`, confirm you have given every `<file>` in <review_files> its own pass. Reviewing an implementation file does not cover its header, interface, or configuration counterpart — a file being the smaller or secondary member of the group is not a reason to skip it.
```

출처: [main_task_system.md L16-L22](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/config/template/prompts/main_task_system.md#L16-L22)

여러 파일을 한 묶음으로 리뷰하면 모델이 큰 구현 파일만 보고 헤더나 설정 파일을 건너뛰기 쉽다. README가 지적한 "불완전한 커버리지" 문제를 prompt 수준에서도 막으려는 문장이다. 코드 수준의 커버리지 기록은 06장에서 다룬다.

### 언어 지시 주입

`ocr config`의 `language` 값은 [`Template.ApplyLanguage`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/config/template/template.go#L288-L295)가 system 메시지 끝에 `Always respond in <언어>.`를 붙이는 방식으로 반영된다. 값이 없으면 `English`다. 적용 대상은 `MAIN_TASK`, `PLAN_TASK`, `MEMORY_COMPRESSION_TASK` 세 작업뿐이고, `REVIEW_FILTER_TASK`, `RE_LOCATION_TASK`, `GROUPING_TASK`에는 붙지 않는다. 세 작업은 결과가 사용자에게 문장으로 보이지 않고 도구 호출, 코드 블록, 묶음 JSON으로 소비되므로 언어 지시가 필요 없다(추정).

## effort 수준과 템플릿 수치

`task_template.json` 끝의 수치 기준 가운데 라운드 수는 effort 수준이 덮어쓴다.

### 코드 분석: effort 프리셋

`internal/config/template/effort.go`, [effort.go L34-L61](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/config/template/effort.go#L34-L61)

```go
var effortPresets = map[Effort]EffortPreset{
	EffortLow:    {MaxReviewRounds: 1},
	EffortMedium: {MaxReviewRounds: 2},
	EffortHigh:   {MaxReviewRounds: 3},
}

// ParseEffort validates a user-supplied effort value.
func ParseEffort(s string) (Effort, error) {
	e := Effort(strings.ToLower(s))
	if _, ok := effortPresets[e]; !ok {
		return "", fmt.Errorf("invalid effort %q: must be one of %s", s, strings.Join(EffortNames(), ", "))
	}
	return e, nil
}

// Preset returns the knob set for e.
func (e Effort) Preset() EffortPreset {
	if p, ok := effortPresets[e]; ok {
		return p
	}
	return effortPresets[EffortDefault]
}
```

`EffortPreset`에는 현재 `MaxReviewRounds` 필드 하나만 있다. 이름은 "effort"지만 계획 단계 기준, 도구 호출 상한, 필터 사용 여부는 바꾸지 않는다. 값은 [`resolveEffort`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/cmd/opencodereview/shared.go#L70-L78)가 `--effort` 플래그, 설정 파일의 `effort`, 기본값 `medium` 순으로 정하고, [`review_cmd.go`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/cmd/opencodereview/review_cmd.go#L200-L204)가 `ApplyEffort`로 템플릿에 덮어쓴다. 템플릿 JSON의 `MAX_REVIEW_ROUNDS: 2`는 effort가 항상 적용되므로 실질적으로 쓰이지 않는다.

| 템플릿 키 | 기본값 | 의미 | 변경 수단 | 다루는 장 |
|---|---|---|---|---|
| `MAX_TOOL_REQUEST_TIMES` | 100 | 묶음당 도구 호출 라운드 상한 | `--max-tools` (최소 50) | 07장 |
| `PLAN_MODE_LINE_THRESHOLD` | 50 | 파일 하나 변경 줄 수가 이 이상이면 계획 단계 실행 | 없음 | 06장 |
| `PLAN_MODE_GROUP_LINE_THRESHOLD` | 100 | 파일 2개 이상 묶음의 합계 변경 줄 수 기준 | 없음 | 06장 |
| `GROUPING_MIN_FILES` | 4 | 이 파일 수 이상이면 LLM 묶음 구성 | 없음 | 04장 |
| `GROUPING_BUNDLE_LINE_THRESHOLD` | 200 | 파일이 적을 때 한 묶음으로 합칠 변경 줄 수 상한 | 없음 | 04장 |
| `MAX_REVIEW_ROUNDS` | 2 | 묶음당 리뷰 라운드 수 | `--effort` | 06장 |
| `MAX_TOKENS` | 200000 | 묶음 prompt 토큰 상한의 기준 | `--max-tokens`, 설정 | 06장, 07장 |
| `MAX_COMPLETION_TOKENS` | 16384 | 응답 토큰 상한 | 없음 | 10장 |

수치 출처: [task_template.json L40-L47](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/config/template/task_template.json#L40-L47)

`PLAN_MODE_*` 두 기준은 [`Template.PlanRequired`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/config/template/template.go#L70-L82)가 해석한다. 기본값으로는 한 파일이 50줄 이상 바뀌었거나, 파일 2개 이상 묶음의 합계가 100줄 이상이면 계획 단계를 실행한다. 기준을 0 이하로 두면 계획이 항상 실행된다. 이 판정이 실제 흐름에서 쓰이는 위치는 06장에서 다룬다.
