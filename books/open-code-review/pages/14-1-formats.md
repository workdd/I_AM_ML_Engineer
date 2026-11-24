진행 로그 라우팅과 JSON, SARIF, 터미널 출력 구조를 다룬다.

## --audience와 진행 로그 라우팅

리뷰 중에는 `[ocr]   ▶ file_read "src/foo.go"` 같은 진행 줄이 계속 출력된다. 사람에게는 유용하지만 JSON 문서와 섞이면 파서가 실패한다. `--audience`는 이 진행 줄을 어디로 보낼지 정한다. 값은 `human`(기본값)과 `agent` 두 가지다([shared_flags.go L36](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/cmd/opencodereview/shared_flags.go#L36)).

### 코드 분석: stdout 패키지

진행 줄을 출력하는 모든 코드는 `os.Stdout`이 아니라 `internal/stdout`의 `Writer()`를 쓴다. 이 패키지는 전역 writer 하나와 그것을 바꾸는 함수 두 개로 구성된다.

[stdout.go L33-L43](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/stdout/stdout.go#L33-L43)

```go
func Quiet() func() {
	mu.Lock()
	old := w
	w = io.Discard
	mu.Unlock()
	return func() {
		mu.Lock()
		w = old
		mu.Unlock()
	}
}
```

`Swap(replacement)`도 같은 구조로 writer를 임의의 대상으로 바꾸고 복원 함수를 돌려준다([stdout.go L57-L67](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/stdout/stdout.go#L57-L67)). 주석은 두 함수를 메인 goroutine에서만 호출하라고 경고한다. 뮤텍스로 메모리 안전성은 보장하지만, 여러 goroutine이 동시에 바꾸면 복원 순서가 섞이기 때문이다.

### 코드 분석: newQuietHandle

[shared.go L388-L395](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/cmd/opencodereview/shared.go#L388-L395)

```go
func newQuietHandle(outputFormat, audience string) *quietHandle {
	h := &quietHandle{}
	switch {
	case audience == "agent":
		h.fn = stdout.Quiet()
	case isMachineReadable(outputFormat):
		h.fn = stdout.Swap(os.Stderr)
	}
```

형식과 대상의 조합별 결과는 테스트 표에 그대로 적혀 있다([shared_test.go L87-L92](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/cmd/opencodereview/shared_test.go#L87-L92)).

| `--format` | `--audience` | 진행 줄 | 결과 문서 |
|---|---|---|---|
| `text` | `human` | stdout | stdout |
| `json`, `sarif` | `human` | stderr | stdout |
| `text` | `agent` | 버림 | stdout, 마지막 요약 줄 포함 |
| `json`, `sarif` | `agent` | 버림 | stdout |

- `--audience agent`는 `--format json`을 뜻하지 않는다. 둘은 독립된 설정이다. 에이전트가 text 출력을 원하면 `--audience agent`만 주면 된다.
- `--format json`만 주면 진행 줄은 stderr로 계속 보인다. `ocr review --format json | jq .summary`가 동작하면서도 터미널에서는 진행 상황을 볼 수 있다.
- text와 `agent` 조합에서는 요약 줄 직전에 `q.Restore()`로 writer를 되살려 `[ocr] Summary:` 한 줄은 출력되게 한다([shared.go L853-L855](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/cmd/opencodereview/shared.go#L853-L855)).

`isMachineReadable`은 `json`, `sarif`에 더해 `html`도 참으로 돌려준다([shared.go L367-L374](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/cmd/opencodereview/shared.go#L367-L374)). `html`은 `--format`으로 받지 않는 값이고 `ocr session export`에서만 들어온다. 내보낸 HTML 안의 ESC 바이트가 색상 제거 필터에 지워지지 않게 하려는 용도다.

## JSON 출력 구조

JSON 출력은 에이전트 연동과 GitHub Action, ACP 어댑터가 모두 소비하는 사실상의 계약이다.

### 코드 분석: jsonOutput

[output.go L352-L370](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/cmd/opencodereview/output.go#L352-L370)

```go
type jsonOutput struct {
	SchemaVersion  string                `json:"schema_version,omitempty"`
	ReviewStatus   string                `json:"review_status,omitempty"`
	Analysis       *string               `json:"analysis,omitempty"`
	AnalysisRaw    *string               `json:"analysis_raw,omitempty"`
	AnalysisMeta   *workflow.Report      `json:"analysis_meta,omitempty"`
	Status         string                `json:"status"`
	LLM            *jsonLLMIdentity      `json:"llm,omitempty"`
	TraceID        string                `json:"trace_id,omitempty"`
	Message        string                `json:"message,omitempty"`
	Summary        *jsonSummary          `json:"summary,omitempty"`
	ToolCalls      *jsonToolCalls        `json:"tool_calls"`
	Comments       []model.LlmComment    `json:"comments"`
	Groups         []agent.FileGroupInfo `json:"groups,omitempty"`
	Warnings       []agent.AgentWarning  `json:"warnings,omitempty"`
	ProjectSummary string                `json:"project_summary,omitempty"`
	Resume         *agent.ResumeInfo     `json:"resume,omitempty"`
	SessionID      string                `json:"session_id,omitempty"`
	Manifest       *session.RunManifest  `json:"manifest,omitempty"`
```

마지막 필드 `retry_report`는 재시도가 있을 때만 붙는다. 주석은 "a first-try-success run emits byte-identical JSON to before"라고 적어, 필드 추가가 기존 소비자의 출력 비교를 깨지 않게 한 의도를 밝힌다([output.go L371-L375](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/cmd/opencodereview/output.go#L371-L375)).

| 필드 | 내용 | 비고 |
|---|---|---|
| `status` | 실행 상태 | manifest가 있으면 그 종료 상태 |
| `summary` | 리뷰 파일 수, 코멘트 수, token 사용량, 경과 시간 | `budget_exceeded`는 예산 도달 시에만 |
| `tool_calls` | 도구별 호출 수와 실패 상세 | 항상 존재 |
| `comments` | `path`, `content`, `start_line`, `end_line`, `category`, `severity` 등 | 0과 0은 파일 단위 코멘트 |
| `manifest` | 커버리지와 실패 분류 | 9장, 13장 참고 |
| `groups` | 파일 묶음 정보 | 4장 참고 |
| `schema_version`, `analysis*`, `review_status` | `--analysis`일 때만 | 12장 참고 |

### status 결정 규칙

[output.go L423-L435](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/cmd/opencodereview/output.go#L423-L435)

```go
	if manifest != nil {
		payload.Status = string(manifest.TerminalState)
		payload.Message = manifestMessage(manifest, len(comments))
	} else if len(comments) == 0 {
		switch {
		case hasSubtaskErrors(warnings):
			payload.Message = "Some files could not be reviewed due to errors."
		case hasReviewLimitWarning(warnings):
			payload.Message = "A review limit was reached before finishing; some findings may be missing."
		default:
			payload.Message = "No comments generated. Looks good to me."
		}
	}
```

- manifest가 있는 review 실행에서 `status`는 manifest의 종료 상태(`complete`, `partial`, `skipped`, `failed`)다. 경고는 `warnings`에 실리지만 `status`를 바꾸지 않는다.
- manifest가 없는 실행(주로 scan)에서는 기본값 `success`에서 시작해, 하위 작업 오류 경고가 있으면 `completed_with_errors`, 그 밖의 경고만 있으면 `completed_with_warnings`가 된다([output.go L436-L443](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/cmd/opencodereview/output.go#L436-L443)).
- token 예산 도달은 일부러 `status`를 건드리지 않는다. 주석은 예산 이유가 `summary.budget_exceeded`, `token_budget_reached` 경고, `coverage.failed[].classification == "budget"` 세 곳으로 확인된다고 적는다([output.go L444-L451](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/cmd/opencodereview/output.go#L444-L451)).

> **주의**: 코멘트가 비어 있고 종료 코드가 0이어도 모든 파일을 리뷰했다는 뜻은 아니다. 네이티브 skill 문서도 "A zero exit code or empty `comments` does not prove all selected files were reviewed"라고 적고, `manifest.coverage`를 확인하라고 안내한다([skills/open-code-review/SKILL.md](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/skills/open-code-review/SKILL.md)).

## SARIF 출력

SARIF(Static Analysis Results Interchange Format)는 정적 분석 도구의 결과를 교환하는 JSON 표준이다. GitHub Code Scanning이 이 형식을 받아 저장소의 Security 탭에 경고로 표시한다. OCR은 SARIF 2.1.0 중 필요한 부분만 구조체로 정의한다([sarif.go L20-L32](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/cmd/opencodereview/sarif.go#L20-L32)).

### 코드 분석: outputSARIF

[sarif.go L131-L151](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/cmd/opencodereview/sarif.go#L131-L151)

```go
func outputSARIF(comments []model.LlmComment, version string, warnings []agent.AgentWarning, manifest *session.RunManifest, out io.Writer) error {
	report := sarifReport{
		Schema:  sarifSchema,
		Version: sarifVersion,
		Runs: []sarifRun{{
			Tool: sarifTool{
				Driver: sarifDriver{
					Name:           sarifToolName,
					Version:        version,
					InformationURI: sarifInformationURI,
					Rules:          sarifRules(),
				},
			},
			Results:     sarifResults(comments),
			Invocations: []sarifInvocation{sarifInvocationFromRun(warnings, manifest, len(comments))},
		}},
	}
	enc := json.NewEncoder(out)
	enc.SetIndent("", "  ")
	return enc.Encode(report)
}
```

문서는 항상 run 하나를 담는다. 규칙(`rules`)은 코멘트 카테고리 8개(`bug`, `security`, `performance`, `maintainability`, `test`, `style`, `documentation`, `other`)로 고정되어 있고, 모든 result의 `ruleId`는 이 중 하나다([sarif.go L155-L166](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/cmd/opencodereview/sarif.go#L155-L166)). 카테고리가 비어 있으면 `other`가 된다.

### 필드 매핑

| OCR 코멘트 필드 | SARIF 위치 | 조건 |
|---|---|---|
| `path` | `locations[].physicalLocation.artifactLocation.uri` | 경로가 있을 때 |
| `start_line`, `end_line` | `region.startLine`, `region.endLine` | `start_line > 0`이고 `end_line >= start_line` |
| `content` | `message.text` | 항상 |
| `category` | `ruleId` | 비면 `other` |
| `severity` | `level` | 아래 표 |
| `suggestion_code` | `fixes[].artifactChanges[].replacements[].insertedContent` | 경로, 유효 범위, `existing_code`, `suggestion_code` 모두 있을 때 |

심각도는 SARIF의 세 단계로 줄어든다([sarif.go L174-L185](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/cmd/opencodereview/sarif.go#L174-L185)).

| OCR severity | SARIF level |
|---|---|
| `critical`, `high` | `error` |
| `medium` | `warning` |
| `low`, 빈 값, 알 수 없는 값 | `note` |

빈 값을 `note`로 보내는 이유는 주석에 있다. 코멘트 도구가 알 수 없는 심각도를 `low`로 정규화하므로, SARIF에서도 같은 낮은 단계로 맞춰 중요도를 부풀리지 않는다.

`fixes`는 유효한 줄 범위가 있을 때만 만든다. SARIF 스키마에서 `replacement.deletedRegion`이 필수라서, 범위 없이 수정 제안을 넣으면 문서 전체가 스키마 검증에 실패하기 때문이다([sarif.go L258-L275](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/cmd/opencodereview/sarif.go#L258-L275)). 범위가 없는 제안은 `message.text`에만 남는다.

### 코드 분석: sarifFingerprints

GitHub Code Scanning은 `partialFingerprints`로 실행 간 같은 경고를 추적한다. 지문이 매번 바뀌면 같은 문제가 닫혔다가 새 경고로 다시 열린다.

[sarif.go L288-L297](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/cmd/opencodereview/sarif.go#L288-L297)

```go
func sarifFingerprints(c model.LlmComment, category string) map[string]string {
	var fingerprintSource string
	if code := strings.TrimSpace(c.ExistingCode); code != "" {
		fingerprintSource = c.Path + "|" + category + "|" + code
	} else {
		fingerprintSource = c.Path + "|" + category + "|" + strconv.Itoa(c.StartLine)
	}
	h := sha256.Sum256([]byte(fingerprintSource))
	return map[string]string{sarifFingerprintKey: hex.EncodeToString(h[:])}
}
```

- 지문 재료는 경로, 카테고리, 기존 코드다. LLM이 매번 다르게 쓰는 `message.text`는 쓰지 않는다. 같은 코드에 대한 지적이면 문장이 바뀌어도 같은 경고로 이어진다.
- 기존 코드가 없으면 시작 줄 번호를 쓴다. 이 경우 위쪽에 줄이 추가되면 지문이 바뀐다.
- 같은 지문이 한 실행에서 여러 번 나오면 두 번째부터 `#1`, `#2`를 붙여 별도 경고로 만든다([sarif.go L195-L211](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/cmd/opencodereview/sarif.go#L195-L211)).
- 키 이름 `ocrFinding/v1`은 상수로 버전이 붙어 있다. 지문 계산 방식을 바꿀 때 키를 바꿔 이전 지문과 섞이지 않게 할 수 있다.

### 실행 상태 전달

`invocations[0].executionSuccessful`은 종료 코드를 결정하는 함수와 같은 `reviewManifestRequiresNonZeroExit`로 정한다. `failed`와 운영 실패가 있는 `partial`은 거짓, 예산 실패만 있는 `partial`과 `skipped`, `complete`는 참이다. manifest가 `complete`가 아니면 그 메시지가 `toolExecutionNotifications`에 경고로 들어가, 소비자가 "깨끗한 결과"와 "일부만 본 결과"를 구분할 수 있다([sarif.go L314-L343](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/cmd/opencodereview/sarif.go#L314-L343)).

CI 문서는 `--format sarif --audience agent > results.sarif`로 파일을 만들고 `github/codeql-action/upload-sarif`로 올리는 예를 든다([integrations/ci.md](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/pages/src/content/docs/en/integrations/ci.md)). 저장소의 `action.yml`은 SARIF를 쓰지 않는다. 이 장의 GitHub Action 절에서 다룬다.

## 터미널 Markdown 렌더링

text 출력에서 Project Summary와 Analysis 보고서는 모델이 쓴 Markdown이다. `markdownForOutput`은 출력 대상이 실제 터미널이고 색상이 켜져 있을 때만 렌더링하고, 그 밖에는 제어 문자만 제거한 원문을 낸다.

[markdown.go L27-L36](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/cmd/opencodereview/markdown.go#L27-L36)

```go
func markdownForOutput(writer io.Writer, markdown string) string {
	if writer != os.Stdout || !colorOn() {
		return sanitizeTerminal(markdown)
	}
	width, _, err := term.GetSize(os.Stdout.Fd())
	if err != nil {
		width = 0
	}
	return renderMarkdown(markdown, width)
}
```

`--output` 파일이나 파이프로 내보내면 Markdown 원문이 그대로 남으므로, 다른 도구가 다시 렌더링할 수 있다. 렌더러는 goldmark 파서에 glamour의 ANSI 렌더러를 결합하며, 렌더링 전에 모델 출력에 섞인 ANSI 이스케이프를 지운다(`sanitizeTerminal(ansi.Strip(markdown))`). 모델 응답으로 터미널 제어 시퀀스가 주입되는 것을 막는 처리다.

표는 glamour 기본 렌더링 대신 `layoutMarkdownTable`이 직접 배치한다. 터미널 폭이 열마다 2칸도 확보하지 못하면 표를 포기하고 행마다 "헤더: 값" 목록으로 바꾸고, 열 너비가 충분하면 셀 안쪽 여백을 1칸 둔다([markdown_table.go L78-L101](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/cmd/opencodereview/markdown_table.go#L78-L101)). 12장의 Analysis 보고서가 `Scenario | Before | After` 표를 기본으로 쓰기 때문에 좁은 터미널 대응이 필요하다.
