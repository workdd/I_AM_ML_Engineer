분석 예산과 라운드, 결과 상태와 산출물을 다룬다.

## 예산, 라운드, 마지막 제출 라운드

Analysis의 한도는 템플릿 값과 CLI 플래그가 함께 정한다.

### 템플릿 기본값

[behavior_template.json](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/config/template/behavior_template.json)은 버전 `ocr.behavior-prompts/v9`이며, 메인 대화(`behavior_system.md`, `behavior_task.md`), 압축 대화, 마지막 라운드의 세 대화와 숫자 세 개로 구성된다.

| 키 | 값 | CLI 경로에서의 실제 값 |
|---|---|---|
| `MAX_TOKENS` | 200000 | Review 템플릿의 `MaxTokens`로 덮어쓴다([analysis.go L142](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/cmd/opencodereview/analysis.go#L142)) |
| `MAX_COMPLETION_TOKENS` | 16384 | 그대로 |
| `MAX_TOOL_REQUEST_TIMES` | 300 | 그대로. `--max-tools`와 무관 |

`MAX_TOKENS`가 덮어써지므로 `--max-tokens`나 저장된 `max_tokens` 설정이 Review와 Analysis에 같이 적용된다. CLI 문서의 "Prompt (input) token ceiling per subtask in both Review and Analysis"와 일치한다. 300 라운드는 Review의 기본 라운드 상한과 별개로, CLI 문서도 "independent of `--max-tools`"라고 적는다.

### 코드 분석: budgetClient

Analysis 전용 token 예산은 llmloop의 공통 예산 검사와 별도로, LLM 클라이언트를 감싸는 `budgetClient`가 관리한다.

[budget.go L52-L67](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/behaviortask/budget.go#L52-L67)

```go
	select {
	case budget.slots <- struct{}{}:
	case <-ctx.Done():
		return nil, false, ctx.Err()
	}
	defer func() { <-budget.slots }()
	if err := ctx.Err(); err != nil {
		return nil, false, err
	}
	budget.mu.Lock()
	budgetReached := budget.maximum > 0 && budget.usage.TotalTokens >= budget.maximum
	budget.mu.Unlock()
	if budgetReached && !finalSubmission {
		return nil, true, llm.ErrTokenBudget
	}
	response, err := client.client.CompletionsWithCtx(ctx, request)
```

- `slots`는 용량 1인 채널이다([budget.go L22-L24](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/behaviortask/budget.go#L22-L24)). 조사 요청과 백그라운드 압축 요청이 겹쳐도 Analysis가 동시에 보내는 모델 요청은 하나뿐이다. README의 "Analysis uses at most one additional concurrent model request"가 이 채널이다.
- 예산 검사는 요청을 보내기 전에만 한다. 이미 진행 중인 요청은 끝까지 기록되므로 사용량이 상한을 넘을 수 있다.
- `finalSubmission`이 참이면 예산을 넘었어도 요청을 보낸다. 마지막 제출 라운드를 위한 예외다. README는 이 두 경우를 "A request in progress and one submission-only final round can take the total above the limit"라고 적는다.
- 공급자가 사용량을 주지 않으면 메시지, 도구 정의, 응답 본문, 추론 내용, tool call 인자를 직접 세어 추정치를 채운다([budget.go L68-L90](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/behaviortask/budget.go#L68-L90)).

### 마지막 제출 라운드

라운드 상한이나 token 예산에 도달하면 llmloop는 Grace 정책(7장)에 따라 한 번 더 요청한다. Analysis는 그 라운드에 `submit_analysis`만 노출한다.

[runner.go L156-L163](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/behaviortask/runner.go#L156-L163)

```go
	runner.budget = newBudget(deps.MaxTokensBudget)
	runner.loop = llmloop.NewRunner(llmloop.Deps{LLMClient: runner.budget.wrap(deps.LLMClient), Model: deps.Model, Session: runner.history, NewRequestMeta: deps.NewRequestMeta, MaxTokensBudget: deps.MaxTokensBudget, ProgressError: FailureSummary}, llmloop.Task{
		ToolDefs: definitions, ExecuteTool: runner.executeTool, MaxRounds: deps.Template.MaxToolRequestTimes,
		MaxContextTokens: deps.Template.MaxTokens, MaxCompletionTokens: deps.Template.MaxCompletionTokens,
		EmptyResponseMessage: "Use an available evidence tool or submit_analysis; prose alone does not complete the analysis.",
		Compression:          llmloop.CompressionPolicy{Messages: compression, SummaryTemplate: "\n\n<previous_behavior_summary>\n{{summary}}\n</previous_behavior_summary>"},
		Grace:                llmloop.GracePolicy{ReportErrors: true, Tools: []llm.ToolDef{submit}, Messages: grace},
	})
```

`ReportErrors: true`는 코드베이스에서 Analysis만 쓰는 설정이다. 마지막 라운드의 오류를 숨기지 않고 결과 오류로 돌려준다. 마지막 라운드 프롬프트는 "Do not start new investigation, fabricate missing evidence, or describe unexecuted verification as passed"라고 지시하고, 미해결 질문이 남았으면 `partial`을 선언하라고 한다([prompts/behavior_final_round.md L1](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/config/template/prompts/behavior_final_round.md#L1)). 이 라운드의 제출이 형식 오류면 다시 시도하지 않는다.

### 초기 prompt가 너무 클 때

변경 파일이 많으면 변경 목록과 배경 자료만으로 context 상한을 넘을 수 있다. `messages`는 한 번 렌더링해 보고, 넘치면 목록을 `context_read`로 읽는 색인으로 바꿔 다시 렌더링한다.

바뀌는 값은 `change_inventory`와 `context_index` 두 개다. 각각 JSON 문자열로 `runner.indexes`에 보관하고, prompt 자리에는 `{"kind": "inventory", "id": "changes", "bytes": N, "read_with": "context_read; follow the continuation cursor"}` 형태의 안내만 남긴다([runner.go L188-L195](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/behaviortask/runner.go#L188-L195)). 배경 자료 자리도 "paginated context index"에서 읽으라는 문장으로 바뀐다.

두 번째 렌더링도 넘치면 `ErrInitialPromptTooLarge`로 실패한다. 이 경우 지시문과 도구 정의만으로 상한을 넘은 것이므로 `--max-tokens`를 늘리는 것 말고는 방법이 없다. 목록을 자르거나 요약하지 않고 색인으로 옮기는 방식이라 변경 파일 정보는 잃지 않는다.

## 결과 상태와 산출물

Analysis 결과에는 서로 다른 두 종류의 "완료"가 있다. 프로그램이 기록하는 실행 상태와, 모델이 스스로 선언하는 완성도다. README는 "`Result.Completed` means the loop accepted a submission, not that the investigation is complete or correct"라고 이 구분을 강조한다.

### 코드 분석: ExecutionStatus

[result.go L123-L135](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/behaviortask/result.go#L123-L135)

```go
func (result Result) ExecutionStatus(runErr error) string {
	if result.Analysis != nil {
		if result.Completed && runErr == nil && len(result.Errors) == 0 && result.StopReason == "none" {
			return "completed"
		}
		return "partial"
	}
	var view *gitcmd.UnsupportedViewError
	if errors.As(runErr, &view) {
		return "skipped"
	}
	return "failed"
}
```

- 보고서가 수락되었더라도 라운드 상한이나 예산으로 마지막 라운드에서 제출했다면 `StopReason`이 `none`이 아니므로 `partial`이다. 마지막 라운드 프롬프트의 "An accepted report does not erase the program-recorded budget stop"과 같은 규칙이다.
- `skipped`는 `gitcmd.UnsupportedViewError`일 때만 나온다. README의 "If the active Git environment overrides the repository, index or command configuration in a way Analysis cannot reproduce, it is skipped"에 해당한다.
- 모델의 `declared_completion`은 상태 계산에 들어가지 않는다. `complete`를 선언해도 예산 정지를 덮을 수 없고, `partial`을 선언해도 실행 상태는 `completed`일 수 있다.

두 축을 조합하면 해석은 다음과 같다.

| execution_status | declared_completion | 의미 |
|---|---|---|
| `completed` | `complete` | 조사가 끝났고 모델도 완료라고 판단 |
| `completed` | `partial` | 실행은 정상, 모델이 미해결 질문을 남김 |
| `partial` | 임의 | 보고서는 있으나 예산 정지, 오류, 취소 중 하나가 있음 |
| `failed` | `unknown` | 수락된 보고서 없음 |
| `skipped` | `unknown` | Git 환경 문제로 실행하지 않음 |

### 코드 분석: workflow.NewReport

CLI는 `behaviortask.Result`를 그대로 출력하지 않고 `workflow.Report`로 바꾼다. 스키마 버전을 `ocr.review-analysis/v2`로 바꾸고, `Completed`를 `submission_accepted`라는 이름으로 내보내며, 산출물 폴더 경로를 더한다. 실패 문구를 정리하는 부분이 핵심이다.

[report.go L46-L53](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/workflow/report.go#L46-L53)

```go
		summary := behaviortask.FailureSummary(runErr)
		if summary == "analysis failed" {
			if result.SessionID != "" {
				summary += "; inspect the local session for diagnostics"
			} else if result.StopReason == "preparation" {
				summary = "analysis input or artifact preparation failed"
			}
		}
```

JSON 필드 이름 `submission_accepted`는 "제출이 수락되었다"는 뜻일 뿐이다. 이름에서 보듯 조사 완료를 뜻하지 않는다.

`FailureSummary`는 오류를 정해진 문구로 바꾼다. 시간 초과, 취소, 초기 prompt 초과, 지원하지 않는 Git 환경, 산출물 위치 오류, token 예산 소진, 공급자 HTTP 4xx와 5xx 상태 코드를 구분하고, 나머지는 모두 `analysis failed`로 묶는다([runner.go L283-L306](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/behaviortask/runner.go#L283-L306)). 원본 오류 메시지를 그대로 내보내지 않으므로, 경로나 응답 본문 같은 세부 정보는 출력에 나오지 않고 로컬 세션에서만 확인할 수 있다.

### 산출물 파일

| 파일 | 내용 | 기록 시점 |
|---|---|---|
| `result.json` | `ocr.behavior-impact/v2` 메타데이터: 실행 상태, 자기 평가, 비교 대상, 범위, 정지 사유, 사용량 | `started`로 먼저 쓰고 끝날 때 임시 파일 rename으로 교체 |
| `report.md` | 수락된 Markdown 원문. 제목을 덧붙이지 않음 | 수락 후 `O_EXCL`로 한 번 |
| `observations.jsonl` | 관찰 기록. 추가만 함 | 도구 실행마다 |
| `trace.jsonl` | 모델 호출, 압축, 도구 결과 | 실행 중 |
| `raw.jsonl` | HTTP 요청 원문 | `OCR_RAW_LOGGING=1`일 때만 |

위치는 `~/.opencodereview/sessions/<인코딩된 저장소>/<세션 ID>/analysis/`다. `result.json` 교체는 `os.CreateTemp`로 같은 폴더에 쓴 뒤 `os.Rename`으로 바꾸는 방식이라 읽는 쪽이 반쯤 쓰인 파일을 보지 않는다([result.go L137-L156](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/behaviortask/result.go#L137-L156)). 관찰 기록과 로그에는 소스 코드 원문이 들어가므로 CLI 문서도 "Evidence and logs can contain source text"라고 경고한다.

### 출력에서의 Analysis

text 출력은 Review 결과와 Project Summary 뒤에 `Analysis Report` 구역을 덧붙인다. 보고서 본문은 터미널 Markdown 렌더러로 표시하고, 상태가 `completed`가 아니면 상태와 정지 사유, 오류를 먼저 보여 준다([analysis_output.go L24-L80](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/cmd/opencodereview/analysis_output.go#L24-L80)). 실패나 `partial`일 때는 증거 폴더 경로를 `Evidence:` 줄로 알려 준다.

화면에 표시할 때는 `` `[[obs-000001]]` `` 인용 표식을 지운다. `analysisMarkdownForDisplay`는 goldmark로 문서를 파싱해 블록 시작점과 강조, 코드 span의 닫는 위치를 구한 뒤, 표식 앞뒤 공백을 문장 부호에 맞게 정리한다([analysis_output.go L103-L140](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/cmd/opencodereview/analysis_output.go#L103-L140)). 중국어 마침표와 쉼표(`。`, `，` 등)도 문장 부호 목록에 들어 있다.

JSON 출력은 스키마 버전을 `ocr.review-with-analysis/v2`로 바꾸고 네 필드를 추가한다([output.go L452-L464](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/cmd/opencodereview/output.go#L452-L464)).

| 필드 | 내용 |
|---|---|
| `analysis` | 인용 표식을 지운 표시용 Markdown |
| `analysis_raw` | 제출 원문. 인용 표식 포함 |
| `analysis_meta` | `workflow.Report` 전체. `execution_status`, `declared_completion`, `usage` 등 |
| `review_status` | Review의 manifest 종료 상태. Review 오류가 있으면 `complete`와 `skipped`를 `partial`로 낮춤([analysis.go L215-L224](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/cmd/opencodereview/analysis.go#L215-L224)) |

JSON 출력의 나머지 구조는 14장에서 다룬다.
