`internal/llmloop` 패키지는 "LLM에 요청하고, 도구 호출을 실행하고, 결과를 대화에 붙여 다시 요청한다"는 반복을 한 곳에 모은 공통 엔진이다. diff 리뷰, 전체 파일 스캔, 동작 영향 분석이 모두 이 루프를 쓰고, 각자는 도구 목록과 도구 실행 함수, 압축 프롬프트만 바꿔 끼운다. 이 장은 턴 루프의 순서, 도구 호출 결과 분류, 다섯 가지 종료 사유, grace round, 그리고 컨텍스트가 커졌을 때의 세 구역 메모리 압축을 코드로 따라간다.

## llmloop 패키지 구성

패키지는 테스트를 빼면 파일 네 개다. 루프 본체와 압축 로직이 대부분을 차지하고, 나머지 두 파일은 타입 정의다.

| 파일 | 줄 수 | 역할 |
|---|---|---|
| `loop.go` | 632 | `Runner`, 턴 루프 `runMainTask`, grace round, 종료 사유 `MainLoopStop`, 토큰 집계 |
| `compression.go` | 467 | 압축 임계값, 세 구역 분할, 압축 요청과 검증, 비동기 압축 |
| `task.go` | 64 | `Task`, `ToolOutcome`, `CompressionPolicy`, `GracePolicy` 등 호출 측이 채우는 구조체 |
| `warnings.go` | 13 | 비치명적 경고 `AgentWarning` |

이 루프를 쓰는 쪽은 세 곳이다. diff 리뷰와 전체 파일 스캔은 `internal/reviewtask`의 `Runner`를 거쳐 같은 설정으로 루프를 만들고(06장, 11장), 동작 영향 분석은 `internal/behaviortask`가 별도 설정으로 루프를 만든다(12장). llmloop는 "어떤 도구가 있는지", "도구가 무엇을 하는지"를 모른다. 호출 측이 넘겨 준 `Task`가 그 정보를 모두 갖고 있다.

### 코드 분석: Task 구조체

호출 측이 루프에 넘기는 작업 정의다. 실행 환경(LLM 클라이언트, 모델 이름, 세션, 누적 토큰 예산)은 별도 구조체 [`Deps`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/llmloop/loop.go#L24-L31)로 넘긴다.

파일: [`internal/llmloop/task.go` L50-L59](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/llmloop/task.go#L50-L59)

```go
type Task struct {
	ToolDefs             []llm.ToolDef
	ExecuteTool          ToolExecutor
	MaxRounds            int
	MaxContextTokens     int
	MaxCompletionTokens  int
	EmptyResponseMessage string
	Compression          CompressionPolicy
	Grace                GracePolicy
}
```

| 필드 | 루프 안의 용도 |
|---|---|
| `ToolDefs` | 매 요청의 `Tools`로 전달하고, 모델이 부른 도구가 허용된 것인지 확인하는 기준 |
| `ExecuteTool` | 도구 호출 하나를 실행해 `ToolOutcome`을 돌려주는 콜백 |
| `MaxRounds` | 라운드 상한. 템플릿의 `MAX_TOOL_REQUEST_TIMES` 값 |
| `MaxContextTokens` | 압축 임계값 계산 기준. 템플릿의 `MAX_TOKENS` 값 |
| `MaxCompletionTokens` | 요청의 `MaxTokens`와 압축 요약 길이 상한 |
| `EmptyResponseMessage` | 모델이 도구를 하나도 부르지 않았을 때 대화에 붙이는 user 메시지 |
| `Compression` | 압축용 메시지 템플릿과 요약을 붙일 틀(`{{summary}}` 자리표시자) |
| `Grace` | 예산이 끝났을 때 한 번 더 주는 마지막 라운드의 메시지와 도구 |

호출 측 세 곳이 넣는 값은 다음과 같다. 수치는 각 템플릿 파일의 기본값이다.

| 호출 측 | 라운드 상한 | `MAX_TOKENS` | 요약 틀 | grace 도구 |
|---|---|---|---|---|
| diff 리뷰 ([`task_template.json`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/config/template/task_template.json#L40-L47)) | 100 | 200000 | `<previous_review_summary>` | `code_comment`, `task_done` |
| 전체 파일 스캔 ([`scan_template.json`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/config/template/scan_template.json#L81-L88)) | 60 | 58888 | `<previous_review_summary>` | `code_comment`, `task_done` |
| 동작 영향 분석 ([`behavior_template.json`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/config/template/behavior_template.json#L20-L22)) | 300 | 200000 | `<previous_behavior_summary>` | `submit_analysis` |

diff 리뷰와 스캔의 값은 [`reviewtask.NewRunner`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/reviewtask/task.go#L70-L116)에서, 동작 영향 분석의 값은 [`behaviortask` L157-L163](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/behaviortask/runner.go#L157-L163)에서 채운다.

`Runner` 하나는 한 세션 동안 여러 하위 작업(파일 묶음이나 파일 하나)에 공유된다. 토큰 카운터와 경고, 도구 호출 횟수는 `Runner`에 누적되고, 압축 상태는 대화(`Run` 호출) 하나마다 따로 둔다. 이 구분이 뒤의 비동기 압축 절에서 중요해진다.

## 정리

- llmloop는 도구의 의미를 모르는 공통 루프다. 호출 측이 `Task`에 도구 정의, 실행 콜백, 라운드와 토큰 상한, 압축 프롬프트, grace 정책을 넣고, 리뷰와 스캔은 `reviewtask`를 거쳐 같은 설정을, 동작 영향 분석은 별도 설정을 쓴다
- 한 라운드는 누적 예산 확인, 세션 기록, LLM 요청, 도구 순차 실행, 다음 메시지 구성 순서다. 도구 실행 전에 "이번 요청에 보낸 도구 목록"으로 허용 여부를 다시 확인해, 제한된 라운드에서 이전 도구를 부르지 못하게 한다
- 종료 사유는 `StopMaxRounds`, `StopEmptyRounds`, `StopCompression`, `StopTokenBudget`, `StopNone` 다섯 가지이고, 선언된 예산 두 가지로 멈춘 경우에만 도구를 줄인 grace round를 한 번 준다. grace round에서 완료해도 원래 종료 사유는 남는다
- 압축은 클라이언트 측 토큰 추정치(tiktoken `cl100k_base`) 기준으로 60%에서 백그라운드, 80%에서 동기로 실행한다. 대화를 frozen, compress, active 세 구역으로 나눠 가운데만 요약하고, 요약은 첫 user 메시지 끝에 붙인다
- 압축의 모든 실패 경로는 원래 대화를 돌려준다. 루프를 멈추는 것은 "압축 후에도 80% 이상"과 "예산 거부" 두 경우뿐이다
- 비동기 압축 상태는 대화 단위로 두어 하위 작업끼리 간섭하지 않게 하고, 세션 마무리 전에 `WaitBackground`로 남은 작업을 기다린다
