파일별 커버리지 기록과 결과 집계, reviewtask.Runner의 역할을 다룬다.

## 커버리지 기록과 결과 집계

README가 말하는 "불완전한 커버리지" 문제에 대한 코드 쪽 답은 파일 단위 run manifest다. 리뷰를 시작하기 전에 대상 파일 전체를 분모로 고정하고, 끝날 때 각 파일을 완료, 재사용, 실패 중 하나로 표시한다. 표시가 없는 파일은 `Finalize`에서 실패로 처리된다. manifest의 저장 형식은 13장에서 다룬다.

### 코드 분석: task_done 없이 멈춘 묶음의 파일별 판정

`internal/agent/agent.go`, [agent.go L790-L803](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/agent/agent.go#L790-L803)

묶음이 `task_done`을 부르지 못하고 멈췄을 때 파일을 하나씩 판정한다.

```go
					var failedCount int64
					for _, d := range g.Diffs {
						fingerprint := reviewItemFingerprint(a.reviewMode(), d)
						if comments := a.args.CommentCollector.CommentsForPath(d.NewPath); len(comments) > 0 {
							a.markCompleted(d)
							a.session.RecordReviewItemDone(d.NewPath, d.OldPath, d.NewPath, fingerprint, comments)
							continue
						}
						a.markFailed(d, stop.class, stop.reason)
						if stop.checkpoint != "" {
							a.session.RecordReviewItemFailed(d.NewPath, d.OldPath, d.NewPath, fingerprint, stop.checkpoint)
						}
						failedCount++
					}
```

코멘트가 하나라도 남은 파일은 완료로, 없는 파일은 실패로 기록한다. 주석([agent.go L783-L789](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/agent/agent.go#L783-L789))은 이렇게 하지 않으면 한 파일이 막혀 묶음 전체가, 나아가 실행 전체가 실패 상태로 떨어진다고 설명한다. 한편 이 기준은 "코멘트가 있으면 리뷰를 마친 것으로 본다"는 가정이다. 묶음의 다른 파일을 보다가 멈췄어도 코멘트가 하나 있으면 완료로 분류된다.

| 묶음 결과 | 파일 처리 | `subtaskFailed` 증가 |
|---|---|---|
| Go 오류 반환 | 묶음 전체 `markFailed`, 분류는 `classifyItemError` | 묶음 파일 수 |
| panic | 묶음 전체 `FailurePanic` | 묶음 파일 수 |
| `task_done` 없이 중단 | 코멘트 있는 파일은 완료, 없는 파일은 실패 | 실패 파일 수 (묶음에 코멘트가 없을 때만) |
| 정상 완료 | 묶음 전체 `markCompleted` | 없음 |

오류 분류는 [`classifyItemError`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/agent/agent.go#L1214-L1227)가 맡는다. 원본 오류 문자열에는 프로바이더 응답, 인증 정보, 절대 경로가 섞일 수 있으므로 manifest에는 고정된 문장만 남기고 원문은 세션 checkpoint에만 저장한다.

| 오류 | 분류 | manifest 사유 문장 |
|---|---|---|
| `llm.ErrRequestTimeout` | `timeout` | LLM request timed out; configure OCR_LLM_TIMEOUT or provider timeout_sec |
| `context.DeadlineExceeded` | `timeout` | file review exhausted its task deadline (--timeout) |
| `context.Canceled` | `cancelled` | file review was cancelled |
| `errMainTaskEmpty` | `configuration` | review template main_task is empty |
| 그 밖 | `provider` | provider or subtask request failed |

도구 호출 라운드 상한(`StopMaxRounds`)과 실행 예산(`StopTokenBudget`)으로 멈춘 경우만 `budget`으로 분류되고, 그 밖의 루프 중단은 `unknown`이다([agent.go L1242-L1247](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/agent/agent.go#L1242-L1247)). 나중에 새 중단 사유가 추가되어도 자동으로 `budget`을 물려받지 않게 하려는 의도다.

### 코드 분석: 최종 집계

`internal/agent/agent.go`, [agent.go L843-L861](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/agent/agent.go#L843-L861)

모든 묶음이 끝난 뒤 결과를 돌려주는 부분이다.

```go
	failed := atomic.LoadInt64(&a.subtaskFailed)
	reused := int64(0)
	if a.resumeInfo != nil {
		reused = a.resumeInfo.ReusedFiles
	}
	// A resumed run can still have usable coverage when every newly dispatched
	// subtask hard-fails. Preserve the dispatcher-level all-failed error only when
	// there is no reused result; otherwise the manifest is partial and the CLI
	// decides the process status from the recorded failure classifications.
	if failed > 0 && failed == dispatched && reused == 0 {
		// Even when all subtasks failed, some may have produced comments before
		// hitting the error. Return those comments instead of discarding them.
		if comments := a.args.CommentCollector.Comments(); len(comments) > 0 {
			return comments, nil
		}
		return nil, fmt.Errorf("all %d file review(s) failed — check your LLM configuration and API key", dispatched)
	}

	return a.args.CommentCollector.Comments(), nil
```

결과 코멘트는 묶음별로 따로 모으지 않고 공유 `CommentCollector`에서 한 번에 꺼낸다. 이 직전에 [`CommentWorkerPool.Await()`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/agent/agent.go#L829-L833)로 남은 후처리 작업을 모두 기다린다. 모든 묶음이 실패해도 코멘트가 하나라도 있으면 오류 대신 그 코멘트를 돌려준다. 오류는 "모든 파일 실패 + 재사용 없음 + 코멘트 없음"일 때만 난다.

돌려주는 코멘트 타입은 [`model.LlmComment`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/model/review.go#L7-L21)다.

| 필드 | 채우는 곳 | 의미 |
|---|---|---|
| `Path` | `code_comment` 인자, 없으면 묶음 키 | 대상 파일 경로 |
| `Content` | `code_comment` 인자 | 지적 내용 |
| `ExistingCode` | `code_comment` 인자, 위치 보정 시 교체 가능 | 지적 대상 코드 발췌 |
| `SuggestionCode` | `code_comment` 인자 | 수정 제안 코드 |
| `StartLine`, `EndLine` | 위치 보정 (09장) | 1부터 시작하는 줄 번호, 찾지 못하면 0 |
| `Category` | `code_comment` 인자, 정규화 | bug, security, performance 등 8종, 그 밖은 `other` |
| `Severity` | `code_comment` 인자, 정규화 | critical, high, medium, low, 그 밖은 `low` |
| `Thinking` | 응답의 추론 텍스트 | JSON 출력에만 포함 |

`Run`이 돌려준 뒤에도 CLI 쪽 `emitRunResult`가 [`diff.ResolveLineNumbers`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/cmd/opencodereview/shared.go#L828-L828)로 줄 번호를 한 번 더 계산하고 출력 형식으로 바꾼다. 위치 계산은 09장, 출력 형식은 14장에서 다룬다.

## reviewtask.Runner의 역할

`reviewtask.Runner`는 `llmloop.Runner`에 리뷰 전용 설정을 넣어 만든다. 대화 루프 자체는 07장의 범위이므로 여기서는 `Agent`가 넘기는 값만 본다.

### 코드 분석: NewRunner의 grace round 설정

`internal/reviewtask/task.go`, [task.go L98-L113](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/reviewtask/task.go#L98-L113)

```go
		MaxRounds:            deps.Template.MaxToolRequestTimes,
		MaxContextTokens:     deps.Template.MaxTokens,
		MaxCompletionTokens:  deps.Template.CompletionTokenLimit(),
		EmptyResponseMessage: "You did not successfully call any tools. Please try again or use task_done if finished.",
		Compression: llmloop.CompressionPolicy{
			Messages:        messages,
			SummaryTemplate: "\n\n<previous_review_summary>\n{{summary}}\n</previous_review_summary>",
		},
		Grace: llmloop.GracePolicy{
			Tools: graceRoundToolDefs(deps.MainToolDefs),
			Messages: []llm.Message{llm.NewTextMessage("user",
				"Your review budget is exhausted. This is your FINAL round. You may ONLY:\n"+
					"- Call code_comment to submit any findings you have identified but not yet reported.\n"+
					"- Call task_done if you have nothing more to report.\n"+
					"No other tools are available. Do not attempt further analysis.")},
		},
```

도구 호출 라운드 상한이나 실행 예산에 걸리면 바로 끊지 않고, `code_comment`와 `task_done`만 남긴 마지막 라운드를 한 번 더 준다. 이미 찾았지만 아직 보고하지 않은 결함을 버리지 않기 위한 장치다. grace round로 작업이 완료되더라도 [`recordGraceRoundLimit`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/reviewtask/task.go#L149-L155)가 `review_limit_reached` 경고를 남긴다. 완료 표시만 보면 잘리지 않은 리뷰처럼 보이기 때문이다.

| `Deps` 필드 | `Agent`가 넘기는 값 | 용도 |
|---|---|---|
| `DiffLookup` | `a.findDiff` | 코멘트 경로의 diff 조회, 위치 보정 |
| `AllDiffs` | `a.allDiffs` | 선별된 diff 전체, cross-file 재배치 (09장) |
| `NewRequestMeta` | `a.newRequestMeta` | 재시도 보고서의 요청 식별 정보, scan에서는 nil |
| `MaxTokensBudget` | `Args.MaxTokensBudget` | 대화 루프 안의 예산 검사 |

`AllDiffs`는 선별을 통과한 diff만 돌려준다. 주석([task.go L31-L38](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/reviewtask/task.go#L31-L38))은 제외된 파일로 코멘트를 옮기면 이번 리뷰가 다루지 않은 파일을 가리키게 된다는 이유를 든다. 같은 `Runner`가 `ocr scan`에서도 쓰이므로(11장) 리뷰 전용 값은 nil이면 꺼지도록 설계되어 있다.
