라운드 반복과 확정 코멘트 전달, 토큰 예산 판정을 다룬다.

## 다중 라운드와 확정 코멘트

본 리뷰는 effort에 따라 1~3라운드 돈다(05장). 라운드 2부터는 계획을 빼고, 이전 라운드에서 확정된 코멘트를 넣어 "이미 찾은 것은 반복하지 말고 나머지를 찾으라"고 요청한다. 공식 문서 [architecture.md L205-L209](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/pages/src/content/docs/en/architecture.md#L205-L209)는 계획을 빼는 이유를 "a plan tends to act as a coverage ceiling once the obvious issues are found"라고 설명한다. 계획에 적힌 위험만 확인하고 끝내는 경향을 두 번째 라운드에서 풀어 준다는 뜻이다.

### 코드 분석: 라운드별 신규 코멘트 계산

`internal/agent/agent.go`, [agent.go L1551-L1566](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/agent/agent.go#L1551-L1566)

라운드가 끝난 뒤 이번 라운드에 새로 남은 코멘트를 계산하는 부분이다.

```go
		// Drain async comment workers before computing deltas.
		if a.args.CommentWorkerPool != nil {
			a.args.CommentWorkerPool.AwaitKey(groupKey)
		}
		a.executeGroupReviewFilter(ctx, g, baseline)

		// Compute newly confirmed comments from this round.
		var newlyConfirmed []model.LlmComment
		for _, d := range g.Diffs {
			all := a.args.CommentCollector.CommentsForPath(d.NewPath)
			if b := baseline[d.NewPath]; len(all) > b {
				newlyConfirmed = append(newlyConfirmed, all[b:]...)
			}
			baseline[d.NewPath] = len(all)
		}
		confirmed = append(confirmed, newlyConfirmed...)
```

- 코멘트 저장소는 실행 전체가 공유하므로 "이번 라운드의 코멘트"를 따로 보관하지 않는다. 대신 파일별 코멘트 개수를 `baseline`에 적어 두고, 라운드 후 개수와의 차이로 신규분을 구한다
- `AwaitKey(groupKey)`는 이 묶음이 제출한 후처리 작업만 기다린다. 풀 전체를 기다리는 `Await`와 달리 다른 묶음의 작업과 경쟁하지 않는다([pool.go L120-L135](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/reviewtask/pool.go#L120-L135))
- 반성 필터는 `baseline` 이후의 코멘트, 즉 이번 라운드 코멘트만 검증한다. 필터가 지운 코멘트는 신규 확정분에 들어가지 않는다. 필터 내부는 09장에서 다룬다
- 필터는 본 리뷰가 `task_done` 없이 멈춘 라운드에서도 실행된다. 완료 여부 검사가 필터 호출 뒤에 있기 때문이다

> **주의**: 코멘트는 다른 묶음의 파일로 옮겨질 수 있다(09장의 cross-file 재배치). 이 경우 다른 묶음이 같은 경로에 쓴 코멘트가 이쪽 묶음의 신규분과 필터 후보에 섞일 수 있다(추정).

### 코드 분석: 확정 코멘트 블록

`internal/agent/util.go`, [util.go L46-L81](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/agent/util.go#L46-L81)

다음 라운드 prompt에 넣는 확정 코멘트 요약을 만든다. 핵심 부분은 다음과 같다.

```go
const (
	confirmedMaxExistingCode = 200
	confirmedMaxContent      = 300
	confirmedCap             = 30
)

// buildConfirmedCommentsBlock renders confirmed findings from prior rounds into
// a compact text block for injection into the next round's prompt. Returns ""
// for an empty slice, which triggers stripEmptyConfirmedBlock.
func buildConfirmedCommentsBlock(comments []model.LlmComment) string {
	if len(comments) == 0 {
		return ""
	}
	var sb strings.Builder
	sb.WriteString("The following issues were already identified and confirmed in a prior review pass. " +
		"Do not repeat them. " +
		"Continue reviewing all files in <review_files> and report any other real issues you find.\n\n")
	sb.WriteString("<confirmed_findings>\n")
```

코멘트마다 경로, 한 줄로 편 `existing_code`(최대 200자), 내용(최대 300자)만 넣는다. `suggestion_code`와 `thinking`은 넣지 않는다. 확정 코멘트가 30개에 이르면 라운드를 더 돌리지 않는다.

| 라운드 종료 조건 | 위치 | 결과 |
|---|---|---|
| `ctx` 취소, 시간 초과 | [agent.go L1497-L1499](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/agent/agent.go#L1497-L1499) | 오류 반환, 묶음 실패 |
| 라운드 2 이상에서 실행 예산 초과 | [agent.go L1501-L1510](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/agent/agent.go#L1501-L1510) | 남은 라운드 생략, 이전 결과 유지 |
| prompt가 `MAX_TOKENS`의 80% 초과 | [agent.go L1521-L1526](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/agent/agent.go#L1521-L1526) | 라운드 1이면 묶음 중단, 이후면 반복 종료 |
| 본 리뷰 오류 | [agent.go L1542-L1549](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/agent/agent.go#L1542-L1549) | 라운드 1이면 오류 반환, 이후면 경고 후 이전 결과 유지 |
| `task_done` 없이 중단 | [agent.go L1568-L1586](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/agent/agent.go#L1568-L1586) | `subtaskStop` 기록 후 반복 종료 |
| 신규 코멘트 0개 | [agent.go L1589-L1592](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/agent/agent.go#L1589-L1592) | 조기 종료, 완료 |
| 확정 코멘트 30개 이상 | [agent.go L1594-L1597](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/agent/agent.go#L1594-L1597) | 조기 종료, 완료 |

라운드 1과 이후 라운드의 처리가 다르다는 점이 중요하다. 라운드 1의 실패는 그 묶음의 리뷰가 아예 없다는 뜻이므로 실패로 기록한다. 라운드 2 이후의 실패는 이미 라운드 1의 결과가 있으므로 경고만 남기고 결과를 유지한다.

## 토큰 예산

토큰 사용을 제한하는 장치는 성격이 다른 두 가지다. 묶음 하나의 prompt 크기 상한과, 실행 전체의 합계 상한이다.

| 장치 | 기준 | 검사 시점 | 초과 시 |
|---|---|---|---|
| 파일 크기 검사 (04장) | diff 하나가 `MAX_TOKENS`의 80% 초과 | 선별 단계 | `too_large`로 제외 |
| prompt 크기 검사 | 렌더링한 메시지가 `MAX_TOKENS`의 80% 초과 | 라운드마다 LLM 호출 직전 | `token_threshold_exceeded` 경고, `failed(budget)` |
| dispatch 예측 검사 | 사용량 + 묶음 추정치 > `--max-tokens-budget` | 묶음 dispatch 직전 | 이 묶음과 남은 묶음 모두 미실행 |
| 라운드 검사 | 사용량 > `--max-tokens-budget` | 라운드 2 이상 시작 직전 | 남은 라운드 생략 |
| 대화 루프 검사 (07장) | 사용량 > `--max-tokens-budget` | 도구 호출 라운드마다 | grace round 후 `StopTokenBudget` |

80%라는 값은 [`llmloop.PromptTokenLimit`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/llmloop/compression.go#L29-L35) 하나에 정의되어 선별, 묶음 구성, 라운드 검사가 함께 쓴다.

### 코드 분석: dispatch 예측 검사

`internal/agent/agent.go`, [agent.go L694-L707](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/agent/agent.go#L694-L707)

다음 묶음을 띄우기 전에 그 묶음의 비용을 미리 더해 보는 부분이다.

```go
		if a.args.MaxTokensBudget > 0 {
			used := a.runner.TotalTokensUsed()
			var groupEst int64
			for _, d := range group.Diffs {
				groupEst += estimateDiffFileTokens(d)
			}
			projected := used + groupEst
			if projected > a.args.MaxTokensBudget {
				firstPath := group.Diffs[0].NewPath
				fmt.Fprintf(stdout.Writer(), "[ocr] token budget reached (used %s + group est %s = projected %s > budget %s) — skipping group %q and remaining\n",
					humanTokens(used), humanTokens(groupEst), humanTokens(projected), humanTokens(a.args.MaxTokensBudget), group.Label)
				a.recordWarning("token_budget_reached", firstPath,
					fmt.Sprintf("stopped dispatch: used %d tokens + group estimate %d = projected %d exceeds budget %d", used, groupEst, projected, a.args.MaxTokensBudget))
				a.budgetExceeded.Store(true)
```

이 검사에 걸리면 반복문을 빠져나오므로 뒤의 묶음은 모두 실행되지 않는다. 이미 떠 있는 묶음은 끝까지 돈다. 따라서 실제 사용량은 예산을 최대 동시 실행 묶음 수만큼 넘을 수 있고, 주석도 "overrun is bounded by in-flight count"라고 밝힌다. 예산 초과는 실행 실패(`SetRunFailure`)가 아니라 "보류된 실패 원인"(`SetPendingFailureCause`)으로 기록된다([agent.go L708-L719](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/agent/agent.go#L708-L719)). 실행되지 않은 파일만 `failed(budget)`이 되고, 최종 상태는 커버리지로 정해진다.

### 코드 분석: 비용 추정식

`internal/agent/estimate.go`, [estimate.go L55-L68](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/agent/estimate.go#L55-L68)

```go
func estimateDiffFileTokens(d model.Diff) int64 {
	if d.IsDeleted || d.Diff == "" {
		return 0
	}
	diffTokens := int64(llm.CountTokens(d.Diff))

	// PLAN phase input/output (small JSON output).
	total := diffTokens + promptOverheadTokens
	total += 400
	// MAIN_TASK: diff content carried across rounds + per-round overhead.
	total += (diffTokens + promptOverheadTokens) * avgMainRoundsPerFile
	total += avgOutputTokensPerRound * avgMainRoundsPerFile
	return total
}
```

상수는 [estimate.go L25-L37](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/agent/estimate.go#L25-L37)에 있다.

| 상수 | 값 | 의미 |
|---|---|---|
| `promptOverheadTokens` | 2000 | 호출 하나의 system prompt, 템플릿, 도구 정의 |
| `avgMainRoundsPerFile` | 7 | 파일 하나의 `MAIN_TASK` 도구 호출 라운드 수 (주석: 실제 저장소에서 약 6 관측, 올림) |
| `avgOutputTokensPerRound` | 700 | 라운드당 응답 토큰 |

추정식은 파일 하나를 계획 1회와 도구 호출 7라운드로 계산한다. 리뷰 라운드 수(effort), 묶음 안에서 diff가 합쳐지는 효과, 반성 필터와 위치 보정 호출은 반영하지 않는다. 주석도 이 값을 "order-of-magnitude"이자 "floor"라고 부른다. 계획 단계 출력을 "small JSON output"으로 적은 주석도 현재 계획 형식(평문 목록)과는 맞지 않는다.

[`budget_test.go`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/agent/budget_test.go#L423-L473)의 `TestDispatchSubtasks_TokenBudgetSkipsNextRoundAfterCompletedRound`는 라운드 1에서 예산을 넘긴 묶음이 라운드 2와 grace round를 모두 건너뛰고, LLM 호출 1회와 코멘트 1개로 "완료" 커버리지를 얻는지 확인한다.
