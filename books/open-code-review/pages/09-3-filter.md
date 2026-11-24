반성 필터의 오탐 제거와 제안 코드 diff 렌더링, 문서와의 차이를 다룬다.

## 반성 필터

반성 단계는 [`executeGroupReviewFilter`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/agent/agent.go#L1843-L1965)다. 리뷰 라운드가 끝날 때마다 그 라운드에 새로 생긴 코멘트를 묶음 diff와 함께 다른 LLM 요청에 보내, diff가 **반증하는** 코멘트만 지운다.

| 항목 | 값 |
|---|---|
| 실행 시점 | 라운드마다 `AwaitKey` 직후 (06장) |
| 후보 | 파일별 `baseline` 이후 코멘트, 곧 이번 라운드 코멘트 |
| 입력 | 묶음 diff 전체, 후보 JSON (`id`, `path`, `content`, `existing_code`) |
| 도구 | `report_incorrect_comments`, `approve_all_comments` (Go 코드에 고정) |
| 도구 강제 | 하지 않음 (`ToolChoice` 비움) |
| 생략 조건 | 템플릿에 작업 없음, `--no-filter`, 후보 0개 |
| 실패 시 | 아무것도 지우지 않음 |

### 필터 prompt의 판단 기준

system prompt는 필터의 역할을 "fact-checker"로 좁히고, 두 가지 실수의 비용이 다르다는 점을 먼저 못박는다.

```text
The two mistakes available to you are not equally bad:

- Keeping an incorrect comment costs a reviewer a few seconds of attention.
- Removing a correct comment silently destroys a real finding. It never reaches anyone, and nobody learns that it was dropped.

So when your evidence falls short of proof, approve. "Suspicious", "I cannot verify this", "low value", "the flagged code looks fine to me", and "I would not have raised this" all mean approve.
```

출처: [review_filter_task_system.md L7-L12](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/config/template/prompts/review_filter_task_system.md#L7-L12)

user prompt는 단계별 판정 순서를 고정한다([review_filter_task_user.md L52-L68](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/config/template/prompts/review_filter_task_user.md#L52-L68)).

| 단계 | 판정 | 결과 |
|---|---|---|
| 1 보호 주제 | 메모리 안전, 동시성, 링크와 선언 일관성, 동작 또는 호환성 변경, 쓰이지 않는 매개변수 | 유지, 판정 종료 |
| 2 가치 판단 | 스타일, 명명, 주석 문구, 가독성이고 주장 내용이 사실 | 유지, 판정 종료 |
| 3 Ground A | 지적한 코드가 코멘트 경로 파일의 diff에 없음 | 제거 |
| 4 Ground B | 묶음의 어떤 diff 한 줄이 핵심 주장을 글자 그대로 반박 | 제거 |
| 5 | 위에 해당 없음 | 유지 |

보호 주제는 정답 여부를 따지기 전에 적용되는 거부권이다. prompt는 "On a protected subject you do not get to be confident"라고 적는다. 필터 모델이 언어나 런타임 동작에 대해 확신할 때 가장 자주 틀리는 분야를 미리 막은 것이다. 필터는 도구 없이 diff만 보는데, 리뷰 에이전트는 코드베이스 전체를 읽었을 수 있다. system prompt의 "Anything you cannot see, the Agent may well have seen"이 이 비대칭을 직접 밝힌다.

### 코드 분석: filterTools의 필드 순서

`internal/agent/agent.go`, [agent.go L1617-L1636](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/agent/agent.go#L1617-L1636)

제거를 보고하는 도구의 매개변수 정의다.

```go
			// Field order matters and is load-bearing. Go serializes these
			// properties alphabetically, so "analysis" is emitted before
			// "comment_ids" and the model reasons before it commits. With the
			// order reversed it picks ids first and cannot retract them: replaying
			// recorded sessions showed it writing "this is a protected subject, I
			// should not remove it" in the later field while the id stayed in the
			// earlier one. Do not rename these fields into a different relative
			// order.
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"analysis": map[string]any{
						"type": "array",
						"description": "Work through every candidate comment BEFORE deciding. One entry per candidate: its id, " +
							"whether its subject hits the protected-subject veto (Step 1) or the value veto (Step 2), " +
							"the exact diff line that refutes it if any, and your final call. " +
							"Only ids you conclude here as removable may appear in comment_ids.",
						"items": map[string]any{"type": "string"},
					},
					"comment_ids": map[string]any{
```

Go의 `map`은 JSON으로 바꿀 때 키를 알파벳 순으로 쓴다. 그래서 `analysis`가 `comment_ids`보다 먼저 직렬화되고, 모델은 후보마다 판단을 먼저 적은 뒤 제거할 id를 적는다. 주석은 순서가 반대였을 때 기록된 세션을 재실행해 보니, 모델이 id를 먼저 적고 나중 필드에 "보호 주제라 지우면 안 된다"고 쓰면서도 id를 철회하지 못했다고 설명한다. 필드 이름이 곧 생성 순서를 정하므로 이름을 바꾸면 동작이 달라진다.

### 코드 분석: 응답 해석과 fail-open

`internal/agent/agent.go`, [agent.go L1931-L1942](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/agent/agent.go#L1931-L1942)

```go
	indices := parseFilterToolCalls(resp.ToolCalls(), len(candidates))
	if indices == nil {
		indices = parseFilterResponse(resp.Content(), len(candidates))
	}
	telemetry.SetAttr(span, "comments.filtered", len(indices))
	if len(indices) == 0 {
		telemetry.Event(ctx, "review_filter.completed",
			attribute.String("group.label", groupKey),
			attribute.Int("total_comments", len(candidates)),
			attribute.Int("removed", 0))
		return
	}
```

| 응답 형태 | 해석 | 결과 |
|---|---|---|
| `approve_all_comments` 호출 | 빈 집합 | 제거 없음 |
| `report_incorrect_comments` 호출 | `comment_ids`의 `c-N`을 후보 번호로 변환, 범위 밖 무시 | 해당 후보 제거 |
| 인자 JSON 파싱 실패 | 그 호출 무시 | 다른 호출이 없으면 텍스트 해석으로 |
| 도구 호출 없음, 본문이 `["c-0", ...]` JSON 배열 | 텍스트 해석 ([agent.go L2023-L2042](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/agent/agent.go#L2023-L2042)) | 해당 후보 제거 |
| 도구 호출 없음, 본문 해석 불가 | `nil` | 제거 없음 |
| 요청 오류 | 오류 출력 | 제거 없음 |

모든 실패가 "지우지 않음"으로 끝난다. [`coverage_test.go`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/agent/coverage_test.go#L276-L311)의 `TestExecuteReviewFilter_OmitsToolChoiceAndFailsOpenWithoutToolCall`이 도구 선택을 강제하지 않는 것과 텍스트만 온 응답에서 코멘트가 남는 것을 함께 검사한다. 필터의 오류가 진짜 결함을 지우는 쪽으로 번지지 않게 하는 설계다.

제거는 후보의 "파일 안 순번"으로 이뤄진다. 후보를 모을 때 파일별 코멘트 목록의 실제 순번(`pathIdx`)을 기억해 두고([agent.go L1860-L1875](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/agent/agent.go#L1860-L1875)), [`RemoveByPathAndIndices`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/tool/comment_collector.go#L94-L116)가 그 순번에 해당하는 코멘트만 지운다. 코멘트에 고유 id가 없어서 택한 방식이다.

## 제안 코드 diff 렌더링

`suggestion_code`는 텍스트 출력에서 `existing_code`와 줄 단위로 비교해 색을 입혀 보여 준다. 이 비교는 [`internal/suggestdiff`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/suggestdiff/diff.go#L28-L80) 패키지가 맡고, 호출하는 곳은 [`output.go`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/cmd/opencodereview/output.go#L294-L301)의 `buildDiffLines` 한 곳이다. 두 필드 중 하나라도 비면 비교하지 않는다.

### 코드 분석: ComputeLineDiff의 역추적

`internal/suggestdiff/diff.go`, [diff.go L55-L73](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/suggestdiff/diff.go#L55-L73)

```go
	for i > 0 || j > 0 {
		if i > 0 && j > 0 && strings.EqualFold(strings.TrimSpace(oldLines[i-1]), strings.TrimSpace(newLines[j-1])) {
			if oldLines[i-1] == newLines[j-1] {
				back = append(back, DiffLine{Type: DiffContext, Content: oldLines[i-1]})
			} else {
				// Fuzzy equality is only for alignment; preserve raw changes in the rendered diff.
				back = append(back, DiffLine{Type: DiffAdded, Content: newLines[j-1]})
				back = append(back, DiffLine{Type: DiffDeleted, Content: oldLines[i-1]})
			}
			i--
			j--
		} else if j > 0 && (i == 0 || lcs[i][j-1] >= lcs[i-1][j]) {
			back = append(back, DiffLine{Type: DiffAdded, Content: newLines[j-1]})
			j--
		} else {
			back = append(back, DiffLine{Type: DiffDeleted, Content: oldLines[i-1]})
			i--
		}
	}
```

줄을 맞출 때는 앞뒤 공백과 대소문자를 무시하지만, 실제 내용이 다르면 같은 줄로 표시하지 않고 삭제와 추가 한 쌍으로 그린다. 들여쓰기만 바꾼 제안도 변경으로 보인다. 역추적 중에는 추가를 먼저 넣고 마지막에 순서를 뒤집으므로 출력에서는 삭제 줄이 추가 줄 앞에 온다.

> **주의**: 함수 주석은 "Myers-style LCS"라고 적지만, 구현은 $(m+1) \times (n+1)$ 크기의 표를 채우는 일반적인 동적 계획법 LCS다. Myers 알고리즘의 탐색 방식은 쓰지 않는다. 입력이 코멘트 하나의 짧은 발췌라 크기 문제는 없다.

SARIF 출력의 수정 제안(`fixes`)은 이 패키지를 쓰지 않고 `existing_code` 위치를 `suggestion_code`로 바꾸는 방식이며, 유효한 줄 범위가 있을 때만 만든다. 14장에서 다룬다.

## 문서와 구현의 차이

이 장에서 확인한 차이를 모으면 다음과 같다.

| 출처 | 설명 | 실제 구현 |
|---|---|---|
| README "External positioning" | 독립 모듈로 위치 정확도 향상 | 대부분 deterministic 문자열 매칭, LLM은 마지막 수단이고 결과도 다시 문자열 매칭으로 검증 |
| `CommentWorkerPool` 주석 | reflection, suggestion validation 포함 | 풀에서는 위치 계산과 재위치만 수행, 필터는 동기, 제안 검증 단계 없음 |
| [architecture.md L309-L312](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/pages/src/content/docs/en/architecture.md#L309-L312) | 필터는 "after the main loop finishes" | 라운드마다 실행, 이번 라운드 코멘트만 대상 |
| [architecture.md L374-L374](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/pages/src/content/docs/en/architecture.md#L374-L374) | `{{path}}`를 `REVIEW_FILTER_TASK`에서 사용 | 코드는 치환하지만([agent.go L1892-L1892](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/agent/agent.go#L1892-L1892)) 필터 prompt 파일에 `{{path}}`가 없음 |
| [architecture.md L313-L317](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/pages/src/content/docs/en/architecture.md#L313-L317) | 두 번째 계산은 `review_cmd.go`, 여러 파일에 걸친 발췌 처리 | `shared.go`에서 호출, 자기 경로 diff만 확인 |
| `suggestdiff` 주석 | Myers-style LCS | 동적 계획법 LCS |
