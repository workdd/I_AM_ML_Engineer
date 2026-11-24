deterministic 줄 번호 계산, 파일 간 재배치, LLM 재위치를 다룬다.

## deterministic 줄 번호 계산

[`ResolveComment`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/diff/resolver.go#L63-L74)는 이미 줄 번호가 있으면 그대로 두고, `existing_code`가 비면 실패하고, 그 외에는 hunk 매칭과 파일 전체 매칭을 차례로 시도한다.

### 코드 분석: resolveFromHunk

`internal/diff/resolver.go`, [resolver.go L161-L182](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/diff/resolver.go#L161-L182)

hunk 안에서 발췌를 찾는 세 번의 시도 중 앞의 두 번이다.

```go
	targetLines := splitAndNormalize(cm.ExistingCode)
	if len(targetLines) == 0 {
		return false
	}

	for i := range hunks {
		newSide := extractSideLines(&hunks[i], true)
		if start, end, ok := matchConsecutive(newSide, targetLines); ok {
			cm.StartLine = start
			cm.EndLine = end
			return true
		}
	}

	for i := range hunks {
		newSide := slices.DeleteFunc(extractSideLines(&hunks[i], true), func(l indexedLine) bool { return l.content == "" })
		if start, end, ok := matchConsecutive(newSide, targetLines); ok {
			cm.StartLine = start
			cm.EndLine = end
			return true
		}
	}
```

| 시도 | 비교 대상 | 줄 번호 기준 | 근거 |
|---|---|---|---|
| 1 | hunk의 새 쪽 (context + 추가 줄) | 새 파일 | `existing_code`는 추가된 줄을 인용해야 한다 |
| 2 | 1과 같되 빈 줄 제거 | 새 파일 | 발췌는 `splitAndNormalize`에서 빈 줄이 빠지므로 대상도 맞춤 |
| 3 | hunk의 옛 쪽 (context + 삭제 줄) | 옛 파일 | 삭제된 코드를 인용한 경우의 마지막 수단 |
| 4 | 새 파일 전체 내용, 빈 줄 제거 | 새 파일 | hunk 밖 코드를 인용한 경우 ([resolver.go L251-L296](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/diff/resolver.go#L251-L296)) |

비교 단위는 [`normalizeLine`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/diff/resolver.go#L314-L319)으로 정규화한 줄이다. 앞뒤 공백을 지우고 맨 앞의 `+`나 `-` 하나를 뗀다. 그래서 모델이 diff 표기를 붙인 채로 인용해도, 들여쓰기를 다르게 해도 맞는다. 반면 줄 안의 공백이나 따옴표 하나라도 다르면 맞지 않는다. [`matchConsecutive`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/diff/resolver.go#L230-L247)는 처음 맞는 위치를 돌려주므로, 같은 발췌가 hunk 안에 두 번 나오면 앞의 것이 선택된다.

> **주의**: 시도 3이 성공하면 `StartLine`, `EndLine`은 **옛 파일** 기준 줄 번호다. `LlmComment`에는 어느 쪽 기준인지 표시하는 필드가 없어서, 출력에서는 새 파일 줄 번호와 구분되지 않는다. 주석([resolver.go L148-L154](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/diff/resolver.go#L148-L154))도 "an old-side match reports old-file line numbers"라고 밝히고 이 시도를 마지막에 둔다.

## cross-file 재배치

에이전트는 `file_read_diff`로 다른 파일을 읽을 수 있으므로, 헤더 파일을 리뷰하다가 구현 파일의 코드를 인용하고 코멘트는 헤더에 다는 일이 생긴다. 이때 같은 파일 매칭은 실패하고, LLM 재위치는 엉뚱한 파일의 diff에서 비슷해 보이는 줄을 골라 그럴듯하지만 틀린 위치를 만든다. [`RelocateAcrossFiles`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/diff/resolver.go#L76-L100)는 이 문제를 문자열 매칭만으로 막는다.

### 코드 분석: RelocateAcrossFiles

`internal/diff/resolver.go`, [resolver.go L112-L135](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/diff/resolver.go#L112-L135)

```go
	for i := range diffs {
		d := &diffs[i]
		if d.NewPath == cm.Path || d.OldPath == cm.Path {
			continue
		}
		probe := *cm
		probe.StartLine, probe.EndLine = 0, 0
		if !ResolveComment(&probe, d) {
			continue
		}
		path := d.NewPath
		if path == "" {
			path = d.OldPath
		}
		hits = append(hits, hit{path: path, start: probe.StartLine, end: probe.EndLine})
		if len(hits) > 1 {
			// Ambiguous already; no verdict can come from looking further.
			return "", false
		}
	}

	if len(hits) != 1 {
		return "", false
	}
```

- 코멘트 자기 경로는 이미 시도했으므로 건너뛴다
- 복사본(`probe`)으로 시험해서, 실패한 후보가 원본에 줄 번호를 남기지 않게 한다
- 일치가 정확히 하나일 때만 `Path`, `StartLine`, `EndLine`을 함께 바꾼다. 같은 상용구가 여러 파일에 있을 수 있으므로 둘 이상이면 아무것도 바꾸지 않는다
- 탐색 범위는 `AllDiffs`, 곧 이번 실행에서 리뷰 대상으로 선별된 diff다. 선별에서 제외된 파일로는 옮기지 않는다(06장)

[`relocate_across_files_test.go`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/diff/relocate_across_files_test.go#L43-L101)가 구현 파일로 옮기는 경우, 모호해서 거절하는 경우, 일치가 없는 경우를 검사한다.

## LLM 재위치 작업

deterministic 시도가 모두 실패하고 코멘트 경로의 diff가 있으면 `RE_LOCATION_TASK`를 부른다. 모델에게 diff와 실패한 발췌, 코멘트 내용을 주고, diff에서 그대로 복사한 코드 블록 하나를 받는다.

```text
Below is a unified diff and a review comment. Identify the minimal contiguous code range in the diff that the comment targets.

Rules:
1. Copy the relevant lines VERBATIM from the diff — do not rewrite, reformat, or add anything.
2. Strip leading diff markers (`+`, `-`, ` `) from each line before outputting.
3. Include only the lines directly related to the issue — no surrounding context.
4. If multiple disjoint locations apply, pick the single most relevant one.
5. Output ONLY a fenced code block. No explanation, no commentary.
```

출처: [re_location_task_user.md L1-L8](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/config/template/prompts/re_location_task_user.md#L1-L8)

system 메시지는 한 줄이고 끝에 `/no_think`를 붙인다([re_location_task_system.md L1-L1](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/config/template/prompts/re_location_task_system.md#L1-L1)). 일부 추론 모델에게 추론 과정을 생략하라는 지시로 보인다(추정).

### 코드 분석: BuildReLocationMessages

`internal/diff/relocation.go`, [relocation.go L33-L41](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/diff/relocation.go#L33-L41)

```go
	messages := make([]llm.Message, 0, len(task.Messages))
	for _, m := range task.Messages {
		content := m.Content
		content = strings.ReplaceAll(content, "{diff}", d.Diff)
		content = strings.ReplaceAll(content, "{existing_code}", cm.ExistingCode)
		content = strings.ReplaceAll(content, "{suggestion_content}", cm.Content)
		messages = append(messages, llm.NewTextMessage(m.Role, content))
	}
	return messages
```

이 작업만 자리표시자가 중괄호 하나(`{diff}`)다. 다른 모든 작업은 `{{...}}`를 쓴다(05장). 공식 문서 [architecture.md L385-L388](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/pages/src/content/docs/en/architecture.md#L385-L388)도 이 차이를 별도 주의 문단으로 밝힌다. 넣는 diff는 묶음 전체가 아니라 코멘트 경로 파일 하나의 diff다.

메시지 구성을 `ReLocateComment`에서 떼어 낸 이유는 주석([relocation.go L23-L27](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/diff/relocation.go#L23-L27))에 있다. 호출자가 요청 전에 세션 기록을 먼저 만들어 `RequestNo`를 알아야 재시도 보고서와 세션 기록을 연결할 수 있기 때문이다.

### 코드 분석: ReLocateComment의 결과 처리

`internal/diff/relocation.go`, [relocation.go L83-L94](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/diff/relocation.go#L83-L94)

```go
	code := extractCodeBlock(resp.Content())
	if code == "" {
		return false, resp
	}

	original := cm.ExistingCode
	cm.ExistingCode = code
	if ResolveComment(cm, d) {
		return true, resp
	}
	cm.ExistingCode = original
	return false, resp
```

LLM 응답을 줄 번호로 직접 쓰지 않는다. 응답에서 첫 번째 코드 블록을 꺼내([`extractCodeBlock`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/diff/relocation.go#L99-L117)) 같은 deterministic 매칭에 다시 넣고, 맞을 때만 받아들인다. 맞지 않으면 `ExistingCode`를 원래 값으로 되돌린다. 모델이 diff에 없는 코드를 만들어 내도 위치가 조작되지 않는 구조다.

| 결과 | `ExistingCode` | 줄 번호 | 세션 기록 |
|---|---|---|---|
| 요청 실패 | 원래 값 | 0 | `re-location LLM call failed` 오류 |
| 코드 블록 없음 | 원래 값 | 0 | 응답 기록 |
| 코드 블록이 diff와 불일치 | 원래 값 | 0 | 응답 기록 |
| 코드 블록이 diff와 일치 | LLM이 돌려준 발췌 | 계산된 값 | 응답 기록 |

요청의 토큰 사용량은 실행 전체 집계에 더해진다([tools.go L154-L159](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/reviewtask/tools.go#L154-L159)). 다만 06장의 실행 예산 검사는 dispatch와 라운드 경계에서만 하므로, 재위치 요청 자체는 예산 초과 여부와 관계없이 실행된다(추정).

## 출력 직전 줄 번호 재계산

`Agent.Run`이 끝나면 CLI의 `emitRunResult`가 [`diff.ResolveLineNumbers`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/cmd/opencodereview/shared.go#L828-L828)를 한 번 더 부른다.

### 코드 분석: ResolveLineNumbers

`internal/diff/resolver.go`, [resolver.go L36-L56](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/diff/resolver.go#L36-L56)

```go
	for i := range result {
		cm := &result[i]
		if cm.StartLine > 0 || cm.EndLine > 0 {
			continue
		}
		if cm.ExistingCode == "" {
			continue
		}
		d, ok := diffByPath[cm.Path]
		if !ok {
			continue
		}

		// Primary: try matching from deleted/context lines in diff hunks
		if resolveFromHunk(d, cm) {
			continue
		}

		// Fallback: scan the new file content for consecutive matches
		resolveFromFileContent(d, cm)
	}
```

이미 줄 번호가 있는 코멘트는 건너뛰고, 코멘트 자기 경로의 diff 하나만 본다. 비동기 처리 경로와 같은 `resolveFromHunk`, `resolveFromFileContent`를 쓰므로, 비동기 경로에서 실패한 코멘트가 여기서 성공할 수 있는 경우는 많지 않다. 재개 실행(13장)에서 이전 세션의 코멘트를 그대로 가져온 경우처럼 비동기 경로를 거치지 않은 코멘트를 위한 안전망으로 보인다(추정).

> **주의**: 공식 문서 [architecture.md L313-L317](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/pages/src/content/docs/en/architecture.md#L313-L317)은 이 재계산이 `review_cmd.go`에 있고 "comments whose `existing_code` spans multiple files"를 잡는다고 설명한다. 실제 호출은 `shared.go`의 `emitRunResult`에 있고, 함수는 코멘트 자기 경로의 diff만 보며 파일을 넘나드는 탐색은 하지 않는다. 다른 파일로 옮기는 일은 앞 절의 `RelocateAcrossFiles`가 비동기 경로에서만 한다. 주석 "try matching from deleted/context lines"도 새 쪽을 먼저 보는 실제 순서와 다르다.
