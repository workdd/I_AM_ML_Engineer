리뷰 전 토큰 비용 추정과 `--preview` 출력을 다룬다.

## 비용 추정

### 코드 분석: 파일당 token 추정

`--max-tokens-budget`을 준 실행은 리뷰 시작 전에 예상 token 사용량을 출력한다.

파일: [internal/agent/estimate.go L55-L68](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/agent/estimate.go#L55-L68)

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

[상수](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/agent/estimate.go#L25-L37)는 고정 prompt 오버헤드 `promptOverheadTokens` = 2000, 파일당 본 리뷰 라운드 `avgMainRoundsPerFile` = 7, 라운드당 출력 `avgOutputTokensPerRound` = 700이다. diff token 수를 $T$라 하면 파일 하나의 추정값은 다음과 같다.

$$
E(T) = (T + 2000) + 400 + 7(T + 2000) + 7 \times 700 = 8T + 21300
$$

라운드 수 7은 주석에 "실제 저장소에서 약 6회를 관찰해 올림"이라고 적혀 있다(자체 발표). 주석은 이 추정이 계획 단계를 항상 포함하고 도구 호출로 prompt가 불어나는 효과를 반영하지 못하므로 "하한(floor)"이며, 목적은 청구 금액이 아니라 규모 경고라고 밝힌다. 같은 상수가 11장의 `internal/scan/estimate.go`에도 있고, 패키지 의존을 피하려고 복제해 두었다는 "KEEP IN SYNC" 주석이 붙어 있다.

추정은 두 곳에서 쓰인다.

| 사용처 | 함수 | 동작 |
|---|---|---|
| 실행 전 경고 | `estimateDiffCost` | 선택된 파일 전체 합계를 `[ocr] estimated cost: ~N file(s), est. ...` 형식으로 출력. 예산보다 크면 `WARNING: ... review will stop partway` 추가 |
| 묶음 배정 전 점검 | `estimateDiffFileTokens` | 이미 쓴 token에 다음 묶음의 추정값을 더해 예산을 넘으면 그 묶음부터 배정 중단 (06장) |

실행 전 경고는 [agent.go L389-L406](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/agent/agent.go#L389-L406)에서 `MaxTokensBudget > 0`일 때만 출력된다. 주석은 예산을 쓰지 않는 사용자의 출력 형식을 바꾸지 않으려고 이렇게 제한했다고 설명한다. 금액은 프로바이더와 모델마다 단가가 달라 표시하지 않는다.

## 리뷰 대상 미리 보기 출력

### 미리보기의 범위

`ocr review --preview`(또는 `-p`)는 LLM을 부르지 않고 어떤 파일이 리뷰될지 보여 준다. 02장에서 본 것처럼 LLM 런타임을 만들기 전에 분기하므로 API 키가 없어도 된다. [`agent.Preview`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/agent/preview.go#L35-L53)는 `agent.New`를 거치지 않고 빈 `Agent`를 만든다. 주석은 `New`를 쓰면 세션이 자동 생성되어 OCR 홈 아래에 마무리되지 않은 JSONL 파일이 남는다고 설명한다.

### 코드 분석: 미리보기 항목 구성

파일: [internal/agent/preview.go L76-L99](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/agent/preview.go#L76-L99)

```go
	set.ForEachInOrder(func(d model.Diff, providerExcluded bool) {
		if providerExcluded {
			result.TotalInsertions += d.Insertions
			result.TotalDeletions += d.Deletions
			result.ExcludedCount++
			result.Entries = append(result.Entries, DiffPreviewEntry{
				Path:          effectivePath(d),
				Insertions:    d.Insertions,
				Deletions:     d.Deletions,
				Status:        diffStatus(d),
				ExcludeReason: ExcludeProviderDirectory,
			})
			return
		}
		dec := decisions[next]
		next++
		entry := DiffPreviewEntry{
			Path:          effectivePath(dec.Diff),
			Insertions:    dec.Diff.Insertions,
			Deletions:     dec.Diff.Deletions,
			Status:        diffStatus(dec.Diff),
			WillReview:    dec.selected(),
			ExcludeReason: dec.Reason,
		}
```

미리보기는 실제 실행이 버리는 내장 제외 디렉터리 파일까지 받기 위해 `GetDiff` 대신 `GetDiffSet`을 부른다(03장). 그 파일들은 `selectFiles`를 거치지 않고 바로 `provider_directory` 사유가 붙는다. 나머지는 실제 실행과 같은 `selectFiles` 결과를 쓴다. `ForEachInOrder`로 두 종류를 git 출력 순서대로 섞어 보여 주므로, 줄 수 합계가 git의 변경 통계와 맞는다.

미리보기 JSON(`--preview --format json`)의 필드는 [model/preview.go](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/model/preview.go#L40-L59)에 정의되어 있다.

| 필드 | 의미 |
|---|---|
| `files[].path` | 파일 경로. 삭제면 이전 경로 |
| `files[].status` | `added`, `modified`, `deleted`, `renamed`, `binary` |
| `files[].insertions`, `files[].deletions` | 추가 줄, 삭제 줄 수 |
| `files[].will_review` | 리뷰 대상 여부 |
| `files[].exclude_reason` | 제외 사유 문자열 (위 판정 표) |
| `total_files`, `total_insertions`, `total_deletions` | 전체 변경 통계 |
| `reviewable_count`, `excluded_count` | 리뷰 대상 수, 제외 수 |

text 형식은 `Preview: N file(s) changed`, `Will review (N):`, `Excluded from review (N):` 세 블록으로 출력된다([output.go L792-L831](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/cmd/opencodereview/output.go#L792-L831)). SARIF는 완료된 리뷰 결과가 필요하다는 이유로 미리보기에서 거부된다.

미리보기와 실제 실행의 차이는 세 가지다.

| 항목 | 미리보기 | 실제 실행 |
|---|---|---|
| 재개 시 완료 파일 제외 | 반영 안 함 (`--preview`와 `--resume`은 함께 쓸 수 없음) | `applyResume`이 제외 |
| 예산 초과로 배정 중단 | 반영 안 함 | 실행 중 결정 |
| workspace의 내장 제외 디렉터리 untracked 파일 | 목록에 없음 | 리뷰 대상 아님 |

마지막 항목은 [`Preview` 주석](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/agent/preview.go#L42-L46)이 밝힌 의도된 제한이다. untracked `node_modules`의 파일까지 diff로 합성하려면 리뷰 때마다 그 폴더 전체를 읽어야 한다. 묶음 구성은 미리보기에 포함되지 않는다. LLM 묶음 결과는 실행해 봐야 알 수 있기 때문이다.
