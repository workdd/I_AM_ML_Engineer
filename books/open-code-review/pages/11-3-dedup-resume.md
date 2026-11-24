배치별 중복 제거와 프로젝트 요약, 중단과 재개를 다룬다.

## 배치별 중복 제거

같은 배치(같은 언어나 같은 디렉터리)의 파일들은 비슷한 지적을 반복하기 쉽다. 배치가 끝나면 그 배치에서 나온 코멘트만 모아 `DEDUP_TASK`로 합친다.

| 설정 | 값 | 비고 |
|---|---|---|
| 실행 조건 | 템플릿에 `DEDUP_TASK` 정의, `--no-dedup` 아님 | |
| `DEDUP_MIN_COMMENTS` | 4 (템플릿) | 배치 코멘트가 이보다 적으면 건너뜀. 0 이하이면 코드 기본값 2 |
| 입력 | `[{id: "c-N", path, content, existing_code}]` JSON | 판단에 필요한 필드만 |
| 출력 | `{"groups": [{"members": [...], "merged_content": ...}]}` | |

### 코드 분석: applyDedupGroupsWithCheckpoints

LLM이 돌려준 그룹을 그대로 믿지 않고, 모든 코멘트가 정확히 한 번씩 그룹에 들어갔는지 확인한다.

파일: [`internal/scan/agent.go` L1105-L1145](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/scan/agent.go#L1105-L1145)

```go
	for _, g := range parsed.Groups {
		if len(g.Members) == 0 {
			return nil, nil, false
		}
		// ... 첫 멤버를 대표(canonical)로 선택, 멤버 id 검사
		canonical := originals[canonicalIdx]
		if len(g.Members) > 1 && g.MergedContent != "" {
			canonical.Content = g.MergedContent
		}
		out = append(out, canonical)
		if len(memberPaths) == 1 {
			checkpoints[canonical.Path] = append(checkpoints[canonical.Path], canonical)
			continue
		}
		for _, idx := range memberIndices {
			original := originals[idx]
			checkpoints[original.Path] = append(checkpoints[original.Path], original)
		}
	}

	if len(seen) != len(originals) {
		return nil, nil, false // some id missing
	}
	return out, checkpoints, true
```

- 빈 그룹, 모르는 id, 두 그룹에 들어간 id, 빠진 id 중 하나라도 있으면 결과 전체를 버리고 원래 코멘트를 유지한다. 주석은 이유를 "we refuse to silently drop comments we can't account for"로 적었다
- 그룹의 대표는 첫 멤버이고, `merged_content`가 있으면 내용만 바꾼다. 위치(`path`, `existing_code`)는 첫 멤버의 것을 쓴다
- 출력되는 코멘트와 재개용 체크포인트가 다르다. 여러 파일에 걸친 그룹은 출력에서 하나로 합치지만, 체크포인트에는 파일마다 원래 코멘트를 남긴다. 다음 재개에서 한 파일만 바뀌어 다시 리뷰되더라도 다른 파일의 발견이 사라지지 않게 하기 위해서다([L615-L618](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/scan/agent.go#L615-L618) 주석)

중복 제거는 배치 경계를 넘지 않는다. `.go` 배치와 `.ts` 배치에 같은 지적이 있어도 합치지 않는다. LLM 호출 오류나 JSON 형식 오류는 "keeping originals"를 출력하고 넘어간다.

## 프로젝트 요약

모든 배치가 끝나면 `PROJECT_SUMMARY_TASK`로 저장소 수준 요약을 한 번 만든다([`maybeRunProjectSummary`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/scan/agent.go#L901-L951)). 입력은 모든 코멘트를 ``- `path`: 내용`` 한 줄씩으로 만든 목록이고, 내용은 280바이트에서 UTF-8 문자 경계로 자른다. 프롬프트는 "Identify cross-cutting patterns and the most important findings; do NOT restate every individual comment."라고 지시하고, 결과는 Markdown 문서다.

| 단계 | 세션 기록 키 | 기록 작업 종류 |
|---|---|---|
| plan | 파일 경로 | `PlanTask` |
| main | 파일 경로 | `MainTask` |
| 배치 중복 제거 | `__scan_dedup_batch_N__` | `MemoryCompressionTask` |
| 프로젝트 요약 | `__scan_project_summary__` | `MemoryCompressionTask` |

중복 제거와 요약은 세션에 기록할 때 전용 작업 종류가 없어 메모리 압축 종류를 빌려 쓴다. 코드 주석이 "reuse existing task type; no scan-specific type to invent"라고 밝힌다. 세션 뷰어(13장)에서 이 두 기록은 압축 요청처럼 보인다. 요약은 실패해도 실행을 막지 않고 결과만 비운다.

## 중단과 재개

스캔은 파일 단위로 완료 체크포인트를 남기고, `--resume <session-id>`로 끝난 파일을 건너뛰고 다시 실행할 수 있다.

### 코드 분석: 파일 지문

파일: [`internal/scan/agent.go` L304-L307](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/scan/agent.go#L304-L307)

```go
func scanItemFingerprint(it model.ScanItem) string {
	sum := sha256.Sum256([]byte(session.ReviewModeFullScan + "\x00" + it.Path + "\x00" + it.Content))
	return fmt.Sprintf("%x", sum)
}
```

- 지문은 모드 이름, 경로, 내용의 SHA-256이다. 파일 내용이 한 바이트라도 바뀌면 지문이 달라져 다시 리뷰한다
- 모델, 프롬프트, 규칙은 지문에 들어가지 않는다. 다른 모델로 재개해도 끝난 파일은 재사용하고, 재개 정보에 이전 모델과 현재 모델을 함께 기록한다(L287-L293)

재개 조건은 명령 단계에서 확인한다([`scan_cmd.go` L262-L277](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/cmd/opencodereview/scan_cmd.go#L262-L277), [`session/resume.go` L300-L315](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/session/resume.go#L300-L315)).

| 확인 | 실패 메시지 요지 |
|---|---|
| 세션에 리뷰 모드 정보가 있음 | missing review mode metadata |
| 리뷰 모드가 `full_scan` | review mode does not match |
| 이전 세션의 `--path` 범위와 현재 범위가 같음 | scan path scope does not match |
| 완료된 파일이 하나 이상 | has no completed scan items |

재개 실행에서 `dispatchBatch`는 지문이 체크포인트에 있는 파일의 코멘트를 수집기에 그대로 넣고 LLM을 부르지 않는다(L693-L708). 재사용한 파일도 같은 배치의 중복 제거 입력에 들어간다. 배치 후처리 뒤 [`recordBatchCheckpoints`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/scan/agent.go#L619-L649)가 새로 끝난 파일은 완료로, 재사용한 파일은 원래 세션 ID와 함께 재사용으로 기록한다. 실행이 취소되어도 그때까지 끝난 파일의 체크포인트는 기록하고 돌아간다(L575-L581).

분배한 파일이 모두 실패하고 재사용한 파일도 없으면 "all N file scan(s) failed — check your LLM configuration and API key" 오류를 낸다([L602-L605](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/scan/agent.go#L602-L605)). 일부만 실패하면 오류 없이 결과를 돌려주고 실패는 `scan_subtask_error` 경고로 남는다. 스캔이 오류로 끝나면 명령은 `--resume <id>`로 다시 시도하라는 안내를 stderr에 출력한다.
