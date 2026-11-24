scan 명령의 진입점과 파일 열거, 선별, 비용 추정을 다룬다.

## scan 명령 진입점

명령 정의와 실행은 [`cmd/opencodereview/scan_cmd.go`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/cmd/opencodereview/scan_cmd.go)에 있다. 별칭은 `ocr s`이고 위치 인자를 받지 않는다. 플래그는 [`registerScanFlags`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/cmd/opencodereview/shared_flags.go#L234-L257)가 등록한다.

| 플래그 | 기본값 | 의미 |
|---|---|---|
| `--path` | 저장소 전체 | 쉼표로 구분한 파일이나 디렉터리. 없는 경로는 오류 |
| `--exclude` | 없음 | 제외할 glob 목록 |
| `--concurrency` | 8 | 배치 안에서 동시에 실행하는 파일 수 |
| `--timeout` | 15 | 파일 하나의 하위 작업 시간 상한(분). 0이면 무제한 |
| `--max-tools` | 0 | 파일당 라운드 상한. 템플릿 값보다 클 때만 적용 |
| `--max-tokens` | 0 | 파일당 프롬프트 토큰 상한(`MAX_TOKENS`) 덮어쓰기 |
| `--max-tokens-budget` | 0 | 실행 전체 누적 토큰 예산. 0이면 무제한 |
| `--batch` | 템플릿 값 | `none`, `by-language`, `by-directory` |
| `--no-plan`, `--no-dedup`, `--no-summary` | 꺼짐 | plan, 중복 제거, 프로젝트 요약 단계 끄기 |
| `--preview`, `-p` | 꺼짐 | LLM 없이 대상 파일만 표시 |
| `--resume` | 없음 | 이전 스캔 세션 ID로 재개. `--preview`와 함께 쓸 수 없다 |
| `--background`, `-b` | 없음 | 요구사항이나 업무 맥락 문자열 |

### 코드 분석: 스캔 템플릿 적용

파일: [`cmd/opencodereview/scan_cmd.go` L131-L153](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/cmd/opencodereview/scan_cmd.go#L131-L153)

```go
	// scan owns its own template (scan_template.json) independent from the
	// diff-review template loaded by loadCommonContext above. Apply --max-tools
	// as an "only raise" override to the scan template's per-file budget.
	scanTpl, err := template.LoadScanDefault()
	if err != nil {
		return fmt.Errorf("load scan template: %w", err)
	}
	if err := scanTpl.Validate(); err != nil {
		return fmt.Errorf("invalid scan template: %w", err)
	}
	if opts.maxTools > scanTpl.MaxToolRequestTimes {
		scanTpl.MaxToolRequestTimes = opts.maxTools
	}
	if opts.batch != "" {
		// CLI override of BATCH_STRATEGY; validated downstream by parseBatchStrategy
		// (unknown values silently fall back to "none").
		scanTpl.BatchStrategy = opts.batch
	}
	// Token budget: --max-tokens-budget overrides the template value when set.
	budget := scanTpl.MaxTokensBudget
	if opts.maxTokensBudget > 0 {
		budget = int64(opts.maxTokensBudget)
	}
```

- `--max-tools`는 올리기만 한다. 60보다 작은 값을 주면 무시된다
- `--batch`의 오타는 오류가 되지 않는다. `validateScanOptions`는 이 값을 검사하지 않고, `parseBatchStrategy`가 모르는 값을 `none`으로 바꾼다. 주석도 "silently fall back"이라고 밝힌다
- 기본 스캔 템플릿에는 `MAX_TOKENS_BUDGET` 키가 없다. 플래그를 주지 않으면 예산은 0, 즉 무제한이다

이어서 명령은 세 가지를 고정한다. 파일 읽기는 항상 작업 트리 모드(`tool.ModeWorkspace`)이고([L199-L205](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/cmd/opencodereview/scan_cmd.go#L199-L205)), 도구 레지스트리는 `CodeAccess` 없이 만들며, MCP 클라이언트는 초기화하지 않는다. 템플릿의 `MAIN_TASK.tools`는 `task_done`, `code_comment`, `code_search`, `file_read`, `file_find` 다섯 개다.

> **주의**: `scan_template.json`의 각 작업에는 `"timeout": 180`, `"timeout": 90` 같은 값이 적혀 있다([`scan_template.json` L14](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/config/template/scan_template.json#L14) 등 여섯 곳). 그러나 이 JSON을 받는 [`LlmConversation`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/config/template/template.go#L379-L383) 구조체에는 `timeout` 필드가 없어 값이 읽히지 않는다. 실제 시간 제한은 `--timeout`(파일 하위 작업, 분 단위)과 LLM 요청 timeout(10장)이 정한다.

## 파일 열거: Provider

`scan.Provider`는 diff 대신 `model.ScanItem{Path, Content, IsBinary, LineCount}` 목록을 만든다. 파일 목록을 얻는 방법은 저장소 여부에 따라 둘이다.

| 조건 | 목록 방법 | `.gitignore` 처리 |
|---|---|---|
| `git rev-parse --git-dir` 성공 | `git ls-files -z`(추적)와 `git ls-files -z --others --exclude-standard`(미추적)의 합집합 | git 규칙 전체 (중첩, 전역, 부정 패턴) |
| git 저장소 아님 | `filepath.WalkDir` 직접 순회 | 루트 `.gitignore`와 내장 제외 디렉터리 목록만 |

`-z` 출력은 NUL로 구분하고 레코드를 trim하지 않는다. 주석은 앞뒤 공백도 유효한 파일 이름 바이트이므로, `" normal.go"`를 `"normal.go"`로 다듬으면 이후 `Lstat`이 없는 경로를 가리켜 파일이 오류 없이 빠진다고 설명한다([`provider.go` L285-L297](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/scan/provider.go#L285-L297)).

### 코드 분석: Enumerate의 파일별 검사

파일: [`internal/scan/provider.go` L126-L156](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/scan/provider.go#L126-L156)

```go
		if diff.IsPathExcluded(p.repoDir, rel, gitignorePatterns) {
			continue
		}
		full := filepath.Join(p.repoDir, rel)
		info, err := os.Lstat(full)
		// ... stat 오류는 경고 후 건너뜀
		if !info.Mode().IsRegular() {
			continue
		}
		if info.Size() > p.maxFileSizeBytes {
			fmt.Fprintf(os.Stderr, "[ocr] WARNING: skipping %s (%d bytes exceeds %d-byte scan limit; raise MaxTokens if the real concern is token budget, not memory)\n",
				rel, info.Size(), p.maxFileSizeBytes)
			continue
		}
		binary, err := isBinaryFile(full)
		// ... 오류 처리
		if binary {
			// Emit placeholder so preview can display [B], but do not
			// read the file body — saves memory on large binaries.
			out = append(out, model.ScanItem{
				Path:     rel,
				IsBinary: true,
			})
			continue
		}
```

| 검사 | 기준 | 결과 |
|---|---|---|
| 기본 제외 경로, `.gitignore` | `diff.IsPathExcluded` (04장) | 목록에서 빠짐 |
| 파일 종류 | `Lstat` 기준 일반 파일만. 심볼릭 링크는 따라가지 않음 | 목록에서 빠짐 |
| 크기 | `MAX_FILE_SIZE_BYTES` 2097152 (2MiB) | 경고 후 빠짐 |
| 바이너리 | 앞 8000바이트 안에 NUL 바이트가 있으면 바이너리 | 내용 없이 `IsBinary=true`로 남김 |
| 그 밖 | 내용 전체를 읽고 줄 수를 셈 | 리뷰 후보 |

- 크기 상한 2MiB는 메모리 보호용이다. 상수 주석은 실제 리뷰 가능 한도는 토큰 상한이고, 기본 `MAX_TOKENS` 58888에서 약 188KB라고 적었다([`provider.go` L32-L37](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/scan/provider.go#L32-L37))
- 바이너리를 목록에 남기는 이유는 `--preview`에서 제외 사유를 보여 주기 위해서다. 본문은 읽지 않는다
- 바이너리 판정은 git과 같은 "NUL 바이트 존재" 휴리스틱이다

### --path 선택자 해석

`--path`가 있으면 열거한 목록에서 해당 파일이나 그 아래 파일만 남긴다([`filterByPaths`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/scan/provider.go#L318-L351)). 규칙은 세 단계다.

1. 선택자를 그대로 비교한다. `src`는 `src`와 `src/...`에 맞고 `srcx.go`에는 맞지 않는다
2. 그대로 맞는 항목이 없으면 작업 트리에서 실제 경로를 찾는다. 심볼릭 링크를 따라가고, 대소문자를 구분하지 않는 파일 시스템(Windows, 기본 macOS)에서는 디스크에 적힌 철자로 바꾼다
3. 실제 경로도 없으면 `--path "x" does not exist in the working tree` 오류를 낸다. 경로가 있지만 대상 파일이 없으면(무시된 디렉터리 등) 오류 없이 빈 선택이 된다

`.`이나 `./`는 저장소 루트를 뜻하므로 다른 선택자와 함께 와도 전체 스캔이 된다([L73-L86](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/scan/provider.go#L73-L86)).

## 리뷰 대상 선별

열거한 파일은 `selectScanItems`에서 제외 사유를 하나씩 받는다. 이 함수는 실제 실행과 `--preview`가 함께 쓰므로 두 경로의 판단이 갈라지지 않는다.

### 코드 분석: whyExcluded와 토큰 상한

파일: [`internal/scan/agent.go` L432-L449](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/scan/agent.go#L432-L449)

```go
func (a *Agent) selectScanItems(items []model.ScanItem) []scanSelection {
	limit := llmloop.PromptTokenLimit(a.args.Template.MaxTokens)
	decisions := make([]scanSelection, 0, len(items))
	for _, it := range items {
		decision := scanSelection{
			item:   it,
			reason: a.whyExcluded(it),
		}
		if decision.reason == model.ExcludeNone && limit > 0 {
			decision.tokens = llm.CountTokens(it.Content)
			if decision.tokens > limit {
				decision.reason = model.ExcludeTooLarge
			}
		}
		decisions = append(decisions, decision)
	}
	return decisions
}
```

[`whyExcluded`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/scan/agent.go#L495-L522)는 다음 순서로 판정하고, 먼저 걸린 사유 하나만 남긴다.

| 순서 | 검사 | 제외 사유 |
|---|---|---|
| 1 | 바이너리 | `ExcludeBinary` |
| 2 | 내장 비밀 정보 경로 (자격 증명 파일 등) | `ExcludeSecret` |
| 3 | 사용자 제외 규칙 | `ExcludeUserRule` |
| 4 | 사용자 포함 규칙에 맞음 | 제외 없음 (확장자 허용 목록을 건너뜀) |
| 5 | 확장자가 허용 목록에 없음 | `ExcludeExtension` |
| 6 | 기본 제외 경로 | `ExcludeDefaultPath` |
| 7 | 내용 토큰 수가 `MAX_TOKENS`의 80% 초과 | `ExcludeTooLarge` |

- 비밀 정보 경로 검사가 사용자 규칙보다 앞선다. 주석은 어떤 포함 규칙도 자격 증명 경로를 받아들일 수 없고, 어떤 제외 규칙도 이를 다른 사유로 바꿀 수 없게 했다고 설명한다(이슈 #1240으로 표시). diff 리뷰의 파일 선별과 같은 순서다(04장)
- 토큰 상한은 규칙 검사 뒤에 둔다. 한 파일이 항상 같은 제외 사유를 받게 하기 위해서다
- 기본값으로 계산하면 `PromptTokenLimit(58888)`은 47110 토큰이다. 이 토큰 수는 tiktoken `cl100k_base`로 센 추정치다(07장)

선별 뒤 남은 파일이 없으면 세션을 마무리하고 빈 결과를 돌려준다([L353-L363](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/scan/agent.go#L353-L363)). 건너뛰는 경우에도 세션 종료 기록은 디스크에 남긴다.

`--preview`는 같은 열거와 선별만 하고 LLM을 부르지 않는다. [`Preview`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/scan/preview.go#L13-L22)는 `NewAgent`를 거치지 않는데, 주석은 `NewAgent`가 세션을 자동으로 만들어 마무리되지 않은 JSONL 파일을 남기기 때문이라고 설명한다.

## 비용 추정

대상이 정해지면 LLM을 부르기 전에 토큰 사용량을 대략 추정해 출력한다. 주석은 이 값이 청구 금액 수준의 정확도가 아니라 "order-of-magnitude warning"이라고 밝힌다.

파일: [`internal/scan/estimate.go` L44-L59](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/scan/estimate.go#L44-L59)

```go
func estimateFileTokens(it model.ScanItem, planEnabled func(model.ScanItem) bool) int64 {
	if it.IsBinary || it.Content == "" {
		return 0
	}
	fileTokens := int64(llm.CountTokens(it.Content))

	var total int64
	if planEnabled(it) {
		total += fileTokens + promptOverheadTokens // PLAN input
		total += 400                               // PLAN output (small JSON)
	}
	// MAIN_TASK: file content carried across rounds + per-round overhead.
	total += (fileTokens + promptOverheadTokens) * avgMainRoundsPerFile
	total += avgOutputTokensPerRound * avgMainRoundsPerFile
	return total
}
```

| 상수 | 값 | 의미 |
|---|---|---|
| `promptOverheadTokens` | 2000 | 요청마다 붙는 system 프롬프트, 템플릿, 도구 정의 |
| `avgMainRoundsPerFile` | 7 | 파일당 메인 라운드 수. 주석은 실제 저장소에서 약 6이 관찰되어 올림했다고 적었다 |
| `avgOutputTokensPerRound` | 700 | 라운드당 출력 토큰 |
| plan 출력 | 400 | 작은 JSON 응답 |
| 중복 제거, 요약 입력 | 코멘트당 120 | 파일당 코멘트 3개로 가정 |

파일 하나의 추정치는 대략 `(파일 토큰 + 2000) × 7 + 4900`이다. 메인 라운드마다 파일 내용 전체가 다시 들어간다고 보고 7을 곱한다. 07장에서 본 대로 루프는 매 라운드 대화 전체를 다시 보내므로 이 가정과 같은 방향이다. 다만 prompt cache로 실제 비용이 줄어드는 효과는 반영하지 않는다. 예산이 있고 추정치가 예산을 넘으면 "scan will stop partway" 경고를 출력한다(L368-L374). 이 경고는 실행을 막지 않는다.
