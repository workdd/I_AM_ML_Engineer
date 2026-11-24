공통 플래그의 정의와 검증, 사용자 입력 오류의 안내 방식을 다룬다.

## 공통 플래그와 옵션 검증

### 플래그 등록 함수

`review`, `scan`, `delegate`는 비슷한 플래그를 공유한다. 중복 선언을 피하려고 [shared_flags.go](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/cmd/opencodereview/shared_flags.go#L15-L73)가 `addRepoFlag`, `addDiffFlags`, `addOutputFlags` 같은 작은 등록 함수를 모아 두고, 명령별 `register*Flags`가 이를 조합한다.

| 플래그 | 짧은 이름 | 기본값 | 의미 | review | scan | delegate |
|---|---|---|---|---|---|---|
| `--repo` | 없음 | 현재 폴더 | git 저장소 루트 | 사용 | 사용 | 사용 |
| `--rule` | 없음 | 없음 | 시스템 리뷰 규칙 JSON 경로 | 사용 | 사용 | 사용 |
| `--from`, `--to` | 없음 | 없음 | 비교 시작 ref, 끝 ref | 사용 | 없음 | 사용 |
| `--commit` | `-c` | 없음 | 단일 commit 리뷰 | 사용 | 없음 | 사용 |
| `--exclude` | 없음 | 없음 | 쉼표로 구분한 gitignore 형식 제외 패턴 | 사용 | 사용 | 사용 |
| `--format` | `-f` | `text` | `text`, `json`, `sarif` | 사용 | 사용 | `text`, `json`만 |
| `--audience` | 없음 | `human` | `human`(진행 표시) 또는 `agent`(요약만) | 사용 | 사용 | 없음 |
| `--output` | `-o` | stdout | 결과를 UTF-8 파일로 저장 | 사용 | 사용 | 없음 |
| `--concurrency` | 없음 | `8` | 동시 하위 작업 수 | 사용 | 사용 | 없음 |
| `--timeout` | 없음 | `15` | 하위 작업 예산(분), 0은 무제한 | 사용 | 사용 | 없음 |
| `--max-tools` | 없음 | `0` | 하위 작업당 도구 호출 라운드 상한 | 사용 | 사용 | 없음 |
| `--max-git-procs` | 없음 | `16` | 동시 git 하위 프로세스 수 | 사용 | 사용 | 사용 |
| `--max-tokens` | 없음 | `0` | 묶음당 prompt token 상한 | 사용 | 사용 | 없음 |
| `--max-tokens-budget` | 없음 | `0` | 실행 전체 token 사용 상한 | 사용 | 사용 | 없음 |
| `--background` | `-b` | 없음 | 요구사항, 업무 배경 문자열 | 사용 | 사용 | 사용 |
| `--background-file` | `-B` | 없음 | 배경으로 쓸 Markdown 파일 | 사용 | 없음 | 사용 |
| `--provider`, `--model` | 없음 | 설정값 | 이번 실행에만 쓰는 프로바이더, 모델 | 사용 | 사용 | 없음 |
| `--preview` | `-p` | `false` | LLM 없이 리뷰 대상 파일만 표시 | 사용 | 사용 | 하위 명령 |
| `--resume` | 없음 | 없음 | 이전 세션 ID에서 재개 | 사용 | 사용 | 없음 |

`review`에만 있는 플래그는 [registerReviewFlags](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/cmd/opencodereview/shared_flags.go#L212-L231)에서 추가된다. `--effort`(low, medium, high), `--no-filter`(9장의 LLM 후처리 필터 생략), `--analysis`와 `--analysis-max-tokens-budget`(12장)이다. `scan`에만 있는 `--path`, `--no-plan`, `--no-dedup`, `--no-summary`, `--batch`는 11장에서 다룬다.

같은 이름이라도 명령마다 의미가 다른 플래그가 있다. `--max-tokens`의 도움말은 `review`에서 "per-group prompt token ceiling", `scan`에서 "per-file prompt token ceiling"이다. `--max-tools`도 `review`는 최솟값 50을 강제하지만 `scan`은 "template default보다 클 때만 적용"으로 다르게 동작한다.


### 코드 분석: validateReviewOptions

`review` 실행의 첫 줄은 플래그 조합을 검증하는 함수다. 앞부분이 비교 모드를 정규화하고 서로 배타적인 조합을 막는다.

파일: [cmd/opencodereview/shared_flags.go L122-L145](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/cmd/opencodereview/shared_flags.go#L122-L145)

```go
func normalizeDiffTarget(from, to, commit string) string {
	if from != "" && to == "" && commit == "" {
		return "HEAD"
	}
	return to
}

func validateReviewOptions(opts *reviewOptions) error {
	opts.to = normalizeDiffTarget(opts.from, opts.to, opts.commit)
	if err := validateDiffMode(opts.from, opts.to, opts.commit); err != nil {
		return err
	}
	if opts.preview && opts.resume != "" {
		return fmt.Errorf("--preview and --resume cannot be used together")
	}
	if err := validateAudience(opts.audience); err != nil {
		return err
	}
	normalizedFormat, err := validateOutputFormat(opts.outputFormat)
	if err != nil {
		return err
	}
	opts.outputFormat = normalizedFormat
	if err := validateAnalysisOptions(*opts); err != nil {
```

- `--from`만 주면 `--to`가 `HEAD`로 채워진다. README의 "Current branch" 예시(`ocr review --from main`)가 이 한 줄로 구현된다
- 정규화가 검증보다 먼저 일어나므로 [`validateDiffMode`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/cmd/opencodereview/shared_flags.go#L83-L101)의 `--to is required when --from is specified` 오류는 `review`와 `delegate`에서 실제로 나올 수 없다. 남는 오류는 `--to`만 준 경우와 `--from/--to`를 `--commit`과 함께 준 경우다
- `--format`은 소문자로 바꾸고 공백을 지운 뒤 비교한다. `--format JSON`도 받아들인다

함수 뒷부분([L148-L170](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/cmd/opencodereview/shared_flags.go#L148-L170))은 정수 플래그를 다룬다.

| 검사 | 음수 | 0 | 양수 |
|---|---|---|---|
| `--max-tools` | 오류 | 템플릿 기본값(100) | 1~49는 경고 후 50으로 올림 |
| `--max-git-procs` | 오류 | 기본값 16 | 그대로 사용 |
| `--max-tokens` | 오류 | 설정값 또는 템플릿 기본값 | 그대로 사용 |
| `--max-tokens-budget` | 오류 | 무제한 | 그대로 사용 |
| `--effort` | 해당 없음 | 빈 문자열은 설정값 또는 `medium` | `low`, `medium`, `high` 외에는 오류 |

`--max-tools`의 최솟값 50은 `const minMaxTools = 50`로 함수 안에 박혀 있다. 템플릿 기본값 100은 [task_template.json L40](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/config/template/task_template.json#L40-L40)의 `MAX_TOOL_REQUEST_TIMES`다.

### 설정값 우선순위

여러 값이 CLI 플래그, 저장된 설정 파일, 내장 템플릿 세 곳에서 정해진다. 우선순위는 함수마다 같은 형태다.

| 값 | 결정 함수 | 우선순위 |
|---|---|---|
| prompt token 상한 | [`resolveMaxTokens`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/cmd/opencodereview/shared.go#L52-L66) | `--max-tokens` > 설정 `max_tokens` > 템플릿 `MAX_TOKENS`(200000) |
| 리뷰 노력 수준 | [`resolveEffort`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/cmd/opencodereview/shared.go#L70-L78) | `--effort` > 설정 `effort` > `medium` |
| 도구 호출 라운드 | [`resolveReviewMaxTools`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/cmd/opencodereview/review_cmd.go#L383-L388) | `--max-tools`(양수일 때) > 템플릿 기본값 |
| 프로바이더, 모델 | `loadLLMRuntime` | `--provider`, `--model` > 설정, 환경 변수 (10장) |

노력 수준은 리뷰 라운드 수로 바뀐다. [`effortPresets`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/config/template/effort.go#L34-L38)는 `low`=1, `medium`=2, `high`=3 라운드를 정한다. `--timeout`은 라운드 수만큼 곱해진다. [agent.go L682](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/agent/agent.go#L682-L682)에서 하위 작업 제한 시간을 `ConcurrentTaskTimeout * 분 * ReviewRounds()`로 계산하므로, 기본 15분은 `medium`에서 실제로 30분이 된다. CLI 문서도 "15/30/45 min for low/medium/high"로 같은 내용을 적었다.

## 사용자 입력 오류 처리

### 코드 분석: 플래그 오타 제안

알 수 없는 플래그를 받으면 cobra가 만든 오류에 비슷한 플래그 이름을 붙인다. 루트 명령에 `SetFlagErrorFunc(flagErrorWithSuggestion)`을 한 번 등록하면 cobra가 부모 명령의 처리기를 상속하므로 모든 하위 명령에 적용된다.

파일: [cmd/opencodereview/flag_suggest.go L39-L59](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/cmd/opencodereview/flag_suggest.go#L39-L59)

```go
	var best string
	bestDist := 3 // max edit distance to consider
	cmd.Flags().VisitAll(func(f *pflag.Flag) {
		d := levenshtein(unknown, f.Name)
		if d < bestDist {
			bestDist = d
			best = f.Name
		}
	})
	cmd.InheritedFlags().VisitAll(func(f *pflag.Flag) {
		d := levenshtein(unknown, f.Name)
		if d < bestDist {
			bestDist = d
			best = f.Name
		}
	})

	if best != "" {
		return fmt.Sprintf("\n\nDid you mean this?\n\t--%s", best)
	}
	return ""
```

편집 거리가 3 미만, 즉 2 이하인 플래그 중 가장 가까운 하나만 제안한다. `--format` 대신 `--fromat`을 치면 `Did you mean this? --format`이 붙는다. 명령 자신의 플래그와 상속 플래그(`--color`)를 모두 후보로 본다. 편집 거리 계산은 같은 파일의 [`levenshtein`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/cmd/opencodereview/flag_suggest.go#L62-L88)이 두 행만 쓰는 동적 계획법으로 구현한다.

### 코드 분석: 인자 개수 오류 메시지

`SilenceUsage` 때문에 cobra 기본 메시지("accepts 2 arg(s), received 1")만으로는 사용자가 무엇을 넣어야 하는지 알 수 없다. [arg_errors.go](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/cmd/opencodereview/arg_errors.go#L14-L19) 주석은 그래서 "사용자가 보는 모든 것이 오류 안에 담겨야 한다"고 적었다.

파일: [cmd/opencodereview/arg_errors.go L57-L78](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/cmd/opencodereview/arg_errors.go#L57-L78)

```go
func argCountError(cmd *cobra.Command, requirement string) error {
	var b strings.Builder
	b.Grow(256)

	path := cmd.CommandPath()
	fmt.Fprintf(&b, "%q %s", path, requirement)
	if sig := positionalSignature(cmd); sig != "" {
		fmt.Fprintf(&b, " (%s)", sig)
	}

	if len(cmd.ValidArgs) > 0 {
		b.WriteString("\n\nValid values: ")
		b.WriteString(strings.Join(validArgNames(cmd.ValidArgs), ", "))
	}

	b.WriteString("\n\nUsage:\n  ")
	b.WriteString(cmd.UseLine())

	if example := strings.TrimRight(cmd.Example, "\n"); example != "" {
		b.WriteString("\n\nExample:\n")
		b.WriteString(example)
	}
```

오류 문구를 명령이 이미 선언한 메타데이터에서 조립한다. `Use` 문자열의 `<key> <value>` 같은 위치 인자 표기, `ValidArgs`, `Example`을 그대로 읽으므로 도움말과 오류 메시지가 서로 어긋날 수 없다. `[flags]`나 `[bash|zsh]`처럼 대괄호로 싼 부분은 사용자가 입력하는 자리 표시가 아니라서 [`positionalSignature`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/cmd/opencodereview/arg_errors.go#L92-L105)가 버린다. `exactArgs`, `minimumArgs`, `maximumArgs` 세 래퍼가 cobra의 같은 이름 검증기를 대신한다.
