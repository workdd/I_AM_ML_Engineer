파일마다 리뷰 여부를 정하는 판정 순서와 확장자 허용 목록을 다룬다.

## 파일별 제외 판정

### 코드 분석: selectFiles

선별의 본체는 파일마다 제외 사유 하나를 정하는 순수 함수다.

파일: [internal/agent/selection.go L43-L63](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/agent/selection.go#L43-L63)

```go
func (a *Agent) selectFiles(diffs []model.Diff) []fileDecision {
	limit := llmloop.PromptTokenLimit(a.args.Template.MaxTokens)
	decisions := make([]fileDecision, 0, len(diffs))
	for _, d := range diffs {
		dec := fileDecision{Diff: d, Reason: a.whyExcluded(d)}
		switch {
		case dec.Reason != ExcludeNone:
		case d.IsDeleted:
			dec.Reason = ExcludeDeleted
		case limit > 0:
			// Counted only when a limit is configured, so the tokenizer pass is
			// skipped entirely for max_tokens=0.
			dec.DiffTokens = llm.CountTokens(d.Diff)
			if dec.DiffTokens > limit {
				dec.Reason = ExcludeTooLarge
			}
		}
		decisions = append(decisions, dec)
	}
	return decisions
}
```

[함수 주석](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/agent/selection.go#L33-L42)은 이 함수가 "순수하다(출력, 변경, git, LLM 접근 없음)"고 강조한다. 예산 소진, 재개 시 재사용, 프로바이더 실패처럼 실행해 봐야 아는 결과는 선별이 아니라 실행 결과로 따로 다룬다. 그래서 선별 결과는 입력 diff와 설정만으로 결정되고, 재개 동일성 계산(13장)도 이 함수를 그대로 다시 부를 수 있다.

판정 순서는 정적 검사, 삭제 여부, 크기 순이다. 크기 검사는 `MAX_TOKENS`가 0보다 클 때만 token을 센다.

크기 상한은 [`PromptTokenLimit`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/llmloop/compression.go#L23-L35)로 계산한다. `MAX_TOKENS`에 `tokenWarningThreshold` 0.80을 곱한 값이다. 템플릿 기본값 200000에서는 160000 token이다. diff 하나가 이보다 크면 그 파일은 prompt에 넣을 수 없으므로 리뷰 대상에서 빠진다. 같은 0.80은 07장의 대화 압축이 즉시 동기 압축을 시작하는 기준과 같은 상수다.

### 코드 분석: whyExcluded

정적 검사는 경로와 diff 헤더만 보고 사유를 정한다.

파일: [internal/agent/selection.go L78-L101](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/agent/selection.go#L78-L101)

```go
	// Ahead of both user rules: no include glob can admit a credential path,
	// and no user exclude can claim it under a different reason.
	if allowedext.IsSecretPath(d.OldPath) || allowedext.IsSecretPath(d.NewPath) {
		return ExcludeSecret
	}

	if f != nil && f.IsUserExcluded(path) {
		return ExcludeUserRule
	}

	if f != nil && f.HasInclude() && f.IsUserIncluded(path) {
		return ExcludeNone
	}

	ext := a.extFromPath(path)
	if ext != "" && !allowedext.IsAllowedExt(ext) {
		return ExcludeExtension
	}

	if allowedext.IsExcludedPath(path) {
		return ExcludeDefaultPath
	}

	return ExcludeNone
```

이 인용 앞에 바이너리 검사(`d.IsBinary`)가 있다. 전체 판정 순서와 각 사유를 정리하면 다음과 같다. 사유 문자열은 [model/preview.go](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/model/preview.go#L10-L26)에 정의되어 미리보기 JSON과 manifest에 그대로 쓰인다.

| 순서 | 사유 문자열 | 조건 | include로 되살림 |
|---|---|---|---|
| 0 | `provider_directory` | 루트의 `vendor/`, `node_modules/` 등 11개 접두사 (03장, diff 단계) | 불가 |
| 1 | `binary` | diff에 `Binary files ... differ` 표식 | 불가 |
| 2 | `secret_exclude` | 이전 경로나 새 경로가 내장 비밀 경로와 일치 | 불가 |
| 3 | `user_exclude` | 규칙 파일 `exclude` 또는 `--exclude` 패턴과 일치 | 해당 없음 |
| 4 | (선택) | include가 정의되어 있고 그 패턴과 일치 | 이 단계가 되살림 |
| 5 | `unsupported_ext` | 확장자가 있는데 허용 목록에 없음 | 가능 |
| 6 | `default_path` | 내장 기본 제외 패턴과 일치 | 가능 |
| 7 | `deleted` | 삭제된 파일 | 불가 |
| 8 | `too_large` | diff token 수가 `MAX_TOKENS`의 80% 초과 | 불가 |

- 비밀 경로 검사가 사용자 규칙보다 앞에 있다. 주석대로 어떤 include 패턴도 비밀 경로를 되살리지 못하고, 사용자 exclude가 그 파일을 다른 사유로 가져가지도 못한다
- 비밀 경로는 이전 경로와 새 경로를 모두 검사한다. `.env`를 `config.txt`로 이름을 바꾼 변경도, 반대로 일반 파일을 `.env`로 바꾼 변경도 막힌다
- 사용자 exclude가 include보다 앞이므로 두 패턴에 모두 걸리면 제외된다
- include는 허용 목록이 아니다. include에 걸리지 않은 파일도 5, 6단계를 통과하면 리뷰된다. 저장소의 [리뷰 규칙 문서](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/pages/src/content/docs/en/review-rules.md#L51-L56)도 "It is not a whitelist"라고 같은 내용을 적었다

### 선택 집합과 유지 집합

[`fileDecision`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/agent/selection.go#L13-L31)은 두 가지 판정 메서드를 가진다. `selected()`는 사유가 없는 파일, 즉 실제로 리뷰할 파일이다. `retained()`는 선택된 파일에 삭제된 파일을 더한 집합이다. 삭제된 파일은 리뷰할 새 내용이 없지만, 리뷰 prompt에 들어가는 "다른 변경 파일" 목록에는 남아야 한다. 어떤 함수가 지워졌는지 알아야 그 함수를 호출하던 코드를 제대로 리뷰할 수 있기 때문이다.

선택된 파일이 0개면 리뷰를 생략한다([agent.go L342-L378](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/agent/agent.go#L342-L378)). 텔레메트리 이벤트 `review.skipped`의 사유는 크기 초과 파일이 있으면 `too_large`, 삭제 파일만 남았으면 `deleted`, 그 밖에는 `no_supported_files`다. 이 경우 manifest 상태는 skipped이고 종료 코드는 0이다(02장).

### 코드 분석: 확장자 추출

파일: [internal/agent/agent.go L2084-L2094](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/agent/agent.go#L2084-L2094)

```go
func (a *Agent) extFromPath(path string) string {
	basename := path
	if idx := strings.LastIndex(path, "/"); idx >= 0 {
		basename = path[idx+1:]
	}
	dot := strings.LastIndex(basename, ".")
	if dot <= 0 {
		return ""
	}
	return strings.ToLower(basename[dot:])
}
```

`dot <= 0` 조건 때문에 확장자가 없는 파일(`Makefile`, `Dockerfile`, `LICENSE`)과 점으로 시작하는 파일(`.gitignore`, `.dockerignore`)은 빈 문자열을 돌려받는다. `whyExcluded`는 `ext != ""`일 때만 허용 목록을 검사하므로, 이 파일들은 확장자 검사를 통과한다. 허용 목록에 없는 `.md`나 `.txt`는 빠지지만 확장자 없는 파일은 기본 제외 패턴에만 걸리지 않으면 리뷰된다. 확장자는 소문자로 바꾸므로 `.GO`와 `.go`는 같다.

## 확장자 허용 목록과 기본 제외 패턴

### 허용 목록

허용 목록 [supported_file_types.json](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/config/allowlist/supported_file_types.json)은 확장자 144개를 담은 JSON 배열이다. `go:embed`로 실행 파일에 포함되고, 처음 쓰일 때 [`initMap`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/config/allowlist/allowed_ext.go#L55-L64)이 소문자 키의 맵으로 바꾼다.

| 분류 | 확장자 예 |
|---|---|
| JVM, .NET | `.java`, `.kt`, `.scala`, `.groovy`, `.cs`, `.vb`, `.fs` |
| 스크립트 | `.py`, `.rb`, `.php`, `.pl`, `.lua`, `.r`, `.sh`, `.ps1` |
| JavaScript 계열 | `.js`, `.jsx`, `.ts`, `.tsx`, `.mjs`, `.vue`, `.svelte`, `.astro` |
| 시스템 언어 | `.c`, `.cpp`, `.h`, `.go`, `.rs`, `.zig`, `.swift`, `.m`, `.cu` |
| 템플릿 | `.html`, `.ftl`, `.hbs`, `.jinja2`, `.erb`, `.ejs`, `.liquid` |
| 설정, 데이터 | `.json`, `.yaml`, `.toml`, `.ini`, `.xml`, `.properties`, `.env`, `.tf`, `.hcl` |
| 스키마, IDL | `.proto`, `.graphql`, `.prisma`, `.thrift`, `.capnp`, `.sql` |
| 하드웨어, 스마트 계약 | `.v`, `.sv`, `.vhd`, `.sol`, `.vy`, `.move` |

목록에 없는 대표적인 확장자는 `.md`, `.txt`, `.csv`, `.lock`이다. 문서와 데이터 파일은 리뷰 대상이 아니라는 기본값이다. `.env` 항목은 이름과 확장자 해석이 겹쳐 직관과 다르게 동작한다. `.env` 파일 자체는 점으로 시작해 확장자가 빈 문자열이고, 그보다 앞선 비밀 경로 검사에서 빠진다. 비밀 경로 예외인 `.env.example`은 `extFromPath`가 마지막 점 이후인 `.example`을 확장자로 보므로 허용 목록에 없어 `unsupported_ext`로 빠진다. 허용 목록의 `.env`가 실제로 통과시키는 파일은 `prod.env`처럼 이름 끝이 `.env`인 파일이다. 이 파일은 `.env`로 시작하지 않아 비밀 경로 검사에도 걸리지 않고 리뷰 대상이 된다.

### 기본 제외 패턴

[default_exclude_patterns.json](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/config/allowlist/default_exclude_patterns.json)은 glob 패턴 115개를 담는다. 파일 안에서 빈 줄로 두 무리가 나뉜다.

| 무리 | 줄 범위 | 예 |
|---|---|---|
| 테스트, 생성 코드 | [L2-L75](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/config/allowlist/default_exclude_patterns.json#L2-L75) | `**/*_test.go`, `**/*.spec.{js,jsx,ts,tsx,mts,cts}`, `**/__tests__/**`, `**/testdata/**`, `**/fixtures/**`, `**/*.pb.go`, `**/*.generated.*` |
| 의존성, 빌드 산출물 | [L77-L122](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/config/allowlist/default_exclude_patterns.json#L77-L122) | `**/node_modules/**`, `**/package-lock.json`, `**/*.min.{js,css}`, `**/dist/**`, `**/vendor/**`, `**/target/**`, `**/__pycache__/**`, `**/.venv/**` |

테스트 파일이 기본으로 빠진다는 점이 실사용에서 가장 눈에 띄는 기본값이다. 테스트 코드를 리뷰하려면 규칙 파일의 `include`에 해당 패턴을 넣어야 한다(05장).

[`IsExcludedPath`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/config/allowlist/allowed_ext.go#L91-L100)는 경로와 패턴을 모두 소문자로 바꾼 뒤 doublestar 라이브러리로 비교한다. 그래서 `**/tests/**/*.R` 패턴은 `.r` 파일에도 걸린다. [패키지 주석](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/config/allowlist/allowed_ext.go#L8-L27)은 `*`가 경로 구분자를 넘지 않고 `**`만 넘는다는 점을 예로 설명한다. `*_test.go`는 루트의 파일에만, `**/*_test.go`는 모든 깊이에 일치한다.

> **주의**: 저장소의 [리뷰 규칙 문서](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/pages/src/content/docs/en/review-rules.md#L202-L203)는 JavaScript 테스트 패턴을 `**/*.test.{js,jsx,ts,tsx}`, `**/*.spec.{js,jsx,ts,tsx}`로 적었다. 코드의 JSON은 두 패턴 모두 `mts`, `cts`까지 포함한다. `.mts`, `.cts` 테스트 파일도 기본으로 제외된다.

### 두 단계 디렉터리 제외의 차이

`vendor`, `node_modules`, `target`은 03장의 내장 제외 디렉터리와 이 장의 기본 제외 패턴 양쪽에 있다. 두 장치는 범위와 되살림 가능 여부가 다르다.

| 구분 | 03장 내장 제외 디렉터리 | 기본 제외 패턴 |
|---|---|---|
| 정의 위치 | `providerDirIgnoreDirs` (Go 코드) | `default_exclude_patterns.json` |
| 비교 방식 | 루트 기준 접두사 | `**/` glob, 모든 깊이 |
| 적용 시점 | diff 수집 직후 | `whyExcluded` 6단계 |
| 사유 문자열 | `provider_directory` | `default_path` |
| include로 되살림 | 불가 | 가능 |
| workspace untracked 파일 | 목록에서 미리 제거 | 판정 후 제외 |

루트의 `vendor/`는 어떤 설정으로도 리뷰할 수 없고, `services/api/vendor/`는 include로 리뷰할 수 있다. 미리보기는 내장 제외 디렉터리에 걸린 파일을 접두사별로 한 줄에 모아 보여 준다. 수천 개일 수 있고 어떤 규칙으로도 되살릴 수 없기 때문이다.
