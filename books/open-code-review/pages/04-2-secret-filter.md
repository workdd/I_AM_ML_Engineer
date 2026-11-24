비밀 경로 차단과 사용자 include, exclude 설정을 다룬다.

## 비밀 경로 차단

### 코드 분석: IsSecretPath

비밀 경로는 기본 제외 패턴과 별도 파일, 별도 함수로 관리된다. [파일 머리 주석](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/config/allowlist/secret_path.go#L16-L21)은 그 이유를 "기본 제외 목록은 include가 되살릴 수 있는 리뷰 잡음이지만, 비밀 경로는 리뷰 범위에 아예 들어가면 안 된다"고 적었다.

파일: [internal/config/allowlist/secret_path.go L51-L72](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/config/allowlist/secret_path.go#L51-L72)

```go
	if isSecretEnvPath(lowerPath) {
		return true
	}

	for _, pattern := range secretPatterns {
		if matched, _ := doublestar.Match(pattern, lowerPath); matched {
			return true
		}
	}
	return false
}

func isSecretEnvPath(lowerPath string) bool {
	base := path.Base(lowerPath)

	switch base {
	case ".env.example", ".env.sample", ".env.template":
		return false
	}

	return base == ".env" || strings.HasPrefix(base, ".env.")
}
```

`.env` 계열은 예외 템플릿이 있어서 glob 대신 코드로 처리한다. `.env`, `.env.local`, `.env.production`은 모두 막히고, 팀이 공유하려고 commit하는 템플릿 세 가지만 통과한다. 나머지 패턴은 [default_secret_patterns.json](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/config/allowlist/default_secret_patterns.json#L1-L12)의 10개다.

| 종류 | 패턴 |
|---|---|
| SSH | `**/.ssh/**`, `**/id_rsa`, `**/id_dsa`, `**/id_ecdsa`, `**/id_ed25519` |
| 원격 인증 | `**/.netrc`, `**/_netrc` |
| 패키지 레지스트리 | `**/.npmrc`, `**/.pypirc`, `**/.dockercfg` |
| 환경 변수 파일 | `.env`, `.env.*` (예외: `.example`, `.sample`, `.template`) |

[`IsSecretPath` 주석](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/config/allowlist/secret_path.go#L40-L46)은 이 검사가 "경로만 보고 파일 내용은 보지 않는다"고 밝힌다. 소스 코드 안에 하드코딩된 키는 이 단계에서 걸리지 않고, 오히려 리뷰 에이전트가 지적해야 할 대상이다. 같은 함수를 12장의 분석 작업과 11장의 전체 스캔도 쓴다. 분석 작업은 여기에 `.git-credentials`와 `.docker/config.json`을 추가로 막는다([behaviortask/code.go L133](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/behaviortask/code.go#L133-L133)). 리뷰 선별의 목록에는 이 두 경로가 없다.

## 사용자 include와 exclude

사용자 규칙은 두 곳에서 온다. 규칙 JSON 파일(`--rule` 또는 저장소의 규칙 파일, 05장)의 `include`, `exclude` 배열과 CLI의 `--exclude` 플래그다. [`applyCLIExcludes`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/cmd/opencodereview/shared.go#L333-L341)는 `--exclude` 패턴을 규칙 파일의 `Exclude` 뒤에 덧붙인다. 그러므로 두 출처의 exclude는 판정에서 구분되지 않고 둘 다 `user_exclude`가 된다. CLI에는 include 플래그가 없다.

패턴 비교는 [`FileFilter.IsUserExcluded`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/config/rules/system_rules.go#L231-L244)와 `IsUserIncluded`가 맡는다. 중괄호 확장(`{a,b}`)을 먼저 펼치고, 경로와 패턴을 소문자로 바꿔 doublestar로 비교한다. 대소문자를 구분하지 않는 비교라서 `**/Generated/**`와 `**/generated/**`는 같은 패턴이다.

| 원하는 결과 | 설정 |
|---|---|
| 테스트 파일도 리뷰 | `include`에 `**/*_test.go` 같은 패턴 추가 |
| 허용 목록 밖 확장자 리뷰 | `include`에 `**/*.md` 같은 패턴 추가 |
| 특정 폴더 제외 | `exclude` 또는 `--exclude '**/generated/**'` |
| 루트 `vendor/` 리뷰 | 불가 (내장 제외 디렉터리) |
| `.env.local` 리뷰 | 불가 (비밀 경로) |
