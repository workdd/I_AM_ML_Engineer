내장 시스템 규칙 표와 규칙 문서, 확장자만으로 언어를 정할 수 없는 파일의 판별을 다룬다.

## 내장 시스템 규칙 표

내장 규칙은 [`system_rules.json`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/config/rules/system_rules.json#L1-L79) 하나에 모여 있다. `default_rule` 하나와 `path_rule_map` 객체 하나로 구성되고, `path_rule_map`의 키는 glob 패턴, 값은 `rule_docs/` 아래 Markdown 파일 이름이다. 패턴은 74개, 규칙 문서는 70개다. 문서 수가 더 적은 이유는 `cuda_hip.md`, `ruby.md`, `starlark.md`, `ocaml.md`처럼 한 문서를 여러 패턴이 공유하기 때문이다.

JSON 객체는 원래 키 순서를 보장하지 않는다. 그런데 이 표는 "먼저 일치한 패턴이 이긴다"는 규칙으로 동작하므로 순서가 의미를 가진다. 그래서 `SystemRule`은 표준 `map` 대신 순서를 지키는 슬라이스로 패턴을 읽는다.

### 코드 분석: SystemRule.UnmarshalJSON

`internal/config/rules/system_rules.go`, [system_rules.go L69-L86](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/config/rules/system_rules.go#L69-L86)

키 순서를 지키기 위해 `json.Decoder`로 토큰을 하나씩 읽는 부분이다.

```go
	for dec.More() {
		// Read key
		keyTok, err := dec.Token()
		if err != nil {
			return fmt.Errorf("read path_rule_map key: %w", err)
		}
		key, ok := keyTok.(string)
		if !ok {
			return fmt.Errorf("expected string key in path_rule_map, got %T", keyTok)
		}
		// Read value
		var value string
		if err := dec.Decode(&value); err != nil {
			return fmt.Errorf("read path_rule_map value for %q: %w", key, err)
		}
		r.PathRules = append(r.PathRules, PathRule{Pattern: key, Rule: value})
	}
	return nil
}
```

`PathRules`는 `[]PathRule`이고 선언 순서대로 쌓인다. 이어서 [`LoadDefault`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/config/rules/system_rules.go#L92-L115)가 `go:embed`로 묶인 `rule_docs/*`에서 파일 이름을 실제 본문으로 바꾼다. 따라서 로드가 끝난 `SystemRule`의 `Rule` 필드에는 파일 이름이 아니라 규칙 본문 전체가 들어 있다. 파일이 없으면 로드 자체가 실패하므로, 바이너리에 포함된 표와 문서가 어긋난 채로 실행되는 경우는 없다.

### 코드 분석: SystemRule.resolveDetail

`internal/config/rules/system_rules.go`, [system_rules.go L166-L177](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/config/rules/system_rules.go#L166-L177)

경로 하나에 규칙을 고르는 핵심 반복문이다.

```go
func (r *SystemRule) resolveDetail(path string) RuleDetail {
	lowerPath := strings.ToLower(path)
	for _, pr := range r.PathRules {
		expanded := expandBraces(pr.Pattern)
		for _, p := range expanded {
			if matched, _ := doublestar.Match(strings.ToLower(p), lowerPath); matched {
				return RuleDetail{Rule: pr.Rule, Source: "system", Pattern: pr.Pattern}
			}
		}
	}
	return RuleDetail{Rule: r.DefaultRule, Source: "system", Pattern: "default"}
}
```

- 경로와 패턴을 모두 소문자로 바꾼 뒤 `doublestar.Match`로 비교한다. 그래서 `**/*.R` 패턴은 `foo.r`에도 일치한다
- [`expandBraces`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/config/rules/system_rules.go#L182-L203)는 첫 번째 `{...}` 묶음만 펼친다. 중괄호가 두 묶음인 패턴은 두 번째 묶음을 그대로 `doublestar`에 넘긴다. 현재 내장 표에는 중괄호가 두 묶음인 패턴이 없다
- 아무 패턴도 일치하지 않으면 `Pattern: "default"`와 `default.md` 본문을 돌려준다. 시스템 계층은 항상 무언가를 돌려주므로 규칙이 비는 파일은 없다

### 순서가 결과를 바꾸는 패턴

first match 규칙 때문에 표의 앞뒤 관계가 실제 동작을 정한다.

| 앞에 오는 패턴 | 뒤에 오는 패턴 | 결과 |
|---|---|---|
| `.github/workflows/**/*.{yaml,yml}` (L14) | `.github/**/*.{yaml,yml}` (L15) | 워크플로 YAML은 `github_workflows.md`, 나머지 `.github` YAML은 `github_config.md` |
| `.github/**/*.{yaml,yml}` (L15) | `**/*.{yaml,yml}` (L16) | 일반 YAML은 마지막의 `yaml.md` |
| `**/*{mapper,dao}*.xml` (L5) | (다른 XML 패턴 없음) | MyBatis 류 매퍼만 전용 규칙, 나머지 XML은 `default.md` |
| `**/package.json` (L8) | `**/*.{json,json5,jsonc}` (L13) | `package.json`은 의존성 규칙, 다른 JSON은 키 철자 검사 |
| `**/*.m` (L69) | `**/*.mm` (L70) | `.m`은 `matlab.md`가 기본, `.mm`은 `objc.md` |

[`rule_docs_table_test.go`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/config/rules/rule_docs_table_test.go#L206-L215)는 네 언어 문서 사이트의 `review-rules.md` 표가 이 JSON과 같은 순서로 패턴을 나열하는지 검사한다. 문서와 코드의 순서가 어긋나면 테스트가 실패하므로, 이 표에 한해서는 공식 문서를 그대로 믿어도 된다.

## 규칙 문서의 구성

`rule_docs/`의 70개 문서는 모두 영어 Markdown이고 합계 3,577줄이다. 길이 편차가 크다. 가장 짧은 `pom_xml.md`, `json.md`, `build_gradle.md`는 한 줄이고, 가장 긴 `starlark.md`는 242줄, `nushell.md`는 219줄이다.

| 문서 | 줄 수 | 구성 방식 |
|---|---|---|
| [`default.md`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/config/rules/rule_docs/default.md#L1-L22) | 22 | Correctness, Security, Performance, Maintainability, Test Coverage 다섯 범주의 질문 목록 |
| [`java.md`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/config/rules/rule_docs/java.md#L1-L10) | 41 | 철자 오류, dead code, 논리 오류, 성능, 스레드 안전성 항목 |
| [`go.md`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/config/rules/rule_docs/go.md#L1-L4) | 69 | 원칙 문단 뒤에 오류, nil, context, 채널, 타이머, 경계값, 보안, 테스트 범주 |
| [`json.md`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/config/rules/rule_docs/json.md#L1-L1) | 1 | JSON 키의 철자만 검사하고 값은 무시 |
| [`pom_xml.md`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/config/rules/rule_docs/pom_xml.md#L1-L1) | 1 | 새로 추가된 버전에 SNAPSHOT 접미사 금지 |

최근에 손본 문서일수록 첫 단락에 판단 원칙을 둔다. `go.md`의 첫 단락은 다음과 같다.

```markdown
#### Go Review Principles
> Favor precision over recall: report only defects that are likely real in the changed code and its reachable context. A false positive costs reviewer trust. Treat correctness and security findings as blocking; style-only suggestions are non-blocking. Focus on language-specific risks that ordinary formatting and deterministic tooling do not already cover.
```

같은 문서의 다음 단락은 `go vet`, Staticcheck, `go test -race`, 컴파일러, `gofmt`가 확정할 수 있는 결함은 중복 보고하지 말라고 적는다. 규칙 문서는 "무엇을 찾을지"와 함께 "무엇을 보고하지 않을지"를 정하는 장치이고, 이 방향은 09장의 반성 필터와 같다. 반대로 `java.md`는 항목 나열 위주라 문서마다 성숙도가 다르다.

규칙 문서는 도구 이름도 직접 부른다. `java.md`는 철자 오류를 확인할 때 `code_search`로 비슷한 식별자의 명명 규칙을 찾으라고 적고, `go.md`는 `file_read`와 `code_search`로 호출 위치와 동기화 범위를 먼저 확인하라고 적는다. 규칙 문서가 에이전트의 도구 사용 방식까지 지정하는 셈이다. 도구 자체는 08장에서 다룬다.

## .m 확장자의 내용 sniffing

`.m`은 MATLAB과 Objective-C가 함께 쓰는 확장자다. 경로만으로는 구분할 수 없어서, 시스템 계층을 감싼 `sniffer`가 파일 첫 줄을 읽어 판단한다.

### 코드 분석: sniffer 구조와 Resolve

`internal/config/rules/sniffer.go`, [sniffer.go L45-L60](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/config/rules/sniffer.go#L45-L60)

시스템 계층을 감싸는 decorator 구조체와 `Resolve`다.

```go
type sniffer struct {
	inner    systemLayer
	repoDir  string
	ref      string // git ref to read at; "" reads the working tree
	runner   *gitcmd.Runner
	objcRule string // rule_docs/objc.md
}

// Resolve returns objc.md for a ".m" file whose content sniffs as
// Objective-C, and otherwise defers to the wrapped system layer.
func (s *sniffer) Resolve(path string) string {
	if s.sniffsAsObjC(path) {
		return s.objcRule
	}
	return s.inner.Resolve(path)
}
```

주석([sniffer.go L37-L44](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/config/rules/sniffer.go#L37-L44))은 두 가지 설계 결정을 밝힌다.

- sniffer는 전체 resolver가 아니라 **시스템 계층만** 감싼다. 사용자가 `.m` 전용 규칙을 두었다면 sniffing보다 사용자 규칙이 먼저 이겨야 하기 때문이다. [`sniffer_test.go`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/config/rules/sniffer_test.go#L157-L160)의 `TestSniffer_UserRuleOutranksSniff`가 이 순서를 검사한다
- 캐시를 두지 않는다. `Resolve`는 묶음별 리뷰 goroutine에서 동시에 불리므로 캐시를 두려면 mutex가 필요한데, `.m` 파일 하나당 읽기 한 번이면 비용이 작다고 판단했다

### 코드 분석: 판별 접두어

`internal/config/rules/sniffer.go`, [sniffer.go L159-L178](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/config/rules/sniffer.go#L159-L178)

첫 줄이 어떤 문자열로 시작하면 Objective-C로 보는지 정한 목록이다.

```go
var objcSniffPrefixes = []string{
	"#import", "#include", "#pragma", "#if", "#define",
	"@import", "@interface", "@implementation", "@class", "@protocol",
	"//", "/*",
}

// looksLikeObjC reports whether the first non-blank line of content looks
// like Objective-C rather than MATLAB. It returns false (keeping the default
// MATLAB behavior) when content is empty or inconclusive.
func looksLikeObjC(firstLine string) bool {
	if firstLine == "" {
		return false
	}
	for _, prefix := range objcSniffPrefixes {
		if strings.HasPrefix(firstLine, prefix) {
			return true
		}
	}
	return false
}
```

| 판단 근거 | 내용 |
|---|---|
| `//`, `/*` 포함 이유 | MATLAB 주석은 `%`로 시작하고 MATLAB 파일은 `/`로 시작할 수 없다. Xcode 템플릿과 라이선스 머리말이 첫 줄에 주석을 두므로 주석 시작 기호를 신호로 쓴다 |
| `#` 단독을 넣지 않은 이유 | Octave도 `.m`을 쓰고 `#`을 주석으로 쓴다. `#` 전체를 넣으면 Octave 파일을 오판한다 |
| 읽기 실패 시 | 빈 문자열이 되어 `matlab.md` 유지 |
| 읽는 위치 | `ref`가 있으면 `git show <ref>:<path>`, 없으면 작업 트리 파일 |
| 시간 상한 | [`sniffTimeout`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/config/rules/sniffer.go#L21-L21) 5초 |

`ref`를 따로 두는 이유는 `--from`, `--to` 범위 리뷰에서 대상 커밋이 체크아웃되어 있지 않을 수 있어서다. 작업 트리를 읽으면 엉뚱한 버전의 첫 줄을 보게 된다. sniffing이 규칙을 바꿨다는 사실은 `RuleDetail.SniffedAs`에 기록되는데, 이 필드는 `json:"-"` 태그로 직렬화에서 빠진다([system_rules.go L127-L138](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/config/rules/system_rules.go#L127-L138)). `Pattern` 필드가 "일치한 glob"이라는 계약을 유지하도록 annotation을 별도 필드로 분리했다.

공식 문서도 이 heuristic이 버전 사이에 바뀔 수 있다고 밝히고, `.m` 경로를 확실하게 고정하려면 프로젝트 규칙을 쓰라고 권한다.
