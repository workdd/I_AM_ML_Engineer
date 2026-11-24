작업 트리 변경과 untracked 파일 수집, 제외 디렉터리와 .gitignore 적용을 다룬다.

## workspace 모드와 untracked 파일

### 코드 분석: workspace 기준 commit 결정

workspace 모드는 커밋 전 변경을 리뷰한다. 기준은 `HEAD`이고, 저장소에 commit이 하나도 없으면 빈 tree다.

파일: [internal/diff/workspace.go L32-L48](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/diff/workspace.go#L32-L48)

```go
	out, stderr, err := provider.runGitSplit(ctx, "rev-parse", "--verify", "--quiet", "--end-of-options", "HEAD^{commit}")
	if err == nil {
		base := InputResolution{ResolvedBase: strings.TrimSpace(out)}
		provider.workspaceBase = &base
		return base, nil
	}
	branch, _, branchErr := provider.runGitSplit(ctx, "symbolic-ref", "--quiet", "HEAD")
	branch = strings.TrimSpace(branch)
	if branchErr == nil && strings.HasPrefix(branch, "refs/heads/") {
		_, refStderr, refErr := provider.runGitSplit(ctx, "show-ref", "--verify", "--quiet", "--", branch)
		var exitErr *exec.ExitError
		if errors.As(refErr, &exitErr) && exitErr.ExitCode() == 1 && refStderr == "" {
			base := InputResolution{BaseIsEmpty: true}
			provider.workspaceBase = &base
			return base, nil
		}
	}
```

`HEAD`가 풀리지 않는 경우를 무조건 "첫 commit 전"으로 보지 않는다. `HEAD`가 `refs/heads/...` 브랜치를 가리키고, 그 브랜치 ref가 존재하지 않으며(종료 코드 1), stderr도 비어 있을 때만 빈 tree로 판정한다. 저장소가 손상되었거나 권한 오류가 난 경우를 빈 저장소로 착각해 모든 파일을 새 파일로 리뷰하는 일을 막는다.

### tracked 변경의 diff 명령

[`workspaceTrackedDiff`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/diff/git.go#L628-L642)는 기준 commit과 작업 트리를 한 번에 비교한다. 빈 tree가 기준이면 `git hash-object -t tree --stdin`에 빈 입력을 넣어 빈 tree 객체 ID를 얻는다.

```
git --no-optional-locks -c core.quotepath=false -c diff.autoRefreshIndex=false \
    diff --no-ext-diff --no-textconv --find-renames --src-prefix=a/ --dst-prefix=b/ \
    --no-color -U3 --end-of-options <HEAD SHA 또는 빈 tree> --
```

- `git diff <commit> --`는 작업 트리를 그 commit과 비교하므로 staged와 unstaged 변경이 한 diff로 합쳐진다
- `--no-optional-locks`와 `diff.autoRefreshIndex=false`는 리뷰가 index 파일을 갱신하지 않게 한다. 사용자가 편집기나 다른 git 명령을 쓰는 동안 리뷰가 index 잠금을 잡아 충돌하는 것을 피한다

> **주의**: 저장소의 [CLI 문서](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/pages/src/content/docs/en/cli-reference.md#L179-L183)는 workspace 모드가 `git diff HEAD`를 쓰고 "결과가 비면 `git diff --staged`로 대체한다"고 적었다. 기준 커밋의 코드에는 이 대체 경로가 없다. tracked 변경은 위 명령 하나로만 수집되고, `HEAD`가 없는 저장소는 `--staged` 대신 빈 tree와 비교한다.

### untracked 파일의 diff 합성

untracked 파일은 git diff에 나오지 않으므로 직접 diff 텍스트를 만든다. 목록은 [`untrackedFilesList`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/diff/git.go#L712-L737)가 `git ls-files -z --others --exclude-standard`로 얻는다. `--exclude-standard`가 `.gitignore`, `.git/info/exclude`, 전역 제외 설정을 git 방식 그대로 적용한다. 그 뒤 루트 `.gitignore`와 내장 제외 디렉터리를 한 번 더 적용한다.

`-z`(NUL 구분)를 쓰는 이유를 주석이 밝혔다. 이 옵션이 없으면 git이 탭, 줄바꿈, 따옴표, 백슬래시가 든 파일 이름을 따옴표로 감싸 이스케이프하므로 디스크에서 같은 파일을 찾지 못한다. 앞뒤 공백도 파일 이름의 일부라서 결과를 trim하지 않는다.

### 코드 분석: untracked 파일 내용 변환

파일: [internal/diff/git.go L679-L698](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/diff/git.go#L679-L698)

```go
		if looksBinary(content) {
			// Emit git's own binary marker so the parser flags the file and
			// the selection layer excludes it, matching the tracked path where
			// `git diff` itself reports "Binary files ... differ".
			results = append(results, untrackedBinaryDiff(f))
			continue
		}

		lineCount := bytes.Count(content, []byte{'\n'})
		if len(content) > 0 && content[len(content)-1] != '\n' {
			lineCount++
		}

		var sb strings.Builder
		sb.WriteString(fmt.Sprintf("diff --git a/%s b/%s\n", f, f))
		sb.WriteString("--- /dev/null\n")
		sb.WriteString(fmt.Sprintf("+++ b/%s\n", f))
		sb.WriteString(fmt.Sprintf("@@ -0,0 +1,%d @@\n", lineCount))

		lines := bytes.Split(content, []byte{'\n'})
```

- 새 파일 전체를 `+` 줄로 만든 hunk 하나(`@@ -0,0 +1,N @@`)를 합성한다. tracked 변경과 같은 파서를 그대로 쓰기 위해서다
- 바이너리 판정은 git과 같은 기준이다. 앞쪽 8000바이트([`binarySniffWindow`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/diff/git.go#L644-L656)) 안에 NUL 바이트가 있으면 바이너리로 본다. 이때는 `Binary files /dev/null and b/<path> differ` 표식만 내보내고, 04장의 선별 단계가 이 표식을 보고 제외한다
- 64 MiB([`maxUntrackedFileSize`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/diff/workspace_file.go#L15-L15) = `64 << 20`)를 넘는 파일은 내용을 읽지 않고 바이너리 표식으로 처리한다
- 읽기에 실패한 파일은 경고 없이 건너뛴다(`if rerr != nil { continue }`)

### 작업 트리 파일의 안전한 조회

untracked 파일과 workspace 모드의 새 파일 내용은 디스크에서 읽는다. [`readWorkspaceFileForDiffWithLimit`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/diff/workspace_file.go#L23-L75)는 저장소 밖을 가리키는 심볼릭 링크로 다른 파일을 읽게 만드는 경로를 막는다. 심볼릭 링크 파일은 따라가지 않고 링크 대상 경로 문자열을 내용으로 돌려준다. git이 심볼릭 링크를 저장하는 방식(대상 경로를 blob으로 저장)과 같다. 그 앞뒤로 저장소 루트를 [`pathutil.CanonicalPath`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/pathutil/path.go#L12-L19)로 정규화하고, 상위 폴더와 최종 경로가 모두 루트 안에 있는지 [`pathutil.WithinBase`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/pathutil/path.go#L21-L32)로 확인한다. `WithinBase`는 `filepath.Rel` 결과가 `..`로 시작하지 않는지 먼저 보고, 문자열로 판정하지 못하면 `os.SameFile`로 상위 폴더를 하나씩 올라가며 다시 확인한다. 대소문자를 구분하지 않는 파일 시스템처럼 경로 문자열과 실제 파일이 어긋나는 환경을 위한 보완으로 보인다(추정).

## 제외 디렉터리와 .gitignore

### 내장 제외 디렉터리

[`providerDirIgnoreDirs`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/diff/git.go#L27-L40)는 diff 단계에서 무조건 빼는 경로 접두사 11개다.

| 분류 | 접두사 |
|---|---|
| 편집기, 버전 관리 | `.idea/`, `.vscode/`, `.svn/`, `.git/` |
| 의존성 | `vendor/`, `node_modules/`, `_packages/` |
| 빌드 산출물, 캐시 | `target/`, `.happypack/`, `.cachefile/`, `rpm/` |

이 접두사는 저장소 루트 기준이다. [`ProviderDirPrefix`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/diff/gitignore.go#L51-L61)는 `strings.HasPrefix(relPath, prefix)`로 비교하므로 `vendor/a.go`는 빠지지만 `pkg/vendor/a.go`는 이 단계에서 빠지지 않는다. 하위 폴더의 `vendor`는 04장의 기본 제외 패턴 `**/vendor/**`가 처리한다. 두 단계의 차이는 사용자의 include 규칙으로 되살릴 수 있는지다. 이 단계에서 빠진 파일은 어떤 규칙으로도 리뷰 대상이 되지 않는다.

### 코드 분석: .gitignore 판정 순서

range와 commit 모드에서는 git이 `.gitignore`를 적용하지 않는다. 이미 commit된 파일의 diff이기 때문이다. 그런데 commit에 생성물이 실수로 들어간 경우를 걸러 내려고, OpenCodeReview가 루트 `.gitignore`를 직접 해석해 적용한다.

파일: [internal/diff/git.go L317-L338](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/diff/git.go#L317-L338)

```go
func (patterns GitignorePatterns) Matches(relPath string) bool {
	excluded := false
	for _, pat := range patterns {
		body, negated := strings.CutPrefix(pat, "!")
		if body == "" {
			continue
		}

		// Directory-only patterns (trailing "/") apply to directories, never to
		// files. Git uses a negated one such as `!*/` to keep descending into
		// subdirectories, not to re-admit the files inside them — honouring it
		// here would readmit everything below the root.
		if negated && strings.HasSuffix(body, "/") {
			continue
		}

		if matchGitignoreBody(relPath, body) {
			excluded = !negated
		}
	}
	return excluded
}
```

git과 같은 "마지막으로 일치한 패턴이 결정한다" 규칙이다. [`isPathExcluded` 주석](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/diff/git.go#L304-L312)은 이 규칙이 필요한 이유로 허용 목록 관용구를 든다. `*`로 모두 무시한 뒤 `!*.go`처럼 다시 포함하는 `.gitignore`에서 부정 패턴을 무시하면, 모든 파일이 제외되어 리뷰가 아무것도 다루지 않게 된다.

개별 패턴 해석은 [`matchGitignoreBody`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/diff/git.go#L362-L408)가 맡는다.

| 패턴 형태 | 해석 |
|---|---|
| 끝이 `/` | 디렉터리 전용. 경로의 상위 폴더 이름과 비교 |
| 앞이 `/` | 루트 고정. 루트의 그 이름만 일치 |
| `**` 포함 | doublestar 라이브러리로 비교 |
| `/` 없음 | 파일 이름(basename)과 비교 |
| `/` 포함 | 전체 경로 또는 `/`로 시작하는 경로 끝부분과 비교 |

### 특정 commit의 .gitignore 조회

range와 commit 모드는 작업 트리의 `.gitignore`가 아니라 리뷰 대상 commit에 있던 `.gitignore`를 읽는다. [`LoadGitignoreAtRef`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/diff/gitignore.go#L16-L40)는 `git ls-tree -z --full-tree <ref>`로 루트 항목을 나열하고, 이름이 `.gitignore`인 blob을 `git cat-file blob`으로 읽어 패턴 목록으로 바꾼다.

- `ls-tree`를 재귀 없이 실행하므로 루트의 `.gitignore`만 찾는다. 하위 폴더의 `.gitignore`는 range와 commit 모드에서 적용되지 않는다
- 일반 파일 모드(`100644`, `100755`)만 읽는다. `.gitignore`가 심볼릭 링크(`120000`)면 무시한다. 링크 대상 경로 문자열을 패턴으로 오해하지 않기 위한 처리로 보인다(추정)
- 작업 트리가 아니라 commit을 기준으로 하므로 같은 `--commit`을 언제 리뷰해도 같은 파일 목록이 나온다

[`partitionDiffs`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/diff/git.go#L438-L461)는 내장 제외 디렉터리에 걸린 diff를 `Excluded`로, `.gitignore`에 걸린 diff는 버리고, 나머지를 `Included`로 나눈다. workspace 모드는 `.gitignore`를 tracked 변경에 적용하지 않는다. 주석은 "workspace에서는 무시된 untracked 파일이 이미 걸러졌고, 여기서 다시 적용하면 tracked 변경이 잘못 빠진다"고 설명한다. git에서 이미 추적 중인 파일은 `.gitignore`에 있어도 변경이 기록되기 때문이다.

`Excluded`는 리뷰에 쓰이지 않지만 미리보기(04장)가 "provider_directory" 사유로 보여 준다. [`DiffSet.ForEachInOrder`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/diff/git.go#L80-L97)는 두 목록을 git이 출력한 원래 순서대로 다시 섞어 순회한다.
