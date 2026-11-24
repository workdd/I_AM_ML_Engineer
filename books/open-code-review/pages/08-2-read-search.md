파일 읽기, diff 조회, 검색 도구의 인자와 상한을 다룬다.

## file_read: 줄 범위와 바이트 상한

`file_read`는 파일의 일부 줄을 줄 번호와 함께 돌려준다. 결과 크기를 세 겹으로 제한한다.

상한 값은 [`file_read.go` L13-L27](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/tool/file_read.go#L13-L27)의 상수 세 개다. 주석은 줄 수 상한만으로는 부족하다고 설명한다. 커밋된 minify 파일이나 lockfile은 한 줄에 수 MB를 담을 수 있어 500줄로 잘라도 결과가 커지기 때문이다.

| 제한 | 값 | 넘었을 때의 표시 |
|---|---|---|
| 한 번에 읽는 줄 수 | 500줄 | `IS_TRUNCATED: true`, "Results truncated to 500 lines. Please narrow your line range." |
| 한 줄 길이 | 2048바이트 (UTF-8 문자 경계에서 자름) | 줄 끝에 `…[line truncated, N more bytes omitted]` |
| 결과 전체 | 256KiB (줄 번호 접두어 포함) | "Output truncated at 262144 bytes. Re-read the remaining lines ..." |

출력은 `File: <path> (Total lines: N)`, `IS_TRUNCATED`, `LINE_RANGE` 세 줄 머리 다음에 `줄번호|내용` 형식의 본문이 온다([L115-L119](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/tool/file_read.go#L115-L119)). `LINE_RANGE`는 바이트 상한으로 잘린 뒤 실제로 보여 준 마지막 줄을 가리킨다. `start_line`이 없거나 0 이하이면 1, `end_line`이 없으면 파일 끝이고, `start_line > end_line`이거나 시작 줄이 파일 길이를 넘으면 `error`를 돌려준다.

> **주의**: `tools.json`의 `file_read` 설명에 있는 출력 예시는 본문 줄에 `N|` 접두어가 없다. 실제 출력에는 매 줄 줄 번호와 `|`가 붙는다. 모델은 설명의 예시와 다른 형식을 받는다.

### FileReader의 읽기 모드

세 읽기 도구는 모두 `FileReader`를 거쳐 파일에 접근한다. 리뷰 입력에 따라 모드가 정해진다([`filereader.go` L21-L43](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/tool/filereader.go#L21-L43)).

| 모드 | 선택 조건 | 읽기 방법 | 파일 목록 (`file_find`) | 검색 대상 (`code_search`) |
|---|---|---|---|---|
| `ModeWorkspace` | 기본, 스캔 | 디스크에서 직접 읽기 | `git ls-files --cached --others --exclude-standard` | 작업 트리 (`--untracked` 포함) |
| `ModeRange` | `--from`과 `--to` | `git show <to>:<path>` | `git ls-tree -r --name-only <ref>` | `git grep ... <ref>` |
| `ModeCommit` | `--commit` | `git show <commit>:<path>` | 위와 같음 | 위와 같음 |
| `ModeEmpty` | 비교 기준이 빈 트리 | 항상 "does not exist in the empty tree" 오류 | 빈 목록 | `No matches found` |

`git show`를 쓰는 모드는 30초 timeout을 걸고, 출력을 스트림으로 읽어 필요한 줄만 모은다(L158-L173). 파일 전체를 메모리에 올리지 않고 총 줄 수만 센다.

### 코드 분석: 작업 트리 경로 검사

작업 트리 모드는 모델이 준 경로를 그대로 열지 않는다.

파일: [`internal/tool/filereader.go` L97-L126](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/tool/filereader.go#L97-L126)

```go
func (fr *FileReader) resolveWorkspacePath(path string) (string, error) {
	repoRoot, err := pathutil.CanonicalPath(fr.RepoDir)
	if err != nil {
		return "", fmt.Errorf("resolve repository path %q: %w", fr.RepoDir, err)
	}

	fullPath := filepath.Join(repoRoot, path)
	if !pathutil.WithinBase(repoRoot, fullPath) {
		return "", fmt.Errorf("file path %q is outside repository", path)
	}

	resolvedPath, err := filepath.EvalSymlinks(fullPath)
	if err != nil {
		if os.IsNotExist(err) {
			return fullPath, nil
		}
		return "", fmt.Errorf("resolve file %q: %w", path, err)
	}
	if !pathutil.WithinBase(repoRoot, resolvedPath) {
		return "", fmt.Errorf("file path %q is outside repository", path)
	}
	// ... os.Stat, validateWorkspaceFileMode
	return resolvedPath, nil
}
```

- 경로 검사는 두 번이다. `../` 같은 경로로 저장소 밖을 가리키는 경우를 먼저 막고, 심볼릭 링크를 따라간 실제 위치가 저장소 밖인 경우를 다시 막는다
- named pipe(FIFO)는 열 때 상대가 없으면 무한히 기다릴 수 있어 거부한다(L128-L134)
- `git show` 모드는 `--end-of-options`를 넣어 `ref:path`가 옵션으로 해석되지 않게 하고, `core.quotepath=false`로 비ASCII 경로를 그대로 다룬다

## 비교 대상 선택: side 인자

diff 리뷰에서 모델은 바뀐 뒤의 코드만이 아니라 바뀌기 전 코드도 볼 필요가 있다. 커밋, 범위, 작업 트리 리뷰에서는 `file_read`, `file_find`, `code_search`가 선택 인자 `side`(`"new"` 또는 `"old"`)를 받는다. 비교 양쪽의 기준점을 정하는 방법은 03장에서 다루고, 여기서는 도구 쪽 구현만 본다.

### 코드 분석: CodeAccess.execute

파일: [`internal/tool/code_access.go` L36-L67](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/tool/code_access.go#L36-L67)

```go
func (access *CodeAccess) execute(ctx context.Context, args map[string]any, current *FileReader, run func(*FileReader) (string, error)) (string, error) {
	rawSide, supplied := args["side"]
	if access == nil {
		if supplied {
			return "", fmt.Errorf("side is not supported for this task")
		}
		return run(current)
	}
	side := "new"
	// ... supplied이면 "new" 또는 "old"인지 검사, 아니면 "side must be new or old"
	if side == "new" {
		return run(&access.newReader)
	}
	prefix := "SIDE: old\nREVISION: " + access.oldReader.Ref + "\n"
	if access.oldReader.Mode == ModeEmpty {
		prefix = "SIDE: old\nSOURCE: empty-tree\n"
	}
	// ... 컨텍스트 확인, run(&access.oldReader), 오류에도 prefix 부착
	return prefix + content, nil
}
```

- `CodeAccess`가 nil인 경우(스캔)는 `side`가 오면 오류를 낸다. 모델이 정의에 없는 인자를 지어내도 조용히 무시하지 않는다
- `"old"` 결과에는 `SIDE: old`와 `REVISION: <sha>` 또는 `SOURCE: empty-tree`가 앞에 붙는다. `"new"` 결과는 기존 형식 그대로다
- 생성자 두 개가 조합을 제한한다. `NewComparisonCodeAccess`는 새 쪽이 고정된 커밋이어야 하고, `NewWorkspaceCodeAccess`는 새 쪽이 살아 있는 작업 트리이고 옛 쪽이 고정된 HEAD 또는 빈 트리여야 한다(L16-L34)

### 정의 쪽 주입

카탈로그 `tools.json`에는 `side`가 없다. 리뷰 설정 단계에서 [`WithComparisonSides` 또는 `WithWorkspaceSides`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/config/toolsconfig/comparison.go#L15-L37)가 원본 정의(`RawDefinition`)를 복사해 세 도구에만 `side` 속성을 덧붙인다([`review_cmd.go` L228-L241](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/cmd/opencodereview/review_cmd.go#L228-L241)).

| 변경 | 내용 |
|---|---|
| 속성 추가 | `"side": {"type": "string", "enum": ["new","old"], ...}`. 이미 `side`가 있으면 오류 |
| 설명 치환 | "This tool can only read file content from the modified version ..." 문장을 "This tool reads either side of the current comparison."으로 바꿈 (`code_search`도 같은 방식) |
| 설명 추가 (비교 리뷰) | 옛 쪽은 대상 브랜치 끝이 아니라 고정된 비교 기준이라는 설명, 이름이 바뀐 파일은 해당 쪽 경로를 써야 한다는 설명 |
| 설명 추가 (작업 트리 리뷰) | 새 쪽은 인덱스가 아니라 스냅숏 없는 작업 트리, 옛 쪽은 리뷰 시작 시점의 HEAD라는 설명 |

원본 카탈로그를 건드리지 않고 파생 정의를 만들기 때문에 스캔은 `side`가 없는 원래 정의를 그대로 쓴다.

## code_search: git grep 기반 검색

`code_search`는 별도 인덱스 없이 매 호출마다 `git grep`을 실행한다. `.gitignore`를 따르고 Git pathspec 문법(`:(exclude)vendor/` 등)을 그대로 쓸 수 있다.

### 코드 분석: buildGrepArgs

파일: [`internal/tool/code_search.go` L69-L96](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/tool/code_search.go#L69-L96)

```go
func (p *CodeSearchProvider) buildGrepArgs(searchText string, caseSensitive bool, usePerlRegexp bool, noIndex bool, pathspec []string) []string {
	cmdArgs := []string{"--no-pager", "-c", "core.quotepath=false", "grep"}

	if noIndex {
		// ... git 저장소가 아닌 디렉터리
		cmdArgs = append(cmdArgs, "--no-index", "--exclude-standard")
	} else if p.FileReader.Ref == "" {
		cmdArgs = append(cmdArgs, "--untracked")
	}

	if !caseSensitive {
		cmdArgs = append(cmdArgs, "-i")
	}
	if usePerlRegexp {
		cmdArgs = append(cmdArgs, "-P")
	} else {
		cmdArgs = append(cmdArgs, "-F")
	}

	cmdArgs = append(cmdArgs, "-n", "--no-color")
	// git grep limits matches per file. Fetch one extra to distinguish an exact
	// limit from truncated results, then enforce the global limit below.
	cmdArgs = append(cmdArgs, "--max-count", fmt.Sprintf("%d", gitGrepMaxCount+1))
```

| 인자 | git grep 옵션 | 기본값 |
|---|---|---|
| `search_text` | `-e <text>` | 필수. 공백뿐이면 `Error: search_text is blank` |
| `case_sensitive` | 거짓이면 `-i` | 거짓 |
| `use_perl_regexp` | 참이면 `-P`, 거짓이면 `-F` (고정 문자열) | 거짓 |
| `file_patterns` | `--` 뒤의 pathspec | 저장소 전체. `..` 경로 요소가 있으면 거부, `\`는 `/`로 바꿈 |
| (작업 트리 모드) | `--untracked` | 추적하지 않는 파일도 검색 |
| (커밋, 범위 모드) | 위치 인자 `<ref>` | ref가 `-`로 시작하면 거부 |

- 기본값이 고정 문자열 검색(`-F`)이라 모델이 `a.b(` 같은 코드 조각을 그대로 넣어도 정규식 오류가 나지 않는다
- ref 위치 인자에는 `--end-of-options`를 쓰지 않는다. 주석은 git 2.45 미만의 `git grep`이 그 위치의 구분자를 지원하지 않기 때문이라고 설명하고, 대신 `-`로 시작하는 ref를 직접 거부한다
- 10초 timeout(`gitGrepTimeout`)을 넘으면 "try narrowing file_patterns to a more specific path" 오류를 돌려준다

git 저장소가 아닌 디렉터리에서 `git grep`이 종료 코드 128과 "not a git repository"로 실패하면, ref가 없는 경우에 한해 `--no-index --exclude-standard`로 다시 실행한다(L158-L165). 스캔이 일반 디렉터리도 다룰 수 있게 하기 위해서다(11장).

### 결과 형식과 상한

출력은 파일별로 묶어 `File: <path>`, `Match lines: <n>`, `줄번호|내용` 순서로 쓴다. 일치하는 줄이 없으면(종료 코드 1, stderr 없음) `No matches found`를 돌려준다.

파일: [`internal/tool/code_search.go` L226-L246](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/tool/code_search.go#L226-L246)

```go
		// Count every file with a text match, including those beyond the render
		// budget, so the truncation note can report the true matched-file count.
		matchedFiles[fname] = true
		if matchCount >= gitGrepMaxCount {
			// Keep scanning to finish counting matched files, but render no more.
			truncated = true
			continue
		}
		m := match{lineNum: ln, content: parts[offset+2]}
		if !seen[fname] {
			seen[fname] = true
			fileOrder = append(fileOrder, fname)
		}
		fileMatches[fname] = append(fileMatches[fname], m)
		matchCount++
	}

	var sb strings.Builder
	if truncated {
		sb.WriteString(fmt.Sprintf("Note: Showing the first %d matches across %d matching files. Some files are partially shown or omitted entirely. Narrow file_patterns to see the rest.\n", gitGrepMaxCount, len(matchedFiles)))
	}
```

- 상한은 두 단계다. `git grep --max-count 101`로 파일마다 101개까지 받고, 그 결과에서 전체 100개(`gitGrepMaxCount`)만 출력한다
- 101번째 일치 이후도 끝까지 읽어 일치한 파일 수를 센다. 잘림 안내 문구에 "N개 파일 중 처음 100개"를 정확히 적기 위해서다

> **주의**: 원본 문서의 [Tools 페이지 Limits 절](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/pages/src/content/docs/en/tools.md)은 "Caps matches at 100 per file via `git grep --max-count 100`, so total output across many files can exceed 100"이라 하고 안내 문구를 "The results have been truncated. Only showing first 100 results."로 적는다. 코드는 파일당 101개를 받아 전체 100개로 자르고, 문구도 위 인용과 다르다. 출력이 100개를 넘는 일은 없다. `tools.json` 설명의 "only the first 100 results will be returned"는 코드와 맞는다. 또 `tools.json` 설명의 출력 예시에는 `Match lines:` 줄이 없다.

## file_find: 파일 이름 검색

`file_find`는 파일 목록을 받아 이름에 검색어가 들어 있는지 부분 문자열로 비교한다. 정규식이나 glob은 쓰지 않는다.

### 코드 분석: MatchFileNames

파일: [`internal/tool/file_find.go` L77-L117](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/tool/file_find.go#L77-L117)

```go
	// 1. First pass: match against base filename (maintains precision for pure filename queries).
	var matched []string
	for _, f := range files {
		// ... 컨텍스트 확인
		base := f
		if idx := strings.LastIndex(f, "/"); idx != -1 {
			base = f[idx+1:]
		}
		if strings.Contains(baseCmp, queryCmp) && (allowed == nil || allowed(f)) {
			matched = append(matched, f)
		}
		if len(matched) >= limit {
			break
		}
	}

	// 2. Fallback pass: if basename match returned no results, match against the full relative path
	// (enables directory-scoped queries such as `pkg/util` or `internal/diff`).
	if len(matched) == 0 {
		// ... 전체 경로에 같은 비교
	}
```

- 1차는 파일 이름(마지막 `/` 뒤)에서 찾는다. 결과가 하나도 없을 때만 2차로 전체 상대 경로에서 찾는다
- 검색어에 `/`가 있으면 파일 이름에는 `/`가 없으므로 1차가 항상 비고, 자연스럽게 경로 검색이 된다. `\`는 `/`로 바꿔 비교한다
- 결과는 100개(`fileFindMaxCount`)에서 멈추고, 잘렸다는 표시 없이 조용히 끝난다. 일치가 없으면 `// The file was not found`를 돌려준다

> **주의**: `tools.json`은 "When the query contains a path separator ('/' or '\\'), also matches against the full repository-relative path"라고 설명한다. 코드는 구분자 여부와 관계없이 1차 결과가 비면 전체 경로로 다시 찾는다. 예를 들어 `agent`를 찾을 때 이름에 `agent`가 들어간 파일이 하나도 없으면 `internal/agent/` 아래 모든 파일이 나온다.

파일 목록은 `git ls-files` 또는 `git ls-tree`로 얻고(L124-L179), git이 아닌 디렉터리는 루트 `.gitignore`와 기본 제외 디렉터리 목록을 지키며 직접 순회한다. 목록에서 확장자 없는 파일은 뺀다. 예외는 `Makefile`, `Dockerfile`, `LICENSE`, `Vagrantfile`, `Containerfile`, `BUILD`, `WORKSPACE` 일곱 개다([L232-L248](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/tool/file_find.go#L232-L248)). 확장자 판정은 이름에 `.`이 있는지만 보므로 `.env` 같은 점 파일은 남는다.

## file_read_diff: 다른 파일의 diff 조회

`file_read_diff`는 이번 리뷰에서 이미 파싱한 diff를 경로로 찾아 돌려준다. git을 다시 실행하지 않고 메모리의 `DiffMap`만 본다.

처리 함수는 [`file_read_diff.go` L65-L101](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/tool/file_read_diff.go#L65-L101)이다. `path_array`가 비면 `Error: no files found`를 돌려준다.

- 찾은 diff는 `==== FILE: <path> ====` 머리와 함께 이어 붙이고, 못 찾은 경로는 끝에 모아 `Error: diff not found for: ...`로 알린다
- 못 찾았을 때 diff가 있는 경로를 정렬해 최대 10개(`diffNotFoundMaxListed`)까지 "Diffs exist for: ..."로 보여 주고, 나머지는 "(and N more)"로 줄인다. 모델이 경로 철자를 고칠 수 있게 하려는 처리다
- diff가 하나도 없으면 "This run has no diffs. Read the file with file_read instead."를 돌려준다. 주석은 여기서 `file_read_diff`를 다시 권하면 방금 실패한 호출로 돌아가게 되므로 `file_read`를 안내한다고 설명한다

> **주의**: diff 리뷰의 `DiffMap`은 파일 선별 전 전체 diff로 만든다. [`agent.go` L315-L322](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/agent/agent.go#L315-L322)에서 `injectDiffMap()`이 `selectFiles` 앞에 있고, 주석도 "Build the read-only DiffMap from ALL parsed diffs (before filtering)"라고 적는다. `internal/tool`에는 비밀 정보 경로 검사(`allowedext.IsSecretPath`)가 없으므로, 선별에서 `ExcludeSecret`으로 빠진 `.env` 같은 파일의 diff도 `file_read_diff`로 모델에 전달될 수 있고, 없는 경로를 요청하면 그 경로가 "Diffs exist for:" 목록에 나올 수 있다. 같은 패키지의 `file_read`와 `code_search`도 이 검사를 하지 않는다. 동작 영향 분석은 별도의 [`safePath`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/behaviortask/code.go#L131-L136) 검사를 둔다. 비밀 경로 선별 규칙은 04장 참고.

## code_comment 개요

`code_comment`는 다른 도구와 달리 결과를 모델에게 돌려주는 것보다 코멘트를 모으는 것이 목적이다. 실행 경로는 `reviewtask.executeToolCall`의 별도 분기([L74-L200](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/reviewtask/tools.go#L74-L200))이고, 인자를 코멘트 목록으로 파싱한 뒤 줄 번호를 찾아 수집기에 넣는다. 작업자 풀이 있으면 비동기로 처리하고 모델에게는 바로 "Successfully commented."를 돌려준다. 줄 번호 찾기, 다른 파일로의 재기록, 위치 재탐색 LLM 호출, 반성 필터는 모두 09장에서 다룬다.

> **주의**: `tools.json`의 `code_comment` 스키마는 `comments` 항목마다 `content`, `existing_code`, `category`(8개 값), `severity`(`critical`, `high`, `medium`, `low`), `path`를 필수로 요구한다. 원본 문서 [Tools 페이지](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/pages/src/content/docs/en/tools.md)의 스키마 예시는 최상위 `path`와 항목별 `content`, `existing_code`, `suggestion_code`, `thinking`만 보여 주고 `category`, `severity`를 언급하지 않는다. 모델에게 실제로 전달되는 것은 `tools.json` 쪽이다.
