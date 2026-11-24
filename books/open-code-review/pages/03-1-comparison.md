비교 양쪽을 commit SHA로 고정하는 과정과 range, commit 모드의 git 명령을 다룬다.

## 비교 대상 고정

### InputResolution 구조

브랜치 이름 같은 ref는 실행 도중에도 움직일 수 있다. 리뷰가 몇 분에서 수십 분 걸리는 동안 누군가 브랜치에 push하면, 처음 읽은 diff와 나중에 도구가 읽는 파일 내용이 서로 다른 commit을 가리키게 된다. 이를 막으려고 실행 초반에 모든 ref를 commit SHA로 바꿔 [`InputResolution`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/diff/git.go#L129-L140)에 담는다.

| 필드 | 의미 | range | commit | workspace |
|---|---|---|---|---|
| `ResolvedBase` | 비교 기준 commit SHA | merge-base | 첫 번째 부모 | `HEAD` SHA |
| `ResolvedHead` | 비교 대상 commit SHA | `--to`의 SHA | `--commit`의 SHA | 비어 있음 |
| `ExactRange` | `base..head` 문자열 | 둘 다 있을 때 | 부모가 있을 때 | 비어 있음 |
| `BaseIsEmpty` | 기준이 빈 tree인지 | 거짓 | 루트 commit이면 참 | 첫 commit 전이면 참 |

구조체 주석은 "빈 필드는 해당 없음이거나 풀 수 없음을 뜻하며, 호출자는 빈 값을 실제 endpoint로 다루거나 값을 지어내면 안 된다"고 적었다. workspace는 작업 트리 자체가 비교 대상이라 고정할 head가 없다.

### 코드 분석: ResolveComparisonInput

실행 전 고정은 `internal/agent`의 함수가 담당한다. 02장의 `prepareReviewCodeTools`와 재개 검증이 이 함수를 부른다.

파일: [internal/agent/identity.go L79-L99](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/agent/identity.go#L79-L99)

```go
	case args.From != "" && args.To != "":
		from, err := resolveCommitHead(ctx, args, args.From)
		if err != nil {
			return nil, err
		}
		head, err := resolveCommitHead(ctx, args, args.To)
		if err != nil {
			return nil, err
		}
		resolved := diff.NewProvider(args.RepoDir, from, head, args.GitRunner).ResolveInput(ctx)
		if resolved.ResolvedBase == "" {
			return nil, fmt.Errorf("resolve merge-base between %q and %q", args.From, args.To)
		}
		return &resolved, nil
	default:
		resolution, err := diff.NewWorkspaceProvider(args.RepoDir, args.GitRunner).ResolveWorkspaceComparison(ctx)
		if err != nil {
			return nil, err
		}
		return &resolution, nil
	}
```

range 모드는 `--from`과 `--to`를 각각 `rev-parse --verify <ref>^{commit}`으로 SHA로 바꾼 뒤, 그 SHA로 merge-base를 계산한다. 공통 조상이 없으면(관련 없는 두 히스토리) 여기서 실패한다.

### 코드 분석: commit 모드의 부모 확인

commit 모드는 부모 commit을 읽을 수 있는지까지 확인한다.

파일: [internal/diff/comparison.go L21-L39](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/diff/comparison.go#L21-L39)

```go
	resolution := InputResolution{ResolvedHead: head, BaseIsEmpty: true}
	for line := range strings.SplitSeq(content, "\n") {
		if line == "" {
			break
		}
		if parent, ok := strings.CutPrefix(line, "parent "); ok {
			if provider.resolveCommit(ctx, parent) != parent {
				return InputResolution{}, fmt.Errorf("comparison parent %s is unavailable", parent)
			}
			parents := provider.commitParents(ctx, head)
			if len(parents) == 0 || parents[0] != parent {
				return InputResolution{}, fmt.Errorf("comparison parent %s is not visible in Git history; deepen the clone before reviewing this commit", parent)
			}
			resolution.ResolvedBase = parent
			resolution.ExactRange = parent + ".." + head
			resolution.BaseIsEmpty = false
			break
		}
	}
```

- `git cat-file commit <SHA>`의 헤더에서 첫 번째 `parent` 줄을 읽는다. 헤더는 첫 빈 줄에서 끝나므로 그 전까지만 본다
- 부모 SHA가 객체 저장소에 실제로 있는지(`resolveCommit`), `rev-list --parents`가 같은 부모를 보여 주는지 두 번 확인한다. 얕은 clone(shallow clone)에서는 commit 객체가 부모 SHA를 기록하고 있어도 그 부모가 히스토리에 없다. 이 경우 "deepen the clone" 안내와 함께 실패한다. CI에서 `fetch-depth: 1`로 받은 저장소를 리뷰할 때 나오는 오류다(추정)
- `parent` 줄이 없으면 루트 commit이다. 초기값 `BaseIsEmpty: true`가 그대로 남아 빈 tree와 비교한다

### 코드 분석: 고정된 SHA로 Provider 생성

실제 diff를 읽는 단계는 사용자가 입력한 ref 대신 고정된 SHA를 쓴다.

파일: [internal/agent/agent.go L566-L586](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/agent/agent.go#L566-L586)

```go
	from, to, commit := a.args.From, a.args.To, a.args.Commit
	if s := a.args.SealedInput; s != nil && s.ResolvedHead != "" {
		switch {
		case commit != "":
			commit = s.ResolvedHead
		case s.ResolvedBase != "":
			from, to = s.ResolvedBase, s.ResolvedHead
		}
	}

	switch {
	case commit != "":
		return diff.NewCommitProvider(a.args.RepoDir, commit, a.args.GitRunner)
	case from != "" && to != "":
		return diff.NewProvider(a.args.RepoDir, from, to, a.args.GitRunner)
	default:
		if a.args.SealedInput != nil {
			return diff.NewWorkspaceProviderWithBase(a.args.RepoDir, *a.args.SealedInput, a.args.GitRunner)
		}
		return diff.NewWorkspaceProvider(a.args.RepoDir, a.args.GitRunner)
	}
```

range 모드에서는 `from`을 merge-base SHA로 바꾼다. 그래도 의미가 바뀌지 않는 이유를 [함수 주석](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/agent/agent.go#L555-L565)이 설명한다. base가 head의 조상이면 `merge-base(base, head)`는 base 자신이다. workspace 모드는 기준 commit만 고정하고 작업 트리는 계속 실시간으로 읽는다.

고정된 `InputResolution`은 실행 manifest와 재개 동일성 검사(13장)에도 그대로 기록된다. [`loadDiffs`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/agent/agent.go#L589-L614)는 diff를 읽은 직후 `ResolveInput`과 `RemoteIdentity`를 다시 호출해 manifest에 넣을 값을 확정한다. `RemoteIdentity`는 `origin` 원격 주소에서 사용자 정보, 쿼리, `.git` 접미사를 지운 정규화 문자열이며 manifest에는 그 SHA-256만 남는다.

## range 모드와 commit 모드의 git 명령

### 코드 분석: GetDiffSet의 diff 명령

range와 commit 모드는 git 명령 하나로 전체 diff를 받는다.

파일: [internal/diff/git.go L222-L242](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/diff/git.go#L222-L242)

```go
	case ModeRange:
		base := p.MergeBase(ctx)
		if base == "" {
			return DiffSet{}, fmt.Errorf("cannot find merge-base between %s and %s", p.from, p.to)
		}
		out, stderr, err := p.runGitSplit(ctx, "-c", "core.quotepath=false", "diff", "--no-ext-diff", "--no-textconv", "--find-renames", "--src-prefix=a/", "--dst-prefix=b/", "--no-color", "-U"+fmt.Sprint(DiffContextLines), "--end-of-options", base, p.to, "--")
		if err != nil {
			return DiffSet{}, gitFailure("git diff", stderr, err)
		}
		combined.WriteString(out)

	case ModeCommit:
		// --diff-merges=first-parent: for merge commits, plain `git show`
		// emits a combined diff ("diff --cc"), which ParseDiffText cannot
		// parse — the commit would silently yield zero reviewable diffs.
		// Diffs against the first parent instead, in regular unified format.
		out, stderr, err := p.runGitSplit(ctx, "-c", "core.quotepath=false", "show", "--no-ext-diff", "--no-textconv", "--find-renames", "--src-prefix=a/", "--dst-prefix=b/", "--no-color", "--diff-merges=first-parent", "-U"+fmt.Sprint(DiffContextLines), "--end-of-options", p.commit)
		if err != nil {
			return DiffSet{}, gitFailure("git show", stderr, err)
		}
		combined.WriteString(out)
```

플래그 하나하나가 사용자의 git 설정이 출력 형식을 바꾸지 못하게 막는다.

| 인자 | 막는 상황 |
|---|---|
| `-c core.quotepath=false` | 한글, 일본어 같은 비ASCII 경로가 `"\355\225\234..."` 형태로 이스케이프되는 것 |
| `--no-ext-diff` | 사용자가 설정한 외부 diff 도구가 unified 형식 대신 다른 출력을 내는 것 |
| `--no-textconv` | `.gitattributes`의 textconv 필터가 파일을 다른 텍스트로 변환해 보여 주는 것 |
| `--find-renames` | 이름 변경이 삭제 하나와 추가 하나로 나뉘어 리뷰 대상이 두 배가 되는 것 |
| `--src-prefix=a/ --dst-prefix=b/` | `diff.noprefix`, `diff.mnemonicPrefix` 설정으로 경로 접두사가 바뀌는 것 |
| `--no-color` | `color.ui=always` 설정으로 ANSI 코드가 섞이는 것 |
| `-U3` | 주변 줄 수. 상수 [`DiffContextLines`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/diff/git.go#L24-L25) = 3 |
| `--end-of-options` | `-`로 시작하는 ref가 옵션으로 해석되는 것 |
| `--diff-merges=first-parent` | merge commit이 파싱할 수 없는 combined diff(`diff --cc`)로 나오는 것 |

merge commit 처리 주석이 중요하다. 일반 `git show`는 merge commit에 대해 `diff --cc` 형식을 출력하는데, 파서가 이 형식을 읽지 못해 리뷰 대상이 조용히 0개가 되던 문제가 있었다. 첫 번째 부모 기준으로 비교하면 "이 merge가 대상 브랜치에 들여온 변경"이 리뷰된다.

merge-base는 [`computeMergeBase`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/diff/git.go#L465-L471)가 `git merge-base --end-of-options from to`로 한 번 계산하고 `Provider.mergeBase`에 저장한다. 실패하면 빈 문자열을 돌려주고, 위 코드가 `cannot find merge-base` 오류를 만든다.

### range 모드의 의미

range 모드는 `git diff A..B`(두 끝점 비교)가 아니라 `git diff A...B`(세 점 표기, 분기 이후 B의 변경)와 같은 결과를 낸다. `main`이 분기 이후 앞으로 나아갔어도 그 변경은 리뷰에 섞이지 않는다. 풀 리퀘스트 화면이 보여 주는 diff와 같은 기준이다(미검증).
