OCR은 리뷰 한 번을 JSONL 파일 하나로 남긴다. 이 파일은 LLM에 보낸 메시지, 응답, 도구 호출, 파일 단위 체크포인트, 실행 manifest를 모두 담고 있어서 세 가지 용도로 다시 읽힌다. 중단된 리뷰를 이어서 돌리는 재개(`--resume`), 두 실행의 지적을 비교하는 `ocr session compare`, 브라우저로 기록을 보는 세션 뷰어(`ocr viewer`)다. 이 장은 `internal/session`과 `internal/viewer`를 따라 저장 형식, 재개 시 동일성 판정, 뷰어의 로컬 보안을 정리한다.

## 세션 저장 구조 개요

모든 세션 데이터는 사용자 홈의 `~/.opencodereview/` 아래에 있다. 저장소 경로를 디렉터리 이름 하나로 바꾼 뒤, 그 안에 세션 ID(UUID v4)를 파일 이름으로 쓴다.

```text
~/.opencodereview/
├── config.json                         # 10장의 프로바이더 설정
├── sessions/
│   └── <인코딩된 저장소 경로>/
│       ├── <session-id>.jsonl          # 리뷰 또는 스캔 한 번의 전체 기록
│       └── <session-id>/analysis/      # --analysis 실행 시 결과, trace.jsonl 등 (12장)
├── raw/
│   └── <인코딩된 저장소 경로>/
│       └── <session-id>.jsonl          # OCR_RAW_LOGGING=1일 때 HTTP 시도 원문
└── auth/chatgpt/accounts.json          # ChatGPT 로그인 토큰 (10장)
```

| 경로 | 쓰는 코드 | 권한 |
|---|---|---|
| `sessions/<repo>/<id>.jsonl` | [persist.go L113-L126](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/session/persist.go#L113-L126) | 디렉터리 0700, 파일 0600 |
| `raw/<repo>/<id>.jsonl` | [raw_writer.go L49-L60](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/session/raw_writer.go#L49-L60) | 같은 규칙 |
| `sessions/<repo>/<id>/analysis/` | [analysis.go L117-L124](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/cmd/opencodereview/analysis.go#L117-L124) | 디렉터리 0700 |

세션 파일은 리뷰를 시작하는 순간 만들어진다. `session.New`가 `session_start` 레코드를 바로 쓰기 때문에, 재개 검증처럼 "실행을 거부할 수도 있는" 판단은 반드시 `session.New`보다 먼저 해야 한다. 이 순서는 재개 절에서 다시 나온다.

### 코드 분석: 저장소 경로 인코딩

파일: `internal/session/persist.go`, [persist.go L92-L110](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/session/persist.go#L92-L110)

저장소 절대 경로를 디렉터리 이름 하나로 바꾸는 규칙이다.

```go
	vol := filepath.VolumeName(p)
	p = p[len(vol):]

	// Trim leading path separators
	p = strings.TrimLeft(p, "/\\")

	// Replace separators with -
	p = strings.ReplaceAll(p, "/", "-")
	p = strings.ReplaceAll(p, "\\", "-")

	// Replace colons (from Windows drive letters)
	vol = strings.ReplaceAll(vol, ":", "_")

	// Handle edge case where path was only separators or volume name
	result := vol + p
	if result == "" {
		return "empty"
	}
	return result
```

`/home/dev/my-app`은 `home-dev-my-app`이 된다. 구분자와 하이픈을 구별하지 않으므로 `/home/a/b`와 `/home/a-b`는 같은 디렉터리를 쓴다. 이 충돌은 코드도 알고 있다. 뷰어의 비교 핸들러 주석([handler.go L194-L203](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/viewer/handler.go#L194-L203))이 이 경우를 명시하고, 두 세션이 기록한 실제 작업 디렉터리(`cwd`)를 [`SameRepoPath`](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/session/delete.go#L147-L158)로 다시 비교해 다른 저장소 쌍을 거부한다. 디렉터리 이름이 아니라 세션 안의 `cwd`가 저장소 식별의 근거다.

## 정리

- 세션은 `~/.opencodereview/sessions/<인코딩된 저장소 경로>/<UUID>.jsonl` 하나이며, 파일 결과 체크포인트와 재개 계보는 즉시 flush하고 대화 기록은 버퍼에 모았다가 쓴다
- 경로 인코딩은 `/`와 `-`를 구별하지 않아 디렉터리가 겹칠 수 있으므로, 비교와 삭제는 세션이 기록한 `cwd`를 다시 확인한다
- 리뷰 세션은 `session_end`에 `ocr.run-manifest/v1` 커버리지를 남기고, 종료 상태는 코멘트 수가 아니라 `selected` 대비 `failed` 비율로만 정한다
- 재개는 부모 manifest가 완료로 기록한 지문만 재사용하며, 입력 모드, 저장소, 입력 산출물, 규칙 해시, 커밋 끝점, 명시하지 않은 provider와 model 변경 중 하나라도 어긋나면 세션을 만들기 전에 전체를 거부한다
- 비교는 줄 번호 대신 경로, 분류, 코드 조각으로 지적을 맞추고, 이후 실행이 실제로 리뷰하지 않은 파일의 지적은 Resolved가 아닌 Not reviewed로 분리한다
- 뷰어는 GET 전용 읽기 서버이며 Host 헤더 허용 목록으로 DNS rebinding을, CSP와 프레임 차단 헤더로 주입과 clickjacking을 막고, fixed와 ignored 표시는 브라우저 `localStorage`에만 둔다
