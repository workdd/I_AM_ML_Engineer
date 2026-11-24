OpenTelemetry 연동과 GitHub Action 구성을 다룬다.

## OpenTelemetry 텔레메트리

`internal/telemetry`는 OpenTelemetry로 span, metric, 이벤트를 내보낸다. 기본값은 꺼짐이다.

### 코드 분석: 설정 우선순위

[telemetry/config.go L45-L62](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/telemetry/config.go#L45-L62)

```go
func resolveEnv(cfg *Config) {
	if os.Getenv("OCR_ENABLE_TELEMETRY") == "1" {
		cfg.Enabled = true
	}
	if v := os.Getenv("OTEL_SERVICE_NAME"); v != "" {
		cfg.ServiceName = v
	}
	if v := os.Getenv("OTEL_EXPORTER_OTLP_ENDPOINT"); v != "" {
		cfg.Exporter = "otlp"
		cfg.OTLPEndpoint = v
	}
	if v := os.Getenv("OTEL_EXPORTER_OTLP_PROTOCOL"); v != "" {
		cfg.OTLPProtocol = v
	}
	if os.Getenv("OCR_CONTENT_LOGGING") == "1" {
		cfg.ContentLog = true
	}
}
```

우선순위는 기본값, `~/.opencodereview/config.json`의 `telemetry` 키, 환경 변수 순이다([telemetry/config.go L113-L125](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/telemetry/config.go#L113-L125)). 환경 변수는 켜는 방향으로만 작동한다. `OCR_ENABLE_TELEMETRY=0`은 설정 파일의 `enabled: true`를 끄지 못한다.

| 설정 | 환경 변수 | 설정 파일 키 | 기본값 |
|---|---|---|---|
| 사용 여부 | `OCR_ENABLE_TELEMETRY=1` | `telemetry.enabled` | 꺼짐 |
| exporter | `OTEL_EXPORTER_OTLP_ENDPOINT`가 있으면 `otlp` | `telemetry.exporter` | `console` |
| endpoint | `OTEL_EXPORTER_OTLP_ENDPOINT` | `telemetry.otlp_endpoint` | 없음 |
| 프로토콜 | `OTEL_EXPORTER_OTLP_PROTOCOL` | 없음 | `grpc` |
| 서비스 이름 | `OTEL_SERVICE_NAME` | 없음 | `open-code-review` |
| 상위 trace | `TRACEPARENT` | 없음 | 없음 |

`TRACEPARENT`는 텔레메트리 문서에 없는 항목이다. `ContextWithTraceParentFromEnv`가 이 환경 변수의 W3C traceparent 값을 읽어 `review.run`과 `scan.run` span을 상위 trace의 하위로 만든다([telemetry/span.go L35-L44](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/telemetry/span.go#L35-L44)). CI 파이프라인이나 ACP 어댑터 같은 상위 프로세스가 OCR 실행을 자기 trace에 연결할 수 있다.

### exporter 초기화

`Init`은 `main.go`에서 명령 실행 전에 한 번 호출되고, 켜져 있으면 종료 시 5초 제한으로 flush한다([main.go L22-L25](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/cmd/opencodereview/main.go#L22-L25)). exporter는 `console`이면 stdout 출력용 exporter를, `otlp`면 프로토콜에 따라 gRPC 또는 HTTP exporter를 만든다. 지원하지 않는 프로토콜은 경고 후 gRPC로 대체한다([telemetry/exporter.go L86-L96](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/telemetry/exporter.go#L86-L96)).

gRPC endpoint는 `http://`로 시작할 때만 평문이고, 스킴이 없는 `host:port`는 TLS로 연결한다([telemetry/exporter.go L41-L50](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/telemetry/exporter.go#L41-L50)). HTTP 프로토콜에서는 endpoint를 기본 URL로 보고 `/v1/traces`, `/v1/metrics`를 붙인다.

> **주의**: 텔레메트리 문서의 Tempo 예제는 TLS 설정이 없는 로컬 collector에 `OTEL_EXPORTER_OTLP_ENDPOINT=localhost:4317`을 지정한다. 코드대로라면 스킴 없는 주소는 TLS로 연결하므로 이 예제는 `http://localhost:4317`이어야 동작할 것으로 보인다(추정, 실행 확인 안 함).

> **주의**: `console` exporter는 stdout에 span을 출력한다. `--format json`의 "stdout에는 JSON 문서 하나"라는 보장은 `stdout.Writer()`를 거치는 진행 줄만 다루므로, 종료 시 flush되는 span 출력과는 충돌할 수 있다(추정, 실행 확인 안 함). 기계 판독 출력과 함께 쓸 때는 `otlp` exporter가 안전하다.

### span, metric, 문서와의 차이

span은 `review.run`(또는 `scan.run`) 아래에 `diff.parse`, 묶음별 `subtask.execute.group.<key>`, 라운드별 `main.loop`, 그리고 `llm.request`, `tool.execute.<이름>`이 달린다. 이벤트는 `event.<이름>` 이름의 즉시 종료 span이다([telemetry/events.go L22-L30](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/telemetry/events.go#L22-L30)). 텔레메트리가 꺼져 있으면 `StartSpan`은 no-op span을 돌려주므로 호출하는 쪽은 켜짐 여부를 검사하지 않는다.

코드를 문서와 대조하면 차이가 세 군데 있다.

| 항목 | 문서 | 코드 |
|---|---|---|
| `review.run` 속성 | `error`만 기재 | `review.repo`, `review.from`, `review.to`, `review.model`도 설정([review_cmd.go L291-L294](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/cmd/opencodereview/review_cmd.go#L291-L294)) |
| `ocr.review.duration_seconds` | 초 단위 histogram | `Int64Histogram`에 `int64(dur.Seconds())`를 기록해 1초 미만은 버림([telemetry/metrics.go L83-L91](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/telemetry/metrics.go#L83-L91)) |
| review 1회의 기록 횟수 | 언급 없음 | 아래 설명 |

세 번째 항목은 코드 읽기로 찾은 것이다. review 경로에서 `RecordReviewDuration`은 `dispatchSubtasks`의 defer([agent.go L636-L638](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/agent/agent.go#L636-L638))와 `emitRunResult`([shared.go L831](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/cmd/opencodereview/shared.go#L831)) 두 곳에서 호출된다. `RecordCommentsGenerated`도 `agent.Run`([agent.go L417-L419](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/agent/agent.go#L417-L419))과 `emitRunResult`([shared.go L832-L834](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/cmd/opencodereview/shared.go#L832-L834)) 양쪽에 있다. 한 번의 review에서 둘 다 실행되면 duration histogram에 표본이 두 개 들어가고 `ocr.comments_generated_total`은 두 배로 늘어난다(추정, 실행 확인 안 함).

`ContentLogging()` 함수는 정의되어 있지만 호출하는 곳이 없다. 문서도 `content_logging`을 "reserved"라고 밝히며, prompt와 응답 본문은 텔레메트리로 나가지 않는다. 본문이 필요하면 13장의 세션 기록이나 `OCR_RAW_LOGGING=1`을 쓴다.

## GitHub Action

저장소 루트의 `action.yml`은 PR에 인라인 코멘트와 요약 코멘트를 다는 composite action이다. 파일이 1119줄로 길지만, OCR 실행 자체는 한 단계다.

### 코드 분석: Run OpenCodeReview 단계

[action.yml L989-L1004](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/action.yml#L989-L1004)

```bash
ARGS=(--from "${RANGE_FROM:-$MERGE_BASE}" --to "${HEAD_SHA}")
# stream_progress 'false' (default): agent audience, stderr captured to
# the log file without a live tee. Opt-in 'true': no --audience agent,
# so human-audience routing sends [ocr] progress lines to stderr; a
# FIFO feeds a background tee so progress streams live into the
# workflow log while still landing in the file for artifacts and the
# posting step. The tee is a real background job rather than process
# substitution so `wait` below guarantees the file is fully flushed
# before anything reads it. Neither mode disturbs $?.
[ "${STREAM_PROGRESS:-false}" = "true" ] || ARGS+=(--audience agent)
ARGS+=(--format json --timeout "$REVIEW_TASK_TIMEOUT")
[ -n "$OCR_REVIEW_CONCURRENCY" ] && ARGS+=(--concurrency "$OCR_REVIEW_CONCURRENCY")
[ -n "$OCR_BACKGROUND" ] && ARGS+=(--background "$OCR_BACKGROUND")
[ -n "$OCR_RULE" ] && ARGS+=(--rule "$OCR_RULE")
[ -n "${EFFORT:-}" ] && ARGS+=(--effort "$EFFORT")
[ -n "${MAX_TOKENS_BUDGET:-}" ] && ARGS+=(--max-tokens-budget "$MAX_TOKENS_BUDGET")
```

- 출력 형식은 SARIF가 아니라 JSON이다. 결과는 `/tmp/ocr-result.json`에, stderr는 `/tmp/ocr-stderr.log`에 저장된다.
- `stream_progress`가 참이면 `--audience agent`를 빼서 진행 줄을 stderr로 받고, FIFO와 `tee`로 워크플로 로그에 실시간 표시하면서 파일에도 남긴다. 앞의 라우팅 표에서 `json`과 `human` 조합이 stderr로 진행 줄을 보내는 성질을 그대로 이용한다.
- `--analysis`는 붙이지 않는다. Action이 다는 PR 코멘트에는 12장의 Behavior/Impact 보고서가 들어가지 않는다.
- 비교 범위는 merge-base부터 PR head SHA까지다. `checkpoint_range`를 켜면 이전 실행이 요약 코멘트에 기록한 head부터 시작하는 증분 범위(`RANGE_FROM`)를 쓴다.

이후 단계는 종료 코드로 갈린다. 0이 아니면 결과 파일을 artifact로 올리고 job을 실패시키며 PR 코멘트는 달지 않는다. 0이면 `actions/github-script`가 `scripts/github-actions/post-review-comments.js`를 불러 JSON을 PR 리뷰로 바꾼다([action.yml L1036-L1119](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/action.yml#L1036-L1119)).

| 입력 | 기본값 | 역할 |
|---|---|---|
| `sticky_summary` | `true` | 요약 코멘트를 새로 달지 않고 기존 것을 갱신 |
| `incremental` | `false` | 기존 봇 코멘트와 줄 범위가 겹치지 않는 새 코멘트만 추가 |
| `incremental_overlap_threshold` | `0.6` | 여러 줄 코멘트의 겹침 판정 IoU 기준 |
| `review_comment_batch_size` | `50` | `createReview` 한 번에 담는 인라인 코멘트 수 |
| `route_severity_below` | 빈 값 | 지정 심각도 이하를 인라인 대신 요약으로 이동 |
| `resolve_outdated` | `false` | GitHub가 outdated로 표시한 자기 스레드를 resolve. `report`는 기록만 |

`route_*` 입력은 설명에 "fail-open: never drops a finding"이라고 적혀 있다. 결과를 버리지 않고 인라인 코멘트에서 요약 코멘트로 옮길 뿐이다. `resolve_outdated: true`는 저장소 쓰기 권한(`contents: write`)이 필요하다고 입력 설명이 따로 경고한다. `review_comment_batch_size` 설명에는 한 요청에 인라인 코멘트 71개를 넣었다가 실패한 사례가 기본값 50의 근거로 적혀 있다.

CI 문서의 SARIF 업로드 예제와 이 Action은 별개의 연동 경로다. Code Scanning 경고로 관리하려면 문서 예제처럼 `--format sarif`를 직접 실행하고, PR 대화에서 코멘트로 받으려면 Action을 쓴다.
