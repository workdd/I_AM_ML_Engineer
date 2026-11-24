`ocr review --analysis`는 라인 단위 리뷰(Review)와 별개로, 커밋이나 브랜치 범위의 변경이 호출자와 사용자 흐름에 어떤 동작 차이를 만드는지 조사하는 두 번째 에이전트(Analysis)를 함께 실행한다. 이 장은 그 조사 작업을 구현한 `internal/behaviortask` 패키지와, Review와 Analysis를 병렬로 묶는 `internal/workflow` 패키지를 코드 수준에서 읽는다. 핵심은 "모델이 무엇을 읽었는지"를 관찰(observation) 기록으로 남기고, 보고서는 그 기록 ID만 인용할 수 있게 한 구조다.

## --analysis 옵션과 실행 조건

Analysis는 기본값이 꺼져 있는 옵션이다. 켜면 Review 파이프라인은 그대로 돌고, 같은 비교 대상에 대해 독립된 조사 작업이 하나 더 붙는다. 패키지 README는 이 작업을 "a single, read-only investigation using `llmloop.Task`"라고 설명하고, `ocr review`나 `ocr scan`의 기본 동작과 비용은 바꾸지 않는다고 밝힌다([behaviortask/README.md](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/behaviortask/README.md)).

Review가 "이 줄에 결함이 있는가"를 묻는다면 Analysis는 "이 비교 전후로 무엇이 달라졌고 누가 영향을 받는가"를 묻는다. 시스템 프롬프트 첫 문단이 이 구분을 명시한다.

```markdown
Analyze behavior and impact for one supplied code comparison. Reconstruct concrete before/after scenarios and follow them to callers, consumers, and user workflows. Do not perform a general finding hunt or paraphrase the diff file by file.
```

출처: [prompts/behavior_system.md L3](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/internal/config/template/prompts/behavior_system.md#L3)

### 코드 분석: validateAnalysisOptions

`cmd/opencodereview/analysis.go`의 검증 함수는 Analysis를 켤 수 있는 조건을 정한다.

[analysis.go L32-L41](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/cmd/opencodereview/analysis.go#L32-L41)

```go
	if opts.resume != "" || opts.preview {
		return fmt.Errorf("--analysis cannot be combined with --resume or --preview")
	}
	if reviewModeFromOptions(opts) == session.ReviewModeWorkspace {
		return fmt.Errorf("--analysis is not available in workspace mode; ...")
	}
	format := strings.ToLower(strings.TrimSpace(opts.outputFormat))
	if format != "text" && format != "json" {
		return fmt.Errorf("--analysis supports only --format text or json")
	}
```

(workspace 오류 문자열은 길어서 `...`로 줄였다.)

- 이 블록 앞에서 `--analysis-max-tokens-budget` 음수 검사가 `--analysis` 여부와 관계없이 먼저 실행된다([analysis.go L26-L28](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/cmd/opencodereview/analysis.go#L26-L28)).
- workspace 모드(커밋되지 않은 변경)는 거부된다. Analysis는 old 쪽과 new 쪽을 모두 불변 커밋 SHA로 고정해야 하는데, 작업 트리는 고정할 대상이 없기 때문이다(입력 준비 절에서 다시 본다).
- `--resume`과 함께 쓸 수 없다. Analysis는 매번 새로 조사하도록 설계되어(추정), 실패 시 CLI는 "--analysis requires a fresh run"이라는 안내를 낸다([review_cmd.go L372-L373](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/cmd/opencodereview/review_cmd.go#L372-L373)).
- 출력은 text와 json만 허용된다. SARIF는 라인 단위 결과만 담는 형식이라 서술형 보고서를 실을 자리가 없다(SARIF는 14장에서 다룬다).

### 관련 플래그

Analysis에 영향을 주는 플래그는 전용 두 개와 Review와 공유하는 세 개다.

| 플래그 | 기본값 | Analysis에서의 의미 | 근거 |
|---|---|---|---|
| `--analysis` | `false` | Behavior/Impact 조사를 Review와 함께 실행 | [shared_flags.go L229](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/cmd/opencodereview/shared_flags.go#L229) |
| `--analysis-max-tokens-budget` | `0` (무제한) | Analysis 전체 입력+출력 token 상한 | [shared_flags.go L228](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/cmd/opencodereview/shared_flags.go#L228) |
| `--max-tokens` | 템플릿 기본값 | 요청 하나의 prompt token 상한. Review와 같은 값 사용 | [analysis.go L142](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/cmd/opencodereview/analysis.go#L142) |
| `--max-git-procs` | `16` | Review와 Analysis가 합쳐서 쓰는 Git 하위 프로세스 수 | [shared_flags.go L58](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/cmd/opencodereview/shared_flags.go#L58) |
| `--timeout` | `15` (분) | Analysis 실행 전체의 deadline | [analysis.go L65](https://github.com/alibaba/open-code-review/blob/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd/cmd/opencodereview/analysis.go#L65) |

마지막 행은 문서에 드러나지 않는 동작이다. CLI 문서는 `--timeout`을 "Per-subtask budget"으로 설명하고 effort의 리뷰 라운드 수만큼 선형으로 늘어난다고 적는다. 그런데 Analysis 요청은 `time.Duration(opts.concurrentTaskTimeout) * time.Minute`을 그대로 받으며 effort 배수를 곱하지 않는다. 즉 기본값에서 Review 묶음 하나는 30분(medium, 2라운드)을 쓸 수 있지만 Analysis 전체는 15분 안에 끝나야 한다.

Review와 Analysis는 비교 대상 해석 결과(`codeTools.resolution`)만 공유하고, 파일 선별 결과나 코멘트는 공유하지 않는다. README는 이를 "It ignores caller-supplied diffs and does not use Review's selected-file list"라고 적는다. Review가 규칙상 건너뛴 파일도 Analysis는 변경 목록에 넣고 볼 수 있다.

## 정리

- `--analysis`는 commit 또는 range 비교에만 쓸 수 있고 출력은 text와 json만 지원한다. workspace, `--resume`, `--preview`, SARIF와는 함께 쓸 수 없다.
- `workflow.Run`은 Review와 Analysis를 두 goroutine으로 병렬 실행한다. Analysis 실패는 종료 코드를 바꾸지 않지만 취소는 두 작업에 모두 적용된다.
- Analysis는 Review의 diff와 파일 선별 결과를 쓰지 않고, 확정된 SHA로 전체 변경 목록을 다시 만든다. 자격 증명으로 보이는 경로, 심볼릭 링크, `--exclude` 대상은 모든 도구에서 읽을 수 없다.
- 모든 도구 결과는 `obs-NNNNNN` ID의 관찰 기록으로 `observations.jsonl`에 먼저 저장된다. 보고서 검증은 인용 ID가 존재하고 실패하지 않았는지만 확인하며, 문장의 사실 여부는 검사하지 않는다.
- Analysis 동시 요청은 하나로 제한되고, 300 라운드와 선택적 token 예산을 넘으면 `submit_analysis`만 가능한 마지막 라운드가 한 번 열린다. 실행 시간은 effort 배수가 붙지 않은 `--timeout` 분 단위로 제한된다.
- 실행 상태(`completed`, `partial`, `skipped`, `failed`)와 모델의 자기 평가(`declared_completion`)는 서로 다른 값이며, 자기 평가는 실행 상태를 덮지 못한다.
