# reef: 서빙하면서 스스로 학습하는 에이전트 인프라

- **레포**: [Human-Agent-Society/reef](https://github.com/Human-Agent-Society/reef) · [reefinfra.ai/docs](https://reefinfra.ai/docs/)
- **제작**: Human-Agent-Society (기여자 9명, 상위 3명이 커밋 60건 중 대부분)
- **공개일**: 2026-08-31 (Python, Apache-2.0, PyPI `reef-infra`, 별 74개 · 2026-09-02 기준)
- **읽은 날짜**: 2026-09-02 (커밋 `d3cb988`, 마지막 푸시 2026-09-02)
- **태그**: #ContinualLearning #SelfImprovingAgent #RL #Inference #HarnessEvolution #ArtifactVersioning #SGLang #slime

## 한 줄 요약

지속학습 백엔드다. 추론 엔진과 RL 학습 프레임워크 사이에 비어 있던 버전 관리, 무중단 반영, 가중치 밖 진화를 하나의 HTTP 서비스로 묶었다. 에이전트는 provider 대신 Reef 엔드포인트로 요청을 보내고 Reef가 그 상호작용을 평가해 가중치나 harness를 계속 갱신한다.

## 핵심 문제의식

기존 도구는 절반씩만 담당한다. vLLM이나 SGLang 같은 추론 엔진은 트래픽을 받지만 학습하지 않는다. slime이나 veRL 같은 RL 프레임워크는 학습하지만 라이브 트래픽을 받지 않는다. 그 사이에 남는 일이 있다. 어떤 버전이 어떤 응답을 냈는지 추적하고, 새 버전을 서비스 중단 없이 밀어 넣고, 가중치가 아닌 프롬프트나 스킬도 같은 방식으로 갱신하는 일이다.

Reef는 자기 위치를 이렇게 밝힌다. **"에이전트는 모델과 harness의 합이고, Reef는 harness와 모델을 실행하는 런타임 사이에 앉는다."**

사용 방식도 이 구도를 따른다. 에이전트가 provider 엔드포인트 대신 Reef의 `/v1/chat/completions`나 `/v1/messages`로 요청을 보내면 된다. OpenAI와 Anthropic 요청 본문을 그대로 받아서 클라이언트 코드를 바꿀 일이 거의 없다.

## 학습 루프 네 단계

| 단계 | 하는 일 | 구현 위치 |
|------|---------|-----------|
| 1 Serve | 에이전트 요청을 처리하고 상호작용을 기록 | `service/`, `runtime/` |
| 2 Observe | 피드백을 기록된 상호작용에 연결 | `records.py`, `train/processors/` |
| 3 Grow | 자격을 갖춘 레코드에서 업데이트를 생성 | `recipe/`, `train/` |
| 4 Commit | 후보를 평가하고 통과한 것만 발행 | `train/evaluation/`, `artifact/`, `surface/` |

## 일반 추론 엔드포인트에 더해진 네 가지

- **Scenario**: 학습 단위. 첫 요청이 시나리오를 만들고 레시피를 지정하며 시작 릴리스를 고정한다. **이 바인딩은 이후 바뀌지 않는다.** 다른 레시피명으로 다시 요청하면 HTTP 409로 거절한다.
- **Receipt**: 응답 헤더 `x-reef-agent-record-id`로 돌아오는 영수증. 나중에 이 상호작용을 지목해 피드백을 붙이는 열쇠다.
- **Report**: `/reef/report`로 보내는 평가. 숫자 `score`, 텍스트나 구조화된 `feedback`, 대상 receipt 목록을 담는다. 스칼라 하나보다 많은 신호를 읽는 레시피가 있어서 `feedback`을 따로 뒀다.
- **Release Chain**: 발행 이력. ID를 셋으로 나눠 둔 게 핵심이다. `release_id`는 Reef의 발행 결정, `content_id`는 선택된 모델이나 harness 콘텐츠, `runtime_load_id`는 구체적인 서빙 엔진의 가중치 로드를 가리킨다.

## 두 가지 학습 표면

시나리오에 묶인 레시피가 무엇을 갱신할지 결정한다.

- **가중치(weights)**: 학습 런타임이 필요하고 GPU를 쓴다. 갱신된 가중치는 서빙 엔진으로 밀려 들어가고 이후 요청은 Reef 재시작 없이 새 버전을 쓴다.
- **harness**: 규칙, 스킬, 설정, 프롬프트, 확장이 담긴 트리를 갱신한다. **GPU가 필요 없고 Reef 자체 프로세스에서 돈다.** 후보 harness와 현재 harness를 설정된 과제로 겨뤄 이겼을 때만 발행한다.

harness 쪽 사용법이 흥미롭다. 코딩 에이전트를 설치하듯 `curl | bash`로 받는다.

```bash
curl -fsS -H "Authorization: Bearer $REEF_TOKEN" \
  'http://localhost:8900/reef/harness/install?adapter=pi' | bash

reef-pi -p "fix the bug"
reef-pi report --score 0 --feedback "missed the empty-token case"
```

`reef-pi`가 실행 중 receipt를 들고 있다가 `report` 명령에 결과만 붙여 보낸다. 어댑터는 `pi`, `opencode`, `claude` 세 가지가 구현돼 있고 Claude Code 어댑터는 2026-09-02에 들어왔다.

## 무중단 반영이 가능한 이유

이 레포에서 가장 공들인 부분이다. 학습이 도는 중에도 응답의 출처를 잃지 않게 만드는 장치가 겹겹이다.

- **요청 시점에 artifact ref를 얼린다.** 요청이 처리되는 중간에 업데이트가 끝나도 그 receipt가 기록하는 내용은 바뀌지 않는다.
- **토큰 span을 검증한다.** 라이브 가중치라면 엔진에 생성된 토큰 구간마다 `runtime_load_id`를 보고하게 한다. 그 span 정보가 생성된 모든 토큰을 설명하고 얼린 릴리스와 일치할 때만 응답을 받아들인다.
- **head는 compare-and-swap으로만 움직인다.** `advance_current`는 예상 head를, `publish`는 예상 parent를 요구한다.
- **롤백이 이력을 다시 쓰지 않는다.** 과거 릴리스의 `content_id`를 활성화해 새 `release_id`로 다시 발행한다. 스텝 번호는 단조 증가를 유지한다.
- **커밋 로그가 append-only JSONL이고 fsync된 append가 커밋 시점이다.** 시나리오마다 하나씩 두고 스텝 번호·artifact ref·체크포인트 플래그·알고리즘 상태·레코드 고수위·컴팩션 삭제·메트릭을 남긴다.
- **레코드는 id 충돌 시 덮어쓰지 않고 conflict를 낸다.** 삭제가 일어나는 유일한 경로는 컴팩션이다.

지속성 아티팩트는 Git 기반이다. 시나리오당 ref 하나에 가중치 파일은 LFS 패턴으로 두고 `reef-artifact.json` 매니페스트를 붙인다. 설치에 `git-lfs`가 필요한 이유다.

## 레시피 카탈로그

레시피는 wheel에 포함되지 않고 `recipes/` 쿡북에 있으며 점 표기 클래스 참조로 선택한다.

| 워크로드 | 레시피 | 갱신 대상 |
|----------|--------|-----------|
| 테스트나 검증기로 채점되는 과제 스트림 | `recipes.sao.recipe:SAORecipe` | 모델 가중치 |
| 명시적 보고 없이 다음 상태 신호가 유용한 에이전트 트래픽 | `recipes.openclawrl.recipe:OpenClawRLRecipe` | 모델 가중치 |
| 한 문제에 대한 반복 채점 시도 | `recipes.tttd.recipe:TTTDRecipe` | 모델 가중치 |
| 학습 가능한 유도 모델 + 고정 실행기의 코드 검색 | `recipes.tttd.recipe:TTTDRecipe` | 유도 모델 가중치 |
| 스킬 풀을 진화시키는 에이전트 피드백 | `recipes.skillclaw.recipe:SkillClawRecipe` | 스킬 풀, GPU 불필요 |

## 코드 규모와 구성

| 항목 | 규모 |
|------|------|
| 전체 파일 | 1,573개 |
| 파이썬 파일 | 401개 (약 3.4MB) |
| 테스트 파일 | 122개 |
| `reef/` 본체 | 181개 파일, `train/` 80 · `service/` 20 · `harness/` 16 · `runtime/` 11 |
| `recipes/` 쿡북 | 729개 (`openclawrl/examples`만 520개) |
| 서브모듈 | `third_party/cordis`, `third_party/reef-client` |

의존하는 외부 프로젝트는 SGLang(추론), slime(가중치 학습), cordis(harness 진화)다.

## 확인한 사실

- **SAO 인용은 정확하다.** README가 예제로 쓰는 arXiv:2607.07508은 `Single-Rollout Asynchronous Optimization for Agentic Reinforcement Learning`(Zhenyu Hou, Yujiang Li, Jie Tang, Yuxiao Dong, 2026-07-08)이 맞다.
- **코드 실체가 있다.** 파이썬 401개 파일에 테스트 122개다. 문서만 있고 구현이 비어 있는 레포가 아니다.
- **활발히 움직인다.** 마지막 푸시가 읽은 당일이고 PR 번호가 이미 129번대다.

## 한계와 리스크

- **공개된 지 이틀 됐다.** 저장소 생성일이 2026-08-31이다. 코드는 그 전부터 개발됐겠지만 공개 이후 실사용 검증 기간이 사실상 없다. 열린 이슈가 37개이고 Q3 로드맵 이슈가 아직 열려 있다.
- **전체 가중치 학습은 프로세스당 하나, 단일 스레드다.** 문서가 명시한 제약이다. 업데이트를 내지 않거나 텍스트 아티팩트만 갱신하는 시나리오는 여럿 돌릴 수 있지만 가중치 학습을 여러 시나리오로 동시에 돌리려면 프로세스를 나눠야 한다.
- **스트리밍에서는 검증이 빠진다.** pass-through 스트리밍은 토큰 span 검사를 할 수 없어 `stream`이 true면 `return_meta_info`를 끄고 평범한 SSE 교환으로 기록한다. **무중단 반영의 핵심 보증이 스트리밍 경로에서는 성립하지 않는다.**
- **진행 중이던 런타임 작업은 복구되지 않는다.** 레코드와 알고리즘 상태는 로그에서 복원되지만 pending runtime work는 durably recoverable하지 않다고 문서가 밝힌다. 그래서 학습 스텝이 성공해야 배치가 확인된다.
- **비교표가 자기에게 유리하게 짜여 있다.** "버전 관리", "업데이트 중 라이브 유지", "가중치 밖 진화" 세 항목에서 추론 엔진과 RL 프레임워크가 모두 X인데, 애초에 그 도구들이 담당하는 범위가 아니다. 없는 기능이 아니라 다른 계층의 일이다. 카테고리를 자기 기능으로 정의하면 비교는 언제나 이긴다.
- **GPU 요구가 걸린다.** 가중치를 학습하는 레시피는 GPU 환경이 필요하다. harness와 스킬 진화만 쓰면 CPU로도 되지만 README 전면에 걸린 그림은 가중치 학습 쪽이다.
- **락인 구조가 강하다.** 시나리오와 레시피 바인딩은 한번 정하면 바뀌지 않고 아티팩트는 Git LFS 저장소에 쌓인다. 설계상 의도된 불변성이지만 초기에 레시피를 잘못 고르면 시나리오를 새로 파야 한다.

## 내 작업과의 연결

1. **버전과 응답의 연결을 끊지 않는 설계**

   receipt를 응답 헤더로 돌려주고 요청 시점에 artifact ref를 얼리는 방식은 평가 파이프라인에 그대로 옮길 만하다. 어떤 버전이 어떤 답을 냈는지 사후에 확실히 말할 수 있어야 A/B 비교든 회귀 추적이든 성립한다.

2. **후보가 이겼을 때만 발행하는 게이트**

   harness 진화가 현재 버전과 후보를 겨뤄 이겼을 때만 올리는 구조는 [AgentCore 노트](../blogs/%5B20260902%5D%20AWS%EB%B8%94%EB%A1%9C%EA%B7%B8_Bedrock_AgentCore_%EB%A9%80%ED%8B%B0%EB%8D%B0%EC%9D%B4%ED%84%B0%EC%86%8C%EC%8A%A4_NLP_%EC%97%90%EC%9D%B4%EC%A0%84%ED%8A%B8.md)의 CI 품질 게이트와 같은 발상이다. 자동 개선을 켤 때 반드시 있어야 하는 제동 장치다.

3. **가중치가 아닌 것도 학습 대상이다**

   스킬 풀과 harness 트리를 GPU 없이 갱신하는 경로는 현실적인 진입점이다. 모델을 건드리지 않고 프롬프트·스킬·규칙만 데이터 기반으로 갱신하는 구조는 우리 환경에서도 바로 시도해볼 수 있다.

4. **피드백을 스칼라 하나로 좁히지 않기**

   `score`와 별개로 텍스트나 구조화된 `feedback`을 받는 스키마는 참고할 만하다. 점수만 남기면 왜 틀렸는지가 사라져 나중에 원인 분석을 못 한다.

## 결론

만들려는 것이 분명한 레포다. **추론과 학습 사이에서 아무도 책임지지 않던 계층**, 즉 버전 이력·무중단 교체·가중치 밖 아티팩트 진화를 한 서비스로 묶었다. 릴리스 ID를 셋으로 쪼개고 토큰 span까지 검증하는 대목을 보면 "동작하는 중에 바뀌는 시스템"의 어려움을 정확히 알고 설계했다.

다만 공개 이틀 차이고 검증 기간이 없다. 스트리밍 경로에서 핵심 보증이 빠지는 점, 전체 가중치 학습의 단일 시나리오 제약, GPU 요구도 실제 도입 전에 따져야 한다. 지금은 **아이디어와 구조를 참고할 대상**으로 두고 harness나 스킬 진화처럼 GPU 없이 되는 경로부터 작게 시험해보는 편이 맞다.
