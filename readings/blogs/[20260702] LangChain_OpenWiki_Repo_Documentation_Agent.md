# OpenWiki: 코딩 에이전트를 위한 오픈소스 repo 문서화 에이전트

- **출처**: [LangChain Blog](https://www.langchain.com/blog/introducing-openwiki-an-open-source-agent-for-repo-documentation)
- **저장소**: [langchain-ai/openwiki](https://github.com/langchain-ai/openwiki)
- **작성자**: Brace Sproul
- **게시일**: 2026-07-01
- **읽은 날짜**: 2026-07-02
- **태그**: #OpenWiki #CodebaseWiki #AIAgent #Documentation #LangChain #DeepAgents #LangSmith

## 한 줄 요약

OpenWiki는 코드베이스를 분석해 `openwiki/` 문서 디렉토리를 만들고, `AGENTS.md`/`CLAUDE.md`가 그 wiki를 참조하도록 연결해 **코딩 에이전트가 repo 구조와 관례를 지속적으로 이해하게 만드는 CLI 에이전트**다.

## 문제의식 — 에이전트는 repo 문맥이 필요하다

코딩 에이전트가 좋은 코드를 쓰려면 단순히 현재 파일만 알아서는 부족하다. 핵심 로직이 어디 있는지, 파일들이 어떻게 연결되는지, 이 repo가 기대하는 패턴이 무엇인지 알아야 한다.

문제는 repo 문서가 유지되기 어렵다는 점이다.

- 처음 문서를 쓰는 데 시간이 든다.
- 코드가 바뀔 때마다 문서를 갱신하기 더 어렵다.
- 큰 repo와 잦은 PR 환경에서는 문서가 빠르게 낡는다.

OpenWiki는 이 작업을 자동화한다. repo wiki를 생성하고, 코딩 에이전트가 그 wiki를 찾을 수 있게 연결하며, 코드 변경에 맞춰 wiki를 갱신한다.

## 왜 `AGENTS.md`/`CLAUDE.md` 하나로는 부족한가?

대부분의 코딩 에이전트는 `AGENTS.md`, `CLAUDE.md` 같은 instruction file을 읽는다. 이 파일들은 작업 규칙, 명령어, 스타일 가이드를 담기에는 좋지만, 수십~수백 페이지짜리 repo 문서를 그대로 넣기에는 맞지 않다.

OpenWiki의 설계는 다음과 같다.

```text
AGENTS.md / CLAUDE.md
  -> "repo 문맥이 필요하면 openwiki/를 참고하라"는 짧은 안내

openwiki/
  -> 실제 코드베이스 구조, 주요 모듈, 흐름, 패턴 문서
```

즉 instruction file은 진입점이고, wiki는 durable context 저장소다. 에이전트는 매번 거대한 문서를 context에 다 넣지 않고, 필요한 시점에 관련 문서를 찾아 읽을 수 있다.

## 사용 방법

설치는 npm 기반이다.

```bash
npm install -g openwiki
```

초기 생성:

```bash
openwiki --init
```

기존 문서 업데이트:

```bash
openwiki --update
```

대화형 CLI:

```bash
openwiki
```

한 번 실행하고 종료:

```bash
openwiki -p "Summarize what you can do"
```

첫 실행 시 inference provider, API key, LLM을 설정한다. 설정과 secret은 로컬의 `~/.openwiki/.env`에 저장된다.

## 지원 provider와 관측성

OpenWiki는 다음 provider를 기본 지원한다.

| Provider | 메모 |
|----------|------|
| OpenRouter | 기본 흐름에서 open model 사용 가능 |
| Fireworks | open model serving provider |
| Baseten | model deployment provider |
| OpenAI | closed model provider |
| Anthropic | Claude 계열 provider |

LangSmith API key를 제공하면 OpenWiki 실행이 `openwiki` 프로젝트로 tracing된다. 문서 생성/갱신 중 에이전트가 어떤 파일을 읽고 어떤 판단을 했는지 확인할 수 있다는 점이 중요하다.

## GitHub Action으로 지속 갱신

OpenWiki는 일회성 문서 생성보다 **지속 갱신**을 핵심 가치로 둔다. 저장소 예시 workflow는 매일 스케줄로 실행된다.

```yaml
- name: Run OpenWiki
  run: openwiki --update --print
```

이후 `peter-evans/create-pull-request` 액션으로 `openwiki/` 변경분만 PR로 만든다.

운영 관점에서 좋은 점:

- 문서 갱신이 코드 변경 PR과 분리된다.
- 자동 변경이라도 PR review를 거칠 수 있다.
- 에이전트 instruction file은 계속 같은 wiki 경로를 참조하므로 workflow를 바꿀 필요가 없다.

## 기술 구성 메모

저장소 기준 OpenWiki는 Node.js 20+ 기반 TypeScript CLI다. 패키지 설명은 "DeepAgents documentation agent로 codebase OpenWiki를 생성·유지하는 CLI"에 가깝다.

주요 의존성:

| 의존성 | 역할 |
|--------|------|
| `deepagents` | 장기 작업형 documentation agent 기반 |
| `langchain` / `@langchain/*` | 모델 provider 연동 |
| `@langchain/langgraph-checkpoint-sqlite` | agent state/checkpoint 계열 |
| `ink` / `react` | 터미널 UI |
| `marked` | Markdown 처리 |

현재 공개 패키지 버전은 `0.0.1`이고 MIT 라이선스다.

## 의미 있는 설계 포인트

### 1. 문서를 사람용 산출물이 아니라 agent memory로 본다

OpenWiki의 문서는 README 보강용이 아니라 코딩 에이전트가 반복해서 참조할 repo memory다. 사람에게도 읽히지만, 1차 사용자는 "repo를 이해해야 하는 agent"에 가깝다.

### 2. instruction file과 wiki의 역할을 분리한다

`AGENTS.md`/`CLAUDE.md`에는 규칙과 wiki pointer만 둔다. 실제 상세 문맥은 `openwiki/`로 분리한다. 이 구조는 context window 낭비를 줄이고, 문서가 커져도 instruction file이 비대해지는 문제를 피한다.

### 3. documentation drift를 PR workflow로 관리한다

자동으로 main에 직접 쓰지 않고, 갱신 PR을 만든다. 이 방식은 hallucinated documentation이나 잘못된 repo 해석을 리뷰 단계에서 걸러낼 여지를 준다.

## 한계와 주의점

| 항목 | 설명 |
|------|------|
| 초기 릴리스 | `0.0.1` 단계라 기능과 포맷이 바뀔 수 있음 |
| 모델 품질 의존 | repo 해석과 문서 품질은 선택한 LLM에 크게 좌우됨 |
| 자동 문서의 신뢰성 | 실제 코드와 어긋난 설명이 생길 수 있어 PR 리뷰 필요 |
| secret 관리 | 로컬은 `~/.openwiki/.env`, GitHub Action은 repository secrets 관리 필요 |
| 문서 표준화 부족 | `openwiki/` 내부 포맷이 장기 표준으로 굳을지는 더 봐야 함 |

## 내 작업과의 연결

최근 정리한 OKF, LogicRAG와 같은 흐름으로 볼 수 있다.

| 자료 | 핵심 관점 |
|------|----------|
| OKF | 지식을 markdown bundle로 표현하는 개방형 포맷 |
| LogicRAG | 질문별 논리 구조를 만들어 필요한 정보를 순서대로 검색 |
| OpenWiki | repo별 durable context를 wiki로 만들어 agent가 검색하게 함 |

이 repo(`I_AM_ML_Engineer`) 자체도 이미 `readings/`, `practical_tips/`, `deep_learning/`처럼 사람이 만든 wiki 구조를 갖고 있다. OpenWiki를 붙이면 코드/문서 저장소의 구조 설명을 자동 생성하고, Codex/Claude 같은 에이전트가 작업 전 필요한 맥락을 더 빨리 찾게 만들 수 있다.

다만 개인 학습 repo에는 자동 wiki가 과할 수 있다. 실제로 더 유용한 곳은 다음 같은 환경이다.

- microservice가 많고 신규 합류자가 구조를 파악하기 어려운 repo
- agentic coding을 여러 명이 같이 쓰는 팀 repo
- PR이 잦아 문서 drift가 빠르게 생기는 product repo
- `AGENTS.md` 하나에 넣기에는 맥락이 너무 많은 monorepo

## 결론

OpenWiki의 핵심은 "에이전트에게 repo context를 매번 프롬프트로 때려 넣지 말고, 지속 갱신되는 wiki를 만들어 필요할 때 찾아보게 하자"는 것이다. `AGENTS.md`/`CLAUDE.md`는 얇은 router로 두고, 실제 지식은 `openwiki/`에 쌓는다.

코딩 에이전트가 일회성 autocomplete에서 repo-aware collaborator로 가려면, 이런 형태의 durable context layer가 점점 중요해질 가능성이 높다.
