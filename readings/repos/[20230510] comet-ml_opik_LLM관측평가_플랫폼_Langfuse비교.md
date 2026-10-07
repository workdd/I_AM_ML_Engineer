# Opik: LLM 관측·평가 플랫폼과 Langfuse 비교

- **레포**: [comet-ml/opik](https://github.com/comet-ml/opik) · [문서](https://www.comet.com/docs/opik/)
- **비교 대상**: [langfuse/langfuse](https://github.com/langfuse/langfuse)
- **제작**: Comet (기여자 171명, 익명 포함 · 2026-10-07 기준)
- **공개**: 2023-05-10 저장소 생성일 (Python·TypeScript SDK, Java 백엔드, Apache-2.0, 별 22,412개 · 2026-10-07 기준)
- **읽은 날짜**: 2026-10-07 (Opik 커밋 `f217a86`, 릴리즈 `2.2.92` / Langfuse 커밋 `1a21a42`, 릴리즈 `v4.53.0`)
- **태그**: #LLMObservability #LLMasJudge #Evaluation #Langfuse #ClickHouse #SelfHosting

## 한 줄 요약

LLM 앱과 에이전트의 호출을 트레이스로 남기고, 데이터셋 실험과 LLM 심판으로 채점하고, 운영 중인 트레이스에 심판 규칙을 걸어 모니터링하는 플랫폼이다. Langfuse와 기능 범위는 거의 같고, 실제 차이는 **라이선스 범위, 상태 DB 종류, 평가 지표 제공 방식** 세 가지다.

## 기능 범위

| 영역 | 내용 |
|---|---|
| 트레이싱 | 함수에 `@track` 을 붙이면 중첩 호출이 트리로 남는다. OpenTelemetry 수신과 프레임워크별 연동을 제공한다 |
| 오프라인 평가 | 데이터셋 → 실험 → 지표. 실험끼리 나란히 비교하고 PyTest 연동으로 CI에 붙인다 |
| LLM 심판 지표 | SDK에 내장 (아래 절) |
| 운영 모니터링 | 운영 트레이스에 심판 규칙을 자동 적용하고 대시보드로 추이를 본다 |
| 그 밖의 기능 | 프롬프트 관리, 플레이그라운드, 프롬프트 자동 최적화기(`opik_optimizer`), 가드레일, MCP 서버 |

README는 하루 4천만 트레이스 처리를 내세우지만 측정 조건은 적혀 있지 않다.

## 내장 심판 지표

`sdks/python/src/opik/evaluation/metrics/llm_judges/` 아래에 있는 지표들이다.

| 지표 | 용도 |
|---|---|
| `hallucination` | 근거에 없는 내용을 답했는지 판정 |
| `context_precision`<br>`context_recall` | 검색된 근거가 정답에 필요한 내용을 얼마나 정확히, 빠짐없이 담았는지 판정 |
| `answer_relevance`<br>`usefulness` | 질문에 맞는 답인지, 쓸모 있는 답인지 판정 |
| `factuality` | 사실 일치 판정 |
| `g_eval`<br>`g_eval_presets` | 평가 기준을 글로 주면 채점 단계를 만들어 채점 |
| `llm_juries` | 여러 심판 모델의 판정을 종합 |
| `trajectory_accuracy` | 에이전트가 거친 행동 경로를 채점 |
| `moderation`<br>`structure_output_compliance`<br>`syc_eval` | 유해성, 출력 형식 준수, 아첨 성향 판정 |

심판 모델 호출은 LiteLLM을 거친다(`litellm_chat_model.py`). 기본 모델은 `gpt-5-nano` 이고, `model_name` 에 LiteLLM 모델명을 넘기면 바뀐다. 그래서 OpenAI 호환 API만 있으면 자체 호스팅 모델도 심판으로 쓸 수 있다.

**서버 없이 지표만 쓸 수 있다.** `pip install opik` 후 지표 클래스를 가져와 점수를 매기면 된다. 이 점이 Langfuse와 가장 크게 갈리는 지점이다.

## 자체 호스팅 구성

`deployment/docker-compose/docker-compose.yaml` 과 Helm 차트 기준이다.

```
frontend ── backend (Java) ──┬── MySQL 8.4    메타데이터 (프로젝트·데이터셋·프롬프트)
                             ├── ClickHouse   트레이스·스팬·점수
                             ├── ZooKeeper    ClickHouse 분산 DDL 조정
                             ├── Redis        캐시·작업 큐
                             └── MinIO (S3)   첨부 파일
python-backend (온라인 평가·플레이그라운드 코드 실행)
guardrails-backend, otel-collector, jaeger (선택)
```

`./opik.sh` 하나로 뜨지만 단일 노드에서도 컨테이너가 10개를 넘는다.

### ClickHouse는 빠지지 않는다

- `apps/opik-backend/config.yml` 의 `databaseAnalytics` 는 ClickHouse 드라이버로 고정돼 있고 다른 DB를 고르는 설정이 없다
- 분석 DB 마이그레이션 136개가 `ReplacingMergeTree` 같은 ClickHouse 엔진으로 작성돼 있다
- 최근 마이그레이션(000126~000129)이 `ON CLUSTER '{cluster}'` 를 쓴다. 이 구문은 클러스터 설정과 조정 서버(ZooKeeper 또는 ClickHouse Keeper)가 있어야 실행되므로, 단일 노드로 띄워도 ZooKeeper가 따라온다. Helm 기본값도 `zookeeper.enabled: true` 다

### 상태 DB는 MySQL로 고정이다

환경변수 `STATE_DB_DRIVER_CLASS`, `STATE_DB_PROTOCOL` 이 있어 바꿀 수 있어 보이지만 PostgreSQL로는 바꿀 수 없다.

- `pom.xml` 의 JDBC 드라이버는 `mysql-connector-j` 하나다
- 상태 DB 마이그레이션 109개가 MySQL 문법이다. 첫 파일에만 `ENUM(` 과 `ON UPDATE CURRENT_TIMESTAMP(6)` 가 들어 있다
- DAO 쿼리가 MySQL 함수를 쓴다. `PromptDAO.java` 하나에 `JSON_OBJECT` 4곳, `JSON_ARRAYAGG` 3곳이 있다
- PostgreSQL 지원 요청은 두 번 닫혔다. [#2402](https://github.com/comet-ml/opik/issues/2402)에서 메인테이너는 "MySQL 전용 문법을 써서 드라이버만으로는 안 되고, 여러 상태 DB를 지원할 계획이 없으며, 장기적으로 상태 일부를 ClickHouse로 옮긴다"고 답했다. [#3886](https://github.com/comet-ml/opik/issues/3886)에서도 "계획 없음, 관리형 MySQL을 쓰라"고 답했다

드라이버 이름만 바꾸면 SQL은 그대로라 실행되지 않는다. 포크해서 옮기면 마이그레이션 109개와 DAO 전체를 다시 써야 하고, 릴리즈가 거의 매일 나오므로 업그레이드할 때마다 충돌이 난다.

## Langfuse 비교

두 저장소를 같은 날 확인했다.

| 항목 | Opik | Langfuse |
|---|---|---|
| 소유 | Comet | ClickHouse, Inc. (LICENSE 저작권 표기 기준) |
| 라이선스 | 전체 Apache-2.0 | 핵심 MIT, `ee/` 폴더는 상용 라이선스 |
| 상용 라이선스 기능 | 없음 | SSO, 감사 로그, 관리자 API, 데이터 보존 기간, UI 커스터마이징, 도메인 검증 |
| 메타데이터 DB | MySQL | PostgreSQL |
| 트레이스 DB | ClickHouse + ZooKeeper | ClickHouse (docker-compose에 ZooKeeper 없음) |
| 기본 구성 | 컨테이너 10개 이상 | web, worker, PostgreSQL, ClickHouse, Redis, MinIO 6개 |
| 백엔드 언어 | Java + Python | TypeScript |
| 평가 지표 | SDK에 심판 지표 내장, 서버 없이 사용 가능 | 화면에서 심판 평가기를 설정. 코드 기반 지표는 Ragas 같은 외부 라이브러리와 조합 (미검증) |
| 그 밖의 기능 | 프롬프트 최적화기, 가드레일 | AI 게이트웨이 |
| 별 | 22,412 | 35,443 |

Langfuse의 상용 기능 목록은 `web/src/ee/features/` 와 `worker/src/ee/` 의 디렉터리 이름에서 읽었다. 디렉터리 안에서 어디까지가 막혀 있는지는 확인하지 않았다.

## 읽을 때 감안할 것

- **README의 경쟁사 비교표는 Comet이 직접 작성했다.** Opik이 모든 칸에서 Yes인 표라 선택 근거로는 약하다
- **자체 호스팅판의 인증 수준은 확인하지 못했다.** Helm 기본값이 `basicAuth: false` 다. 사용자·권한 관리가 없다고 가정하고 네트워크 접근 제한부터 설계하는 편이 안전하다
- **SDK가 사용 통계를 외부로 보내는지 확인하지 못했다.** 폐쇄망이나 민감 데이터 환경이면 도입 전에 끄는 설정을 찾아야 한다
- **심판 프롬프트는 영어다.** 한국어 답변에 매긴 점수가 사람 판정과 맞는지는 따로 대조해야 한다
- **업그레이드 빈도가 높다.** 릴리즈가 거의 매일 나오고 분석 DB 마이그레이션만 129번까지 쌓였다. 버전을 고정하고 올리기 전에 검증해야 한다
- GitHub 언어 통계에 Java가 나오지 않지만 백엔드는 `pom.xml` 을 쓰는 Java다

## 가져갈 지점

- **평가와 운영 모니터링은 따로 고를 수 있다.** 평가만 필요하면 Opik SDK 지표를 가져다 쓰고 서버는 띄우지 않는다. ClickHouse·ZooKeeper 운영 비용은 운영 트레이스를 쌓을 때만 값을 한다
- **운영 모니터링 플랫폼을 고를 때 기준은 이미 운영 중인 DB다.** PostgreSQL을 이미 운영한다면 Langfuse가 DB 하나를 덜 들인다. 대신 트레이스에 고객 데이터가 쌓이는데 데이터 보존 기간 기능이 상용 쪽이라, ClickHouse 정리 배치를 직접 만들어야 할 수 있다
- **LLM 호출을 HTTP 클라이언트로 직접 하는 코드에는 자동 연동이 붙지 않는다.** OpenAI SDK나 LangChain을 거치지 않는 파이프라인이라면 `@track` 이나 OpenTelemetry 스팬을 직접 심어야 한다
- **자체 호스팅 모델을 심판으로 쓸 때 LiteLLM 설정이 경로가 된다.** OpenAI 호환 엔드포인트면 `openai/<모델명>` 과 `api_base` 로 붙는 것이 LiteLLM의 일반 경로다. 직접 붙여 보지는 않았다. 사설 인증서를 쓰는 게이트웨이라면 LiteLLM 쪽 SSL 검증 설정도 맞춰야 한다
- **`llm_juries` 와 `context_precision`/`context_recall` 은 RAG 평가에 바로 옮길 만하다.** 심판 한 명의 편향을 줄이는 방식과 근거 품질을 따로 재는 방식을 직접 짜지 않아도 된다

## 결론

Opik은 "전부 공개된 Langfuse"에 가깝다. 플랫폼 전체를 들일 거라면 상태 DB가 MySQL로 고정돼 있고 ZooKeeper까지 필요하다는 점을 먼저 따져야 하고, PostgreSQL 중심 환경이면 Langfuse가 가볍다. 반대로 지금 필요한 것이 평가뿐이라면 Opik의 서버 없이 쓰는 심판 지표가 가장 싸게 시작하는 길이다. 다만 영어 심판 프롬프트의 판정이 자기 데이터에서 사람 판정과 맞는지부터 확인해야 한다.
