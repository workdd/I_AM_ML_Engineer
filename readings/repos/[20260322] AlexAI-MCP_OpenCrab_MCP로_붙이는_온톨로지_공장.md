# OpenCrab — 에이전트에 MCP로 붙이는 온톨로지 공장

- **레포**: [AlexAI-MCP/OpenCrab](https://github.com/AlexAI-MCP/OpenCrab) · [opencrab.sh](https://opencrab.sh)
- **제작**: AlexAI-MCP (컨트리뷰터 5명 · 커밋 37개 중 25개가 `claude` 명의)
- **공개일**: 2026-03-22 (Python, README·pyproject는 MIT 표기이나 LICENSE 파일 없음, ⭐102 · 2026-08-20 기준)
- **읽은 날짜**: 2026-08-20 (커밋 `d34352c`, 마지막 푸시 2026-06-03)
- **태그**: #MCP #Ontology #KnowledgeGraph #GraphRAG #Neo4j #BM25 #ReBAC #Crawler

## 핵심 문제의식

에이전트에 붙는 지식은 대개 벡터 인덱스 한 겹이다. 문서를 잘라 임베딩하고 비슷한 조각을 꺼내오는 구조로는 "이 주장의 근거가 무엇인가", "이 값을 바꾸면 어디가 흔들리는가", "이 사용자가 이 문서를 볼 권한이 있는가"를 물을 수 없다. OpenCrab은 그 답을 검색 계층이 아니라 문법 계층에서 찾는다. 노드와 엣지가 아무렇게나 생기지 못하도록 9개 공간(space)과 공간 사이에 허용된 관계만 상수로 못 박아 뒀다. 그 문법을 MCP 도구 30개로 에이전트에 그대로 노출한다.

레포는 두 덩어리를 한 제품처럼 설명한다. LocalCrab은 로컬에서 도는 온톨로지 공장이고 opencrab.sh는 완성된 팩을 유통하는 SaaS다. 공개된 것은 앞쪽뿐이며 SaaS 구현은 비공개라고 문서에 밝혀 뒀다.

## 설계 축

| 축 | 선택 | 이유 |
| --- | --- | --- |
| 문법 우선 | 스키마 없는 그래프 대신 9 space · 노드 타입 30 · 관계 38개를 파이썬 상수로 고정 | 검증기가 거절할 수 있어야 에이전트가 쓰레기를 못 넣는다 |
| 로컬 우선 | 기본 저장소가 SQLite + Chroma 임베디드 + JSON 파일 | Docker도 Neo4j도 없이 `pip install` 하나로 돈다 |
| 근거 결속 | Claim은 Evidence에 supports/contradicts로 묶여야 승격된다 | 출처 없는 주장을 그래프에 남기지 못하게 막는다 |
| 승격 수명주기 | extracted → candidate → validated → promoted, 아니면 rejected | LLM이 뽑아낸 것을 본 그래프에 바로 쓰지 않는다 |
| 권한 내장 | ReBAC 객체 타입 7종 × 권한 6종을 문법에 포함 | 못 볼 것을 걸러내는 일을 앱이 아니라 온톨로지가 한다 |

## 9개 공간

| 공간 | 담는 것 |
| --- | --- |
| subject | User · Team · Org · Agent |
| resource | Project · Document · File · Dataset · Tool · API · CrawlRun |
| evidence | TextUnit · LogEntry · Evidence |
| concept | Entity · Concept · Topic · Class |
| claim | Claim · Covariate · CollectionCompleteness |
| community | Community · CommunityReport |
| outcome | Outcome · KPI · Risk |
| lever | Lever |
| policy | Policy · Sensitivity · ApprovalRule |

공간 쌍 11개에만 관계를 허용한다. subject→resource는 owns·can_edit 같은 권한 관계, evidence→claim은 supports·contradicts·timestamps, lever→outcome은 raises·lowers·stabilizes·optimizes 식이다. 문법에 없는 조합은 검증기가 거절한다. 여기에 메타데이터 차원 12개가 모든 노드·엣지에 직교로 얹힌다. 존재(identity·provenance·lineage), 품질(confidence·freshness·completeness), 관계(dependency·sensitivity·maturity), 행동(usage·mutation·effect) 네 층이다.

## 구조

```
mission.json
    |
    v
CrabHarness  planner -> registry -> delegation -> runtime
    |                                                |
    |                                        Codex 워커 (서브프로세스)
    v
검증 3관문  완결성 + 의미 점수 + autoresearch 판정
    |
    v
promotion-package.json
    |
    v
OpenCrab  MCP 도구 30개 / 문법 검증 / 하이브리드 질의
    |
    +-- 로컬 저장소  SQLite graph.db · Chroma · JSON 문서 · SQLite 상태
    |
    v
Neo4j 리플레이 -> OpenCrab Pack v1 ZIP -> opencrab.sh
```

| 경로 | 줄 수 | 역할 |
| --- | --- | --- |
| `opencrab/mcp/tools.py` | 1,590 | MCP 도구 30개의 스키마와 디스패치 |
| `apps/api/main.py` | 1,033 | 데모 FastAPI. 라우트 12개, `POST /mcp` Streamable HTTP 포함 |
| `opencrab/ontology/query.py` | 549 | 하이브리드 질의 엔진 |
| `opencrab/cli.py` | 493 | `opencrab` 명령 8개 |
| `opencrab/ontology/identity.py` | 400 | 별칭 등록, 중복 후보 제안·해소 |
| `opencrab/stores/local_graph_store.py` | 356 | SQLite 그래프 저장소 |
| `opencrab/ontology/impact.py` | 322 | I1~I7 영향 분석 |
| `opencrab/grammar/` | 722 | manifest · validator · glossary |
| `crabharness/` | 2,879 | 미션 플래너, 런타임, 검증, 승격 패키지 |
| `apps/web/` | 1,179 | Obsidian 풍 그래프 UI (Next.js) |

파이썬 75개 파일 15,437줄, 테스트 5개 파일 1,435줄에 테스트 함수 134개다.

## 질의 경로

`ontology_query`는 세 갈래를 합친다. Chroma 벡터 검색, 인메모리 BM25, 그래프 이웃 확장이다. 합치는 방식은 RRF(k=60)다. 텍스트가 무거운 질의에는 BM25 교차 점수를 섞어 `0.7 × RRF + 0.3 × BM25`로 최종 점수를 낸다. 그래프 확장은 엣지 타입별 가중치(SUPPORTS·DEPENDS_ON 0.7, CONTAINS·INFLUENCES 0.65, RELATED_TO 0.6, CONTRADICTS 0.5)에 홉당 0.85 감쇠를 곱한다.

토크나이저는 한글 토큰을 2·3그램으로 쪼갠다. 질의 힌트 목록에는 "이유·변경·개정·배경·불가·위험·법규·관계"가 들어 있다. 한국어 코퍼스를 상정하고 만든 흔적이다. 커밋 메시지에도 한국어가 섞여 있고 예제 워커 하나는 나라장터(G2B) 입찰 상세를 긁는다.

## I1~I7 영향 분석

`ontology_impact`는 변경이 번지는 방향을 일곱 갈래로 나눠 묻는다. I1 데이터(어떤 값·레코드가 바뀌나), I2 관계(어떤 엣지가 영향받나), I3 공간(어느 온톨로지 공간이 닿나), I4 권한(어떤 ReBAC 정책이 바뀌나), I5 로직(어떤 규칙·추론 사슬이 무효가 되나), I6 캐시·인덱스(무엇을 다시 만들어야 하나), I7 다운스트림(어떤 외부 시스템이 영향받나).

## CrabHarness

수집 쪽 통제면이다. `mission.json` 하나가 무엇을 어디까지 얼마나 긁을지를 선언하면, 플래너가 `codex_workers/*/worker.manifest.json`을 훑어 맞는 워커를 고르고 서브프로세스로 위임한다. 결과 번들은 완결성 점수(기본 임계 0.8), 의미 점수, autoresearch 판정 세 관문을 통과해야 승격 패키지가 된다.

의미 점수는 미션에 적힌 질문마다 증거를 0.0~1.0으로 채점해 매기는데, `ANTHROPIC_API_KEY`가 있으면 Claude Haiku 4.5로 채점하고 없으면 규칙 기반으로 떨어진다. `harness_loop.py`는 여기서 한 걸음 더 나가서 run → validate → analyze → evolve를 돌며 미션 파라미터를 스스로 조정하고 상태를 `.harness-loop-state.json`에 남긴다.

핵심 원칙은 세 가지다. 크롤러가 아니라 미션이 먼저다. 워커는 매니페스트와 어댑터만 추가하면 되고 코어를 고칠 필요가 없다. 하네스는 패키지만 만들고 그래프에 직접 쓰지 않는다.

## 사용법

```bash
pip install -e ".[dev]"

opencrab status      # 저장소·인덱스 상태
opencrab manifest    # 문법 전문 덤프
opencrab ingest ./docs --recursive
opencrab query "system performance and error rates"

# 에이전트에 붙이기
claude mcp add opencrab -- opencrab serve

# 수집 파이프라인
crabharness catalog
crabharness plan missions/examples/github-trending-harvest.json
crabharness run  missions/examples/github-trending-harvest.json
```

MCP 서버는 줄 단위 JSON-RPC를 stdin/stdout으로 주고받는 직접 구현이다. SDK를 쓰지 않고 initialize·tools/list·tools/call 세 메서드만 처리한다.

## 제약

- **LICENSE 파일이 없다.** README와 pyproject는 MIT라고 적었지만 파일이 없어 GitHub는 라이선스 미표기로 잡는다. 사내에 들이려면 확인이 필요하다.
- **Pack v1은 계약서만 있고 구현이 없다.** 문서는 필수 엔트리 11개와 `validate → Neo4j 임포트 → 익스포트 → 정규화 → ZIP` 5단계를 규정하지만 코드에는 Neo4j 내보내기 한 단계(`export_neo4j_opencrab_ingest`)만 있다. 레포 어디에도 zipfile 호출이 없다.
- **`.env.example`의 `STORAGE_MODE`는 죽은 설정이다.** Settings에 그 필드가 없고 `extra="ignore"`라 조용히 버려진다. 팩토리는 언제나 로컬 저장소를 돌려준다. Neo4j·Mongo 어댑터는 남아 있지만 `export-neo4j-pack` 명령과 Obsidian 임포터에서만 쓰인다. `query.py` 주석은 여전히 그래프 순회를 Neo4j가 한다고 적어 뒀다.
- **예제 워커 3개 중 github_trending은 `echo` 스텁이다.** 실제로 도는 것은 landscape와 soeak 둘이다.
- **의미 점수가 API 키 유무에 따라 달라진다.** 게이트 통과 여부가 환경에 좌우되므로 재현 실험에는 주의가 필요하다.
- **테스트가 한쪽에 몰려 있다.** 134개 중 문법 48개, MCP 41개, 저장소 40개인데 질의·검색은 3개뿐이다. 정작 손이 많이 간 랭킹 로직이 가장 얇게 덮여 있다.
- pyproject의 Homepage·Repository가 실제 레포가 아니라 `github.com/opencrab/opencrab`을 가리킨다.

## 메모 — 내 관심사와의 접점

- **CMP 온톨로지와 대비된다.** 이쪽은 TTL과 SPARQL로 쓰는데 OpenCrab은 OWL/RDF가 아니라 프로퍼티 그래프에 파이썬 상수로 문법을 박는다. 추론기가 없는 대신 "에이전트가 노드를 넣으려 할 때 거절당한다"는 쓰기 경로 검증이 있다. IaaS 온톨로지에 없는 축이 정확히 이 쓰기 검증과 승격 수명주기다.
- **evidence → claim 결속은 Agentic RAG의 근거 추적과 같은 문제를 다룬다.** Claim에 `status`를 달아 candidate → validated → promoted로 굴리는 방식은 평가 파이프라인의 판정 상태 관리에 그대로 옮겨 써도 되겠다.
- **I1~I7은 온톨로지 변경 리뷰 체크리스트로 쓸 만하다.** 특히 I6 캐시·인덱스와 I7 다운스트림은 TTL만 고칠 때 놓치기 쉬운 자리다.
- **RRF k=60 + alpha 0.7 조합은 그대로 베낄 만한 기본값이다.** 한글 2·3그램 토크나이저도 형태소 분석기 없이 굴릴 때의 현실적인 절충으로 참고할 만하다.
- **커밋 로그에 분업이 그대로 남아 있다.** 37개 중 25개가 `claude` 명의이고 사람 컨트리뷰터 4명이 붙은 커밋은 대부분 실행 경로 버그다. BFS 허브 노드 성능, 로컬 모드에서 엣지 타입이 뭉개지던 문제, Docker 기동 실패 같은 것들이다. 에이전트가 뼈대를 세우고 사람이 실제로 돌려보며 고친 흔적으로 읽힌다.
