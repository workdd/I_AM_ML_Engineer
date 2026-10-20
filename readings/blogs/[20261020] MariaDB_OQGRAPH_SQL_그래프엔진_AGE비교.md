# MariaDB OQGRAPH: 일반 엣지 테이블 위에 얹는 SQL 그래프 연산 엔진, Apache AGE와 비교

- **출처**: [OQGRAPH Overview](https://mariadb.com/docs/server/server-usage/storage-engines/oqgraph-storage-engine/oqgraph-overview) · [OQGRAPH Examples](https://mariadb.com/docs/server/server-usage/storage-engines/oqgraph-storage-engine/oqgraph-examples) · [Installing OQGRAPH](https://mariadb.com/docs/server/server-usage/storage-engines/oqgraph-storage-engine/installing-oqgraph) · [OQGRAPH System and Status Variables](https://mariadb.com/docs/server/ha-and-performance/optimization-and-tuning/system-variables/oqgraph-system-and-status-variables)
- **제작**: Open Query의 Arjen Lentz, Antony T Curtis와 Andrew McDonnell. MariaDB 서버에 들어간 커뮤니티 기여 엔진이다
- **문서**: MariaDB 공식 문서 (OQGRAPH v3 기준, 2026-10-20 확인 시 "19일 전 갱신")
- **비교 대상**: [Apache AGE](https://github.com/apache/age) (PostgreSQL 확장, 별 4,904개, 최신 태그 `PG15/v1.8.0-rc0` · 2026-10-20 GitHub API 기준)
- **읽은 날짜**: 2026-10-20
- **태그**: #GraphDB #MariaDB #OQGRAPH #ApacheAGE #PostgreSQL #ShortestPath #Hierarchy #SQL

## 한 줄 요약

OQGRAPH는 데이터를 저장하는 엔진이 아니라, **일반 테이블에 들어 있는 엣지 목록(출발 id, 도착 id, 가중치)을 그래프로 보고 최단 경로·도달 가능 노드·잎 노드를 SQL `SELECT`로 돌려주는 읽기 전용 가상 테이블**이다. 노드 라벨도 속성도 질의 언어도 없다. 쓰는 법은 `WHERE latch='dijkstras' AND origid=1 AND destid=6` 한 줄이다. 반대로 AGE는 PostgreSQL 안에 라벨과 속성을 가진 속성 그래프를 두고 openCypher로 패턴을 질의하는 그래프 DB다. **"관계형 DB에 그래프 알고리즘 몇 개를 붙인 것"과 "관계형 DB 안에 그래프 DB를 둔 것"의 차이**로 보면 된다.

## 구조: 원본 테이블과 OQGRAPH 테이블

```
oq_backing (일반 테이블, InnoDB 등)          oq_graph (ENGINE=OQGRAPH, 읽기 전용)
┌────────┬────────┬────────┐                ┌───────┬────────┬────────┬────────┬─────┬────────┐
│ origid │ destid │ weight │  ── 읽음 ──▶   │ latch │ origid │ destid │ weight │ seq │ linkid │
└────────┴────────┴────────┘                └───────┴────────┴────────┴────────┴─────┴────────┘
  INSERT / UPDATE / DELETE 는 여기에만          SELECT 할 때 latch 값에 따라 그래프 연산 결과를 행으로 반환
```

- **원본 테이블(backing table)**: 실제 엣지를 담는 일반 테이블이다. 뷰는 안 된다. 쓰기는 모두 여기에 한다.
- **OQGRAPH 테이블**: `data_table='oq_backing' origid='origid' destid='destid' [weight='weight']`로 원본 테이블의 열을 연결한다. 열 6개(`latch`, `origid`, `destid`, `weight`, `seq`, `linkid`)와 해시 키 두 개의 형식이 정해져 있고, 다르면 생성 오류가 난다.
- 가중치 열을 연결하지 않으면 모든 엣지의 가중치는 1이다.
- 엣지는 **방향이 있다.** 예제에서 1→6 경로는 있지만 6→1은 빈 결과다.
- 노드는 정수 id(`BIGINT UNSIGNED`)뿐이다. 이름이나 속성은 다른 테이블과 `JOIN`해서 붙인다.

## latch 명령

`latch` 열에 넣는 문자열이 곧 연산 종류다. `origid`, `destid` 중 무엇을 조건에 넣는지에 따라 의미가 바뀐다.

| latch | 조건 열 | 연산 |
|---|---|---|
| `NULL` | 없음 | 원본 엣지 목록 그대로 |
| `''` (빈 문자열) | 없음 | 모든 노드 나열 |
| `''` | origid | origid에서 1홉 이웃 |
| `dijkstras` | origid, destid | 다익스트라 최단 경로. 경로의 노드가 `seq` 순서로 `linkid`에 나옴 |
| `dijkstras` | origid | origid에서 도달 가능한 모든 노드와 거기까지의 가중치 합 |
| `dijkstras` | destid | destid로 갈 수 있는 모든 노드와 가중치 합 (역방향) |
| `breadth_first` | origid / destid / 둘 다 | 너비 우선으로 도달 가능 노드, 역방향 도달 노드, 홉 기준 최단 경로 |
| `leaves` | origid | origid에서 닿는 노드 중 들어오는 엣지만 있는 노드(끝점) |
| `leaves` | destid | destid로 갈 수 있는 노드 중 나가는 엣지만 있는 노드(시작점) |
| `leaves` | 둘 다 | 지원 안 함 (빈 결과) |

경로를 한 줄로 보려면 `GROUP_CONCAT(linkid ORDER BY seq)`로 묶는다.

```sql
SELECT GROUP_CONCAT(linkid ORDER BY seq) AS path FROM oq2_graph
 WHERE latch='dijkstras' AND origid=1 AND destid=6;
-- 1,2,3,4,5,6  (2→6 엣지 가중치가 10이라 홉이 많은 쪽이 최단)
```

정수 latch(`1`=dijkstras, `2`=breadth_first, `4`=leaves)는 옛 형식이다. 문자열 안의 숫자는 별칭으로 계속 받지만, latch 열을 정수형으로 만드는 옵션(`oqgraph_allow_create_integer_latch`)은 MariaDB 11.5에서 제거됐다.

## 설치와 운영

- 서버에 기본으로 켜져 있지 않다. 별도 패키지(`mariadb-plugin-oqgraph`, `MariaDB-oqgraph-engine`)를 설치하고 `INSTALL SONAME 'ha_oqgraph';`로 올린다.
- v3는 Boost와 Judy 라이브러리가 필요하다. Judy는 Red Hat 계열 공식 저장소에 없어 EPEL을 먼저 붙여야 한다.
- 쿼리 캐시를 쓰지 않는다.
- 상태 변수는 `Oqgraph_boost_version`, `Oqgraph_compat_mode`, `Oqgraph_verbose_debug` 세 개뿐이다. 성능 튜닝용 설정은 문서에 없다.

## Apache AGE와 비교

AGE 쪽 사실은 [AGE README](https://github.com/apache/age), 1.8.0 업그레이드 스크립트(`age--1.7.0--1.8.0.sql`), 이슈 [#2349](https://github.com/apache/age/issues/2349) · [#2350](https://github.com/apache/age/issues/2350)에서 확인했다.

| 항목 | MariaDB OQGRAPH | PostgreSQL Apache AGE |
|---|---|---|
| 정체 | 스토리지 엔진 플러그인. 원본 테이블 위의 **읽기 전용 계산 테이블** | PostgreSQL 확장. 그래프를 저장하고 질의하는 **그래프 DB** |
| 데이터 모델 | 엣지 목록(정수 출발·도착 id, 가중치 하나) | 속성 그래프. 노드·엣지마다 라벨과 속성(`agtype`, JSON과 비슷한 형식) |
| 노드 속성 | 없음. 다른 테이블과 `JOIN` | 노드·엣지 속성으로 직접 저장 |
| 엣지 종류 | 테이블 하나 = 엣지 종류 하나. 여러 관계는 OQGRAPH 테이블을 여러 개 만들거나 원본에 종류 열을 두고 뷰가 아닌 테이블로 나눔 | 라벨로 구분해 한 그래프에 여러 관계 |
| 질의 언어 | 평범한 SQL. `latch` 값으로 연산 지정 | `cypher()` 함수 안의 openCypher. 결과를 SQL과 섞어 조인 가능 |
| 패턴 매칭 | 없음. "A가 B를 지배하고 B가 C를 판매" 같은 다중 관계 패턴은 SQL 조인으로 직접 작성 | `MATCH (a)-[:CONTROLS*1..]->(b)-[:SELLS]->(p)` 형태로 작성 |
| 최단 경로 | 다익스트라(가중치), 너비 우선(홉 수) 내장 | Cypher 문법에는 `shortestPath()`가 없다(master 문법 파일 기준). 1.8.0에서 SQL 함수 `age_shortest_path`, `age_all_shortest_paths`(너비 우선)를 추가했고, 둘 다 **가중치 없는 홉 기준**이다. 가중치 최단 경로 함수는 없음 |
| 가변 길이 경로 | `dijkstras`/`breadth_first`로 도달 가능 노드 전부 | VLE(`[*1..n]`). #2349는 순환을 피하며 경로를 전부 펼치는 비용이 깊이에 따라 급격히 커진다고 보고 |
| 그 밖의 알고리즘 | 잎 노드(`leaves`) 찾기 | 내장 그래프 알고리즘 라이브러리는 없음. 필요하면 SQL 재귀 CTE나 애플리케이션에서 구현 |
| 쓰기 | 원본 테이블에 일반 `INSERT/UPDATE/DELETE` | Cypher `CREATE`, `MERGE`, `SET`, `DELETE` 또는 라벨 테이블에 직접 |
| 트랜잭션·인덱스 | 원본 테이블 엔진(InnoDB 등)을 따름 | PostgreSQL 트랜잭션. 라벨마다 실제 테이블이라 속성에 인덱스 생성 가능 |
| 설치 | 별도 패키지 + `INSTALL SONAME`. Boost, Judy 필요 | 확장 설치 + 세션마다 `LOAD 'age'`와 `search_path` 설정 |
| 유지 상태 | MariaDB에 포함된 커뮤니티 기여 엔진. 문서 범위가 작고 예제 문서에 오류가 남아 있음(아래) | Apache 프로젝트. PostgreSQL 버전별 브랜치로 1.7, 1.8 릴리스 후보를 계속 냄 |

### 같은 질문을 두 엔진으로

"노드 1에서 노드 6까지 최단 경로"

```sql
-- OQGRAPH: 가중치 반영 다익스트라
SELECT GROUP_CONCAT(linkid ORDER BY seq) FROM oq2_graph
 WHERE latch='dijkstras' AND origid=1 AND destid=6;
```

```sql
-- AGE 1.8: 홉 수 기준 최단 경로 (AGE 회귀 테스트 age_shortest_path.sql의 형태)
-- 인자: 그래프 이름, 시작, 끝, [엣지 종류, 방향, 최소 홉, 최대 홉]
SELECT path FROM age_shortest_path(
    '"g"'::agtype,
    (SELECT id FROM cypher('g', $$ MATCH (n {nid: 1}) RETURN id(n) $$) AS (id agtype)),
    (SELECT id FROM cypher('g', $$ MATCH (n {nid: 6}) RETURN id(n) $$) AS (id agtype))
) AS path;
-- 가중치를 반영하려면 라벨 테이블에 재귀 CTE를 직접 작성해야 한다
```

"도토리홀딩스가 직·간접 지배하는 회사가 판매하는 1등급 상품" ([OntoBricks ① 노트](%5B20261015%5D%20OntoBricks_온톨로지_개념_지식그래프_첫걸음.md)의 예)

```sql
-- AGE: 패턴 하나
SELECT * FROM cypher('fin', $$
  MATCH (:Company {name:'도토리홀딩스'})-[:CONTROLS*1..]->(c:Company)-[:SELLS]->(p:Product {riskGrade: 1})
  RETURN DISTINCT p.name
$$) AS (name agtype);
```

```sql
-- OQGRAPH: 지배 관계만 그래프로 두고, 판매·등급은 일반 테이블 조인
SELECT DISTINCT p.name
  FROM control_graph g
  JOIN sells s   ON s.company_id = g.linkid
  JOIN product p ON p.id = s.product_id AND p.risk_grade = 1
 WHERE g.latch = 'breadth_first' AND g.origid = 101   -- 101 = 도토리홀딩스
   AND g.linkid <> 101;
```

엣지 종류가 하나인 계층 탐색은 OQGRAPH가 더 짧고, 관계 종류가 여럿 섞인 패턴은 AGE가 훨씬 짧다.

### 어느 쪽을 고를 상황인가

| 상황 | 맞는 쪽 |
|---|---|
| 이미 MariaDB에 부모-자식, 출발-도착 테이블이 있고 최단 경로·하위 전체·도달 가능 여부만 필요 (조직도, 부품표, 경로 탐색) | OQGRAPH |
| 가중치 최단 경로가 핵심 (비용, 거리) | OQGRAPH. AGE에서는 직접 구현 |
| 노드·엣지 종류가 여럿이고 속성 조건이 붙는 다단계 질의 (지식 그래프, GraphRAG) | AGE |
| LLM이 질의를 생성 (Text-to-Cypher) | AGE. Cypher는 LLM이 학습한 예제가 많고, OQGRAPH의 `latch` 문법은 드물다 |
| PostgreSQL 기반 | AGE (OQGRAPH는 MariaDB 전용) |

## 한계와 주의

- **예제 문서에 오류가 있다.** 최단 경로 예제의 질의는 `latch='breadth_first'`인데 결과 표에는 `dijkstras`가 찍혀 있고, 설명의 `oq_graph2`는 `oq2_graph`의 오타다. 문서를 그대로 믿기보다 직접 돌려 확인하는 편이 낫다.
- **노드는 정수 id만 된다.** 문자열 키(UUID, 이름)를 쓰는 데이터는 정수 매핑 테이블을 따로 둬야 한다.
- **엣지 종류 하나, 가중치 하나다.** 관계별로 OQGRAPH 테이블을 따로 만들고 원본도 테이블로 나눠야 한다(뷰는 원본으로 쓸 수 없다).
- **성능과 규모에 대한 공식 수치가 없다.** 질의마다 원본 테이블을 읽어 그래프를 계산하는 구조로 보이지만, 메모리 사용과 대형 그래프 동작은 문서에 없다(미검증).
- **AGE의 최단 경로 함수는 1.8.0부터다.** 2026-10-20 기준 1.8.0은 릴리스 후보(`-rc0`) 태그이고, 가중치 없는 홉 기준만 있다. 쓰고 있는 AGE 버전에 이 함수가 있는지 먼저 확인해야 한다.
- AGE 비교의 VLE 비용 수치(이슈 #2350의 23ms 대 0.09ms)는 이슈 작성자의 벤치마크이고 프로젝트가 확인한 값이 아니다.

## 가져올 수 있는 부분

| 대상 | 가져올 수 있는 부분 |
|---|---|
| 계층 질의 설계 | OQGRAPH는 "엣지 테이블은 그대로 두고, 계층·도달 질의만 계산 계층에 맡긴다"는 설계다. 계층을 깊이·루트·경로 속성으로 미리 풀어 0홉으로 질의하는 방식은 같은 문제(가변 길이 탐색 비용)를 적재 시점으로 옮겨 푼 것이라, 두 방식을 비교 근거로 쓸 수 있다 |
| AGE 최단 경로 대안 | AGE 1.7 이하에서 최단 경로가 필요하면 VLE로 경로를 전부 펼치는 대신 라벨 테이블 위의 재귀 CTE 너비 우선 탐색을 쓴다(#2350의 예). 가중치 최단 경로도 같은 방식으로 직접 짠다 |
| 질의 도구 분담 | 엣지 종류가 하나인 계층·경로 질의는 그래프 알고리즘 함수로, 관계 종류가 섞인 패턴 질의는 Cypher로 나누는 기준. OntoBricks 노트의 "벡터 / SQL / 그래프" 분담 표에 한 단계 더 붙일 수 있다 |
| latch 방식의 도구 설계 | 연산 종류를 열 값 하나(`latch`)로 고르고 결과를 늘 같은 열 구조로 돌려주는 방식. 에이전트에 그래프 도구를 줄 때 "연산 이름 + 시작/끝 노드"만 받는 단순한 MCP 도구 형태로 응용할 수 있다 |
| AGE 1.8 확인 항목 | 1.8.0에 전역 인접 정보 캐시(`age_invalidate_graph_cache`)와 너비 우선 최단 경로 함수가 들어왔다. AGE를 올릴 때 최단 경로·계층 질의를 이 함수로 바꿀 수 있는지 점검할 항목이다 |
