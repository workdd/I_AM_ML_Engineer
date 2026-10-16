# Omnigraph: 에이전트가 브랜치로 쓰는 타입 그래프 DB

- **레포**: [ModernRelay/omnigraph](https://github.com/ModernRelay/omnigraph) · [웹사이트](https://omnigraph.dev) · [예제 그래프](https://github.com/ModernRelay/omnigraph-cookbooks)
- **제작**: ModernRelay (기여자 15명, 익명 포함 · 2026-10-16 기준)
- **공개**: 2026-04-10 저장소 생성일 (Rust, MIT, 별 1,316개 · 2026-10-16 GitHub API 기준)
- **읽은 날짜**: 2026-10-16 (커밋 `32aa366`, 릴리즈 `v0.13.0`)
- **태그**: #GraphDB #Ontology #GraphRAG #HybridSearch #Branching #Lance #MCP #AgentMemory

## 한 줄 요약

여러 에이전트가 **하나의 타입이 정해진 도메인 모델(온톨로지)을 공유하며** 읽고 쓰는 그래프 DB다. 그래프 탐색, 벡터 검색, 전문 검색(BM25), RRF 융합을 **질의 하나**에서 하고, 에이전트는 main에 바로 쓰지 않고 **Git처럼 브랜치에 쓴 뒤 병합**한다. 데이터는 Lance 열 지향 포맷으로 로컬 디스크나 S3 호환 저장소에 둔다.

## 기능 범위

| 기능 | 내용 |
|---|---|
| 타입 온톨로지 | `.pg` 스키마로 노드·엣지 타입, 키, 제약을 선언하고 쓰기마다 엔진이 강제 |
| 통합 검색 | 그래프 탐색 + 벡터 근사 최근접(ANN) + 전문 검색 + RRF를 한 질의 안에서 실행 |
| 브랜치 | 에이전트·작업마다 브랜치, 3-way 병합, 그래프 전체 시점 조회(time travel) |
| 설정을 코드로 | `cluster.yaml`에 그래프, 스키마, 저장 질의, 임베딩 공급자, 정책을 선언. `cluster plan`으로 미리 보고 `cluster apply`로 반영 (Terraform 방식) |
| 권한을 코드로 | Cedar 정책을 서버에서 모든 쓰기마다 검사. 그래프별·브랜치별·서버 전체 범위 |
| 저장소 | 로컬, S3 호환(RustFS, MinIO, AWS S3, R2, GCS), Azure(미리보기). 열린 포맷 Lance |
| 클라이언트 | CLI, HTTP(OpenAPI), TypeScript SDK, MCP 서버. Python SDK는 예정 |

README가 내세우는 용도는 사내 지식 그래프(company brain), 에이전트 메모리, 의사결정 이력 그래프, 이슈·의존성 그래프, 실험 데이터 계층이다.

## 스키마 언어 (`.pg`)

```pg
node Person {
  email: String @key
  display_name: String
  age: I32?
  @range(age, 0..150)
}

node Company {
  slug: String @key
  name: String
}

edge WorksAt: Person -> Company @card(0..) {
  role: String?
  @unique(@src, @dst)
}
```

| 제약 | 의미 |
|---|---|
| `@key` | 이 속성 조합으로 id를 만든다. 같은 키로 넣으면 같은 노드가 된다 |
| `@unique` | 같은 값 조합 금지 (엣지는 `@src`, `@dst` 포함 가능) |
| `@range`<br>`@check` | 숫자 범위, 문자열 정규식 |
| `@card(min..max)` | 엣지 개수 제한 |
| `@index` | 색인 의도 선언. 정확성에는 영향 없음 |

- 타입: 문자열, 정수, 실수, 날짜, `Vector(N)`, `Blob`, `enum(...)`, 리스트, null 허용(`T?`)
- `interface`로 공통 속성을 재사용하고, `@rename_from`으로 이름을 바꿔도 id와 이력을 유지한다
- 단일 속성 선언에는 색인이 자동으로 붙는다: 정렬 가능한 값은 스칼라 색인, 자유 텍스트는 전문 색인, 벡터는 벡터 색인

## 질의 언어 (`.gq`)

이름과 타입이 있는 질의를 파일에 저장하고 이름으로 부른다.

```gq
query engineers($title: String) @description("People with a title") {
  match {
    $p: Person { title: $title }
    $p worksAt $c
  }
  return { $p.name, $c.name as company }
  order { company asc, $p.name asc }
  limit 50
}
```

- 탐색: `$a knows{1,3} $b`(1~3홉), `$a (knows | likes) $b`(엣지 종류 선택), `$a <related> $b`(방향 무시), `not { }`, `exists { }`, `count { } > 2`, `sum(...) { }`
- 여러 홉 탐색은 상한을 반드시 적어야 하고, 홉 수는 시작 노드에서의 최단 거리다
- `@description`, `@instruction`을 달면 MCP 같은 클라이언트가 저장 질의를 **도구로 노출**할 때 그대로 쓴다
- 브랜치 조작(`branch create`, `merge`, `diff`)과 실행 계획 확인(`explain`)도 같은 언어의 문장이다

### 한 질의 안의 하이브리드 검색

```gq
query hybrid($vector: Vector(4), $text: String) {
  match { $d: Document }
  return { $d.slug, $d.title }
  order { rrf(nearest($d.embedding, $vector), bm25($d.body, $text)) }
  limit 10
}
```

| 함수 | 의미 |
|---|---|
| `nearest` | 벡터 L2 거리 순위. 텍스트를 넣으면 설정한 임베딩 공급자가 벡터로 바꾼다 |
| `search`<br>`fuzzy`<br>`match_text` | 전문 검색 (오타 허용 포함) |
| `bm25` | BM25 점수 |
| `rrf(a, b, k)` | 두 순위를 RRF로 융합. 기본 k = 60 |

- `match` 안의 그래프·속성 조건을 **순위 매기기 전에** 적용한다. `limit 10`은 조건을 만족한 것 중 상위 10개다
- 탐색 조건이 후보를 많이 줄이면, 그 후보만 먼저 골라 검색 범위를 좁힌다(전체 대비 10% 미만, 10만 개 이하일 때 기본 동작)
- 벡터 검색은 IVF 파티션을 기본 20개까지만 읽고, 결과가 모자라면 4배 → 무제한 → 전체 정확 탐색 순으로 넓힌다
- 임베딩 공급자: OpenAI 호환(기본, OpenRouter의 `openai/text-embedding-3-large`), OpenAI, Gemini

## 브랜치와 병합

```bash
omnigraph branch create --from main agent/ingest-42
omnigraph load --data batch.jsonl --mode append --branch agent/ingest-42 graph.omni
omnigraph branch merge agent/ingest-42 --into main
```

- 공통 조상(merge base)을 기준으로 양쪽 변경을 비교하는 3-way 병합이다. `dry run`으로 미리 볼 수 있다
- 충돌은 구조화된 목록으로 돌려주고(HTTP 409) **아무것도 반영하지 않는다**

| 충돌 종류 | 의미 |
|---|---|
| `divergent_insert` | 양쪽이 같은 id를 다른 값으로 넣음 |
| `divergent_update` | 같은 엔티티를 양쪽이 다르게 수정 |
| `delete_vs_update` | 한쪽은 삭제, 한쪽은 수정 |
| `orphan_edge` | 끝점이 삭제된 엣지가 남음 |
| `unique_violation`<br>`cardinality_violation`<br>`value_constraint_violation` | 병합 결과가 스키마 제약 위반 |

- 엣지를 양쪽에서 같이 넣었을 때의 결과가 선언에 따라 다르다. 선언이 없으면 둘 다 남고, `@unique(@src, @dst)`면 충돌, `@key(@src, @dst)`면 하나로 합쳐진다. **같은 사실을 여러 에이전트가 넣는 경우를 스키마로 정한다.**

## 구조

`docs/dev/architecture.md` 기준이다.

```
CLI · omnigraph-server (HTTP, MCP) · 내장 SDK
        │  모든 쓰기가 같은 Cedar 정책 검사를 지남
engine ── compiler (.pg/.gq 파싱·타입 검사) ── planner (실행 계획)
   │
catalog · core · storage
   │
Lance 데이터셋 (노드·엣지 타입마다 따로) ── 로컬 / S3 / Azure
```

- 노드·엣지 타입마다 Lance 데이터셋이 따로 있고, Omnigraph가 그 위에 **그래프 전체 단위의 스냅숏**을 만든다. 쓰기가 성공하면 영향을 받은 타입이 한꺼번에 보이고, 읽는 쪽은 절반만 반영된 커밋을 보지 않는다
- 버전·브랜치·색인은 Lance 기능을 쓰고, Omnigraph는 스키마, 그래프 의미, 정책, 배포 상태를 더한다
- 설계 결정을 RFC로 남긴다. `docs/rfcs/`에 78개가 있다(템플릿과 README 제외)

## 에이전트 연동

- **MCP 서버**: npm `@modernrelay/omnigraph-mcp`와 서버 내장 `/mcp` 엔드포인트. 서버 쪽 RFC 0003에 따르면 MCP 계층은 HTTP 핸들러를 그대로 도구로 노출하고, **자체 권한 경로를 두지 않는다**(정책은 엔진에서 한 번만 검사). 요청 64KiB, 결과 1MiB, 읽기 30초, 동시 16개 제한이 있다
- **에이전트 스킬 내장**: `omnigraph skill export ~/.claude/skills/omnigraph`로 설치된 CLI 버전과 맞는 작업 지침을 내보낸다
- 실행 계획(`explain`)과 진단 문서가 따로 있어, 에이전트가 질의가 왜 느린지·왜 비었는지 확인할 수 있다

## 한계와 주의

- **아직 0.x다.** v0.12.0 → v0.13.0이 이틀 간격이고, 스키마 예약어 변경처럼 이전 그래프와 새 그래프의 규칙이 다른 부분이 문서 곳곳에 있다.
- **Python SDK가 없다.** Python 쪽은 HTTP(OpenAPI)로 붙어야 한다.
- **임베딩 공급자가 좁다.** OpenAI 호환, OpenAI, Gemini만 있다. 로컬 모델은 OpenAI 호환 엔드포인트로 띄워야 한다.
- **`cleanup`은 이력을 지운다.** 오래된 저장 버전을 영구 삭제해 이전 커밋을 못 읽게 될 수 있다. `optimize`와 다르다.
- **Azure는 미리보기다.** 쓰기 프로세스가 전용 래퍼를 거쳐야 한다.
- **복합 속성, 엣지 속성 색인은 아직 만들어지지 않는다.** 선언은 받지만 성능에 반영되지 않는다.
- 성능 수치는 README에 없다. 벤치마크 도구(`omnigraph-bench`)만 있고 결과 숫자는 확인하지 않았다(미검증).

## 가져올 수 있는 부분

| 대상 | 가져올 수 있는 부분 |
|---|---|
| 거버넌스 지식그래프 스키마 | `@key`로 id를 속성에서 만들고, `@card`·`@unique`·`@range`로 제약을 **쓰기 시점에** 강제하는 방식. 스키마 밖 데이터가 들어온 뒤 적재기에서 걸러내는 대신 DB가 막는다 |
| GraphRAG 검색 | 그래프 조건으로 후보를 먼저 거르고, 벡터·BM25 순위를 RRF로 합치는 흐름을 **질의 하나**로 표현. 후보가 적을 때만 먼저 거르는 게이트(10%, 10만 개)와 벡터 검색이 모자랄 때 넓히는 순서도 그대로 참고할 만하다 |
| 에이전트 쓰기 안전장치 | 에이전트가 main 대신 브랜치에 쓰고, 검토 후 병합하는 흐름. 충돌을 구조화된 종류로 돌려주고 하나라도 있으면 아무것도 반영하지 않는 규칙 |
| 같은 사실의 중복 적재 | 엣지 identity 선언(`없음` / `@unique` / `@key`)으로 여러 에이전트가 같은 관계를 넣을 때의 결과를 미리 정하는 방식 |
| MCP 도구 설계 | 저장 질의에 `@description`·`@instruction`을 달아 그대로 도구로 노출. 권한은 MCP 계층이 아니라 엔진 한 곳에서만 검사 |
| 배포와 정책 | 스키마·저장 질의·정책을 디렉터리 하나에 선언하고 plan → apply로 반영하는 방식 |
| 에이전트 지침 배포 | CLI가 자기 버전에 맞는 스킬을 내보내는 방식(`skill export`). 도구와 지침의 버전이 어긋나지 않는다 |
