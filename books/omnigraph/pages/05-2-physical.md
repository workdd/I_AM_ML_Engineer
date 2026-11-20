비용 함수와 물리 노드 카탈로그를 본다.

## 비용 모델과 물리 선택

물리 선택은 `Lowering`이 논리 노드를 아래에서부터 물리 노드로 바꾸는 과정이다 ([optimizer.rs L2771-L2801](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-planner/src/optimizer.rs#L2771-L2801)). 대부분은 그대로 대응되지만 세 지점에서 `cost.rs`의 비용 모델이 선택한다. `cost.rs`의 함수는 모두 I/O 없는 순수 함수이고 입력은 manifest에 이미 있는 행 수와 바이트 수뿐이다.

### 행 수 추정

공통 입력은 `estimate_rows`다 ([cost.rs L406-L408](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-planner/src/cost.rs#L406-L408)). 스캔의 기본값은 노드 타입의 manifest 행 수(`entity_count`)이고, 필터가 `@key` 전체나 unique 속성을 유한한 값 개수로 묶으면 그 개수로 줄어든다. 등식은 1, 리스트 포함은 원소 수다 ([cost.rs L321-L355](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-planner/src/cost.rs#L321-L355)). `Expand`는 홉마다 평균 fanout을 곱하고 목적지 타입 행 수에서 멈춘다 ([cost.rs L468-L476](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-planner/src/cost.rs#L468-L476)). 추정이 없으면(`None`) 비용 모델은 결정하지 않고 보수적인 기본값으로 간다.

> **주의**: `docs/dev/execution.md`의 Expand 절은 행 수 추정이 "`@key` 속성 등식이면 1로 줄어든다"고만 적는다 ([docs/dev/execution.md L190-L195](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/docs/dev/execution.md#L190-L195)). 코드는 unique 속성 등식과 리스트 포함(원소 수)도 반영한다. 이 책은 코드를 따른다.

### 코드 분석: choose_expand_mode

`crates/omnigraph-planner/src/cost.rs`, [cost.rs L220-L246](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-planner/src/cost.rs#L220-L246)

`Expand`가 홉마다 BTREE로 이웃을 찾을지(`IndexedScan`), 그래프 전체의 CSR을 메모리에 만들지(`Csr`) 고른다.

```rust
pub fn choose_expand_mode(i: &ExpandCostInputs) -> ExpandMode {
    if i.effective_max_hops > i.max_hops_cap || i.frontier_rows > i.max_frontier_cap {
        return ExpandMode::Csr;
    }

    let hops = i.effective_max_hops.max(1) as f64;
    let frontier = i.frontier_rows as f64;
    let edges = i.edge_count as f64;
    let src = i.src_node_count.max(1) as f64;
    let fanout = edges / src;

    let indexed_cost = match i.coverage {
        IndexCoverage::Indexed => hops * frontier * fanout * i.probe_factor,
        IndexCoverage::Degraded => hops * edges * i.probe_factor,
    };
    let csr_cost = if i.csr_cached {
        0.0
    } else {
        CSR_BUILD_FACTOR * edges
    };
```

$h$를 홉 수, $F$를 입력 frontier 행 수, $|E|$를 엣지 수, $|V_{src}|$를 출발 타입 노드 수, $p$를 방향 계수라 하면 두 비용은 다음과 같다.

$$
\text{indexed} = h \cdot F \cdot \frac{|E|}{|V_{src}|} \cdot p, \qquad \text{csr} = 1.5 \cdot |E|
$$

인덱스가 키 열을 완전히 덮고 CSR이 아직 없으면 양변을 $|E|$로 나눠 $h \cdot F \cdot p < 1.5 \cdot |V_{src}|$일 때 인덱스 경로가 선택된다. 함수 주석의 표현대로 결정은 $|E|$와 무관한 비율이 된다. 같은 계획에서 앞선 `Expand`나 `AntiJoin`이 CSR을 이미 만들었으면 CSR 비용은 0이다. `CSR_BUILD_FACTOR`는 1.5이고 ([cost.rs L174-L178](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-planner/src/cost.rs#L174-L178)), 방향 계수는 단방향 1, 양방향 2다.

| 입력 | 출처 | 기본값 |
|---|---|---|
| `frontier_rows` | `estimate_rows`, 없으면 `u64::MAX` | 없음 |
| `edge_count`, `src_node_count` | manifest의 `entity_count` | 없음 |
| `max_frontier_cap` | `OMNIGRAPH_EXPAND_INDEXED_MAX_FRONTIER` | [1024](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/engine/plan_source.rs#L56) |
| `max_hops_cap` | `OMNIGRAPH_EXPAND_INDEXED_MAX_HOPS` | [6](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/engine/plan_source.rs#L60) |
| `coverage` | 고정 index facts의 BTREE 전체 포함 여부 | 없음 |
| `csr_cached` | 같은 계획의 앞선 CSR 사용 여부 | `false` |

두 상한은 엔진이 계획 입력을 모을 때 환경 변수에서 한 번 읽고 계획의 `assumptions.env`에 기록한다 ([plan_source.rs L71-L84](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/engine/plan_source.rs#L71-L84)). 실행은 환경을 다시 읽지 않는다.

`Lowering::expand_mode`는 결과를 `ExpandPolicy`와 묶는다 ([optimizer.rs L3998-L4010](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-planner/src/optimizer.rs#L3998-L4010)). 세션이 모드를 고정했으면 `Pinned`, 통계가 없으면 `Csr`과 `Uncosted`, 비용 모델이 골랐으면 `Costed`다. 정책은 실행 중 모드를 바꿀 수 있는지를 선언하며 `Costed`만 다른 모드를 대안으로 가진다. 엔진은 홉마다 관측한 frontier로 `should_switch_to_csr`를 다시 묻는다 ([cost.rs L264-L296](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-planner/src/cost.rs#L264-L296)). 작업 예산이 걸린 계획은 함수 시작에서 `IndexedScan`과 `Budgeted`로 확정된다 ([optimizer.rs L3948-L3954](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-planner/src/optimizer.rs#L3948-L3954)). 실행 중 전환은 6장에서 다룬다.

### 코드 분석: choose_access_path

`crates/omnigraph-planner/src/cost.rs`, [cost.rs L116-L130](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-planner/src/cost.rs#L116-L130)

`Expand` 위의 목적지 스캔을 읽는 방법을 정한다.

```rust
pub fn choose_access_path(
    frontier_rows: Option<u64>,
    table_rows: Option<u64>,
    build_bytes: Option<u64>,
    build_bytes_budget: u64,
) -> Option<AccessPath> {
    let (frontier, rows) = (frontier_rows?, table_rows?);
    let few_rows = rows <= frontier.saturating_mul(HASH_JOIN_RATIO);
    let fits = build_bytes.is_some_and(|bytes| bytes <= build_bytes_budget);
    Some(if few_rows && fits {
        AccessPath::HashJoin
    } else {
        AccessPath::IdLookup
    })
}
```

`IdLookup`은 입력 배치마다 `id IN (...)`으로 Lance를 읽고, `HashJoin`은 목적지 테이블을 한 번 읽어 해시 테이블을 만든다. 테이블 행 수가 frontier의 8배 이하이고 build 측 추정 바이트가 메모리 풀의 4분의 1 이하일 때만 해시 조인이다 ([cost.rs L90-L96](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-planner/src/cost.rs#L90-L96)). `HashJoin`을 고르면 목적지 스캔은 입력 없는 `Scan`이 되어 build 측에 놓인다 ([optimizer.rs L3184-L3203](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-planner/src/optimizer.rs#L3184-L3203)). 해시 조인은 `IdLookup`을 `fallback`으로 선언하므로 실행 중 한도를 넘으면 id 조회로 돌아갈 수 있다.

### 상관 블록 전략

`not { ... }`, `count { ... } > 2` 같은 상관 블록은 두 방식으로 실행할 수 있다 ([cost.rs L132-L145](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-planner/src/cost.rs#L132-L145)). `Correlated`는 외부 행에서 출발해 내부 패턴을 따라가고, `KeySet`은 블록 자신의 가장 좁은 바인딩에서 출발해 상관 바인딩에서 끝나는 트리를 한 번만 실행한다. `key_set_tree`는 블록이 상관 바인딩의 id 외에 외부 행을 읽지 않고, 어떤 바인딩이 자기 필터로 테이블 행 수보다 좁혀질 때만 재구성 트리를 만든다 ([key_set.rs L88-L104](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-planner/src/key_set.rs#L88-L104)). 선택은 두 트리의 행 수 합(`chain_rows`)이 더 작은 쪽이다 ([cost.rs L149-L155](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-planner/src/cost.rs#L149-L155)). 논리 계획은 항상 상관 트리를 블록의 의미로 유지하고 재구성 트리는 별도 아레나에 둔다.

## 물리 계획: 노드 카탈로그와 속성

### 코드 분석: physical_nodes! 선언

`crates/omnigraph-planner/src/physical.rs`, [physical.rs L913-L935](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-planner/src/physical.rs#L913-L935)

물리 노드는 매크로 선언 한 곳에 필드와 분류 토큰을 함께 적는다. 아래는 `Expand`다.

```rust
    Expand {
        input: NodeId,
        src: String,
        dst: String,
        edges: EdgeSelection,
        src_type: String,
        dst_type: String,
        min_hops: u32,
        max_hops: Option<u32>,
        edge_binding: Option<String>,
        mode: ExpandMode,
        frontier_estimate: Option<u64>,
        policy: ExpandPolicy,
        versions: BTreeMap<String, Option<u64>>,
    } => "Expand" [input: NodeId]
        shape: if *max_hops == Some(1) || matches!(policy, ExpandPolicy::Budgeted) {
            PipelineShape::Streaming
        } else {
            PipelineShape::Breaker
        },
        sides: vec![],
        cut: None,
        order: None;
```

`=>` 뒤는 explain 이름, 대괄호는 자식 필드, 그 뒤 네 토큰이 분류다. 매크로는 이 토큰으로 `shape`, `collected_sides`, `row_cut`, `order`를 변형마다 한 팔씩 생성한다 ([physical.rs L734-L789](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-planner/src/physical.rs#L734-L789)). 토큰이 빠진 줄은 그 줄에서 컴파일되지 않으므로 새 노드가 분류 없이 추가될 수 없다. `Expand`는 한 홉이거나 예산 정책이면 스트리밍이고, 그 외 다중 홉은 BFS 전체를 돌아야 하므로 breaker다.

| 토큰 | 의미 | 예 |
|---|---|---|
| `shape:` | 스트리밍 입력에 대한 방출 시점 | `Sort`, `Aggregate`, `AntiJoin`은 `Breaker` |
| `sides:` | 전체를 먼저 수집하는 입력 | `HashJoin`의 build, `CrossJoin`의 왼쪽 |
| `cut:` | 조건과 무관하게 남기는 최대 행 수 | `Limit`의 `rows`, `Sort`의 `fetch` |
| `order:` | 출력 순서 | 필터는 입력 순서 유지, 순위 스캔은 점수 순서 |

물리 노드는 24개다 ([physical.rs L794-L800](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-planner/src/physical.rs#L794-L800)). 논리 노드와 1대1이 아닌 지점이 있다. `Nearest`와 `TextSearch`는 `Scan`의 `ranked` 필드가 되고 ([physical.rs L801-L814](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-planner/src/physical.rs#L801-L814)), `HashJoin`, `HydrateColumns`, `ContainsJoin`은 물리 선택에서만 생긴다.

### 파이프라인 분해

`Breaker` 노드에서 계획은 파이프라인으로 잘린다. `pipelines`는 루트에서 내려가며 스트리밍 노드는 현재 파이프라인에 넣고, breaker와 수집 측은 자식 파이프라인으로 떼어 낸다 ([physical.rs L1635-L1651](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-planner/src/physical.rs#L1635-L1651)). 결과 `PipelineSpec`은 원본, 연산자 목록, 싱크, 먼저 소진할 파이프라인을 가진다 ([physical.rs L1933-L1947](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-planner/src/physical.rs#L1933-L1947)). 변경 피드와 병합은 이 분해로 push 실행기를 구성하고, 쿼리는 explain에 같은 분해를 출력한다.

### 코드 분석: Properties

`crates/omnigraph-planner/src/physical.rs`, [physical.rs L502-L513](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-planner/src/physical.rs#L502-L513)

물리 선택이 끝나면 3단계 `derive_properties`가 아래에서 위로 노드마다 속성을 계산한다.

```rust
/// The properties a physical node declares: derived after selection,
/// recomputed when a later pass changes the node, never recomputed by an
/// executor. Each is a bound the executor adapts within, never above.
#[derive(Debug, Clone)]
pub struct Properties {
    pub schema: SchemaRef,
    pub ordering: Option<Vec<OrderKey>>,
    pub rows: Estimate,
    pub work_bytes: Estimate,
    pub retained_limit: Option<u64>,
    pub sources: Vec<StatisticSource>,
}
```

`rows`와 `work_bytes`는 `Known(n)` 또는 `Unknown`인 추정치이며 그 자체로는 아무것도 제한하지 않는다. 제한은 `retained_limit`가 한다. 예를 들어 `HydrateColumns` 청크는 쿼리 메모리 풀의 8분의 1까지만 보유한다 ([cost.rs L104-L110](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-planner/src/cost.rs#L104-L110)). `sources`는 추정의 출처 통계 목록이고, explain이 이를 모두 출력하므로 "the optimizer opened no data file"을 확인할 수 있다.

### 코드 분석: 스캔 접근 확정

`crates/omnigraph-planner/src/scan_access.rs`, [scan_access.rs L52-L75](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-planner/src/scan_access.rs#L52-L75)

마무리 단계는 모든 `Scan`에 `ScanAccess`를 기록한다. 의존 스캔은 그 전에 `IdLookup`으로 확정된다.

```rust
        } else if spec.runtime_filter.is_some() {
            ScanAccess::Runtime {
                input: RuntimeInput::JoinFilter,
            }
        } else if let Some(ranked) = ranked {
            ScanAccess::Runtime {
                input: match ranked.kind {
                    RankKind::Nearest => RuntimeInput::Nearest,
                    RankKind::Bm25 => RuntimeInput::FullText,
                },
            }
        } else if let Some(input) = source.scan_runtime_input(spec) {
            ScanAccess::Runtime { input }
        } else if spec.filter.is_none() || !facts.iter().any(IndexFact::may_narrow) {
            ScanAccess::Sequential
        } else {
            if facts.iter().any(|fact| {
                fact.column == spec.columns.id
                    && matches!(fact.kind, IndexKind::Btree { usable: true })
            }) {
                narrowed.extend(narrow_key(spec, source)?);
            }
            source.index_split(spec).await?
        };
```

실행 시 필터, 순위 스캔, 소스가 지정한 실행 시 입력은 계획 시점에 접근을 정할 수 없어 `Runtime`이다. 필터나 좁힐 수 있는 인덱스가 없으면 `Sequential`이고, 그 외에만 소스의 `index_split`을 `await`해 Lance 스캐너가 스칼라 인덱스 경로(`IndexProbe`)를 쓸지 받는다. 네 가지 접근은 `ScanAccess`에 정의된다 ([logical.rs L338-L350](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-planner/src/logical.rs#L338-L350)).

id 열에 사용 가능한 BTREE가 있으면 `narrow_key`가 단일 문자열 `@key` 등식을 찾아 정규화된 id 등식을 필터에 덧붙인다 ([scan_access.rs L101-L162](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-planner/src/scan_access.rs#L101-L162)). 원래 등식은 남는다. 사용자 문서는 다른 키 타입을 좁히지 않는 이유를 "supported historical IDs may use different spellings"라고 적는다 ([docs/user/queries/explain.md L48-L50](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/docs/user/queries/explain.md#L48-L50)). index facts의 수집은 8장에서 다룬다.
