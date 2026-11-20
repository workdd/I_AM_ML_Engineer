수락된 계획이 연산자 트리가 되는 과정과 쿼리 메모리를 본다.

## 읽기 진입점과 실행 순서

세션의 공개 읽기 API 는 `exec/query_doors.rs` 에 모여 있고, 모든 읽기가 `engine::execute_query` 를 지난다([exec/query_doors.rs L1-L3](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/exec/query_doors.rs#L1-L3)).

| 진입점 | 대상 | 그래프 인덱스 | 반환 |
|---|---|---|---|
| `query` | branch 또는 snapshot | 런타임 캐시 공유 | `QueryResult` |
| `query_with_head` | branch 또는 snapshot | 런타임 캐시 공유 | 결과와 head commit id |
| `run_query_at` | 과거 graph 버전 | 쿼리 전용 빌드 | `QueryResult` |
| `query_inspected` | branch 또는 snapshot | 런타임 캐시 공유 | `Executed` (계획, explain, 보고) |
| `replay_bound_plan` | branch 또는 snapshot | 런타임 캐시 공유 | `PlanRun` |

### 코드 분석: query_with_head

`query` 는 `query_with_head` 를 부르고 head 를 버린다. 앞부분의 순서를 본다.

[exec/query_doors.rs L70-L78](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/exec/query_doors.rs#L70-L78)

```rust
let settings = self.effective(query_source)?;
let (resolved, catalog) = self.capture_read_view(target).await?;

let compiled = self.compile_named_query(&catalog, query_source, query_name)?;
admit_wildcard_target(
    matches!(&resolved.requested, ReadTarget::Snapshot(_)),
    compiled.ir().has_wildcard_traversal(),
)?;
let head = resolved.graph_commit_id.clone();
```

순서는 설정 결정, 읽기 시점 고정(`capture_read_view`), 컴파일, 와일드카드 검사, 실행이다. snapshot 은 여기서 한 번 정해지고 실행 도중 branch head 를 다시 읽지 않는다. 반환되는 head commit id 는 같은 snapshot 에서 나온 값이라 읽은 뒤 쓰는 compare-and-swap 의 기대 head 로 쓸 수 있다(9장에서 다룬다). 와일드카드 순회는 과거 snapshot 대상에서 거부된다([exec/query_doors.rs L31-L38](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/exec/query_doors.rs#L31-L38)). 컴파일 결과는 쿼리 소스와 이름을 키로 캐시되고, 적중에는 schema gate 가 메모이즈한 같은 catalog `Arc` 가 필요하다([exec/query_doors.rs L397-L425](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/exec/query_doors.rs#L397-L425)). 파싱과 타입 검사는 4장에서 다룬다.

### 그래프 인덱스 handle 선택

`execute_on_route` 는 IR 에 `Expand` 나 `AntiJoin` 이 있을 때만 그래프 인덱스 handle 을 만든다([exec/query_doors.rs L365-L373](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/exec/query_doors.rs#L365-L373)). handle 은 아직 아무것도 빌드하지 않는다. CSR 은 순회 연산자가 처음 요청할 때 쿼리가 쓰는 간선 타입(`referenced_edge_types`)만으로 만들어진다. `Cached` 는 런타임 캐시 항목을 쿼리 사이에 공유하고, `Direct` 는 과거 snapshot 에 대해 한 번 빌드한다. `test-util` 기능에서는 세션에 설치된 `ReadExecutor` 가 엔진 대신 실행할 수 있으며 참조 엔진이 이 통로를 쓴다.

### 코드 분석: execute_query

엔진 쪽 진입 함수는 수집, 수락, 결합, 실행 네 단계를 잇는다.

[engine/mod.rs L358-L367](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/engine/mod.rs#L358-L367)

```rust
let source = QuerySource::gather(query, catalog, snapshot, params, settings).await?;
let accepted = accept_query(&source).await?;
let bound = bind(accepted, &source, embedding).await?;
let context = EngineContext {
    snapshot,
    catalog,
    graph_index: Arc::new(graph_index),
    load: None,
};
let run = Box::pin(execute(bound, &context)).await?;
```

`settings` 는 `gather` 에만 전달된다. `gather` 는 `engine/` 아래에서 `std::env` 를 부르는 유일한 위치로, 순회 경로 상한 두 값과 RRF gate 정책을 여기서 읽어 계획에 싣는다([engine/plan_source.rs L87-L108](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/engine/plan_source.rs#L87-L108)). `accept_query` 는 계획기의 수락 검사를 부르고([engine/plan_source.rs L648-L653](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/engine/plan_source.rs#L648-L653)), 그 내용은 5장에서 다룬다. `bind` 는 embedding client 를 쓰는 유일한 위치다([engine/bind.rs L1-L5](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/engine/bind.rs#L1-L5)). embedding 호출은 7장에서 다룬다.

| 환경 변수 | 기본값 | 쓰임 |
|---|---:|---|
| `OMNIGRAPH_EXPAND_INDEXED_MAX_FRONTIER` | 1024 | 이보다 큰 추정 frontier 는 CSR 로 순회 |
| `OMNIGRAPH_EXPAND_INDEXED_MAX_HOPS` | 6 | 이보다 많은 홉은 CSR 로 순회, 0 이면 기본값 |

두 기본값은 [engine/plan_source.rs L53-L60](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/engine/plan_source.rs#L53-L60) 에 있다.

재실행 진입점 `replay_bound_plan` 은 직렬화된 계획을 같은 `execute` 로 실행한다([exec/query_doors.rs L291-L311](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/exec/query_doors.rs#L291-L311)). 실행 전에 `plan_pins_snapshot` 이 계획이 기록한 엔진 카탈로그 지문과 dataset 의 경로, Lance branch, 버전을 현재 snapshot 과 비교하고, 하나라도 다르면 충돌 오류로 거부한다([engine/mod.rs L171-L210](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/engine/mod.rs#L171-L210)). 이 검사를 통과하면 실행 선택은 계획과 snapshot 의 함수가 되고 재실행이 같은 선택을 반복한다.

## 계획 하강: 노드 하나에 연산자 하나

계획기는 물리 계획을 루트부터 입력 우선 순서로 방문하며 `Lower` trait 의 메서드를 노드 종류별로 부른다(5장에서 다룬다). 엔진의 구현체가 `engine/lower.rs` 의 `Walk` 이고 `Op` 타입은 `Arc<dyn ExecutionPlan>` 이다.

### 하강 결과

하강 결과 `Lowered` 는 루트 연산자, 노드별 연산자 표, 이번 pass 의 `nearest` 스캔 보고를 담는다([engine/lower.rs L70-L79](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/engine/lower.rs#L70-L79)). `operators` 는 노드 id 에서 연산자로 가는 표로, 실행 보고가 이 표로 노드와 연산자를 짝짓는다. `lower_query` 는 검색 재실행 단계(pass)마다 다시 불리며, 첫 줄에서 `validate_plan_values` 로 결합 값이 저장 column 타입에 맞는지 검사하고, 끝에서 채울 join 이 없는 런타임 filter slot 이 남았는지 확인한다([engine/lower.rs L137-L155](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/engine/lower.rs#L137-L155)).

### 코드 분석: Walk::built

모든 노드 메서드는 연산자를 만든 뒤 `built` 로 등록한다.

[engine/lower.rs L484-L496](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/engine/lower.rs#L484-L496)

```rust
fn built(&mut self, id: NodeId, operator: impl ExecutionPlan) -> Lowers<Plan> {
    let plan: Plan = Arc::new(operator);
    let node = self.lowering.node(id)?;
    if matches!(node, PhysicalNode::Aggregate { .. })
        && plan.properties().emission_type != EmissionType::Final
    {
        return Err(OmniError::manifest_internal(format!(
            "node {id} `Aggregate` is a pipeline breaker in the plan but `{}` emits {:?}",
            plan.name(),
            plan.properties().emission_type,
        ))
        .into());
    }
```

계획기는 노드마다 pipeline 형태(전체 입력을 모으는 breaker, batch 단위로 흘려보내는 streaming)를 선언한다. `built` 는 연산자의 `EmissionType` 이 선언과 같은지 확인하고, 반환 노드(`Projection`, `Aggregate`, `MetadataCount`, `HydrateColumns`)는 출력 schema 를 계획의 선언과 대조한다([engine/lower.rs L512-L552](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/engine/lower.rs#L512-L552)). 어긋나면 실행 전에 내부 오류로 멈춘다.

### 하강 대상 노드와 연산자

push 파이프라인 전용 변형은 읽기 하강에서 명시적으로 거부된다([engine/lower.rs L671-L730](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/engine/lower.rs#L671-L730)).

| 물리 노드 | 연산자 | 형태 |
|---|---|---|
| `Scan` (저장소, 순위 없음) | `ScanExec` | Source |
| `Scan` (순위 있음) | `ScanExec` | Breaker |
| `Scan` (dependent) | `ScanExec` | Streaming |
| `Scan` (엔진 카탈로그) | `CatalogueExec` | Source |
| `MetadataCount` | `MetadataCountExec` | Source |
| `Expand` | `ExpandExec` | 1홉, budgeted 는 Streaming, 그 밖은 Breaker |
| `HashJoin` | `HashJoinExec` | Streaming |
| `CrossJoin`, `ContainsJoin` | `CrossJoinExec`, `ContainsJoinExec` | Streaming |
| `AntiJoin`, `OuterReference` | `AntiJoinMaskExec`, `OuterReferenceExec` | Breaker, Source |
| `Filter`, `Projection`, `Limit` | `FilterExec`, `ProjectionExec`, `LimitExec` | Streaming |
| `Aggregate` | DataFusion `AggregateExec` | Final |
| `Sort` | `SortExec` | Breaker |
| `HydrateColumns` | `HydrateExec` | Streaming |
| `RankFuse` | `RankFuseExec` (7장에서 다룬다) | Breaker |
| `SortMergeJoin`, `HydrateByAddress`, `RowCompare`, `ClassifyThreeWay`, `Page`, `DeltaSource` | 없음 (거부) | 해당 없음 |

### 코드 분석: scan 하강

스캔 노드 하강은 출처에 따라 분기한다. 엔진 카탈로그 스캔에 걸린 제약을 본다.

[engine/lower.rs L593-L604](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/engine/lower.rs#L593-L604)

```rust
if spec.table.is_catalogue() {
    if ranked.is_some()
        || spec.runtime_filter.is_some()
        || !matches!(source, ScanInput::Table)
        || input.is_some()
    {
        return Err(OmniError::manifest_internal(format!(
            "scan {id} of `{}` asks the catalogue for a ranked, runtime-filtered or dependent read",
            spec.table.type_key
        ))
        .into());
    }
```

엔진 카탈로그(`@` 타입) 스캔은 바이너리의 레지스트리 행을 메모리에서 읽는 `CatalogueExec` 이 되므로 순위, 런타임 filter, dependent 입력을 받지 않는다. 저장소 스캔은 순위가 있으면 이번 pass 의 `SearchMode`(gate 가 정한 후보 집합, 넓힌 k)를 받고, 위에 `Limit` 이 있으면 `with_gather_rows` 로 batch 를 일찍 내보낸다. 런타임 filter slot 은 `ContainsJoin` 이 실행 시점에 채운다.

## 연산자의 형태와 쿼리 메모리

모든 자체 연산자는 `operators/shape.rs` 의 세 형태 중 하나를 갖고 출력 stream 을 그 형태를 통해서만 만든다([engine/operators/shape.rs L1-L8](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/engine/operators/shape.rs#L1-L8)). 그래서 선언한 emission 과 실제 동작이 어긋날 수 없다.

| 형태 | `EmissionType` | 동작 |
|---|---|---|
| `Breaker` | `Final` | 입력을 모두 받은 뒤 결과를 batch 크기로 잘라 내보냄 |
| `Streaming` | `Incremental` | 입력 batch 마다 처리해 바로 내보냄 (`map` 은 inline, `produce` 는 producer task) |
| `Source` | `Incremental` | 입력 없는 leaf, 읽는 대로 내보냄 |

모든 형태는 partition 이 하나다. breaker 의 stream 을 버리면 body, 보관 중인 batch, 메모리 예약이 함께 해제되므로 상위 `Limit` 이 읽기를 멈추면 아래 작업도 정리된다([engine/operators/shape.rs L91-L117](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/engine/operators/shape.rs#L91-L117)). 형태는 poll 횟수(`polls`)와 끝까지 읽혔는지(`drained`)를 metric 으로 세고([engine/operators/shape.rs L38-L45](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/engine/operators/shape.rs#L38-L45)), 실행 보고의 `executed`, `skipped` 판정이 이 값에서 나온다.

### Streamed trait 과 producer channel

`Streaming::produce` 를 쓰는 연산자는 `Streamed` trait 의 `open`, `batch`, `finish` 를 구현한다([engine/operators/shape.rs L225-L236](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/engine/operators/shape.rs#L225-L236)). `open` 은 입력을 실행하기 전에 불린다. join 이 수집 쪽 입력을 모으거나 `ContainsJoin` 이 아래 스캔의 런타임 filter 를 채우는 작업이 여기서 끝나야 입력 스캔이 그 결과를 본다. 그 뒤 batch 마다 `batch` 가, 마지막에 `finish` 가 불린다. producer 는 별도 task 로 돌고 소비자와 용량 2 의 channel 로 연결되므로([engine/operators/shape.rs L620-L621](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/engine/operators/shape.rs#L620-L621)) 소비자가 느리면 두 batch 앞에서 기다린다.

### QueryContext 설정

쿼리 하나의 DataFusion session 은 `QueryContext` 가 만든다([engine/context.rs L216-L221](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/engine/context.rs#L216-L221)).

| 설정 | 값 | 출처 |
|---|---|---|
| target partition | 1 | 연산자 순서 보존 |
| batch 크기 | 8,192행 (`ORDERED_SCAN_EXECUTION_BATCH_ROWS`) | [table_store.rs L135-L136](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/table_store.rs#L135-L136) |
| 디스크 spill 상한 | 100 GiB (`ORDERED_SCAN_SCRATCH_BYTES`) | 같은 위치 |
| 메모리 상한 | 계획의 `Assumptions.memory_limit` | 세션 설정 `query_memory_bytes` |

DataFusion 의 정렬과 집계는 spill 할 수 있지만 그래프 breaker 는 예약을 넘으면 거부된다([engine/context.rs L1-L4](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/engine/context.rs#L1-L4)).

### WorkMemory 예약

그래프 연산자의 할당은 `WorkMemory::grow` 로 쿼리 pool 에 먼저 예약된다([engine/operators/memory.rs L683-L698](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/engine/operators/memory.rs#L683-L698)). 예약이 거부되면 이름이 붙은 오류가 나서 어느 연산자가 거부되었는지 남는다. `share` 는 CSV 적재가 pool 을 나눠 쓸 때의 몫이고, 읽기에서는 `None` 이다(추정). 컬렉션 항목은 `entries::<T>` 로 항목당 `(size_of::<T>() + 1) * 4` 바이트를 보수적으로 잡는다([engine/operators/memory.rs L741-L750](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/engine/operators/memory.rs#L741-L750)). budgeted 순회의 작업량 상한(`charge_traversal`)도 같은 구조체가 관리한다.
