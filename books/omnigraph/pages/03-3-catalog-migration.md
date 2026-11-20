SchemaIR에서 카탈로그를 만들고 스키마 변경을 계획하는 과정을 본다.

## 카탈로그 생성

`Catalog`는 실행 시점에 쓰는 조회 구조다. 타입 이름으로 속성, 키, 제약, Arrow 스키마를 바로 찾을 수 있게 펼쳐 둔다.

### 코드 분석: Catalog 구조

[`crates/omnigraph-compiler/src/catalog/mod.rs` L16-L29](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/catalog/mod.rs#L16-L29)

```rust
pub struct Catalog {
    pub node_types: HashMap<String, NodeType>,
    pub edge_types: HashMap<String, EdgeType>,
    /// This graph's system column spellings (RFC 0040); consumers must not
    /// hardcode any spelling.
    pub system_columns: schema_ir::SystemColumns,
    /// Maps normalized lowercase edge name -> EdgeType key (e.g. "knows" -> "Knows")
    pub edge_name_index: HashMap<String, String>,
    /// Interface declarations (for Phase 2 polymorphic queries)
    pub interfaces: HashMap<String, InterfaceType>,
    /// Source-only callers are intentionally unbound. Runtime catalogs are a
    /// direct projection of one validated accepted SchemaIR and retain that
    /// authority so no consumer can reconstruct IDs from mutable names.
    pub identity: CatalogIdentity,
```

- `system_columns`는 그래프의 철자다. 주석은 소비자가 어떤 철자도 하드코딩하지 말라고 적는다
- `edge_name_index`는 소문자로 바꾼 엣지 이름에서 실제 이름으로 가는 표다. 쿼리에서 `worksAt`으로 써도 `WorksAt`을 찾는 이유다([`crates/omnigraph-compiler/src/catalog/mod.rs` L192-L200](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/catalog/mod.rs#L192-L200)). 대소문자만 다른 두 엣지 이름은 카탈로그 생성에서 거부한다([`crates/omnigraph-compiler/src/catalog/mod.rs` L828-L836](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/catalog/mod.rs#L828-L836))
- `identity`는 원본에서만 만든 카탈로그(`SourceUnbound`)인지, 수락된 IR에 묶인 카탈로그(`Bound`)인지 구분한다. 실행에 쓰는 카탈로그는 항상 `Bound`다([`crates/omnigraph-compiler/src/catalog/mod.rs` L47-L51](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/catalog/mod.rs#L47-L51))

카탈로그를 만드는 함수는 두 개다. `build_catalog`는 원본 AST에서 바로 만들고 항상 현재 세대 철자를 쓴다([`crates/omnigraph-compiler/src/catalog/mod.rs` L362-L366](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/catalog/mod.rs#L362-L366)). `build_catalog_from_ir`는 수락된 IR에서 만들고 그 IR의 철자를 쓴다([`crates/omnigraph-compiler/src/catalog/mod.rs` L589-L591](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/catalog/mod.rs#L589-L591)). 엔진이 그래프를 열 때 쓰는 쪽은 후자다(02장).

### 코드 분석: 엣지 키 열 순서

IR에서 만든 카탈로그는 키 열 순서를 식별자로 정한다. [`crates/omnigraph-compiler/src/catalog/mod.rs` L794-L807](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/catalog/mod.rs#L794-L807)

```rust
let mut columns = vec![
    system_columns.src.to_string(),
    system_columns.dst.to_string(),
];
columns.extend(
    stable_fields
        .into_iter()
        .map(|(_, property_name)| property_name),
);
// No implied index: edge keys are always composite and index
// maintenance builds single-column indices only; the fixed
// id/src/dst BTREEs already cover endpoints.
key = Some(columns);
continue;
```

- 엣지 키는 항상 출발, 도착 컬럼이 먼저 오고, 나머지 속성은 바로 앞에서 속성 id 순으로 정렬한 순서로 붙는다([`crates/omnigraph-compiler/src/catalog/mod.rs` L787-L793](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/catalog/mod.rs#L787-L793)). 속성 이름을 바꿔도 물리 튜플 순서가 바뀌지 않는다
- 엣지 키에는 인덱스를 자동으로 만들지 않는다. 주석은 인덱스 유지가 단일 열 인덱스만 만들고, id, 출발, 도착의 BTREE 인덱스가 이미 끝점을 덮는다고 적는다

Arrow 스키마는 시스템 컬럼을 앞에 두고 선언 속성을 뒤에 붙인다. 시스템 컬럼은 모두 null 불가 `Utf8`이다([`crates/omnigraph-compiler/src/catalog/mod.rs` L816-L827](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/catalog/mod.rs#L816-L827)).

## 마이그레이션 계획

스키마 변경은 수락된 IR과 원하는 IR을 비교한 계획 단계를 반드시 거친다. 계획은 `SchemaMigrationStep` 목록과, 지원하지 않는 단계가 하나라도 있는지를 나타내는 `supported` 값이다([`crates/omnigraph-compiler/src/catalog/schema_plan.rs` L23-L27](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/catalog/schema_plan.rs#L23-L27)).

### 코드 분석: 계획 함수

[`crates/omnigraph-compiler/src/catalog/schema_plan.rs` L210-L219](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/catalog/schema_plan.rs#L210-L219)

```rust
let mut steps = Vec::new();
plan_interfaces(&accepted.interfaces, &desired.interfaces, &mut steps);
plan_nodes(&accepted.nodes, &desired.nodes, &mut steps, system_columns);
plan_edges(
    &accepted.nodes,
    &accepted.edges,
    &desired.edges,
    &mut steps,
    system_columns,
);
```

- 그 앞에서 두 IR이 같은 식별자 도메인인지, 시스템 컬럼 철자가 같은지 확인한다. 세대는 스키마 변경으로 바뀌지 않는다([`crates/omnigraph-compiler/src/catalog/schema_plan.rs` L190-L209](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/catalog/schema_plan.rs#L190-L209))
- 비교는 이름이 아니라 식별자로 한다. 속성은 `property_id`, 타입은 `type_id`로 짝을 찾는다([`crates/omnigraph-compiler/src/catalog/schema_plan.rs` L648-L655](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/catalog/schema_plan.rs#L648-L655))

| 단계 | 의미 | 데이터 재작성 |
|---|---|---|
| `AddType` | 타입 추가 | 없음 |
| `RenameType`, `RenameProperty` | 식별자가 같고 이름만 다름 | 없음 |
| `AddProperty` | null 허용 속성 추가 | 없음 |
| `AddConstraint` | 인덱스 선언, 노드 `@unique` 추가 | 없음. `@unique`는 저장된 행을 먼저 검사 |
| `ExtendEnum` | enum 값 추가(상위 집합) | 없음 |
| `RequireProperty` | null 허용에서 필수로 | null 행이 있으면 거부 |
| `RelaxCardinality` | `@card` 범위 넓히기 | 없음 |
| `UpdateTypeMetadata`, `UpdatePropertyMetadata` | 설명 같은 주석 변경 | 없음 |
| `DropType`, `DropProperty` | 선언 삭제. 저장 공간은 cleanup 전까지 회수하지 않음 | 없음 |
| `UnsupportedChange` | 지원하지 않는 변경과 사유, 진단 코드 | 적용 거부 |

단계 정의와 각 단계의 주석은 [`crates/omnigraph-compiler/src/catalog/schema_plan.rs` L29-L153](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/catalog/schema_plan.rs#L29-L153)에 있다. 데이터 재작성 열과 검사 동작은 [스키마 변경 문서](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/docs/user/schema/index.md#L222-L256)를 따랐다.

### 코드 분석: 기수 변경 분류

엣지 `@card` 변경은 넓히기만 허용한다. [`crates/omnigraph-compiler/src/catalog/schema_plan.rs` L581-L599](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/catalog/schema_plan.rs#L581-L599)

```rust
if existing.cardinality != edge.cardinality {
    if existing.cardinality.is_within(&edge.cardinality) {
        steps.push(SchemaMigrationStep::RelaxCardinality {
            type_name: edge.name.clone(),
            from: existing.cardinality.clone(),
            to: edge.cardinality.clone(),
        });
    } else {
        steps.push(SchemaMigrationStep::UnsupportedChange {
            entity: format!("edge:{}", edge.name),
            reason: format!(
                "tightening cardinality on edge '{}' from {} to {} needs a check of the \
                 stored edge counts and is not supported in schema migration v1",
                edge.name, existing.cardinality, edge.cardinality
            ),
            code: None,
        });
    }
}
```

- `is_within`은 기존 범위의 모든 개수가 새 범위 안에 드는지 본다([`crates/omnigraph-compiler/src/schema/ast.rs` L175-L185](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/schema/ast.rs#L175-L185)). 그렇다면 저장된 엣지 개수를 다시 셀 필요가 없다
- 최솟값을 올리거나 최댓값을 내리면 저장된 개수 검사가 필요하므로 거부한다
- 기존 노드 타입에서 출발하는 새 엣지 타입에 0보다 큰 최솟값이 있으면, 기존 노드에 그 엣지가 하나도 없으므로 역시 거부한다(`OG-MF-107`, [`crates/omnigraph-compiler/src/catalog/schema_plan.rs` L538-L550](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/catalog/schema_plan.rs#L538-L550))

같은 원리가 속성에도 적용된다. 새 필수 속성은 기존 행을 채울 방법이 없어 `OG-MF-103`으로 거부되고, 타입 변경은 enum 확장과 필수화를 빼면 `OG-MF-106`으로 거부된다([`crates/omnigraph-compiler/src/catalog/schema_plan.rs` L656-L678](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/catalog/schema_plan.rs#L656-L678), [`crates/omnigraph-compiler/src/catalog/schema_plan.rs` L691-L718](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/catalog/schema_plan.rs#L691-L718)). 사용자 문서는 필수 속성을 넣으려면 null 허용으로 추가하고 값을 적재한 뒤 필수로 바꾸라고 안내한다. 계획이 적용되는 과정과 테이블 메타데이터 커밋은 11장에서 다룬다.
