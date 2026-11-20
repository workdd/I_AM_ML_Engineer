`.pg` 문법과 파서, 이름 검증 규칙을 본다.

## 문법과 선언

스키마 문법은 67줄짜리 pest 파일 하나다. 최상위 선언은 `interface`, `node`, `edge` 세 가지다.

### 코드 분석: 선언 규칙

[`crates/omnigraph-compiler/src/schema/schema.pest` L10-L24](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/schema/schema.pest#L10-L24)

```text
schema_decl = { interface_decl | node_decl | edge_decl }

// interface Named { name: String @key }
interface_decl = { "interface" ~ type_name ~ "{" ~ prop_decl* ~ "}" }

// node Person implements Named, Described { ... }
node_decl = { "node" ~ type_name ~ annotation* ~ implements_clause? ~ "{" ~ (prop_decl | body_constraint)* ~ "}" }
implements_clause = { "implements" ~ type_name ~ ("," ~ type_name)* }

// edge Knows: Person -> Person @card(0..1) { ... }
// edge Knows: Person -> Person
edge_decl = { "edge" ~ type_name ~ ":" ~ type_name ~ "->" ~ type_name ~ cardinality? ~ annotation* ~ ("{" ~ (prop_decl | body_constraint)* ~ "}")? }

// @card(0..1), @card(1..), @card(0..)
cardinality = { "@card" ~ "(" ~ integer ~ ".." ~ integer? ~ ")" }
```

- 노드는 `implements`로 인터페이스 여러 개를 받을 수 있다. 인터페이스 자체는 속성 목록만 가진다
- 엣지의 `@card`는 다른 주석보다 앞에 와야 한다. 문법이 `cardinality? ~ annotation*` 순서로 적혀 있기 때문이다
- 본문 제약은 `@key`, `@unique`, `@index`, `@range`, `@check` 다섯 가지 이름만 받는다. 일반 주석 규칙은 이 다섯 이름 뒤에 `(`가 오면 주석으로 읽지 않는다([`crates/omnigraph-compiler/src/schema/schema.pest` L49-L53](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/schema/schema.pest#L49-L53))

타입 이름은 대문자, 속성 이름은 소문자나 밑줄로 시작한다([`crates/omnigraph-compiler/src/schema/schema.pest` L66-L67](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/schema/schema.pest#L66-L67)). 속성 타입은 기본 타입 11개, `Vector(N)`, `enum(...)`, 목록 `[T]`, 끝의 `?`로 표현하는 null 허용이다([`crates/omnigraph-compiler/src/schema/schema.pest` L40-L47](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/schema/schema.pest#L40-L47)). 각 타입은 Arrow 타입으로 대응된다.

| 스키마 타입 | Arrow 타입 | 비고 |
|---|---|---|
| `String`, `Bool` | `Utf8`, `Boolean` | |
| `I32`, `I64`, `U32`, `U64` | `Int32`, `Int64`, `UInt32`, `UInt64` | |
| `F32`, `F64` | `Float32`, `Float64` | |
| `Date`, `DateTime` | `Date32`, `Date64` | |
| `Blob` | `LargeBinary` | 목록 불가, 키와 인덱스 불가 |
| `Vector(N)` | `FixedSizeList<Float32, N>` | N은 1부터 `i32::MAX`까지 |
| `[T]` | `List<T>` | 요소는 기본 타입만 |

대응 규칙은 [`crates/omnigraph-compiler/src/types.rs` L84-L106](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/types.rs#L84-L106), 목록의 Blob 거부와 Vector 차원 검사는 [`crates/omnigraph-compiler/src/schema/parser.rs` L696-L731](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/schema/parser.rs#L696-L731)에 있다. 사용자 문서의 예시 스키마는 [스키마 언어 문서](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/docs/user/schema/index.md#L8-L25)에서 볼 수 있다.

## 파서와 AST

`SchemaParser`는 `pest_derive`가 문법 파일에서 만드는 파서다([`crates/omnigraph-compiler/src/schema/parser.rs` L15-L17](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/schema/parser.rs#L15-L17)). 공개 진입점은 여럿이지만 모두 `parse_schema_diagnostic_with_modes` 하나로 모이고, 두 가지 모드 값만 다르다.

| 진입점 | 제약 검증 모드 | 속성 이름 규칙 | 쓰는 곳 |
|---|---|---|---|
| `parse_schema` | `Admission` | `CurrentVintage` | 새 그래프 생성, 원본 검사 |
| `parse_schema_for_vintage` | `Admission` | 그래프 세대에 따라 `CurrentVintage` 또는 `LegacyVintage` | 기존 그래프의 스키마 변경 |
| `parse_persisted_schema_contract` | `PersistedContract` | `PersistedContract` | 이미 수락된 원본을 다시 읽기 |

`PersistedContract` 모드는 예전 버전이 받아들였던 본문 `@unique(Blob)`과 I64 범위를 넘어 Float로 저장된 `@range` 경계를 계속 읽기 위한 호환 경로다. 새 스키마 수락에는 쓰지 않는다([`crates/omnigraph-compiler/src/schema/parser.rs` L40-L52](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/schema/parser.rs#L40-L52)). 속성 이름 규칙의 세 값은 [`crates/omnigraph-compiler/src/schema/parser.rs` L64-L77](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/schema/parser.rs#L64-L77)에 정의되어 있다.

### 코드 분석: 파싱 뒤 검증 순서

pest가 선언 목록을 만든 뒤의 처리다. [`crates/omnigraph-compiler/src/schema/parser.rs` L120-L132](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/schema/parser.rs#L120-L132)

```rust
// Resolve implements clauses on nodes
let iface_refs: Vec<&InterfaceDecl> = interfaces.iter().collect();
for decl in &mut declarations {
    if let SchemaDecl::Node(node) = decl {
        resolve_interfaces(node, &iface_refs).map_err(compiler_error_to_diagnostic)?;
    }
}

let schema = SchemaFile { declarations };
validate_property_names(&schema, name_rules).map_err(compiler_error_to_diagnostic)?;
validate_schema_annotations(&schema).map_err(compiler_error_to_diagnostic)?;
validate_constraints(&schema, constraint_mode).map_err(compiler_error_to_diagnostic)?;
Ok(schema)
```

- 인터페이스 해석이 먼저 돌아 노드의 속성 목록을 완성한다. 그 뒤에 이름, 주석, 제약 검증이 완성된 목록을 대상으로 돈다

### 코드 분석: 제약 표현

본문 제약과 속성 주석은 같은 `Constraint` 값이 된다. [`crates/omnigraph-compiler/src/schema/ast.rs` L128-L142](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/schema/ast.rs#L128-L142)

```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Constraint {
    Key(Vec<String>),
    Unique(Vec<String>),
    Index(Vec<String>),
    Range {
        property: String,
        min: Option<ConstraintBound>,
        max: Option<ConstraintBound>,
    },
    Check {
        property: String,
        pattern: String,
    },
}
```

- `email: String @key` 같은 속성 주석은 파싱 중 `Constraint::Key(vec!["email"])`로 바뀐다. 이 변환은 `@key`, `@unique`, `@index`에만 적용된다([`crates/omnigraph-compiler/src/schema/parser.rs` L574-L591](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/schema/parser.rs#L574-L591))
- 엣지 기수 `@card`는 제약이 아니라 `EdgeDecl`의 별도 필드 `Cardinality`이고, 생략하면 `0..`이다([`crates/omnigraph-compiler/src/schema/ast.rs` L163-L173](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/schema/ast.rs#L163-L173)). 최솟값이 최댓값보다 크면 파싱에서 거부한다([`crates/omnigraph-compiler/src/schema/parser.rs` L381-L388](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/schema/parser.rs#L381-L388))

### 코드 분석: 인터페이스 해석

노드가 구현한 인터페이스의 속성은 노드에 없으면 주입되고, 있으면 타입이 같아야 한다. [`crates/omnigraph-compiler/src/schema/parser.rs` L618-L637](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/schema/parser.rs#L618-L637)

```rust
if let Some(existing) = node.properties.iter().find(|p| p.name == iface_prop.name) {
    // Property exists — verify type compatibility
    if existing.prop_type != iface_prop.prop_type {
        return Err(CompilerError::Parse(format!(
            "node {} property '{}' has type {} but interface {} declares it as {}",
            node.name,
            iface_prop.name,
            existing.prop_type.display_name(),
            iface_name,
            iface_prop.prop_type.display_name()
        )));
    }
} else {
    // Property missing — inject it from the interface
    node.properties.push(iface_prop.clone());
    // Also desugar any constraint annotations from the injected property
    desugar_property_constraints(
        std::slice::from_ref(iface_prop),
        &mut node.constraints,
    );
```

- 주입된 속성의 `@key` 같은 주석도 노드 제약으로 바뀐다
- 속성마다 `NodePropertyOrigin`이 기록된다. 노드 본문에 직접 썼는지(`declared_directly`), 어느 인터페이스의 어느 속성을 만족하는지를 담는다([`crates/omnigraph-compiler/src/schema/ast.rs` L50-L60](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/schema/ast.rs#L50-L60)). 이 출처는 SchemaShape와 SchemaIR까지 그대로 전달되고, 마이그레이션 계획은 출처 변경을 지원하지 않는 변경으로 분류한다

## 이름과 주석 검증 규칙

검증은 이름 규칙, 주석 규칙, 제약 규칙의 세 함수가 나눠 맡는다.

### 코드 분석: 예약된 속성 이름

현재 세대 그래프는 밑줄로 시작하는 모든 속성 이름을 시스템용으로 예약한다. [`crates/omnigraph-compiler/src/schema/parser.rs` L174-L181](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/schema/parser.rs#L174-L181)

```rust
PropertyNameRules::CurrentVintage => {
    if super::is_reserved_system_column_name(name) {
        return Err(CompilerError::Parse(format!(
            "property name '{name}' is reserved for system columns \
             (names starting with '_')"
        )));
    }
}
```

- 예약 범위는 Lance의 가상 컬럼 다섯 개(`_rowid`, `_rowaddr`, `_rowoffset`, `_row_created_at_version`, `_row_last_updated_at_version`)와 Omnigraph의 시스템 컬럼을 모두 덮는다([`crates/omnigraph-compiler/src/schema/mod.rs` L10-L30](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/schema/mod.rs#L10-L30))
- 구형 세대 그래프는 Lance 가상 컬럼 이름과 자기 물리 컬럼 이름(`id`, `src`, `dst`)만 거부하고, 업그레이드 대상 이름 `__id`, `__src`, `__dst`도 거부한다([`crates/omnigraph-compiler/src/schema/parser.rs` L182-L203](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/schema/parser.rs#L182-L203))
- 검색 결과 열 이름도 새 선언에서 거부한다. 코드는 `_distance`, `_score`, `_rrf` 세 이름을 막는다([`crates/omnigraph-compiler/src/schema/mod.rs` L38-L42](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/schema/mod.rs#L38-L42))

> **주의**: [사용자 문서](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/docs/user/schema/index.md#L71-L75)는 검색 출력용 예약 이름으로 `_distance`와 `_score` 두 개만 적는다. 코드는 `_rrf`(`RRF_COLUMN`)도 예약한다. 현재 세대에서는 밑줄 접두 규칙이 어차피 세 이름을 모두 막으므로 차이가 드러나는 곳은 구형 세대 그래프뿐이다.

주석별 허용 위치는 아래와 같다. `validate_schema_annotations`와 `validate_property_annotations`가 이 표를 강제한다([`crates/omnigraph-compiler/src/schema/parser.rs` L828-L902](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/schema/parser.rs#L828-L902), [`crates/omnigraph-compiler/src/schema/parser.rs` L935-L978](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/schema/parser.rs#L935-L978)).

| 주석 | 허용 위치 | 거부 조건 |
|---|---|---|
| `@key` | 노드 속성, 노드 본문, 엣지 본문(양 끝 포함, 하나만) | 목록, 벡터, Blob, null 허용 속성. 엣지 속성 수준 |
| `@unique` | 노드와 엣지의 속성, 본문 | 목록, 벡터, Blob |
| `@index` | 노드와 엣지의 속성, 본문 | 목록, Blob |
| `@embed("src", model=...)` | 노드의 벡터 속성 | 원본 속성이 String이 아님, `model` 외 인자 |
| `@range`, `@check` | 노드 본문 | |
| `@description` | 타입, 속성 | 빈 값, 중복 |
| `@instruction` | 노드, 엣지 타입 | 속성 수준 |
| `@rename_from("old")` | 타입, 속성 | 중복, 빈 값, 키워드 인자 |

### 코드 분석: 엣지 키의 양 끝 요구

엣지 `@key`는 본문 제약으로만 쓸 수 있고 반드시 두 끝점을 포함해야 한다. [`crates/omnigraph-compiler/src/schema/parser.rs` L1182-L1196](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/schema/parser.rs#L1182-L1196)

```rust
if is_edge {
    for (meta, bare) in [
        (META_SRC, SYSTEM_COLUMNS_LEGACY.src),
        (META_DST, SYSTEM_COLUMNS_LEGACY.dst),
    ] {
        if !cols.iter().any(|col| {
            col == meta || (col == bare && !prop_names.contains_key(col.as_str()))
        }) {
            return Err(CompilerError::Parse(format!(
                "@key on edge {} must include both endpoints (missing '{}')",
                type_name, meta
            )));
        }
    }
}
```

- 끝점은 메타 필드 철자(`@src`, `@dst`)로 쓰거나, 같은 이름의 사용자 속성이 없을 때 맨 이름(`src`, `dst`)으로 쓴다
- RFC 0044는 키 있는 엣지 타입이 키 튜플에서 id를 파생해, 두 브랜치에 같은 엣지를 넣어도 병합에서 한 행으로 만난다고 설명한다([RFC 0044 요약](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/docs/rfcs/0044-edge-keys.md#L21-L43)). id 파생과 병합은 10장과 12장에서 다룬다
