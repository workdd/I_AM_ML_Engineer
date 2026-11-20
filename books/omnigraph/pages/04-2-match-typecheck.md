바인딩과 순회, 식의 이름 범위, 타입 검사를 본다.

## match 절: 바인딩, 순회, 상관 블록

`match` 블록의 절은 네 종류다. 노드 바인딩, 엣지 순회, 불 필터, 상관 블록(`not`, `exists`, 집계 비교)이다([`crates/omnigraph-compiler/src/query/ast.rs` L502-L510](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/query/ast.rs#L502-L510)).

### 코드 분석: 순회 문법

[`crates/omnigraph-compiler/src/query/query.pest` L169-L175](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/query/query.pest#L169-L175)

```text
traversal = { variable ~ edge_binding? ~ (undirected_edge | edge_alternation | edge_wildcard | edge_ident | string_lit) ~ traversal_bounds? ~ variable }
edge_binding = { variable ~ ":" }
undirected_edge = { "<" ~ (edge_wildcard | edge_alternatives | edge_ident | string_lit) ~ ">" }
edge_alternation = { "(" ~ edge_alternatives ~ ")" }
edge_alternatives = { (edge_ident | string_lit) ~ ("|" ~ (edge_ident | string_lit))+ }
edge_wildcard = { "*" }
traversal_bounds = { "{" ~ integer ~ "," ~ integer? ~ "}" }
```

- `$p knows $f`는 방향 있는 순회, `$p <knows> $f`는 방향 없는 순회다. 방향 없는 순회는 양 끝 타입이 같은 엣지에만 쓸 수 있다(`T22`)
- `(knows | likes)`는 엣지 타입 선택, `*`는 모든 엣지 타입이다
- `{1,3}`은 홉 수 범위다. 생략하면 `{1,1}`이다([`crates/omnigraph-compiler/src/query/parser.rs` L1193-L1206](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/query/parser.rs#L1193-L1206))
- `$p $w:knows $f`처럼 엣지에 변수를 붙이면 엣지 속성을 `$w.since`로 읽을 수 있다. 이때는 한 홉만 허용한다
- 엣지 이름은 소문자로 시작해야 한다. 예약어와 같은 이름의 엣지(`In`)는 `$i "in" $b`처럼 따옴표로 쓴다([`crates/omnigraph-compiler/src/query/parser.rs` L1241-L1261](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/query/parser.rs#L1241-L1261))

바인딩 `$p: Person { name: $who }`의 중괄호 안 값은 상수, 선언된 매개변수, `now()`만 받는다([`crates/omnigraph-compiler/src/query/ast.rs` L575-L581](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/query/ast.rs#L575-L581)). 타입 이름 앞의 `@`는 엔진 카탈로그 타입(`@Operator` 등)을 가리키며, 읽기 전용이고 사용자 스키마가 선언할 수 없다([`crates/omnigraph-compiler/src/query/query.pest` L261-L263](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/query/query.pest#L261-L263)).

### 코드 분석: 상관 블록

`not`, `exists`, 집계 블록은 모두 같은 `Subquery` 값으로 바뀐다. [`crates/omnigraph-compiler/src/query/ast.rs` L527-L547](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/query/ast.rs#L527-L547)

```rust
impl Subquery {
    /// `not { … }`: `count = 0`.
    pub fn not_block(clauses: Vec<Clause>) -> Self {
        Self::row_count(BlockKeyword::Not, clauses, CompOp::Eq)
    }

    /// `exists { … }`: `count > 0`.
    pub fn exists_block(clauses: Vec<Clause>) -> Self {
        Self::row_count(BlockKeyword::Exists, clauses, CompOp::Gt)
    }

    fn row_count(keyword: BlockKeyword, clauses: Vec<Clause>, op: CompOp) -> Self {
        Self {
            keyword,
            clauses,
            func: AggFunc::Count,
            arg: None,
            op,
            right: Expr::Literal(Literal::Integer(0)),
        }
    }
```

- `not { ... }`는 "블록 안 일치 개수 = 0", `exists { ... }`는 "개수 > 0"이다
- `count { ... } > 2`, `sum($m.size) { ... } > 100`도 같은 구조에 집계 함수와 비교값만 다르다
- 블록은 바깥에서 묶인 변수를 하나 이상 참조해야 한다(`T39`). 하강 단계에서 이 블록은 `AntiJoin` 연산이 된다

## 식과 이름 범위

필터, 반환, 정렬, 변경 대입과 조건은 모두 같은 `Expr` 타입을 쓴다([`crates/omnigraph-compiler/src/query/ast.rs` L660-L723](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/query/ast.rs#L660-L723)). 문법은 우선순위가 낮은 것부터 사다리 형태로 적혀 있다.

### 코드 분석: 식 우선순위

[`crates/omnigraph-compiler/src/query/query.pest` L210-L219](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/query/query.pest#L210-L219)

```text
expr              = { or_expr }
or_expr           = { and_expr ~ (kw_or ~ and_expr)* }
and_expr          = { not_expr ~ (kw_and ~ not_expr)* }
not_expr          = { (kw_not ~ not_expr) | comparison }
comparison        = { operand ~ ((filter_op ~ operand) | (kw_in ~ operand) | null_test)? }
null_test         = { kw_is ~ kw_not? ~ kw_null }
operand           = { "(" ~ expr ~ ")" | now_call | nearest_ordering | search_call
                    | fuzzy_call | match_text_call | bm25_call | rrf_call | agg_call
                    | prop_access | meta_field | variable | literal
                    | reserved_property | expr_ident }
```

- `or`가 가장 느슨하고 `and`, `not`, 비교 순으로 단단해진다. 사용자 문서도 `not $p.email is null`을 `not ($p.email is null)`로 읽는다고 설명한다([쿼리 언어 문서](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/docs/user/queries/index.md#L53-L62))
- 비교는 결합하지 않는다. `$a < $b < $c`는 파싱 오류다
- 검색 호출(`search`, `fuzzy`, `match_text`, `bm25`, `rrf`, `nearest`)도 피연산자 자리에 온다. 검색 함수의 의미는 07장에서 다룬다
- `and`, `or`, `not`, `is`, `null`, `in` 여섯 단어는 단어 경계까지 확인하는 예약어다. 그래서 `nothing`, `android`는 식별자로 남는다

맨 이름(`$` 없는 이름)의 뜻은 위치에 따라 다르다. 읽기 절에서는 `return`의 별칭이고, 변경 문장과 바인딩 안에서는 그 타입의 속성이다. 파서는 이 차이를 `NameScope` 값으로 들고 다닌다([`crates/omnigraph-compiler/src/query/parser.rs` L17-L34](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/query/parser.rs#L17-L34)).

`$p.@id`, `$e.@src`, `$e.@dst`, `$e.@type`은 메타 필드다([`crates/omnigraph-compiler/src/query/query.pest` L239-L242](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/query/query.pest#L239-L242)). 맨 이름 `$p.id`는 언제나 사용자 속성 `id`를 뜻한다. 타입 검사기는 메타 필드를 역할로 해석하고 항상 null 불가 `String`으로 본다([`crates/omnigraph-compiler/src/query/typecheck.rs` L1382-L1396](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/query/typecheck.rs#L1382-L1396)). 물리 철자로 바꾸는 일은 하강 단계가 한다.

## 타입 검사

타입 검사기는 카탈로그를 받아 선언 하나를 검사하고, 모든 식에 타입을 붙인 검사 결과를 만든다. 4천 줄이 넘는 `typecheck.rs`가 맡는다.

### 코드 분석: 읽기와 변경의 분기

[`crates/omnigraph-compiler/src/query/typecheck.rs` L706-L723](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/query/typecheck.rs#L706-L723)

```rust
pub fn typecheck_query_decl(catalog: &Catalog, query: &QueryDecl) -> Result<CheckedQuery> {
    if !query.mutations.is_empty() {
        let (params, param_types) = checked_params(&query.params)?;
        typecheck_unattached_base(&query.write_mode, &param_types)?;
        let mut ops = Vec::with_capacity(query.mutations.len());
        for mutation in &query.mutations {
            ops.push(typecheck_mutation(catalog, mutation, &param_types)?);
        }
        Ok(CheckedQuery::Mutation(CheckedMutationQuery {
            name: query.name.clone(),
            params,
            ops,
            write_mode: query.write_mode.clone(),
        }))
    } else {
        Ok(CheckedQuery::Read(typecheck_read_query(catalog, query)?))
    }
}
```

- 변경 문장이 하나라도 있으면 변경 쿼리로 검사한다. 결과 타입이 `CheckedQuery::Read`와 `CheckedQuery::Mutation`으로 갈리므로, 하강 함수가 다른 종류를 받을 수 없다
- 읽기 진입점 `CheckedRead::check`는 검사 결과와 원래 선언을 `Arc`로 함께 보관한다. 계획기가 선언에 쓰인 철자에서 요구 사항을 끌어내야 하기 때문이다([`crates/omnigraph-compiler/src/query/checked.rs` L1-L31](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/query/checked.rs#L1-L31))

변수 하나는 노드 바인딩 아니면 엣지 바인딩이고, 같은 이름 공간을 쓴다. 노드 타입과 엣지 타입은 이름이 같을 수 있으므로 종류가 판별자다([`crates/omnigraph-compiler/src/query/typecheck.rs` L20-L30](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/query/typecheck.rs#L20-L30)). 검사가 끝난 읽기 쿼리는 매개변수, 완료된 match 범위, 반환, 정렬, `limit`을 담는다([`crates/omnigraph-compiler/src/query/typecheck.rs` L70-L84](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/query/typecheck.rs#L70-L84)).

### 코드 분석: 순회 홉 범위 검사

[`crates/omnigraph-compiler/src/query/typecheck.rs` L2218-L2239](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/query/typecheck.rs#L2218-L2239)

```rust
if traversal.min_hops == 0 {
    return Err(CompilerError::typed(
        T15,
        "traversal min hop bound must be >= 1".to_string(),
    ));
}
if let Some(max_hops) = traversal.max_hops {
    if max_hops < traversal.min_hops {
        return Err(CompilerError::typed(
            T15,
            format!(
                "invalid traversal bounds {{{},{}}}; max must be >= min",
                traversal.min_hops, max_hops
            ),
        ));
    }
} else {
    return Err(CompilerError::typed(
        T15,
        "unbounded traversal is disabled; use bounded traversal {min,max}".to_string(),
    ));
}
```

- 최소 홉은 1 이상, 최대 홉은 최소 이상이어야 한다. 최대 홉 없는 무한 순회는 문법에는 있지만(`{1,}`) 타입 검사에서 거부한다
- 엣지 변수가 붙은 순회는 홉 범위를 쓸 수 없다. 여러 홉은 엣지 하나가 아니라 경로이기 때문이다([`crates/omnigraph-compiler/src/query/typecheck.rs` L2240-L2249](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/query/typecheck.rs#L2240-L2249))
- 엣지 선택은 이름 하나, 선택, 와일드카드에 따라 `EdgeSelection`의 세 변형이 된다. 선택의 구성원은 이름 순으로 정렬된다([`crates/omnigraph-compiler/src/query/typecheck.rs` L2262-L2298](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/query/typecheck.rs#L2262-L2298), [`crates/omnigraph-compiler/src/traversal.rs` L14-L27](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/traversal.rs#L14-L27))

### 코드 분석: 정렬과 limit의 관계

[`crates/omnigraph-compiler/src/query/typecheck.rs` L948-L963](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/query/typecheck.rs#L948-L963)

```rust
let has_rrf = query
    .order_clause
    .iter()
    .any(|ord| expr_contains_rrf(&ord.expr));
if has_rrf && query.limit.is_none() {
    return Err(CompilerError::typed(
        T21,
        "rrf ordering requires a limit clause".to_string(),
    ));
}
if has_standalone_nearest && query.limit.is_none() {
    return Err(CompilerError::typed(
        T17,
        "nearest ordering requires a limit clause".to_string(),
    ));
}
```

- `rrf`나 단독 `nearest`로 정렬하는 쿼리는 `limit`이 있어야 한다. 순위 검색은 상위 몇 개를 고르는 연산이라 범위가 필요하다
- 같은 함수 안에서 결과 열 이름이 겹치는 반환(`T25`), 집계 반환에 붙은 `distinct`(`T56`), 집계와 함께 쓴 일반 식(`T9`)도 거부한다([`crates/omnigraph-compiler/src/query/typecheck.rs` L869-L882](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/query/typecheck.rs#L869-L882), [`crates/omnigraph-compiler/src/query/typecheck.rs` L967-L994](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/query/typecheck.rs#L967-L994))

변경 검사도 같은 방식이다. 삽입은 대입이 하나 이상 있어야 하고(`T10`), null 불가 속성을 모두 채워야 한다(`T12`). 단 `@embed` 벡터 속성은 원본 텍스트 속성이 대입되면 생략할 수 있다([`crates/omnigraph-compiler/src/query/typecheck.rs` L1153-L1181](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/query/typecheck.rs#L1153-L1181)). 엣지 타입에는 `update`를 쓸 수 없다(`T16`, [`crates/omnigraph-compiler/src/query/typecheck.rs` L1276-L1287](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/query/typecheck.rs#L1276-L1287)).

| 코드 | 뜻 |
|---|---|
| `T1`, `T4` | 알 수 없는 노드 타입, 엣지 타입 |
| `T5` | 변수가 순회한 엣지의 끝점이 아님 |
| `T15` | 순회 홉 범위 또는 `nearest` 피연산자 오류 |
| `T17`, `T21` | `nearest`, `rrf` 정렬에 `limit` 없음 |
| `T23` | 노드나 엣지 변수를 맞지 않는 역할로 사용 |
| `T25` | 두 반환이 같은 결과 열을 만듦 |
| `T27` | 전문 검색 호출에 전문 인덱스가 없음 |
| `T41` | 불 식의 피연산자가 불 타입이 아님 |

전체 목록은 [`crates/omnigraph-compiler/src/query/codes.rs` L18-L84](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/query/codes.rs#L18-L84)에 있다.
