`.gq` 파일의 문장 종류와 쿼리 선언 형식을 본다.

## 파일 구조와 문장 종류

파일 최상위 규칙은 설정 줄 다음에 문장 하나 또는 쿼리 선언 여러 개가 온다는 것이다([`crates/omnigraph-compiler/src/query/query.pest` L8](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/query/query.pest#L8)). 문장은 다른 문장이나 선언과 함께 쓸 수 없고, 뒤에 무언가 더 오면 `statement_trailer` 규칙이 받아 `Q004`로 거부한다([`crates/omnigraph-compiler/src/query/query.pest` L34-L38](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/query/query.pest#L34-L38)).

### 코드 분석: 본문 종류

[`crates/omnigraph-compiler/src/query/ast.rs` L20-L31](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/query/ast.rs#L20-L31)

```rust
#[derive(Debug, Clone)]
pub enum FileBody {
    Queries(Vec<QueryDecl>),
    Branch(BranchStmt),
    Commit(CommitWrite),
    Load(LoadStmt),
    Show(Option<SettingId>),
    Explain(QueryDecl),
    /// `explain load csv …`: the read doors answer the accepted load plan
    /// as explain rows, reading no payload byte.
    ExplainLoad(LoadStmt),
}
```

| 본문 | 예 | 들어가는 곳 |
|---|---|---|
| `Queries` | `query a(...) { ... } query b(...) { ... }` | 읽기는 `query`, 변경은 `mutate` |
| `Branch` | `branch create b0 from main`, `branch diff b0` | 쓰기는 `mutate`, 읽기는 `query` |
| `Commit` | `commit publish "<id>"`, `commit drop "<id>"` | `mutate` |
| `Load` | `load csv "<uri>" into node:Person mode append` | `mutate` |
| `Show` | `show all;` | `query` |
| `Explain`, `ExplainLoad` | `explain query a(...) { ... }` | `query` |

브랜치 문장은 여섯 가지이며 `BranchStmt`가 쓰기(`Write`), `List`, 읽기(`Read`)로 나눈다([`crates/omnigraph-compiler/src/query/ast.rs` L330-L338](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/query/ast.rs#L330-L338)). `branch diff`, `branch merge-base`, `branch merge ... dry run`이 브랜치를 바꾸지 않는 읽기다([`crates/omnigraph-compiler/src/query/ast.rs` L340-L360](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/query/ast.rs#L340-L360)).

> **주의**: [RFC 0055 요약](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/docs/rfcs/0055-gq-branch-statements.md#L27-L48)은 `branch create`, `delete`, `merge`, `list` 네 문장만 정의한다. 코드는 문법 주석이 RFC 0030 §8을 근거로 드는 `diff`, `merge-base`와 `dry run` 절, `from commit "<id>"`, `as commit` 절까지 받는다([`crates/omnigraph-compiler/src/query/query.pest` L57-L78](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/query/query.pest#L57-L78)). RFC 0055는 상태가 draft이고 코드가 더 넓다.

### 코드 분석: 파일 파싱

`parse_query_diagnostic`은 문법을 돌리기 전에 중첩 깊이를 먼저 잰다. 괄호와 연속된 `not`은 64단계, `and`와 `or` 연결까지 센 식 깊이는 256단계가 상한이다([`crates/omnigraph-compiler/src/query/depth.rs` L18-L26](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/query/depth.rs#L18-L26)). 파서와 타입 검사기가 식 단계마다 재귀하므로, 상한 없이 두면 컴파일 스레드의 스택이 넘쳐 요청 거부 대신 프로세스가 끝날 수 있기 때문이다([`crates/omnigraph-compiler/src/query/depth.rs` L1-L13](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/query/depth.rs#L1-L13)). 그 뒤 pest 결과를 본문 종류별로 나눈다. [`crates/omnigraph-compiler/src/query/parser.rs` L100-L115](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/query/parser.rs#L100-L115)

```rust
Rule::setting_stmt => settings.push(parse_setting_stmt(inner)?),
Rule::branch_stmt => {
    statement = Some(FileBody::Branch(parse_branch_stmt(inner)?));
}
Rule::commit_stmt => {
    statement = Some(FileBody::Commit(parse_commit_stmt(inner)?));
}
Rule::load_stmt => {
    statement = Some(FileBody::Load(parse_load_stmt(inner)?));
}
Rule::show_stmt => {
    statement = Some(FileBody::Show(parse_setting_target(inner)?));
}
Rule::explain_stmt => {
    statement = Some(parse_explain_stmt(inner)?);
}
```

- 설정 줄은 몇 개든 모으고, 문장은 하나만 기록한다. 선언은 `queries`에 쌓였다가 문장이 없을 때 `FileBody::Queries`가 된다
- `query name { ... }`처럼 괄호를 빼먹은 선언은 `missing_param_list` 규칙이 받아 `Q002`와 함께 `()`를 넣는 수정안을 돌려준다. 수정안은 실제로 다시 파싱해 성공할 때만 기계 적용 가능으로 표시한다([`crates/omnigraph-compiler/src/query/parser.rs` L132-L161](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/query/parser.rs#L132-L161))

### 코드 분석: 설정 접두부

[`crates/omnigraph-compiler/src/query/query.pest` L21-L26](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/query/query.pest#L21-L26)

```text
setting_stmt  = { set_stmt | reset_stmt }
set_stmt      = { kw_set ~ setting_name ~ "=" ~ setting_value ~ ";" }
reset_stmt    = { kw_reset ~ (kw_all | setting_name) ~ ";" }
show_stmt     = { kw_show ~ (kw_all | setting_name) ~ ";" }
setting_name  = @{ ident ~ ("." ~ ident)* }
setting_value = { integer | ident | string_lit }
```

- 설정 이름은 문법이 아니라 정의 표에서 찾는다. 그래서 `set`, `show` 같은 단어도 본문 안에서는 식별자로 쓸 수 있다
- 정의 표 `DEFINITIONS`는 설정마다 종류, 기본값, 범위, 환경 변수, 설명을 담는다([`crates/omnigraph-compiler/src/settings.rs` L45-L58](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/settings.rs#L45-L58)). 범위가 `Process`인 설정은 요청이 바꿀 수 없다([`crates/omnigraph-compiler/src/settings.rs` L235-L240](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/settings.rs#L235-L240))
- `Session::effective`가 이 접두부를 호출 하나에만 적용한다(02장)

| 설정 | 기본값 | 범위 | 정의 |
|---|---|---|---|
| `engine` | `v2` | request | [L61-L68](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/settings.rs#L61-L68) |
| `rrf_plan` | `auto` | process | [L69-L76](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/settings.rs#L69-L76) |
| `merge_lineage`, `change_lineage` | `on` | request | [L77-L92](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/settings.rs#L77-L92) |
| `ann_nprobes` | `20` | request | [L93-L100](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/settings.rs#L93-L100) |
| `stage_write_concurrency` | `8` (1~64) | process | [L101-L111](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/settings.rs#L101-L111) |
| `traversal_work_limit` | `1000000` | request | [L112-L122](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/settings.rs#L112-L122) |
| `write_max_bytes`, `load_source_max_bytes` | `33554432` | request | [L123-L138](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/settings.rs#L123-L138) |
| `query_memory_bytes` | `157286400` | request | [L139-L146](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/settings.rs#L139-L146) |
| `history_release_bytes` | `262144` (1024~262144) | request | [L147-L157](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/settings.rs#L147-L157) |
| `explain_dump_dir` | 빈 값 | process | [L158-L165](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/settings.rs#L158-L165) |

`engine`의 값은 `v2` 하나뿐이다. 사용자 문서는 `show` 결과의 `source` 열이 `default`, `env`, `request`, `file` 중 하나라고 설명한다([세션 설정 문서](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/docs/user/queries/settings.md#L1-L16)).

## 선언, 매개변수, 쓰기 모드

### 코드 분석: 쿼리 선언 규칙

[`crates/omnigraph-compiler/src/query/query.pest` L113-L129](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/query/query.pest#L113-L129)

```text
query_decl = {
    "query" ~ ident ~ "(" ~ param_list? ~ ")" ~ query_annotation* ~ (as_commit_clause | as_unattached_clause)? ~ "{"
        ~ query_body
    ~ "}"
}
query_annotation = { description_annotation | instruction_annotation }
description_annotation = { "@description" ~ "(" ~ string_lit ~ ")" }
instruction_annotation = { "@instruction" ~ "(" ~ string_lit ~ ")" }

query_body = { read_query_body | mutation_body }
mutation_body = { mutation_stmt+ }
read_query_body = {
    match_clause
    ~ return_clause
    ~ order_clause?
    ~ limit_clause?
}
```

- 선언 하나는 이름, 매개변수, 주석, 선택적인 쓰기 절, 본문으로 이뤄진다. 본문은 읽기(`match`, `return`, `order`, `limit`)와 변경(`insert`, `update`, `delete` 문장 목록) 중 하나다
- 주석은 `@description`과 `@instruction` 두 가지뿐이다. 저장 쿼리를 도구로 노출하는 클라이언트가 이 값을 쓴다
- 매개변수 타입은 스키마의 기본 타입, 목록, `Vector(N)`이고 끝의 `?`는 null 허용이다([`crates/omnigraph-compiler/src/query/query.pest` L138-L144](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/query/query.pest#L138-L144))

### 코드 분석: 쓰기 모드

변경 선언의 결과가 어디로 가는지는 본문 앞의 절로 정한다. [`crates/omnigraph-compiler/src/query/ast.rs` L476-L493](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/query/ast.rs#L476-L493)

```rust
/// Where a mutation's result goes: the branch head (`Publish`), an
/// unpublished commit (`as commit`), or an unattached snapshot
/// (`as unattached branch [on <base>]`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MutationWriteMode {
    Publish,
    AsCommit,
    Unattached { base: Option<UnattachedBase> },
}

/// The `on` operand of `as unattached branch`: the base snapshot's id as a
/// quoted literal, or the name (without `$`) of a `String` parameter that
/// carries it per invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnattachedBase {
    Literal(String),
    Param(String),
}
```

- 절이 없으면 `Publish`로 브랜치 헤드를 옮긴다. `as commit`은 게시되지 않은 커밋, `as unattached branch [on <base>]`는 어느 브랜치에도 붙지 않은 스냅샷을 만든다
- 읽기 선언에 쓰기 절을 붙이면 파서가 거부한다([`crates/omnigraph-compiler/src/query/parser.rs` L824-L845](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/query/parser.rs#L824-L845))
- `on $base`의 매개변수는 선언되어 있고 null 불가 `String`이어야 한다(`T55`). 엔진은 이 모드로 쓰기 경로를 고른다(02장, 09장)

### 코드 분석: 선언 선택

엔진의 읽기 진입점은 원본 전체를 파싱한 뒤 이름으로 선언 하나를 고른다. [`crates/omnigraph-compiler/src/query_input.rs` L283-L295](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/query_input.rs#L283-L295)

```rust
pub fn find_read_statement(query_source: &str, query_name: &str) -> RunInputResult<ReadStatement> {
    let body = parse_query(query_source)?.body;
    let explain = body.is_explain();
    let queries = body
        .into_read_declarations()
        .map_err(RunInputError::message)?;
    let decl = named_declaration(queries, query_name)?;
    Ok(if explain {
        ReadStatement::Explain(decl)
    } else {
        ReadStatement::Query(decl)
    })
}
```

- `explain` 문장이면 감싼 선언을 고르고 `Explain` 표시를 붙인다. 감싼 선언에 변경 문장이 있으면 거부한다([`crates/omnigraph-compiler/src/query/ast.rs` L260-L270](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/query/ast.rs#L260-L270))
- 변경 진입점은 `find_named_query`를 쓴다. 문장 파일이면 "쿼리 선언이 아니다"라는 메시지로 거부한다([`crates/omnigraph-compiler/src/query_input.rs` L246-L258](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/query_input.rs#L246-L258))

HTTP와 CLI가 받은 JSON 매개변수는 `json_params_to_param_map`이 선언 타입에 맞춰 `Literal`로 바꾼다. null 허용 매개변수가 빠지면 `Null`로 채우고, null 불가 매개변수에 `null`이 오면 거부한다([`crates/omnigraph-compiler/src/query_input.rs` L325-L354](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/query_input.rs#L325-L354)). `JavaScript` 모드에서는 정수가 JS 안전 정수 범위(±9007199254740991)를 넘으면 거부하고 10진 문자열로 넘기라고 안내한다([`crates/omnigraph-compiler/src/query_input.rs` L12-L17](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/query_input.rs#L12-L17), [`crates/omnigraph-compiler/src/query_input.rs` L669-L674](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/query_input.rs#L669-L674)).

RFC 0041은 요청 본문에 원본을 담는 인라인 쿼리와 서버가 시작할 때 미리 파싱해 두는 저장 쿼리를 구분하고, 두 경로가 같은 엔진 코드로 모인다고 정리한다([RFC 0041 비교 표](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/docs/rfcs/0041-inline-stored-queries.md#L77-L85)). 컴파일러 입장에서는 둘 다 같은 `QueryDecl`이다. 저장 쿼리 등록과 Cedar 동작은 14장에서 다룬다.
