insert, update, delete 연산이 행을 만드는 과정을 본다.

## 변경 연산 실행

공개 진입점은 `Session`의 `mutate` 계열 함수다. `mutate`, `mutate_as`, `mutate_with_receipt` 등은 모두 `mutate_as_with_optional_branch_receipt`로 모이고, 이 함수가 쿼리 선언을 찾아 쓰기 경로(`WriteRoute`)를 정한 뒤 세션 설정에서 스테이징 동시성과 쓰기 예산을 꺼낸다.

### 쓰기 경로 (`WriteRoute`)

`resolve_write_route`는 선언의 쓰기 모드에 따라 경로를 정하고, 대상 브랜치에 대한 정책 `change` 검사를 먼저 수행한다 ([mutation.rs L1108-L1166](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/exec/mutation.rs#L1108-L1166)). 경로는 이후 게시 방식으로 옮겨진다.

| `WriteRoute` | 쿼리 모드 | 게시 방식 (`Publication`) | 브랜치 헤드 |
|---|---|---|---|
| `Publish` | 기본 | `Publish` | 전진 |
| `AsCommit` | `as commit` | `Propose(None)` | 유지 |
| `UnattachedRoot` | `as unattached branch` | `Unattached(Head)` | 유지 |
| `UnattachedChild` | `as unattached branch on <id>` | `Unattached(base)` | 유지 |

unattached 자식 쓰기는 기준 스냅샷이 기록한 대상 브랜치로 향하며, 정책 거부를 "커밋 없음"으로 가린다. 게시 전 커밋과 unattached 스냅샷의 의미는 11장에서 다룬다.

### 코드 분석: `mutate_one_attempt`

파일: `crates/omnigraph/src/exec/mutation.rs`, [mutation.rs L1322-L1479](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/exec/mutation.rs#L1322-L1479)

시도 하나의 흐름은 권한 캡처, 호출자 CAS 확인, 하향(lowering), 실행, 스테이징과 게시 순서다. 호출자가 `--if-commit`으로 헤드를 지정했을 때의 확인을 본다.

```rust
        if let Some(expected) = expected_head {
            let actual = txn.effective_graph_head.as_deref();
            if actual != Some(expected) {
                return Err(OmniError::precondition_failed(
                    requested.as_deref().unwrap_or("main"),
                    expected,
                    actual.map(str::to_string),
                ));
            }
            txn.caller_expected_graph_head = Some(expected.to_string());
        }
```

호출자 헤드는 별도 조회가 아니라 이 시도가 캡처한 뷰와 비교한다. 별도 조회를 하면 조회와 실행 사이에 경쟁이 다시 생기기 때문이다. 불일치는 `ReadSetChanged`가 아닌 `PreconditionFailed`라 재준비 루프가 다시 시도하지 않는다. 이어서 쿼리를 캡처한 카탈로그로 하향하고, 모든 연산이 Insert인지로 `selects_rows`를 정해 `attempt.admit`에 넘긴다.

실행 결과가 아무것도 스테이징하지 않았다면 게시를 하지 않는다. 이때 호출자 CAS가 있으면 효과 있는 쓰기와 같은 게이트 순서로 권한을 다시 확인해, 효과 없는 호출에도 선형화 지점을 만든다. 스테이징한 것이 있으면 `commit_staged_mutation`과 `publish_committed_mutation`을 차례로 부른다.

### 코드 분석: 구성과 파괴의 분리 (D₂ 규칙)

파일: `crates/omnigraph/src/exec/mutation.rs`, [mutation.rs L866-L892](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/exec/mutation.rs#L866-L892)

하나의 변경 쿼리가 삽입, 갱신과 삭제를 섞으면 I/O 전에 거부한다. 거부 조건과 오류 문구를 본다.

```rust
    if has_constructive && has_delete {
        return Err(OmniError::manifest(format!(
            "mutation '{}' on the same query mixes inserts/updates and deletes; \
             split into separate mutations: (1) inserts and updates, then (2) deletes. \
             A query is deliberately constructive or destructive, not both, so its \
             read-your-writes stays unambiguous; run the two on a branch and merge \
             if you need them in one atomic commit.",
            ir.name
        )));
    }
```

삽입과 갱신은 메모리의 배치로, 삭제는 predicate로 쌓인다. 둘을 섞지 않으면 쿼리 안의 읽기가 대기 중인 삽입과 삭제 predicate를 맞춰 볼 필요가 없고, 테이블 하나가 쿼리당 최대 한 버전만 커밋한다. 함수 위 주석은 이것이 임시 구조가 아니라 의도한 의미 경계라고 밝힌다. 두 종류를 한 커밋으로 묶어야 하면 브랜치에서 나눠 실행한 뒤 병합하라고 안내하며, 병합은 12장에서 다룬다.

### 연산별 실행

`execute_named_mutation`은 IR의 연산을 순서대로 실행하고 영향받은 노드와 엣지 수를 더한다. 실행 전에 노드 타입마다 몇 번 쓰이는지 세어 두는데, 한 타입을 한 문장만 쓸 때에만 `update`가 열 패치(column patch)를 쓸 수 있기 때문이다.

| 연산 | 함수 | 스테이징 방식 |
|---|---|---|
| 노드, 엣지 삽입 | `execute_insert` ([mutation.rs L1685-L1866](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/exec/mutation.rs#L1685-L1866)) | `@key`로 `id`를 만들고 배치 추가 |
| 노드 갱신 | `execute_update` ([mutation.rs L1867-L2155](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/exec/mutation.rs#L1867-L2155)) | 일치 행을 읽어 전체 행 또는 열 패치 추가 |
| 노드 삭제 | `execute_delete_node` ([mutation.rs L2177-L2388](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/exec/mutation.rs#L2177-L2388)) | predicate, 삭제 id, 행 주소 기록 후 엣지 연쇄 삭제 |
| 엣지 삭제 | `execute_delete_edge` ([mutation.rs L2389-L2456](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/exec/mutation.rs#L2389-L2456)) | predicate와 행 주소 기록 |

갱신은 엣지 타입을 거부하고, `@key`를 구성하는 속성을 바꾸는 것도 거부한다. 키 일부만 바꾸면 행이 자기 키로 찾아지지 않기 때문이다. 열 패치 여부는 `column_patch_plan`이 정한다. Blob을 할당하지 않고, 할당하지 않은 Blob 또는 벡터 열이 하나 이상 있고, 키와 유일성 그룹이 빠진 열을 가리키지 않을 때 후보가 된다 ([mutation.rs L513-L588](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/exec/mutation.rs#L513-L588)). 실제 선택은 바로 아래 소절의 `TableStore::scan_update_matches` 가 비용을 비교해 정한다.

### 코드 분석: 갱신 쓰기 방식 선택과 merge-insert 분기

파일: `crates/omnigraph/src/table_store.rs`, [table_store.rs L7531-L7541](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/table_store.rs#L7531-L7541)

`execute_update` 가 부르는 `scan_update_matches`([table_store.rs L6709-L6715](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/table_store.rs#L6709-L6715))는 일치 행을 읽으면서 `UpdateWriteCost` 를 채우고, 이 함수로 쓰기 방식을 고른다.

```rust
    pub fn mode(&self) -> UpdateWriteMode {
        if self.external_blob
            || self.rows_disallowed
            || self.rows_over_budget
            || self.patch_bytes() < self.rows_bytes()
        {
            UpdateWriteMode::ColumnPatch
        } else {
            UpdateWriteMode::Rows
        }
    }
```

| 방식 | Lance 동작 | 비용 추정 |
|---|---|---|
| `ColumnPatch` | `id` 와 할당 열만 닿은 fragment 에 덧쓴다(`RewriteColumns`) | 원본 행(`id` 와 할당 값) 크기에 닿은 fragment 의 물리 행 수를 곱한 값, 인덱스가 있는 할당 열은 그 열을 한 번 더 읽는 비용 추가 |
| `Rows` | 일치 행 전체를 새 fragment 로 옮긴다(`RewriteRows`) | 옮겨지는 열, 벡터, 관리형 Blob 페이로드를 합한 일치 행 크기 |

비용 정의는 [table_store.rs L7509-L7526](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/table_store.rs#L7509-L7526) 에 있다. 열 패치는 닿은 fragment 의 모든 행을 덮는 열 파일을 쓰므로 일치 행이 적어도 크기가 거의 줄지 않고, 행 재작성은 일치 행만큼 읽고 쓴다. 비용과 무관하게 열 패치로 가는 경우가 세 가지 있다. 일치 행의 생략된 Blob 열이 외부 참조를 담으면 열 패치만 그 참조를 읽지 않고 보존한다. 문장이 여럿인 변경 연산은 다른 테이블과 쓰기 예산을 공유하므로 `rows_allowed` 가 꺼진다. 행 재작성이 쓰기 예산을 넘으면 열 패치로 내려간다. 주석([table_store.rs L7477-L7480](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/table_store.rs#L7477-L7480))은 이 선택이 물리 계획의 결정일 뿐 결과를 바꾸지 않는다고 적는다. 레거시 Blob 이나 원시 타입이 아닌 벡터처럼 폭을 정할 수 없는 열이 생략되면 추정 없이 행 재작성으로 간다.

키 있는 쓰기 쪽의 분기는 `stage_keyed_write` 안에 있다([table_store.rs L4007-L4043](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/table_store.rs#L4007-L4043)). 엄격한 삽입은 id 확인 뒤 부재가 증명된 삽입 작성기로 바로 간다. upsert 는 기본적으로 Lance merge-insert 가 원본을 테이블의 모든 id 와 join 하는 경로를 탄다. 다만 id BTREE 가 있고 BTREE 가 덮지 못한 fragment 수와 원본 id 수의 곱이 `KEYED_WRITE_MAX_ROWS` 이하이면, 먼저 id 존재 수를 센다.

- 모두 새 id 이면 엄격한 삽입과 같은 부재 증명 경로로 스테이징한다.
- 모두 있는 id 이고, 덮인 fragment 가 덮이지 않은 fragment 보다 많고, 덮이지 않은 꼬리의 데이터 파일이 `KNOWN_PRESENT_TAIL_MAX_BYTES`(1 MiB, [table_store.rs L156](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/table_store.rs#L156-L156)) 이하이면 `KnownPresentUpdate` 로 바꿔 인덱스 merge 를 탄다.
- 섞여 있으면 join 경로를 유지한다.

인덱스 merge 는 BTREE 페이지를 읽은 뒤 덮이지 않은 fragment 를 원본의 모든 열에 대해 스캔하므로, 꼬리가 작을 때만 join 보다 싸다는 것이 주석의 설명이다. join 경로의 결과 통계가 모든 행을 삽입했음을 증명하면 upsert 에도 삽입 부재 인증서를 붙여, 이후 브랜치 병합이 대상 확인을 생략할 수 있게 한다. 인증서의 쓰임은 12장에서 다룬다.

노드 삭제는 그 노드를 끝점으로 가진 엣지 타입을 모두 찾아 연쇄 삭제한다. 카탈로그의 엣지 타입을 돌며 시작점과 끝점 열을 고르는 부분이다 ([mutation.rs L2261-L2271](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/exec/mutation.rs#L2261-L2271)).

```rust
        for (edge_name, from_type, to_type) in &edge_info {
            let mut endpoints = Vec::new();
            if from_type == type_name {
                endpoints.push(txn.catalog.system_columns.src);
            }
            if to_type == type_name {
                endpoints.push(txn.catalog.system_columns.dst);
            }
            if endpoints.is_empty() {
                continue;
            }
```

일치한 엣지가 있으면 삭제 id, predicate, 행 주소를 엣지 테이블에 기록한다. 연쇄 삭제도 같은 쿼리의 같은 스테이징에 쌓이므로 노드 삭제와 엣지 삭제가 한 그래프 커밋으로 게시된다.
