쿼리 동안 쓰기를 모으고 테이블별로 스테이징하는 구조를 본다.

## 스테이징 누적기와 키 충돌 fence

`MutationStaging`은 쿼리 하나 동안 테이블별 입력 배치와 삭제 predicate를 메모리에 모은다. 쿼리 끝에서 테이블마다 Lance 트랜잭션 하나를 스테이징하고, 이후 `commit_all`로 넘어간다. 같은 누적기를 대량 적재기도 쓰며 ([staging.rs L1-L24](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/exec/staging.rs#L1-L24)), 적재기 쪽 사용법은 10장에서 다룬다. 쿼리 안의 읽기는 커밋된 Lance 스캔과 대기 중인 Arrow 배치를 DataFusion `MemTable`로 합쳐 읽으므로, 같은 쿼리에서 앞 문장이 삽입한 행을 뒤 문장이 볼 수 있다.

### 코드 분석: `MutationStaging` 구조체

파일: `crates/omnigraph/src/exec/staging.rs`, [staging.rs L122-L163](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/exec/staging.rs#L122-L163)

삽입과 갱신 쪽 필드를 본다.

```rust
    pub(crate) expected_versions: HashMap<String, u64>,
    /// Per-table identifiers captured on first touch.
    pub(crate) paths: HashMap<String, StagedTablePath>,
    /// In-memory accumulated batches per table (insert/update path).
    pub(crate) pending: HashMap<String, PendingTable>,
    /// Monotonic retained-batch accounting, updated only by `append_batch`.
    pending_bytes: u64,
    pending_payload_bytes: u64,
    write_budget: WriteBudget,
```

`expected_versions`는 테이블을 처음 건드릴 때의 manifest 버전이며 게시 CAS의 기준이 된다. `ensure_path`는 처음 건드릴 때만 경로와 기대 버전을 기록하고, 같은 테이블을 다른 식별자로 다시 건드리면 `ReadSetChanged`를 낸다. 삭제 쪽은 predicate(`delete_predicates`) 외에 삭제한 id(`deleted_ids`)와 pin 시점의 행 주소(`deleted_row_addrs`)를 함께 모은다. id는 검증과 변경 기록에, 행 주소는 테이블을 다시 스캔하지 않고 삭제 벡터를 쓰는 데 쓰인다.

테이블별 누적은 `PendingMode`로 구분된다 ([staging.rs L50-L60](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/exec/staging.rs#L50-L60)). 모드 이름은 Lance 연산이 아니라 그래프 의미를 나타내며, 키가 있는 그래프 테이블은 Lance의 단순 `Append`를 쓰지 않는다.

| 모드 | 의미 | Lance 쓰기 방식 |
|---|---|---|
| `StrictInsert` | 기존 `id`가 있으면 거부 | `KeyedWriteSemantics::StrictInsert` |
| `Upsert` | 새 `id`는 삽입, 기존 `id`는 전체 행 교체 | `KeyedWriteSemantics::Upsert` |
| `Overwrite` | 테이블 전체 교체 | 스테이징된 overwrite |
| `ColumnPatch` | 기존 행의 일부 열만 교체 | `KeyedWriteSemantics::KnownPresentColumnPatch` |

### 코드 분석: `append_batch`의 크기 상한

파일: `crates/omnigraph/src/exec/staging.rs`, [staging.rs L324-L333](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/exec/staging.rs#L324-L333)

`append_batch`는 배치를 추가하면서 모드 혼합과 스키마 불일치를 막는다 ([staging.rs L255-L366](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/exec/staging.rs#L255-L366)). Overwrite는 다른 모드와 섞이지 않고, `ColumnPatch`는 같은 테이블의 다른 쓰기와 함께 올 수 없다. `StrictInsert` 뒤에 `Upsert`가 오면 테이블 모드가 `Upsert`로 올라간다. 키 있는 모드는 행 수를 누적해 상한과 비교한다.

```rust
            let rows = existing_rows
                .checked_add(batch.num_rows())
                .ok_or_else(|| OmniError::manifest_internal("pending keyed row count overflow"))?;
            if rows > KEYED_WRITE_MAX_ROWS {
                return Err(OmniError::resource_limit(
                    format!("keyed entities for {table_key}"),
                    KEYED_WRITE_MAX_ROWS as u64,
                    rows as u64,
                ));
            }
```

바이트도 같은 방식으로 `write_budget`과 비교한다. 행 데이터와 Blob 페이로드는 같은 크기의 독립된 예산을 따로 가진다.

| 상한 | 값 | 정의 위치 |
|---|---|---|
| 테이블당 키 있는 행 수 | 8192 | [storage_layer.rs L71](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/storage_layer.rs#L71-L71) |
| 키 있는 쓰기 바이트 | 32 MiB | [storage_layer.rs L72](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/storage_layer.rs#L72-L72) |
| `write_max_bytes` 세션 설정 기본값 | 33554432 | [settings.rs L123-L130](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/settings.rs#L123-L130) |
| `stage_write_concurrency` 기본값 | 8 (1 이상 64 이하) | [settings.rs L101-L111](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/settings.rs#L101-L111) |
| Blob 셀 쓰기 페이로드 | `KEYED_WRITE_MAX_BYTES`와 동일 | [blob.rs L50](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/blob.rs#L50-L50) |

### 코드 분석: `stage_all_with_concurrency`

파일: `crates/omnigraph/src/exec/staging.rs`, [staging.rs L503-L724](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/exec/staging.rs#L503-L724)

먼저 `prepare_pending_table`이 테이블별 배치를 Lance writer가 받을 입력 하나로 줄인다. `Upsert`와 `ColumnPatch`는 `dedupe_merge_batches_by_id`로 같은 `id`의 마지막 행만 남기고(last-write-wins), `StrictInsert`는 배치를 이어 붙인다 ([staging.rs L879-L960](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/exec/staging.rs#L879-L960)). 다음으로 모든 테이블의 외부 Blob URI를 모아 한 번에 정책 확인과 원본 확인을 하고, 그 뒤에야 페이로드를 읽는다 ([staging.rs L560-L604](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/exec/staging.rs#L560-L604)). 마지막으로 테이블별 스테이징을 설정된 동시성으로 실행한다.

```rust
                    stage_pending_table(
                        db,
                        table_key,
                        table,
                        path,
                        expected,
                        write_budget,
                        overwrite_writers,
                    )
                    .await
                },
            ))
            .buffered(concurrency)
```

([staging.rs L652-L664](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/exec/staging.rs#L652-L664)) 테이블마다 독립된 dataset이라 테이블 사이에는 공유 상태가 없다. 중간에 실패하면 일부 테이블만 스테이징 파일을 남기지만, 이 파일은 어떤 manifest도 참조하지 않으므로 그래프에 보이지 않고 `cleanup`이 회수한다. `stage_pending_table`은 pin을 열고 키 있는 쓰기 또는 overwrite를 스테이징만 하며, 커밋은 하지 않는다.

### 삭제 predicate 결합

삭제는 쓰기가 끝난 뒤 두 번째 단계에서 같은 스테이징 목록에 들어간다. 한 테이블에 predicate 가 여럿이면 `combine_delete_predicates`([staging.rs L1016-L1021](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/exec/staging.rs#L1016-L1021))가 각 predicate 에 `IS TRUE` 를 붙여 `OR` 로 묶는다. `p1 OR p2` 대신 이렇게 묶는 이유는 함수 주석에 있다. Lance의 단순화기가 nullable 열에 대한 서로 다른 두 `NOT IN` 조건을 `true`로 접어 null 행까지 지우는 문제(#1277)를 피하기 위해서다. `stage_delete_table`은 실행 단계가 기록한 행 주소의 버전이 지금 연 버전과 같으면 주소로 바로 삭제 벡터를 쓰고, 아니면 Lance의 스캔 삭제로 돌아간다 ([staging.rs L1043-L1053](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/exec/staging.rs#L1043-L1053)). 일치하는 행이 없으면 아무것도 스테이징하지 않는다.

### 키 충돌 fence (RFC 0023)

변경 쿼리가 스테이징하는 키 있는 쓰기는 RFC 0023 "Substrate-native key-conflict fencing"의 경로를 따른다. 결정 요약은 같은 `id`를 삽입할 수 있는 두 동시 작업이 조용히 둘 다 커밋되지 않아야 한다는 보장이다 ([0023-key-conflict-fencing.md L77-L93](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/docs/rfcs/0023-key-conflict-fencing.md#L77-L93)). 이를 위해 Lance가 `Operation::Update`에 붙이는 삽입 행 키 필터(`KeyExistenceFilter`)를 충돌 기본 수단으로 쓰고, 별도의 잠금 테이블이나 트랜잭션 관리자를 만들지 않는다.

### 코드 분석: `KeyedWriteSemantics`

파일: `crates/omnigraph/src/storage_layer.rs`, [storage_layer.rs L286-L295](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/storage_layer.rs#L286-L295)

```rust
pub enum KeyedWriteSemantics {
    /// Insert every source row and reject an `id` already present at the pinned
    /// base.  A proven pre-existing match surfaces `OmniError::KeyConflict`.
    StrictInsert,
    /// Insert new ids and replace the full row for ids already present.
    Upsert,
    /// Replace full rows only for ids already proven present by a coherent
    /// merge classification. Missing ids are an internal read-set change,
    /// never inserts.
    KnownPresentUpdate,
```

네 번째 값 `KnownPresentColumnPatch`는 열 패치용이다. 저장소 어댑터가 이 모드를 Lance merge-insert 동작으로 대응시키므로, 호출자가 키를 `id` 외의 열로 바꾸거나 엄격한 삽입을 upsert로 약하게 만들 수 없다. 스테이징 함수 `stage_keyed_write`는 대상 dataset이 정확히 `id` 하나를 unenforced primary key로 선언했는지 확인한다 ([storage_layer.rs L812-L827](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/storage_layer.rs#L812-L827)). RFC 0023의 경로 표에 따르면 엄격한 삽입은 `id` 기준 merge-insert에 `WhenMatched::Fail`, upsert는 `WhenMatched::UpdateAll`을 쓰고 둘 다 필터 있는 경로를 탄다. 삭제는 스테이징된 Lance 삭제이며 PK 필터를 삭제 충돌 수단으로 쓰지 않는다 ([0023-key-conflict-fencing.md L246-L255](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/docs/rfcs/0023-key-conflict-fencing.md#L246-L255)).

id BTREE 유무와 존재 수에 따른 upsert 의 경로 분기는 앞의 "갱신 쓰기 방식 선택과 merge-insert 분기" 소절에서 보았다.
