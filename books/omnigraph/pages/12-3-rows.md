행 단위 충돌 판정과 검증을 건너뛰는 순수 삽입 경로를 본다.

## 3방향 행 분류

target 도 base 이후 테이블을 바꿨으면 행 단위로 비교한다. 비교 규칙은 `run_three_way_classification` 하나이고, 전체 순서 스캔(ordered walk)과 lineage 후보 경로가 이 함수를 공유한다 ([merge.rs L1995-L2015](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/exec/merge.rs#L1995-L2015)). 세 cursor 는 모두 `id` 순서로 행을 내고, 매 단계에서 가장 작은 id 를 골라 그 id 를 가진 cursor 만 전진시킨다.

### 코드 분석: 행 선택 규칙

한 id 에 대해 결과 행을 고르는 부분이다. [merge.rs L2041-L2058](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/exec/merge.rs#L2041-L2058)

```rust
        let source_changed = !cursor_rows_equal(source_row.as_ref(), base_row.as_ref()).await?;
        let target_changed = !cursor_rows_equal(target_row.as_ref(), base_row.as_ref()).await?;

        let selection = if !source_changed {
            target_row.as_ref()
        } else if !target_changed {
            source_row.as_ref()
        } else if cursor_rows_equal(source_row.as_ref(), target_row.as_ref()).await? {
            // (주석 1줄 생략: 양쪽이 같게 바꿨으면 충돌이 아니다)
            target_row.as_ref()
        } else {
            conflicts.push(classify_merge_conflict(
                table_key,
                &next_id,
                base_row.is_some(),
                source_row.is_some(),
                target_row.is_some(),
            ));
```

행이 없는 것(`None`)도 하나의 값으로 비교하므로 삭제도 같은 규칙으로 처리된다. 한쪽만 바꿨으면 바꾼 쪽을, 양쪽이 같게 바꿨으면 target 을 고른다. 선택 결과가 `None` 인데 target 에 행이 있으면 삭제 목록에 id 와 target 쪽 `_rowid` 를 넣는다. 선택 행이 target 과 다르면, target 에 행이 있을 때 갱신 writer 로, 없을 때 삽입 writer 로 보낸다. target 과 같은 행은 delta 에 넣지 않는다.

### 코드 분석: 충돌 종류 판정

양쪽이 다르게 바꾼 id 의 충돌 종류는 세 쪽의 행 존재 여부로 정한다. [merge.rs L3351-L3371](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/exec/merge.rs#L3351-L3371)

```rust
fn classify_merge_conflict(
    table_key: &str,
    row_id: &str,
    base_present: bool,
    source_present: bool,
    target_present: bool,
) -> MergeConflict {
    let (kind, message) = match (base_present, source_present, target_present) {
        (false, true, true) => (
            MergeConflictKind::DivergentInsert,
            format!("divergent insert for id '{}'", row_id),
        ),
        (true, false, true) | (true, true, false) => (
            MergeConflictKind::DeleteVsUpdate,
            format!("delete/update conflict for id '{}'", row_id),
        ),
        _ => (
            MergeConflictKind::DivergentUpdate,
            format!("divergent update for id '{}'", row_id),
        ),
    };
```

| base | source | target | 충돌 종류 |
|---|---|---|---|
| 없음 | 있음 | 있음 | `DivergentInsert` |
| 있음 | 없음 | 있음 | `DeleteVsUpdate` |
| 있음 | 있음 | 없음 | `DeleteVsUpdate` |
| 있음 | 있음 | 있음 | `DivergentUpdate` |

나머지 네 종류(`OrphanEdge`, `UniqueViolation`, `CardinalityViolation`, `ValueConstraintViolation`)는 행 비교가 아니라 다음 절의 제약 검증이 만든다. 전체 목록은 [error.rs L131-L139](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-core/src/error.rs#L131-L139) 에 있다. 충돌이 하나라도 있으면 효과를 내기 전에 `OmniError::MergeConflicts` 를 돌려준다. 엣지를 양쪽에서 같은 끝점으로 넣은 경우의 결과는 엣지 타입의 `@key`, `@unique` 선언에 따라 달라진다 ([user/branching/merge.md L151-L170](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/docs/user/branching/merge.md#L151-L170)).

### lineage 후보와 전체 순서 스캔

모든 행을 세 번 정렬 스캔하면 비용이 테이블 크기에 비례한다. `merge_lineage` 설정이 `on` 이면 먼저 fragment 와 deletion 메타데이터에서 바뀐 id 후보만 찾는다. 기본값은 release 빌드에서 `On`, debug 빌드에서 `Verify` 다 ([settings.rs L441](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/settings.rs#L441)).

| `merge_lineage` | 동작 |
|---|---|
| `off` | 전체 순서 스캔만 사용 |
| `on` | lineage 후보 계획 시도, 실패하면 전체 순서 스캔 |
| `verify` | 두 경로를 모두 실행하고 행 결과가 다르면 merge 실패 |

설정 분기는 `stage_streaming_table_merge` 의 `match lineage` 에 있다 ([merge.rs L2986-L3063](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/exec/merge.rs#L2986-L3063)).

### lineage 계획의 진입 조건

`plan_lineage_merge` 는 증명할 수 없는 모양이면 `None` 을 돌려 전체 스캔으로 넘긴다. 조건은 Blob 스키마가 아님, 세 등록의 테이블 경로가 같음, 각 dataset 이 등록된 pin 에서 열렸고 stable row id 를 씀, `id` 가 null 불가 정확 기본 키임, 세 pin 의 스키마가 같음이다 ([merge.rs L2617-L2713](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/exec/merge.rs#L2617-L2713)). 스키마 조건은 메타데이터만 바꾼 스키마 변경에서 fragment 는 바이트가 같은데 행 비교로는 모든 행이 바뀐 것으로 보이는 경우를 막는다. 후보를 찾는 `lineage_side_candidates` 는 base 와 한쪽의 fragment 를 id 로 대응시켜, 데이터 파일과 deletion 파일이 모두 같은 fragment 는 건너뛰고 deletion vector 만 다른 fragment 는 차이 나는 offset 의 행만 후보로 넣는다.

후보 집합은 `LINEAGE_CANDIDATE_MAX_BYTES`(32 MiB, `KEYED_WRITE_MAX_BYTES` 와 같음)를 넘으면 전체 스캔으로 돌아간다 ([storage_layer.rs L71-L72](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/storage_layer.rs#L71-L72)). 예산 초과는 merge 실패가 아니라 최적화 실패로 취급한다. 후보는 512개 단위 `id IN (...)` 필터 청크로 나뉘어 세 dataset 에서 읽힌다.

### 읽을 fragment 범위

후보 id 를 읽을 때도 전체 테이블이 아니라 후보가 있을 수 있는 fragment 만 스캔한다 (`three_way_fragments`, [merge.rs L2230-L2250](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/exec/merge.rs#L2230-L2250)). 한쪽만 바꾼 후보는 다른 쪽에서 base 위치에 그대로 살아 있다. 그래서 base 는 양쪽의 base 범위를, 각 쪽은 자기가 바꾼 fragment 와 상대가 바꾼 base fragment 를 스캔한다. 범위 밖의 행은 양쪽에서 같은 위치에 살아 있고, 정확 id 기본 키 덕분에 한 버전 안에서 살아 있는 id 는 하나뿐이라는 것이 이 범위의 근거다.

> **주의**: lineage 후보 방식은 [RFC 0027](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/docs/rfcs/0027-lineage-merge-deltas.md#L1-L16) 에 `status: draft`, `implementation: not-started` 로 남아 있지만, 코드에는 `plan_lineage_merge` 와 `merge_lineage=on` 기본값이 이미 들어가 있다. RFC 0027 이 제안한 `_row_last_updated_at_version` 필터 방식이 아니라 fragment 와 deletion 파일 동일성 비교로 구현돼 있다. 이 책은 코드를 기준으로 쓴다.

### 전체 순서 스캔의 자원 상한

후보 방식이 실패하면 `OrderedTableCursor` 가 각 snapshot 을 `id` 순서로 스트리밍한다. 정렬에는 `id`, `_rowid`, `_rowaddr` 의 좁은 키만 들어가고, 전체 행은 정렬된 키를 청크 단위로 다시 읽어 채운다 ([docs/dev/merge.md L155-L170](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/docs/dev/merge.md#L155-L170)). 정렬 실행 하나의 메모리 풀은 150 MiB, spill 공간은 100 GiB 다 ([table_store.rs L134-L135](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/table_store.rs#L134-L135)). 청크 채우기의 유지 상한은 `HYDRATION_CHUNK_HARD_BYTES`(32 MiB 의 2배) 다 ([ordered_cursor.rs L45-L61](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/ordered_cursor.rs#L45-L61)). 같은 cursor 를 변경 피드와 export 도 쓰며, 그쪽 사용은 13장에서 다룬다.

## 증명된 순수 삽입 경로

source 가 base 이후 정확한 id 삽입만 했다는 것을 Lance 트랜잭션 이력으로 증명하면, base 를 스캔하거나 행을 비교하지 않고 source 의 삽입 구간만 읽어 적용할 수 있다. 증명에 쓰는 표식은 쓰기 경로가 남기는 트랜잭션 속성 `omnigraph.insert_absence = "v1"` 이다 ([docs/dev/merge.md L99-L102](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/docs/dev/merge.md#L99-L102)). 이 표식을 남기는 삽입 경로와 key 충돌 fence 는 9장에서 다룬다.

### 증명 전 빠른 거절

`try_proven_pure_insert_history` 는 비용이 드는 이력 조회 전에 등록 정보만으로 거를 수 있는 경우를 먼저 거른다. base 와 source 의 pin 이 같거나 행 수(`entity_count`)가 늘지 않았으면 순수 삽입일 수 없다 ([merge.rs L1004-L1015](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/exec/merge.rs#L1004-L1015)). `pin_version` 은 등록이 가리키는 staged detached 커밋 버전, 없으면 linear 버전이다. 그 뒤 두 dataset 을 pin 에서 열고 stable row id 를 확인하며, 같은 native branch 인데 branch identifier 가 다르면 수명이 바뀐 것으로 보고 read-set 변경 오류를 낸다 ([L1020-L1050](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/exec/merge.rs#L1020-L1050)). 기본 키가 null 불가 `Utf8` 타입의 `id` 하나가 아니면 증명을 포기한다.

### 이력 사슬 검사

`proven_chain_fragments` 는 source pin 에서 base pin 까지 `read_version` 링크를 하나씩 따라가며 각 버전의 트랜잭션을 확인한다 ([merge.rs L1125-L1174](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/exec/merge.rs#L1125-L1174)). 조사할 버전 수의 상한은 `PURE_INSERT_HISTORY_MAX_VERSIONS` 1,024 이고, 주석은 이것이 정확성 한계가 아닌 최적화 예산이라고 적는다. 증명된 삽입 행 수가 등록의 행 수 차이와 다르면 증명을 버린다.

| 증명 조건 | 내용 |
|---|---|
| 이전 버전 링크 | 각 버전이 정확한 이전 버전과 트랜잭션 UUID 를 가짐 |
| 트랜잭션 모양 | 물리 `id` 위의 삽입 전용 필터 Update, 또는 행을 옮기지 않는 index 전용 커밋 |
| 부수 효과 | 제거, 갱신된 fragment 없음, 관련 없는 유지보수 효과 없음 |
| 메타데이터 | 중첩 스키마와 index coverage 메타데이터 완비 |
| 행 수 | 물리 행 합계가 manifest delta 와 일치 |

표는 [docs/dev/merge.md L104-L119](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/docs/dev/merge.md#L104-L119) 을 옮긴 것이다. 문서는 이 표식이 서명이 아니며, Omnigraph 를 거치지 않은 Lance writer 는 지원하지 않는다고 밝힌다.

### 증명 결과의 두 용도

증명은 두 곳에 쓰인다. HEAD 전진이 필요한 테이블(main 대상, pre-v11 fork source)에서는 `AdoptPureInserts` 가 되어 증명된 구간을 target lineage 위에 삽입 chain 으로 재생한다 ([merge.rs L3782-L3795](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/exec/merge.rs#L3782-L3795)). 그 밖의 adopt 에서는 게시가 포인터 교체이고, 증명은 `AdoptValidation::PureInserts` 로 검증 단계에만 전달된다. 증명이 반환된 뒤 스캔 행 수가 맞지 않으면 경로를 조용히 바꾸지 않고 실패한다. Blob 컬럼이 있는 테이블은 외부 Blob 선택을 연산 전체 예산에 먼저 등록해야 하므로 같은 판정을 두 번째 루프에서 따로 수행한다.
