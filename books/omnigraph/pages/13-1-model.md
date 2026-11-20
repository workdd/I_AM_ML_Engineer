공개 변경 모델과 두 스냅샷 비교 방식을 본다.

## 공개 변경 모델

`model.rs` 의 타입은 그래프 용어만 쓴다. 머리 주석은 테이블 키, incarnation 번호, Lance 버전, 행 주소 같은 물리 용어가 공개 타입에 나타나지 않는다고 적는다 ([model.rs L1-L12](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/changes/model.rs#L1-L12)).

변경 하나는 `GraphEntityChange`, 커밋 하나의 변경 묶음은 `GraphChangeBlock`, 그 원인은 `ChangeCause` 다. 타입은 불투명 id 와 스키마 이름을 묶은 `GraphTypeRef` 로, 행 값은 `EntityImage` 로 나타낸다. 필터는 kind, 타입 이름, 연산 세 차원을 가진 `ChangeFeedScope` 이고, 빈 차원은 모두 통과시킨다.

### 코드 분석: GraphEntityChange

파일: `crates/omnigraph/src/changes/model.rs`, [model.rs L115-L129](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/changes/model.rs#L115-L129)

변경 하나가 어떤 필드로 구성되는지 본다.

```rust
/// One entity change. Cause is stated once on the enclosing block.
///
/// An insert has only `after`; an update has exact `before` and `after`; a
/// delete has only `before`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GraphEntityChange {
    pub kind: ChangeEntityKind,
    pub entity_type: GraphTypeRef,
    pub id: String,
    pub op: ChangeOpKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub before: Option<EntityImage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub after: Option<EntityImage>,
}
```

Insert 는 `after` 만, Delete 는 `before` 만, Update 는 둘 다 가진다. 원인은 블록의 `ChangeCause` 에 한 번 적는다 ([model.rs L131-L149](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/changes/model.rs#L131-L149)). `authored_at` 은 커밋 시각이 아니라 효과 이전에 정해지고 publisher 재시도에도 바뀌지 않는 작성 시각이다. 엣지의 끝점은 이미지 안에 들어가므로 끝점을 옮기는 update 는 before 와 after 의 끝점이 다르다.

### 페이지 한도

페이지 크기는 변경 수와 직렬화 바이트 두 축으로 제한한다 ([model.rs L18-L38](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/changes/model.rs#L18-L38)).

| 상수 | 값 | 의미 |
|---|---|---|
| `COMMIT_CHANGES_DEFAULT_CHANGES` | 1,000 | 페이지당 기본 변경 수 |
| `COMMIT_CHANGES_MAX_CHANGES` | 8,192 | 요청 가능한 변경 수 상한 |
| `COMMIT_CHANGES_DEFAULT_BYTES` | 4 MiB | 기본 바이트 packing 목표 |
| `COMMIT_CHANGES_MAX_BYTES` | 32 MiB | 요청 가능한 packing 목표 상한 |
| `CHANGE_FEED_DEFAULT_COMMITS_PER_POLL` | 128 | poll 한 번이 살피는 기본 커밋 수 |
| `CHANGE_FEED_MAX_COMMITS_PER_POLL` | 512 | poll 당 커밋 수 상한 |

바이트 값은 상한이 아니라 packing 목표다. Blob 을 base64 로 인라인한 update 하나가 32 MiB 를 넘을 수 있고, 그런 변경은 혼자 한 페이지로 전달된다. 0 이나 상한 초과 요청은 `validate_change_page_limits` 가 타입이 있는 오류로 거절한다.

## 커밋 단위 열거기

`enumerate.rs` 는 첫 부모 간선 `P -> C` 하나의 엔티티 변경을 계산한다. 두 끝 버전을 `id` 순으로 병합해 한쪽에만 있으면 insert 또는 delete, 양쪽에 있으면 구조 비교로 update 여부를 정한다. 블록 안 출력 순서는 `(kind: 노드 먼저, 불투명 타입 id, id, 연산 rank)` 로 고정되어 있다 ([enumerate.rs L1-L14](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/changes/enumerate.rs#L1-L14)).

### 코드 분석: next_emit

파일: `crates/omnigraph/src/changes/enumerate.rs`, [enumerate.rs L190-L201](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/changes/enumerate.rs#L190-L201)

정렬된 두 행 스트림을 병합해 다음 변경을 고르는 부분을 본다.

```rust
        let left_id = from.peek().await?.map(|row| row.id.clone());
        let right_id = to.peek().await?.map(|row| row.id.clone());
        let emit = match (left_id, right_id) {
            (None, None) => return Ok(None),
            (Some(_), None) => Emit::Delete(from.pop().await?.expect("peeked row present")),
            (None, Some(_)) => Emit::Insert(to.pop().await?.expect("peeked row present")),
            (Some(left), Some(right)) if left < right => {
                Emit::Delete(from.pop().await?.expect("peeked row present"))
            }
            (Some(left), Some(right)) if left > right => {
                Emit::Insert(to.pop().await?.expect("peeked row present"))
            }
```

부모 쪽 id 가 작으면 그 행은 자식에 없으므로 delete, 자식 쪽 id 가 작으면 insert 다. id 가 같으면 `rows_equal_across_vintages` 로 비교해 같으면 건너뛰고 다르면 update 를 만든다. 범위 밖 연산은 이미지를 만들기 전에 버린다 ([enumerate.rs L202-L229](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/changes/enumerate.rs#L202-L229)). 비교 함수는 시스템 열 이름이 시점마다 다를 수 있어서 `id`, `src`, `dst` 열 이름을 시점별로 바꿔 읽는다 ([row_compare.rs L497-L527](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/changes/row_compare.rs#L497-L527)). 비 Blob 열은 Arrow 논리 동등성으로, Blob 열은 descriptor 정체성으로 먼저 비교하고 descriptor 가 다를 때만 바이트를 비교한다. compaction 이 같은 바이트를 새 파일로 옮겨도 가짜 update 가 생기지 않게 하는 규칙이다.

### 코드 분석: plan_intervals 의 스키마 경계

파일: `crates/omnigraph/src/changes/enumerate.rs`, [enumerate.rs L381-L393](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/changes/enumerate.rs#L381-L393)

열거 전에 커밋 쌍 전체를 거절하는 조건을 본다.

```rust
    for interval in intervals {
        let table_key = interval.type_key();
        match (interval.from, interval.to) {
            (None, Some(added)) => {
                if added.entity_count > 0 {
                    return Err(schema_boundary(graph_commit_id, table_key));
                }
            }
            (Some(removed), None) => {
                if removed.entity_count > 0 {
                    return Err(schema_boundary(graph_commit_id, table_key));
                }
            }
```

데이터가 있는 타입이 추가되거나 삭제되면 `ChangeSchemaBoundary` 로 커밋 전체를 거절한다. 양쪽에 있는 수명도 사용자 스키마 fingerprint 가 다르면 같은 오류다 ([enumerate.rs L394-L408](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/changes/enumerate.rs#L394-L408)). 스키마 변경을 합성 insert, delete 로 바꾸지 않는다는 RFC 0030 의 결정이 여기서 구현된다. 검사는 요청 scope 와 무관하게 모든 구간에 먼저 돈다. 경계는 커밋 쌍의 성질이지 필터된 보기의 성질이 아니라는 이유다. 마지막으로 계획을 `(kind rank, 불투명 타입 id)` 로 정렬해 출력 순서를 정한다 ([enumerate.rs L453-L460](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/changes/enumerate.rs#L453-L460)).

### 코드 분석: 바이트 예산과 전진 보장

파일: `crates/omnigraph/src/changes/enumerate.rs`, [enumerate.rs L709-L723](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/changes/enumerate.rs#L709-L723)

한 변경이 남은 바이트보다 클 때의 처리를 본다.

```rust
            let over_bytes = encoded_bytes > budget.remaining_bytes;
            if over_bytes && budget.has_emitted() {
                return Ok(match last_emitted {
                    Some(key) if emitted_this_call => CommitEnumeration::Truncated(key),
                    _ => CommitEnumeration::Exhausted {
                        required_bytes: encoded_bytes,
                    },
                });
            }
            budget.remaining_rows -= 1;
            // Saturating: a solo over-budget change drives remaining_bytes to 0,
            // so the next change (if any) ends the page here rather than
            // overflowing it further.
            budget.remaining_bytes = budget.remaining_bytes.saturating_sub(encoded_bytes);
            budget.record_emitted();
```

이미 변경이 있는 페이지라면 큰 변경은 다음 페이지로 넘긴다. 페이지의 첫 변경이면 예산을 넘어도 혼자 내보낸다. 피드는 한 예산을 여러 커밋에 걸쳐 쓰므로 `PageBudget` 의 내보낸 수는 페이지 전체 기준이다 ([enumerate.rs L73-L80](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/changes/enumerate.rs#L73-L80)). 결과는 `CommitEnumeration` 세 가지로 끝난다 ([enumerate.rs L115-L125](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/changes/enumerate.rs#L115-L125)).

| 결과 | 뜻 | 호출자 처리 |
|---|---|---|
| `Complete` | 범위 안 변경을 모두 냄 | 블록 완료, 다음 커밋 |
| `Truncated(key)` | 예산이 페이지를 닫음 | key 로 page token 발급 |
| `Exhausted` | 첫 변경도 받을 수 없음 | 이전 블록 경계에서 종료, 빈 페이지면 자원 한도 오류 |

이미지는 페이지에 들어가기로 정해진 행만 `emitted_image` 가 만든다. Blob 내용을 읽는 곳은 이곳뿐이다 ([enumerate.rs L127-L160](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/changes/enumerate.rs#L127-L160)).

## 후보 축소와 row identity

전체 ordered merge 는 정확하지만 비용이 dataset 크기에 비례한다. `candidate_scan.rs` 는 한 커밋이 남긴 Lance transaction 의 물리 흔적에서 같은 논리 변경을 더 싸게 얻는다. 판단이 조금이라도 불확실하면 전체 merge 로 돌아간다 ([candidate_scan.rs L1-L13](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/changes/candidate_scan.rs#L1-L13)).

### 코드 분석: operation_is_row_set_preserving

파일: `crates/omnigraph/src/changes/candidate_scan.rs`, [candidate_scan.rs L76-L90](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/changes/candidate_scan.rs#L76-L90)

어떤 Lance 연산이 후보 경로를 쓸 수 있는지 정하는 분류의 앞부분을 본다.

```rust
pub(crate) fn operation_is_row_set_preserving(operation: &Operation) -> bool {
    match operation {
        // Append only adds fragments; it never removes or reuses a logical id.
        Operation::Append { .. } => true,
        // OmniGraph's keyed writes (strict-insert / upsert / known-present
        // update) are all `RewriteRows` merge_insert and never delete an
        // unmatched-by-source row — no delete-capable by-source merge arm exists
        // in the engine (locked by the write-path guard in forbidden_apis.rs). A
        // different update mode is a foreign or unknown shape, so fall back.
        Operation::Update { update_mode, .. } => update_mode == &Some(UpdateMode::RewriteRows),
        // Everything that can remove, reuse, or re-stamp rows falls back to the
        // exact ordered merge. Listed explicitly so a new variant fails to
        // compile here.
        Operation::Delete { .. }
        | Operation::Overwrite { .. }
```

나머지 열네 개 변형은 모두 `false` 다 ([candidate_scan.rs L91-L104](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/changes/candidate_scan.rs#L91-L104)). wildcard 가지가 없으므로 Lance 에 새 `Operation` 변형이 생기면 컴파일 오류가 나서 분류를 다시 검토하게 된다. 모양만으로는 부족해서 `transaction_is_row_set_preserving` 이 한 번 더 확인한다. `RewriteRows` update 는 엔진이 남긴 `no_by_source_delete` 표식이나 `insert_absence` 인증서가 있을 때만 믿는다 ([candidate_scan.rs L136-L153](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/changes/candidate_scan.rs#L136-L153)). 외부 Lance merge 가 같은 모양으로 행을 지웠을 가능성을 막는 장치다.

`interval_proof` 는 transaction 을 읽기 전에 manifest 등록 정보만으로 구간을 증명한다 ([candidate_scan.rs L301-L332](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/changes/candidate_scan.rs#L301-L332)).

| 증명 | 조건 | 결과 계획 |
|---|---|---|
| `Stamp` | 양 끝이 선형 버전이고 정확히 1 차이 | `_row_last_updated_at_version` 창 `(from, to]` 로 후보 행 필터 |
| `Commit` | `to` 가 detached pin 이고 기록된 transaction 이 `from` 의 열린 버전을 base 로 지목 | 새 fragment 의 모든 행이 이 커밋 것이므로 창 불필요, 기록된 delete id 사용 가능 |

결정 결과인 `CandidatePlan` 은 `Upserts`, `Deletes`, `RowIdentity` 세 형태다 ([candidate_scan.rs L157-L184](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/changes/candidate_scan.rs#L157-L184)). Delete 의 기록된 id 가 1,024 개를 넘으면 부모 스캔에 `IN` 목록을 넣지 않고 id 집합만으로 거른다.

### 코드 분석: interval_candidate_plan 의 대체 경로

파일: `crates/omnigraph/src/changes/candidate_scan.rs`, [candidate_scan.rs L242-L248](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/changes/candidate_scan.rs#L242-L248)

인접 증명이 없을 때 row identity 로 넘어가는 분기를 본다.

```rust
    let Some(proof) = interval_proof(from_entry, to_entry, from_opened, to_opened, to_dataset)
    else {
        if !pair_rows_by_identity {
            return Ok(None);
        }
        return row_identity_plan(from_dataset, to_dataset).await;
    };
```

증명이 없으면 `row_identity_plan` 이 fragment 를 id 로 짝짓고, 파일과 deletion 파일이 같은 쌍은 건너뛰고, 나머지는 `_rowid` 로 행을 짝지어 stamp 가 같은 행을 변하지 않은 행으로 본다 ([row_identity.rs L29-L45](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/changes/row_identity.rs#L29-L45)). 이 규칙은 한쪽 lineage 가 다른 쪽을 거쳐 왔을 때만 성립한다. 갈라진 두 head 의 형제 커밋은 같은 stamp 값을 쓸 수 있다. 그래서 `pair_rows_by_identity` 는 merge 커밋을 지나지 않는 선형 하강이고 `change_lineage` 가 `off` 가 아닐 때만 참이다 ([enumerate.rs L505-L523](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/changes/enumerate.rs#L505-L523)). 후보 목록이 65,536 개를 넘거나 stamp 를 담을 행이 8 Mi 개를 넘으면 전체 merge 로 돌아간다 ([row_identity.rs L23-L26](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/changes/row_identity.rs#L23-L26)). 이 선택은 세션 설정 `change_lineage` 가 조정한다 ([settings.rs L370-L384](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/settings.rs#L370-L384)).

| 값 | 동작 |
|---|---|
| `off` | 인접 transaction 이 없는 구간은 항상 전체 ordered merge |
| `on` | 안정 row id 로 짝짓고 stamp 가 같은 쌍은 건너뜀 (정의 기본값) |
| `verify` | 두 방식을 함께 돌리고 어긋나는 변경에서 페이지를 거절 |

정의의 기본값은 `on` 이고 환경 변수 `OMNIGRAPH_CHANGE_LINEAGE` 로 바꾼다 ([settings.rs L85-L92](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/settings.rs#L85-L92)). 디버그 빌드에서는 `SessionSettings::default` 가 `verify` 로 덮어쓴다. `verify` 일 때 열거기는 row identity 이미터 옆에 전체 merge 를 oracle 로 붙인 `EmitSource::Verified` 를 만든다 ([candidate_scan.rs L740-L752](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/changes/candidate_scan.rs#L740-L752)).
