병합 결과의 제약 검증과 최종 게시를 본다.

## 제약 검증

행 비교로 충돌이 없어도 결과 그래프가 스키마 제약을 어길 수 있다. 예를 들어 source 가 노드를 지우고 target 이 그 노드로 가는 엣지를 넣으면 행 단위로는 충돌이 없지만 결과에 끊긴 엣지가 생긴다. `validate_merge_candidates` 는 모든 후보 테이블의 delta 를 target 의 pin 버전에 대고 공용 검증기로 판정한다 ([merge.rs L6386-L6400](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/exec/merge.rs#L6386-L6400)). 검증기의 계획(`plan_write`)과 판정(`WriteJudge`, `Sweep`) 구조는 10장에서 다루고, 여기서는 merge 가 붙이는 예산과 생략 규칙만 본다.

### 코드 분석: delta 스트리밍과 예산 청구

각 테이블의 추가 행과 변경 행을 배치 단위로 읽어 sweep 에 넣는다. [merge.rs L6465-L6482](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/exec/merge.rs#L6465-L6482)

```rust
            // Validation reads only id/src/dst + scalar constraint columns:
            // vectors and Blobs are projected out of every read.
            let projection = validation_projection(catalog, &delta.table_key);
            let projection: Vec<&str> = projection.iter().map(String::as_str).collect();
            for rows in delta.added.iter().chain(&delta.changed) {
                Box::pin(sweep_validation_rows(
                    &delta.table_key,
                    *rows,
                    &projection,
                    |batch| {
                        // Charge before the sweep: a refusal never extracts
                        // the facts of the batch that crosses the ceiling.
                        budget.reserve(sweep.fact_bytes(index, batch)?)?;
                        sweep.observe(index, batch)
                    },
                ))
                .await?;
            }
```

투영에서 벡터와 Blob 은 빠진다. 각 배치는 sweep 이 사실(fact)을 뽑은 뒤 바로 놓이므로, 값 범위나 enum 처럼 행 하나로 판정하는 검사의 값은 유지되지 않는다. 유지되는 것은 여러 행에 걸친 검사가 끝까지 필요로 하는 키 컬럼(id, 유일성 그룹 값, `@card` 의 엣지 source, 끝점)과 삭제된 id 다. 이 바이트를 sweep 전에 예산에 청구하므로, 상한을 넘는 배치의 사실은 추출되지 않는다.

### 검증 키 예산

예산 상한은 `MERGE_VALIDATION_MAX_BYTES` 이고 값은 `KEYED_WRITE_MAX_BYTES`, 즉 32 MiB 다 ([merge.rs L25-L31](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/exec/merge.rs#L25-L31)). `MergeValidationBudget::reserve` 는 누적값이 상한을 넘으면 `branch-merge retained validation key bytes` 자원 한도 오류를 낸다 ([L3419-L3432](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/exec/merge.rs#L3419-L3432)). 이 예산은 merge base 이후 전체 delta 에 걸리고 `write_max_bytes` 설정으로 바뀌지 않는다. 주석은 이 수치가 검증이 받아들이는 키의 양을 제한할 뿐, 그 키로 만든 사실(문자열 복사, 해시 항목)의 메모리는 키 바이트의 배수라고 밝힌다. 상한을 넘는 merge 는 단계적으로 해야 한다. source 의 중간 커밋에서 branch 를 만들어 먼저 merge 하면 다음 merge 의 base 가 그 커밋 뒤로 옮겨진다 ([user/branching/merge.md L203-L249](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/docs/user/branching/merge.md#L203-L249)).

### 코드 분석: 검증 생략 조건

fast forward 이면서 모든 후보가 특정 노드 타입의 증명된 순수 삽입이면 검증을 건너뛴다. [merge.rs L3706-L3722](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/exec/merge.rs#L3706-L3722)

```rust
fn proven_fast_forward_needs_no_validation(
    catalog: &Catalog,
    candidates: &HashMap<String, CandidateTableState>,
) -> bool {
    !candidates.is_empty()
        && candidates.iter().all(|(table_key, candidate)| {
            matches!(
                candidate,
                CandidateTableState::AdoptPureInserts(_)
                    | CandidateTableState::AdoptSourceState {
                        validation_delta: Some(AdoptValidation::PureInserts(_))
                    }
            ) && table_key
                .strip_prefix("node:")
                .is_some_and(|type_name| node_inserts_need_no_validation(catalog, type_name))
        })
}
```

노드 타입 조건은 `@range`, `@check`, `@unique` 가 없고 enum 속성이 없으며, 그 타입에서 출발하는 엣지 타입 중 `@card` 최솟값이 0 보다 큰 것이 없는 경우다 ([L3730-L3745](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/exec/merge.rs#L3730-L3745)). 이때 검증기가 할 일은 새 행끼리 key 가 겹치는지 보는 것뿐인데, source 의 정확 id 삽입 증명이 이미 그것을 배제한다. 엣지 테이블이 하나라도 있으면 끝점 검사 때문에 검증을 유지한다. 호출 지점은 [L6006-L6024](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/exec/merge.rs#L6006-L6024) 이다.

> **주의**: 개발자 문서는 이 생략이 결과를 따라가므로 32 MiB 검증 키 예산도 건너뛰고, 그래서 `FastForward` 는 `Merged` merge 라면 거부될 크기의 delta 를 게시할 수 있다고 적는다 ([docs/dev/merge.md L83-L95](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/docs/dev/merge.md#L83-L95)). 같은 크기의 데이터라도 merge 결과 종류에 따라 성공 여부가 달라질 수 있다.

dry run 은 검증이 끝난 바로 이 지점에서 결과 종류만 담아 반환한다 ([merge.rs L6026-L6040](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/exec/merge.rs#L6026-L6040)).

## 최종 재검증과 게시

검증을 통과하면 첫 물리 효과 직전에 target 의 모든 테이블 큐를 잡고 입력을 다시 읽는다. 분류는 테이블 큐 밖에서 했으므로, 오래된 target snapshot 으로 만든 계획이 물리 효과로 넘어가지 않게 하는 단계다 ([merge.rs L6042-L6061](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/exec/merge.rs#L6042-L6061)). source 의 테이블 큐는 잡지 않는다.

### 재검증 항목

| 대상 | 비교 항목 | 근거 |
|---|---|---|
| target | branch identifier, graph head, schema identity, manifest 버전 | [merge.rs L6072-L6081](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/exec/merge.rs#L6072-L6081) |
| source | native ref 수명(branch identifier) | [L6087-L6113](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/exec/merge.rs#L6087-L6113) |
| source | schema identity domain, version, IR hash | [L6117-L6138](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/exec/merge.rs#L6117-L6138) |
| 증명된 삽입 테이블 | source 구간과 target 테이블 수명 | `revalidate_proven_pure_insert_source`, `revalidate_proven_pure_insert_target_incarnation` |

source head 가 그 사이 전진한 것은 문제가 아니다. merge 입력은 태그로 고정한 snapshot 이기 때문이다. 문제는 source 를 지우고 같은 이름으로 다시 만든 경우(ABA)이며, 이것을 branch identifier 비교로 잡는다.

### 예상 테이블 버전

게시 CAS 에 넘길 테이블별 기대값(`ExpectedTableVersions`)을 만든다 ([merge.rs L6155-L6181](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/exec/merge.rs#L6155-L6181)). 후보 테이블과 효과 없이 끝난 테이블(`fenced_but_unpublished_table_keys`) 모두에 대해, target 이 캡처 시점에 게시한 버전과 native ref 를 기대값으로 건다. 게시기는 이 기대값과 현재 상태를 비교해 다르면 게시를 거부한다. 이어서 lineage 의도에 `merged_parent` 로 source 의 버퍼 커밋, head, 빠진 레지스트리 행을 담는다 ([graph_coordinator.rs L141-L152](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/db/graph_coordinator.rs#L141-L152)). 게시기가 이 레코드를 `__history` 와 블록 레지스트리에 기록하는 방식은 11장에서 다룬다.

> **주의**: [RFC 0058](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/docs/rfcs/0058-retained-merged-ancestry.md#L1-L16) (retained merged ancestry)은 `draft`, `not-started` 로 표시돼 있고 동기 절은 merge 가 source 의 커밋 집합을 복사하지 않는다고 적는다. 현재 코드는 `CapturedLineage::into_records` 로 source 의 base 이후 버퍼 커밋과 head 를 target 에 함께 넘긴다. RFC 본문은 이 동작이 들어오기 전 상태를 서술한 것으로 보인다 (추정).

### chain 길이 상한

테이블 하나가 한 merge 에서 쌓을 수 있는 detached 커밋 수는 `MAX_BRANCH_MERGE_DATA_TRANSACTIONS` 1,024 로 제한된다 ([merge.rs L32-L35](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/exec/merge.rs#L32-L35)). 청크 수는 삽입 청크, 갱신 청크, 삭제 청크의 합으로 `plan_merge_chain_length` 가 계산하고, 계획이 상한을 넘으면 `enforce_merge_transaction_ceiling` 이 첫 효과 전에 거부한다 ([L4194-L4208](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/exec/merge.rs#L4194-L4208)). 청크 하나는 키 쓰기 상한인 8,192 행, 32 MiB 로 묶인다.

### 코드 분석: 3방향 결과 테이블 게시

`publish_rewritten_merge_table` 은 삽입 청크(`StrictInsert`), 갱신 청크(`KnownPresentUpdate`), 삭제 청크를 차례로 detached 커밋으로 잇는다 ([merge.rs L4656-L4733](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/exec/merge.rs#L4656-L4733)). 각 커밋은 `commit_detached_merge_stage` 로 이전 링크 뒤에 붙고 테이블의 linear HEAD 를 움직이지 않는다. chain 이 끝나면 manifest 에 넘길 등록을 만든다. [merge.rs L4747-L4757](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/exec/merge.rs#L4747-L4757)

```rust
    let (_, tip_uuid) = chain.tip()?;
    let update = crate::db::DatasetUpdate {
        identity,
        type_key: table_key.to_string(),
        published_dataset_version: expected_version + chain.len(),
        native_dataset_branch: table_branch,
        entity_count: final_state.row_count,
        version_metadata: final_state
            .version_metadata
            .with_table_fork_owner(captured_merge_target_ref(target_txn)?)
            .with_staged(final_state.version, tip_uuid.to_string())
```

게시 버전은 target 의 기존 버전에 chain 링크 수를 더한 값이고, chain 끝의 detached 버전과 트랜잭션 UUID 가 `staged_version` 이 된다. index 재구성은 기다리지 않고 이후 `ensure_indices` 나 `optimize` 가 맞춘다. detached 커밋과 staged 버전의 저장 형식은 9장, index 재구성은 13장에서 다룬다.

### 원자성과 실패 처리

모든 테이블의 chain 과 포인터 교체는 한 번의 manifest 게시(`commit_updates_on_branch_with_expected`)에 함께 실린다 ([merge.rs L6347-L6368](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/exec/merge.rs#L6347-L6368)). CAS 전에 어디서 실패해도 target 은 바뀌지 않고, 커밋된 chain 은 어떤 manifest 도 가리키지 않는 detached 버전으로 남아 cleanup 이 회수한다. 실패 지점 주석과 failpoint 이름이 이 동작을 적는다. 그 사이 target 이 전진했으면 CAS 에서 지고 일반 충돌을 돌려준다. CAS 뒤에는 복구 단계가 없다 ([docs/dev/merge.md L411-L416](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/docs/dev/merge.md#L411-L416)).

`branch_merge_as_with_options` 는 게시 이전의 모든 실패를 `OmniError::unpublished` 로 감싸 효과 없음이 확정된 실패로 표시하고, 게시기가 스스로 표시한 결과 불확실 상태는 덮어쓰지 않는다 ([merge.rs L5355-L5371](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/exec/merge.rs#L5355-L5371)).

### 반환값

게시가 끝나면 바뀐 엣지 테이블의 그래프 index 를 무효화하고 `MergeResult` 를 돌려준다 ([merge.rs L6372-L6383](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/exec/merge.rs#L6372-L6383)).

| `MergeOutcome` | 의미 | `commit` |
|---|---|---|
| `AlreadyUpToDate` | source 가 target 에 더할 변경 없음 | `None` |
| `FastForward` | target 상태가 base 와 같음, 그래도 새 커밋을 씀 | `Some` |
| `Merged` | 3방향 merge 로 새 커밋 게시 | `Some` |

정의는 [omnigraph.rs L111-L115](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/db/omnigraph.rs#L111-L115), 결과 구조 `MergeResult` 도 같은 파일에 있다. `commit` 은 게시 시점에 캡처한 영수증이라, 서버와 CLI 는 HEAD 를 다시 읽지 말고 이 값을 그대로 돌려줘야 한다. 그 사이 다른 writer 가 target 을 전진시켰을 수 있기 때문이다 ([docs/dev/merge.md L435-L440](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/docs/dev/merge.md#L435-L440)). `plans` 는 검증한 테이블마다 계획기 문서를 담는다.

> **주의**: [RFC 0057](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/docs/rfcs/0057-bounded-merge-preparation.md#L22-L28) 은 Blob 이 없는 테이블을 최대 4개까지 동시에 준비하는 설계를 accepted, in-progress 로 기록한다. 기준 커밋의 `branch_merge_on_captured_target` 는 테이블 키를 정렬한 순서대로 하나씩 `await` 하며 분류한다 ([merge.rs L5716-L5770](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/exec/merge.rs#L5716-L5770)). 동시 준비는 아직 이 경로에 들어가지 않았다. 같은 맥락에서 [RFC 0001](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/docs/rfcs/0001-fragment-adopt-branch-merge.md#L1-L16) 의 `base_paths` 기반 fragment 참조 채택도 `draft`, `not-started` 이며, 현재 포인터 채택은 같은 dataset 안에서 target 등록이 source 의 pin 을 그대로 가리키는 방식이다.
