저장 형식 스탬프와 미게시 커밋, 설계 문서와의 차이를 본다.

## 저장 형식 스탬프

`__manifest` 스키마 메타데이터의 `omnigraph:internal_schema_version` 값이 저장 형식 스탬프다. 이 바이너리는 14부터 17까지를 연다 ([migrations.rs L172](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-catalog/src/migrations.rs#L172), [migrations.rs L181](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-catalog/src/migrations.rs#L181)). 스탬프는 열기 경로, 읽기 전용 열기, 게시기 모두가 같은 함수 `refuse_if_stamp_unsupported` 로 검사한다 ([migrations.rs L425-L480](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-catalog/src/migrations.rs#L425-L480)).

| 스탬프 | 추가된 것 | 이전 바이너리가 거부해야 하는 이유 |
|---|---|---|
| 14 | 버퍼와 `__history` 불변 객체 | v13 은 버퍼를 전체 이력으로 오해한다 |
| 15 | `history_block`, `history_merge` 행, `unpublished_` 태그 | v14 는 행 종류를 거부하고 태그를 영원히 보존한다 |
| 16 | `unattached_` 태그 | v15 는 태그를 사용자 태그로 보고 영원히 보호한다 |
| 17 | `writer_epoch` 같은 control fact 행 | v16 은 행 종류를 거부한다 |

표의 근거는 [versioning.md L100-L179](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/docs/dev/versioning.md#L100-L179) 와 [migrations.rs L16-L35](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-catalog/src/migrations.rs#L16-L35) 다. 기능별 최소 스탬프는 `UNPUBLISHED_COMMITS_STAMP` 15, `UNATTACHED_BRANCHES_STAMP` 16, `CONTROL_FACTS_STAMP` 17 로 정의돼 있다.

### 코드 분석: 게시가 쓰는 스탬프

게시가 새 버전에 어떤 스탬프를 쓰는지 정하는 함수다. [migrations.rs L261-L266](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-catalog/src/migrations.rs#L261-L266)

```rust
/// The stamp a publish writes over a version it loaded at `loaded`: that
/// stamp, and 16 for a v14 or v15 branch. No publish raises a branch to
/// [`CONTROL_FACTS_STAMP`].
pub(crate) fn stamp_after_publish(loaded: u32) -> u32 {
    loaded.max(CONTROL_FACTS_STAMP - 1)
}
```

열기는 스탬프를 바꾸지 않는다. v14, v15 브랜치는 첫 게시 CAS 에서 16 이 되고, 16 이상은 읽은 값을 그대로 쓴다. 17 로 올리는 길은 모든 프로세스를 멈춘 상태에서 `omnigraph upgrade` 의 스탬프 전용 경로뿐이다. 새로 만든 그래프는 17 로 태어난다. 이 규칙 때문에 control fact 를 담은 행 집합은 17 미만 버전에 쓰일 수 없고, `overwrite_probed` 가 그 경우를 내부 오류로 막는다 ([commit.rs L134-L141](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-catalog/src/commit.rs#L134-L141)).

`guard_stamp` 는 스탬프가 없거나 숫자가 아니거나 업그레이드 대기 키가 있으면 열기를 거부한다 ([migrations.rs L371-L402](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-catalog/src/migrations.rs#L371-L402)). 현대식 레이아웃인데 스탬프만 없는 경우를 따로 구별한다. 이전 바이너리의 중단된 `init` 일 수도, 메타데이터 손상일 수도 있어 판별할 수 없으므로, 제자리 재초기화를 하지 말고 루트를 보존하라고 안내한다.

> **주의**: 스탬프는 저장 형식의 fence 이지 시스템 열 표기(`id` 와 `__id`)를 나타내지 않는다. 표기는 스키마 IR 의 기능 집합이 정한다. 스키마 쪽 내용은 3장에서 다룬다.

## 미게시 커밋과 비연결 브랜치

스탬프 15, 16 은 브랜치 head 를 움직이지 않는 커밋 두 종류를 더했다. 둘 다 대상 브랜치 `__manifest` 의 detached 버전에 완전한 행 집합을 담고, 태그 하나로 주소를 붙인다. detached 버전 쓰기는 앞의 `Placement::Detached` 경로를 쓰며 CAS 가 없고 무작위 버전 id 충돌에 대해서만 최대 20번 재시도한다 ([commit.rs L64-L77](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-catalog/src/commit.rs#L64-L77)).

| 구분 | 미게시 커밋 | 비연결 브랜치 |
|---|---|---|
| 용도 | 나중에 게시하거나 버리는 제안 | 게시하지 않는 스냅샷 체인 |
| 태그 | `unpublished_<C>` | `unattached_<root>_<C>` |
| 속성 버전 | 1 ([unpublished.rs L44-L48](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-catalog/src/unpublished.rs#L44-L48)) | 2 ([unattached.rs L44-L46](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-catalog/src/unattached.rs#L44-L46)) |
| 최소 스탬프 | 15 | 16 |
| 커밋 ID | 게시 때 할당될 `hb1` ID | 새 ULID |

미게시 커밋의 detached 버전은 결과 `table` 행, ID 가 후보 ID `C` 인 head 행, 빈 버퍼, 캡처한 스키마 계약을 담는다. 트랜잭션 속성 `UnpublishedCommitProperty` 에는 나중 게시가 일반 게시기로 재생할 정확한 재료가 들어 있다 ([unpublished.rs L1-L12](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-catalog/src/unpublished.rs#L1-L12)). 비연결 스냅샷은 루트에서 시작해 자식으로 이어지는 체인을 만들고, 모든 기록이 루트가 캡처한 좌표를 반복한다.

### 코드 분석: 미게시 커밋 게시

미게시 커밋을 게시하는 함수는 새 메커니즘을 만들지 않고 일반 게시기를 다시 쓴다. [unpublished.rs L763-L772](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-catalog/src/unpublished.rs#L763-L772)

```rust
        let intent = LineageIntent {
            graph_commit_id: property.intent_nonce.clone(),
            branch: property.graph_branch.clone(),
            actor_id: property.actor_id.clone(),
            merged_parent,
            created_at: property.created_at,
            history_release_bytes: HistoryReleaseBytes(property.history_release_bytes),
        };
        let precondition = PublishPrecondition::ExactCandidate {
            authority: property.authority(),
```

후보를 만들 때와 같은 nonce, 같은 바이트 예산을 넘기므로 게시기는 같은 `hb1` ID 를 계산해야 한다. 다른 ID 가 나오면 `ExactCandidate` 검사가 CAS 전에 거부한다 ([publisher.rs L1835-L1845](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-catalog/src/publisher.rs#L1835-L1845)). 그 전에 head 가 후보의 부모와 같은지도 확인한다 ([unpublished.rs L748-L757](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-catalog/src/unpublished.rs#L748-L757)). 사용자가 검토한 커밋 ID 와 실제로 게시된 커밋 ID 가 다를 수 없다는 보장이다.

후보의 수명은 세 값 중 하나로 판정한다. `Live` 는 게시 가능, `Expired` 는 태그 제거 대상, `Undecidable` 은 근거가 모자라 보존하되 모든 작업을 거부한다 ([unpublished.rs L272-L293](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-catalog/src/unpublished.rs#L272-L293)). head 가 부모를 지나 움직였거나, 대상 브랜치가 폐기됐거나, 이미 게시된 경우가 만료 사유다. 태그 정리와 보존 정책은 13장에서 다룬다.

## 설계 문서와 현재 코드

카탈로그 관련 RFC 는 여러 단계의 설계를 기록한다. 일부는 대체됐고 일부는 현재 코드와 행 형식이 다르다. 읽을 때 어느 문서가 현재 상태인지 구별해야 한다.

| 문서 | 상태 표기 | 현재 코드와의 관계 |
|---|---|---|
| [RFC 0019 Heads and fences](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/docs/rfcs/0019-heads-and-fences.md#L19-L21) | superseded | RFC 0023, 0024 로 나뉘었다. MemWAL 관련 논의는 기각된 이력이다 |
| [RFC 0024 Durable table heads](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/docs/rfcs/0024-durable-table-heads.md#L20-L24) | draft, not-started | 독립 head 형식은 출하되지 않았다. 문서는 "current binary serves manifest v6" 라고 적지만 현재는 v14 에서 v17 이다 |
| [RFC 0062 Manifest version clock](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/docs/rfcs/0062-manifest-version-clock.md#L24-L45) | draft, in-progress | 등록 행 키 `table_version:...:{manifest_version}` 을 제안했다. 현재는 identity 당 `table` 행 하나에 `manifest_version` 필드를 둔다 |
| [RFC 0063 Self-contained branch lineage](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/docs/rfcs/0063-self-contained-branch-lineage.md#L25-L47) | draft, in-progress | 브랜치마다 `graph_head:<branch>` 행을 두는 안을 제안했다. 현재는 `graph_commit` 행 하나가 head 이고, merge 게시가 원본 브랜치 기록을 `__history` 에 덧붙인다 |
| [RFC 0068 Graph commit record](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/docs/rfcs/0068-graph-commit-record.md#L1-L17) | draft, not-started | 커밋 기록 객체의 조건부 생성을 게시 지점으로 삼는 안이다. 현재 게시 지점은 여전히 `__manifest` 버전 CAS 다 |

RFC 0062 가 정한 원칙 하나는 그대로 살아 있다. 테이블의 현재 등록을 고르는 시계는 테이블 자신의 Lance 버전이 아니라 `__manifest` 버전이다. `DatasetEntry::manifest_version` 주석이 이 RFC 를 인용한다 ([state.rs L42-L44](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-catalog/src/state.rs#L42-L44)). 현재 형식이 어떻게 바뀌어 왔는지는 versioning 문서의 스탬프별 항목이 기록한다 (

> **주의**: RFC 0068 은 front matter 에 `implementation: not-started` 로 남아 있지만 `__history` 의 `legacy/` 배치는 그 RFC 의 보충 절을 근거로 이미 구현돼 있다 ([versioning.md L124-L131](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/docs/dev/versioning.md#L124-L131), [history.rs L25-L30](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-catalog/src/history.rs#L25-L30)). RFC 상태 표기만으로 구현 여부를 판단하면 안 된다.
