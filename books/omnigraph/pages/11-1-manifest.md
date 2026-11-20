`__manifest` 한 버전이 담는 행과 그 버전을 읽는 스냅샷을 본다.

## `__manifest` 행 구조

`__manifest` 한 버전은 행 몇 종류의 집합이다. 행 종류는 `object_type` 열 값으로 구분하며 상수는 [lib.rs L133-L158](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-catalog/src/lib.rs#L133-L158) 에 있다.

| `object_type` | 행 수 | 담는 내용 | `object_id` 형식 |
|---|---|---|---|
| `table` | 테이블 identity 마다 1 | 등록 정보와 현재 pin 또는 drop 상태 | `table:{stable:016x}:{incarnation:016x}` |
| `graph_commit` | 버전마다 1 | head 커밋의 기록 | 커밋 ID |
| `settled_commit` | 버퍼 커밋마다 1 | 아직 `__history` 로 내보내지 않은 조상 커밋 | 커밋 ID |
| `replaced_table` | 버퍼 기간에 바뀐 `table` 행마다 1 | 바뀌기 전 `table` 행 | `table:...@{replaced_at}` |
| `schema_contract` | 0 또는 1 | 수락된 스키마 원문, IR, identity | `schema_contract` |
| `history_block` | 내보낸 블록 구간마다 1 | 블록, slot 범위, 첫 slot 의 generation 과 부모 | `history_block:{block}:{first}-{last}` |
| `history_merge` | 내보낸 구간 안의 merge 커밋마다 1 | merge 커밋의 두 번째 부모 | `history_merge:{block}:{slot}` |
| `writer_epoch` | 0 또는 1 | 브랜치에 쓸 수 있는 프로세스의 세대 번호 | `writer_epoch` |

`object_id` 문자열을 만드는 함수는 [layout.rs L246-L267](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-catalog/src/layout.rs#L246-L267) 에 모여 있다. `table` 행이 테이블 identity 마다 하나라는 점이 중요하다. 한 테이블의 현재 상태를 알려고 이력을 접을 필요가 없고, 행 하나를 읽으면 된다.

### 코드 분석: 커밋 행의 필드

커밋 행이 갖는 논리 필드 목록을 본다. [record.rs L103-L116](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-catalog/src/record.rs#L103-L116)

```rust
pub(crate) const COMMIT_FIELDS: [(&str, FieldType); 12] = [
    ("graph_branch", FieldType::Utf8),
    ("native_branch", FieldType::Utf8),
    ("graph_manifest_version", FieldType::UInt64),
    ("generation", FieldType::UInt64),
    ("parent_commit_id", FieldType::Utf8),
    ("merged_parent_commit_id", FieldType::Utf8),
    ("actor_id", FieldType::Utf8),
    ("created_at", FieldType::Int64),
    ("schema_ir_hash", FieldType::Utf8),
    ("schema_identity_version", FieldType::UInt64),
    ("schema_identity_domain", FieldType::Utf8),
    ("schema_content_hash", FieldType::Utf8),
];
```

부모는 `parent_commit_id` 와 `merged_parent_commit_id` 두 개까지다. 일반 커밋은 첫 부모만, merge 커밋은 두 번째 부모까지 채운다. `generation` 은 부모들 generation 의 최댓값에 1 을 더한 값이고 genesis 는 0 이다. merge base 탐색이 이 값을 우선순위로 쓴다. `graph_manifest_version` 은 이 커밋을 기록한 `__manifest` 버전 번호이며, 커밋과 테이블 상태를 잇는 시계 역할을 한다.

`table` 행 필드는 [record.rs L89-L100](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-catalog/src/record.rs#L89-L100) 의 10개다. `table_version`, `table_branch`, `row_count` 가 pin 이고 `manifest_version` 이 그 pin 을 쓴 `__manifest` 버전이다. `dropped_at` 에 값이 있으면 drop 된 identity 다.

### packed record 저장 형태

논리 스키마와 저장 스키마가 다르다. 논리 스키마는 필드가 모두 펼쳐진 형태이고, 저장 형태는 `object_id`, `object_type`, `record` 세 열에 스키마 원문 두 열을 더한 형태다. `record` 는 Lance 의 packed struct(`lance-encoding:packed`) 하나에 모든 필드를 행 단위로 담는다 ([record.rs L1-L21](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-catalog/src/record.rs#L1-L21)).

packed struct 자식에는 null 을 넣을 수 없다. 그래서 모든 자식을 non-null 로 선언하고, 마지막 자식 `present` 비트마스크에 "논리적으로 null 인 필드" 를 표시한다. null 자리에는 `""` 나 `0` 을 채우고, 읽는 쪽은 비트가 켜진 자리에 다른 값이 있으면 거부한다. 필드 순서는 `TABLE_FIELDS`, `COMMIT_FIELDS`, `REPLACED_FIELDS`, `REGISTRY_FIELDS` 순으로 이어 붙이며, 필드 수가 `u32` 비트 수를 넘지 않는지 컴파일 시점에 검사한다.

저장 스키마 생성은 [record.rs L207-L228](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-catalog/src/record.rs#L207-L228) 의 `manifest_storage_schema` 가 한다. `object_id` 에는 Lance 의 `unenforced-primary-key` 표시가 붙는다. 스탬프는 열이 아니라 dataset 스키마 메타데이터에 들어간다.

### hot fragment 와 cold fragment

한 버전의 행은 fragment 두 종류에 나뉜다. id 가 가장 큰 fragment 가 hot fragment 이며 head 커밋, 버퍼, `table` 행, 스키마 계약, 블록 레지스트리 행 최대 `REGISTRY_HOT_ROWS` 개를 담는다. 나머지 cold fragment 는 오래된 레지스트리 행만 담는다 ([commit.rs L12-L27](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-catalog/src/commit.rs#L12-L27)). `REGISTRY_HOT_ROWS` 는 64 다 ([commit.rs L62](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-catalog/src/commit.rs#L62)).

열기와 일반 게시는 hot fragment 만 읽는다. `hot_scanner` 가 fragment 가 둘 이상일 때 스캔 대상을 hot 하나로 좁힌다. cold fragment 는 merge 가 레지스트리를 읽을 때만 연다. cold fragment 를 다시 쓸지 여부는 git `commit-graph` 의 chain 규칙을 따른다. 새로 쓰는 행 수의 두 배보다 작은 fragment 를 작은 것부터 합친다 ([commit.rs L239-L256](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-catalog/src/commit.rs#L239-L256)). 그래서 cold fragment 수가 레지스트리 크기에 대해 로그 수준으로 유지된다고 모듈 문서가 설명한다.

## 매니페스트 상태와 스냅샷

`__manifest` 한 버전을 읽으면 세 단계 표현을 거친다. 저장 batch 를 `ManifestRows` 로 디코드하고, 이를 다시 `ManifestState` 로 접고, 쿼리용 `Snapshot` 을 만든다.

| 타입 | 위치 | 역할 |
|---|---|---|
| `ManifestRows` | [state.rs L812-L821](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-catalog/src/state.rs#L812-L821) | 한 버전의 모든 행: `table` 행, head, 버퍼, 계약, 레지스트리, writer epoch |
| `ManifestState` | [state.rs L73-L87](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-catalog/src/state.rs#L73-L87) | 보이는 테이블 목록, 정확한 head, 계약 head, writer epoch |
| `DatasetEntry` | `state.rs` | 살아 있는 테이블 하나의 경로, 버전, 브랜치, 행 수 |
| `Snapshot` | [lib.rs L278-L319](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-catalog/src/lib.rs#L278-L319) | 한 쿼리의 모든 읽기가 거치는 고정 시점 view |
| `ManifestCoordinator` | [lib.rs L1433-L1457](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-catalog/src/lib.rs#L1433-L1457) | 브랜치 하나의 현재 버전을 붙잡고 게시와 갱신을 수행하는 핸들 |

### 코드 분석: 보이는 상태로 접기

`table` 행을 `DatasetEntry` 목록으로 바꾸는 함수를 본다. [state.rs L1030-L1063](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-catalog/src/state.rs#L1030-L1063) 중 앞부분이다.

```rust
    let mut entries: Vec<DatasetEntry> = tables
        .iter()
        .filter_map(|table| match &table.state {
            TableState::Pinned(pin) => Some(dataset_entry(&table.registration, pin)),
            TableState::Registered | TableState::Dropped { .. } => None,
        })
        .collect();
    let mut aliases = HashMap::<&str, TableIdentity>::new();
    for entry in &entries {
        if let Some(existing) = aliases.insert(&entry.type_key, entry.identity) {
            return Err(OmniError::manifest_internal(format!(
                "manifest has two live table identities ({existing} and {}) bound to alias '{}'",
                entry.identity, entry.type_key
            )));
        }
    }
```

pin 이 있는 행만 보인다. 등록만 됐거나 drop 된 identity 는 빠진다. `TableState` 의 세 값은 [state.rs L195-L206](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-catalog/src/state.rs#L195-L206) 에 정의돼 있고, drop 주석에 "A dropped identity never returns" 가 적혀 있다. 같은 타입 이름(alias)에 살아 있는 identity 가 둘이면 내부 오류다. 타입 이름은 바뀔 수 있어도 한 시점에 한 identity 만 가리킨다.

### 정확한 head 와 유효 head

새로 만든 브랜치는 아직 자기 커밋이 없다. 그 브랜치의 `__manifest` 에는 원본 브랜치가 쓴 head 행이 그대로 복사돼 있다. 코드는 이 head 를 "유효 head" 로 쓰되 "정확한 head" 로는 인정하지 않는다. [state.rs L1079-L1087](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-catalog/src/state.rs#L1079-L1087)

```rust
/// `head` when the branch incarnation `native_branch` wrote it. A fork that has
/// not published holds the head its source wrote, which is its effective head
/// and not its exact one.
pub(crate) fn exact_head<'a>(
    head: &'a GraphLineageRow,
    native_branch: Option<&str>,
) -> Option<&'a GraphLineageRow> {
    (head.native_branch.as_deref() == native_branch).then_some(head)
}
```

판정 기준은 커밋 행의 `native_branch` 가 지금 읽는 브랜치의 Lance native ref 와 같은지 하나다. `ManifestState::graph_heads` 는 정확한 head 만 담는다. 게시 전 검사에서 "head 가 없음" 을 하나의 유효한 기대값으로 다루는 이유가 여기 있다. 브랜치 생성과 native ref 이름 규칙은 12장에서 다룬다.

### 스냅샷과 같은 버전 원칙

`Snapshot` 주석은 한 쿼리의 모든 읽기가 하나의 `Snapshot` 을 거쳐야 타입 사이 일관성이 보장된다고 적는다. `graph_heads` 와 `schema_contract` 필드 주석은 둘 다 "from this SAME pinned manifest version" 을 강조한다. head 를 `CommitGraph` 로 따로 풀면 다른 버전의 head 와 이 스냅샷의 테이블이 짝지어질 수 있기 때문이다.

### 코디네이터 갱신과 incarnation

`ManifestCoordinator::refresh` 는 브랜치의 최신 `__manifest` 를 다시 연다. Lance 브랜치 식별자가 같고 버전 번호도 같으면 저장된 manifest 이미지가 같은지까지 비교하고, 같으면 아무것도 읽지 않는다. 같은 버전 번호인데 이미지가 다르면 루트가 교체된 것으로 보고 히스토리 캐시를 비운다 ([lib.rs L2095-L2132](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-catalog/src/lib.rs#L2095-L2132)).

이 비교에 쓰는 `ManifestIncarnation` 은 버전 번호, e-tag, 타임스탬프, Lance 브랜치 식별자를 묶는다. [lib.rs L929-L944](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-catalog/src/lib.rs#L929-L944)

```rust
    pub fn matches(&self, held: &Self) -> bool {
        if self.version != held.version || self.branch_identifier != held.branch_identifier {
            return false;
        }
        match (&self.e_tag, &held.e_tag) {
            (Some(latest), Some(current)) => latest == current,
            _ => match (self.timestamp_nanos, held.timestamp_nanos) {
                (Some(latest), Some(current)) => latest == current,
                // Some object stores can omit both e_tag and manifest timestamp
                // from the reachable API. In that narrow case the version-number
                // probe is the strongest available identity.
                _ => true,
            },
        }
    }
```

브랜치 식별자는 같은 이름으로 지웠다 다시 만든 브랜치(ABA)를 가려낸다. e-tag 와 타임스탬프가 모두 없는 객체 저장소에서는 버전 번호가 가장 강한 근거라고 주석이 한계를 밝힌다.

### genesis 버전

새 그래프의 `__manifest` 는 Lance `Create` 한 번으로 태어난다. genesis 커밋, 빈 테이블의 `table` 행, 스키마 계약, 스탬프가 모두 이 commit 에 들어간다. [graph.rs L177-L183](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-catalog/src/graph.rs#L177-L183)

```rust
    // The internal-schema stamp rides the Create write's schema metadata, so
    // the stamp is atomic with manifest birth: no crash window can leave
    // `__manifest` durable but unstamped. The Create commit is the manifest's
    // entire birth — entries, genesis lineage, and the stamp in one commit,
    // with nothing failable after it. (A `table_version_management` config
    // key is deliberately not written: neither the pinned Lance substrate nor
    // this crate reads it.)
```

genesis 버전 번호는 1, genesis generation 은 0 으로 고정된다 ([graph.rs L29-L37](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-catalog/src/graph.rs#L29-L37)). 테이블별 genesis 행은 pin 을 생성 시점의 Lance 버전에 걸고 `manifest_version` 을 1 로 둔다.
