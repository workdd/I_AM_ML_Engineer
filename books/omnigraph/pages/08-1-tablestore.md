테이블 저장소 트레이트와 핀 해석, 스캔 경로를 본다.

## TableStore 와 봉인된 TableStorage

`TableStore` 는 그래프 루트 하나에 대한 Lance 접근을 모은 구조체다. 엔진 코드는 이 구조체를 직접 부르지 않고 봉인된 `TableStorage` 트레이트를 거친다.

### 코드 분석: TableStore 구조체

파일: `crates/omnigraph/src/table_store.rs`, [table_store.rs L2170-L2181](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/table_store.rs#L2170-L2181)

필드는 루트 URI, 그래프 공용 Lance `Session`, 외부 Blob 정책 세 개다.

```rust
#[derive(Debug, Clone)]
pub struct TableStore {
    root_uri: String,
    /// The graph's shared Lance `Session` (one per graph — LanceDB's
    /// one-session-per-connection pattern). Held non-optionally so every open
    /// this store performs attaches it; the read path's handle cache shares
    /// the same instance via `ReadCaches`.
    session: Arc<lance::session::Session>,
    /// Immutable graph-level resource policy. The default is deny; an engine
    /// builder replaces it before exposing the handle to writers.
    external_blob_policy: Arc<ExternalBlobPolicy>,
}
```

`Session` 은 Lance 의 인덱스 캐시와 메타데이터 캐시를 담는다. 그래프마다 하나를 두고 모든 열기에 넘기므로 같은 테이블을 여러 문장에서 열어도 manifest 디코딩과 인덱스 로드를 반복하지 않는다. 생성자 `new` 는 정책을 `Deny` 로 두고, `with_external_blob_policy` 는 정책 검증과 그래프 루트 겹침 검사를 통과해야 정책을 교체한다([table_store.rs L2197-L2212](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/table_store.rs#L2197-L2212)). 외부 읽기는 프로세스의 저장소 권한으로 수행되므로, 허용 경로가 그래프 루트와 겹치면 쓰기 권한자가 manifest 나 테이블 파일을 Blob 셀로 복사해 읽어 낼 수 있다.

### 코드 분석: sealed 모듈

파일: `crates/omnigraph/src/storage_layer.rs`, [storage_layer.rs L212-L220](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/storage_layer.rs#L212-L220)

봉인 패턴은 비공개 모듈의 마커 트레이트로 구현된다.

```rust
pub(crate) mod sealed {
    /// Sealed marker — only types defined in `omnigraph-engine` can
    /// implement `TableStorage`. Combined with the trait being the only
    /// route to write APIs from engine code, this gives type-system
    /// enforcement of the staged-write invariant.
    pub trait Sealed {}

    impl Sealed for crate::table_store::TableStore {}
}
```

`TableStorage`([storage_layer.rs L530-L533](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/storage_layer.rs#L530-L533))는 `sealed::Sealed` 를 상위 트레이트로 요구한다. 다른 크레이트는 `Sealed` 를 볼 수 없으므로 자기 타입에 `TableStorage` 를 구현할 수 없다. 모듈 머리 주석은 이 트레이트가 쓰기를 스테이징과 commit 두 단계로만 노출하고 스테이징은 Lance HEAD 를 움직이지 않는다고 적는다. 엔진 코드가 `lance::*` 를 직접 쓰는 자리는 `tests/forbidden_apis.rs` 가 파일과 호출 수로 고정한다.

메서드는 스냅샷 열기(`open_snapshot_at_entry`), 읽기(`scan_with_pending`), 스테이징(`stage_append`, `stage_keyed_write`, `stage_delete`, `stage_overwrite`), commit(`commit_staged_detached`), 유지보수(`stage_compaction`, `stage_index_reconciliation`)로 묶인다.

트레이트 시그니처에 `lance::Dataset` 이 드러나지 않도록 `SnapshotHandle`([storage_layer.rs L377-L380](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/storage_layer.rs#L377-L380))이 `Arc<Dataset>` 을, `StagedHandle`([storage_layer.rs L448-L450](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/storage_layer.rs#L448-L450))이 `StagedWrite` 를 감싼다. `SnapshotHandle` 의 공개 접근자는 버전 번호와 stable row ID 사용 여부뿐이고, 내부 `Dataset` 을 꺼내는 `into_arc`, `into_dataset` 은 `pub(crate)` 다. 스테이징과 commit 의 호출 순서는 9장, 압축과 인덱스 재구성은 13장에서 다룬다.

## 핀 해석과 핸들 캐시

그래프 스냅샷은 테이블마다 어느 Lance 버전을 읽을지 적은 항목을 가진다. 이 항목이 핀이고, 핀의 물리 증거는 `TableVersionMetadata` 에 담긴다. 핀이 가리키는 버전은 대부분 다음 절에서 볼 detached 버전이다.

### 코드 분석: TableVersionMetadata

파일: `crates/omnigraph-core/src/metadata.rs`, [metadata.rs L105-L116](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-core/src/metadata.rs#L105-L116)

핀 하나가 기록하는 물리 증거의 앞부분이다. 뒤에 `transaction_uuid` 와 `last_linear_version` 이 이어진다.

```rust
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TableVersionMetadata {
    manifest_path: String,
    manifest_size: Option<u64>,
    e_tag: Option<String>,
    naming_scheme: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    table_fork_owner: Option<String>,
    /// RFC 0067: the detached Lance version this pin was staged
    /// as before publication. `None` on a linear pin.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    staged_version: Option<u64>,
```

`manifest_path`, `manifest_size`, `e_tag` 는 핀이 가리키는 manifest 객체를 지목한다. `staged_version` 과 `transaction_uuid` 는 게시 전에 스테이징된 detached 버전과 그 트랜잭션이다. `last_linear_version` 은 테이블의 linear 이력이 멈춘 버전이다. `table_fork_owner` 는 v11 이전에 만들어진 테이블 fork 의 소유자를 기록한다.

### 코드 분석: TableVersionMetadata::witnesses

파일: `crates/omnigraph-core/src/metadata.rs`, [metadata.rs L192-L205](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-core/src/metadata.rs#L192-L205)

열린 dataset 이 핀이 가리키던 manifest 인지 판정한다.

```rust
    pub fn witnesses(&self, dataset: &Dataset) -> bool {
        match self.e_tag.as_deref() {
            None => return true,
            Some(expected) if dataset.manifest_location().e_tag.as_deref() == Some(expected) => {
                return true;
            }
            Some(_) => {}
        }
        match self.transaction_uuid.as_deref() {
            Some(uuid) => crate::staging::StagedTransactionIdentity::recorded_by(dataset)
                .is_some_and(|identity| identity.uuid == uuid),
            None => false,
        }
    }
```

ETag 가 일치하면 같은 객체다. ETag 가 달라도 manifest 가 기록한 트랜잭션 uuid 가 핀의 uuid 와 같으면 같은 효과로 본다. v10 형식의 핀은 detached 버전과 그 linear 복제본(linear twin)을 둘 다 가질 수 있어서 두 번째 판정이 필요하다. uuid 를 꺼내는 `recorded_by` 는 manifest 가 기록한 트랜잭션 파일 이름 `{read_version}-{uuid}.txn` 을 분해할 뿐이라 추가 요청이 없다([staging.rs L41-L49](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-core/src/staging.rs#L41-L49)).

### 코드 분석: open_pinned_dataset

파일: `crates/omnigraph-core/src/instrumentation.rs`, [instrumentation.rs L2117-L2122](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-core/src/instrumentation.rs#L2117-L2122)

핀을 실제 Lance 버전으로 해석하는 첫 두 분기다.

```rust
    let Some(staged) = staged_version else {
        return open_dataset(uri, VersionResolution::At(target_version), session, wrapper).await;
    };
    if last_linear_version.is_some_and(|last| target_version > last) {
        return open_dataset(uri, VersionResolution::At(staged), session, wrapper).await;
    }
```

`staged_version` 이 없는 핀은 오래된 linear 핀이므로 번호 그대로 연다. 핀 번호가 `last_linear_version` 보다 크면 linear 이력에 그 번호가 있을 수 없으므로 곧바로 detached 버전을 연다. 그 이하이면 같은 번호의 linear 버전을 열어 기록된 트랜잭션 uuid 가 핀과 같은지 확인하고, 다른 쓰기의 버전이면 경고를 남기고 detached 버전으로 돌아간다([instrumentation.rs L2123-L2143](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-core/src/instrumentation.rs#L2123-L2143)). 정리 작업이 버전을 회수한 경우에는 최신 버전을 확인한 뒤 detached 버전을 다시 시도한다.

### 코드 분석: TableHandleKey

파일: `crates/omnigraph-core/src/handle_cache.rs`, [handle_cache.rs L122-L131](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-core/src/handle_cache.rs#L122-L131)

열린 `Dataset` 핸들을 재사용하는 캐시의 키다.

```rust
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TableHandleKey {
    pub table_path: String,
    pub table_branch: Option<String>,
    pub version: u64,
    /// The detached pin `version` resolves to; `version` alone is a
    /// per-lineage counter two branches share.
    pub staged_version: Option<u64>,
    pub e_tag: Option<String>,
}
```

키에 버전, detached 핀, ETag 가 모두 들어 있어서 쓰기가 일어나면 새 키가 생길 뿐 기존 항목을 무효화할 필요가 없다. 브랜치를 지우고 같은 이름으로 다시 만들어 버전 번호가 재사용되어도 ETag 가 다르면 다른 키다. `get_or_open`([handle_cache.rs L206-L270](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-core/src/handle_cache.rs#L206-L270))은 잠금을 잡고 키를 찾은 뒤, 없으면 잠금을 놓고 `open_pinned_dataset` 으로 연다. 두 요청이 동시에 놓치면 두 번 열리지만 한 버전의 dataset 은 불변이라 결과는 같다.

캐시는 핸들 64 개와 디코딩된 manifest 64 MiB 로 이중 제한된다([handle_cache.rs L115-L120](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-core/src/handle_cache.rs#L115-L120)). 개수 상한만 있으면 자주 쓰이는 한 테이블의 fragment 목록 64개가 메모리를 차지한다(주석의 issue 859). 그래서 핸들마다 manifest 의 `DeepSizeOf` 크기를 기록해 바이트 상한도 둔다. 새 핀을 게시한 쓰기는 스테이징 기준으로 삼았던 핀을 `demote` 해 압박이 생기면 가장 먼저 내보내게 한다.

named 브랜치를 읽는 변경 피드는 `open_at_entry_verified`([table_store.rs L2412-L2453](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/table_store.rs#L2412-L2453))로 연다. 캐시 크기 0 의 세션으로 새로 열어 캐시된 핸들이 브랜치 재생성을 가리지 않게 하고, `witnesses` 로 ETag 를 확인한다. 주석은 이 ETag 비교가 보조 장치이고 실제 판정은 변경 피드의 논리적 head 재확인이 맡는다고 밝힌다. 변경 피드는 13장에서 다룬다.

## 스캔 경로와 메모리 상한

`TableStore` 의 스캔은 모두 `scan_stream_with`([table_store.rs L3096-L3134](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/table_store.rs#L3096-L3134))로 모인다. 이 함수는 Lance `Scanner` 를 동기로 구성하고, 전문 검색 인덱스 요구를 검증하고, 정렬이 있으면 상한 있는 spill 실행기로, 없으면 Lance 기본 스트림으로 보낸다. 구성을 동기로 끝내는 이유를 주석은 rustc 의 레이아웃 질의 한계로 설명한다. 저장소와 변경 연산의 future 가 이미 깊게 중첩돼 있어 비동기 계층을 하나 더 쌓으면 통합 테스트 크레이트가 컴파일되지 않는다.

### 코드 분석: ScanTuning

파일: `crates/omnigraph/src/table_store.rs`, [table_store.rs L244-L252](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/table_store.rs#L244-L252)

호출자가 스캐너를 조정할 때 받는 설정 객체다.

```rust
pub(crate) struct ScanTuning<'a> {
    scanner: &'a mut Scanner,
    full_text_columns: Option<HashSet<String>>,
    /// FTS-index demand of the typed filters set through this surface, unioned
    /// over every `filter_expr` call although Lance keeps only the last filter:
    /// a fail-closed over-approximation.
    filter_demand: FtsFilterDemand,
}
```

`ScanTuning` 은 `Scanner::order_by` 를 노출하지 않는다. 정렬은 실행기 선택을 바꾸므로 `scan_stream_with` 가 미리 결정해야 하고, 콜백이 나중에 정렬을 끼워 넣으면 그 결정을 우회하게 된다. 콜백이 쓸 수 있는 메서드는 필터, fragment 지정, 행 수와 바이트 기준 배치 목표, 파이프라인 읽기, BM25 와 ANN 검색 설정, Blob 처리 방식이다([table_store.rs L320-L448](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/table_store.rs#L320-L448)). `filter_demand` 는 Lance 가 마지막 필터만 유지하더라도 모든 `filter_expr` 호출의 전문 검색 요구를 합쳐, 검증 단계가 실제보다 넓게 확인하도록 한다.

### 코드 분석: execute_bounded_ordered_scan

파일: `crates/omnigraph/src/table_store.rs`, [table_store.rs L3176-L3189](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/table_store.rs#L3176-L3189)

정렬 스캔은 spill 이 꺼져 있으면 실행하지 않는다.

```rust
    async fn execute_bounded_ordered_scan(
        scanner: Scanner,
        options: LanceExecutionOptions,
    ) -> Result<DatasetRecordBatchStream> {
        // Lance permits LANCE_BYPASS_SPILLING to override even an explicit
        // option. Ordered graph scans fail closed instead of falling back to
        // an unbounded SortExec resident set.
        if !options.use_spilling() {
            return Err(OmniError::ResourceLimitExceeded {
                resource: "ordered_scan_spilling_disabled".to_string(),
                limit: 0,
                actual: 1,
            });
        }
```

Lance 의 일반 스캐너는 `SortExec` 을 상한 없는 메모리 풀에서 spill 없이 실행한다. Omnigraph 는 환경 변수 `LANCE_BYPASS_SPILLING` 이 spill 을 꺼도 상한 없는 정렬로 돌아가지 않고 오류를 낸다. 이어서 실행마다 자기 `FairSpillPool` 과 디스크 관리자를 가진 런타임을 새로 만든다. Lance 의 공용 헬퍼는 컨텍스트를 캐시해, LRU 방출 뒤 오래된 컨텍스트와 새 컨텍스트가 공존할 수 있기 때문이다.

| 상수 | 값 | 위치 |
|---|---|---|
| `ORDERED_SCAN_MEMORY_BYTES` | 150 MiB | [table_store.rs L134](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/table_store.rs#L134-L134) |
| `ORDERED_SCAN_SCRATCH_BYTES` | 100 GiB | [table_store.rs L135](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/table_store.rs#L135-L135) |
| `ORDERED_SCAN_EXECUTION_BATCH_ROWS` | 8,192 행 | [table_store.rs L136](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/table_store.rs#L136-L136) |

### 문장 범위 memo

`RangeMemoStore::memo_key`([read_memo.rs L285-L311](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-core/src/read_memo.rs#L285-L311))는 한 문장 안에서 같은 바이트 범위를 다시 읽을 때 저장소 요청을 생략할지 정한다. memo 대상은 `data`, `_indices`, `_deletions` 아래의 불변 파일뿐이다([read_memo.rs L28](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-core/src/read_memo.rs#L28-L28)). 조건부 요청, `head`, 버전 지정 요청은 결과가 객체의 현재 상태에 따라 달라지므로 그대로 통과시킨다. 키는 저장소 접두어, 경로, 범위(전체, 구간, 오프셋, 접미)의 조합이다. `RangeReadMemo` 는 바이트 예산을 가지며 예산이 모자라면 저장하지 않고 통과시킨다. 항목마다 페이로드 외에 128 바이트를 더 계산한다. 핸들 캐시는 이 memo 의 래퍼를 계측 래퍼 바깥에 쌓아, memo 적중이 요청 수 계측에 잡히지 않게 한다.
