최상위 핸들이 담는 상태와 설정 범위, 캐시를 본다.

## Omnigraph 핸들의 구조

`Omnigraph`는 그래프 루트 하나를 연 핸들이다. 필드는 저장 접근, 설치된 상태, 캐시, 동시성 제어, 정책으로 나뉜다.

### 코드 분석: 핸들의 첫 필드

[`crates/omnigraph/src/db/omnigraph.rs` L407-L419](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/db/omnigraph.rs#L407-L419)

```rust
pub struct Omnigraph {
    root_uri: String,
    storage: Arc<dyn StorageAdapter>,
    /// Split Lance access context: data tables receive a graph-scoped cached
    /// session, while mutable control metadata uses a zero-cache session. Both
    /// share one process-wide object-store registry/client pool.
    lance_access: crate::lance_access::LanceAccessContext,
    /// The installed state of the bound branch: an immutable value readers
    /// load with no lock, replaced whole by `install_coordinator` once a
    /// publish or refresh has done its storage round trips on a clone.
    coordinator: Arc<ArcSwap<GraphCoordinator>>,
    table_store: TableStore,
    runtime_cache: RuntimeCache,
```

- `coordinator`는 `ArcSwap<GraphCoordinator>`다. 읽기는 잠금 없이 포인터 하나를 읽고, 게시나 새로 고침은 복사본에서 저장소 왕복을 끝낸 뒤 포인터를 바꾼다
- `lance_access`는 데이터 테이블용 캐시 있는 세션과 제어 메타데이터용 캐시 없는 세션을 나눠 든다

| 필드 | 역할 | 정의 |
|---|---|---|
| `table_store` | 노드, 엣지 테이블 열기와 스테이징(08장) | [L418](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/db/omnigraph.rs#L418) |
| `runtime_cache` | CSR 그래프 인덱스 캐시(06장) | [L419](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/db/omnigraph.rs#L419) |
| `feed_cut_cache` | 변경 피드 절단점 캐시(13장) | [L420-L435](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/db/omnigraph.rs#L420-L435) |
| `read_caches` | 공유 Lance 세션, 테이블 핸들, 수락 카탈로그, 컴파일 쿼리 캐시 | [L436-L441](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/db/omnigraph.rs#L436-L441) |
| `schema_view` | 스키마 원본과 카탈로그를 한 쌍으로 보관 | [L442-L446](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/db/omnigraph.rs#L442-L446) |
| `write_queue` | 같은 루트를 연 모든 핸들이 공유하는 쓰기 게이트 | [L447-L454](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/db/omnigraph.rs#L447-L454) |
| `merge_authority_cache` | 묶이지 않은 브랜치 하나의 코디네이터(12장) | [L455-L479](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/db/omnigraph.rs#L455-L479) |
| `policy` | 엔진 계층 정책 검사기, `None`이면 검사 없음 | [L484-L499](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/db/omnigraph.rs#L484-L499) |
| `embedding` | 첫 문자열 `nearest`에서 만드는 임베딩 클라이언트(07장) | [L500-L505](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/db/omnigraph.rs#L500-L505) |

### 코드 분석: 열기 절차

`open`, `open_read_only`, `open_clone`, `open_with_storage`는 모두 `open_with_storage_and_mode`로 모인다([`crates/omnigraph/src/db/omnigraph.rs` L811-L842](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/db/omnigraph.rs#L811-L842)). 그 앞부분이다. [`crates/omnigraph/src/db/omnigraph.rs` L943-L956](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/db/omnigraph.rs#L943-L956)

```rust
let control_session = lance_access.control_session();
let prepared = crate::db::manifest::ManifestCoordinator::prepare_open_with_contract(
    &root,
    &control_session,
)
.await?;
let schema_contract_guard = write_queue.acquire_schema_exclusive().await?;
let (coordinator, contract) =
    GraphCoordinator::open_with_contract(&root, Arc::clone(&storage), prepared).await?;
crate::db::schema_state::refuse_unsupported_schema_versions(&contract.ir)?;
validate_schema_contract_row(&contract)?;
if matches!(mode, OpenMode::ReadWrite) {
    verify_local_create_if_absent(&root, storage.as_ref()).await?;
}
```

- `prepare_open_with_contract`는 `__manifest`의 내부 형식 버전이 이 바이너리가 읽을 수 있는 범위인지 확인한다. 주석은 제자리 이전이 없으며 범위 밖이면 export와 import로 다시 만들라고 적는다([`crates/omnigraph/src/db/omnigraph.rs` L937-L942](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/db/omnigraph.rs#L937-L942))
- 스키마 계약을 읽는 동안 스키마 게이트를 배타적으로 잡는다. 다른 핸들의 스키마 적용과 겹치지 않기 위해서다
- 읽기 쓰기 모드에서만 로컬 생성 확인과 구형 복구 파일 검사를 한다

뒤이어 수락된 SchemaIR에서 카탈로그를 만들고(`build_catalog_from_ir`, 03장) 필드를 조립한다([`crates/omnigraph/src/db/omnigraph.rs` L966-L1011](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/db/omnigraph.rs#L966-L1011)). `policy`는 `None`으로 시작하고 서버, 클러스터 배포, CLI의 쓰기 열기가 `with_policy`로 정책을 끼운다([`crates/omnigraph/src/db/omnigraph.rs` L1085-L1088](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/db/omnigraph.rs#L1085-L1088)). `write_queue`는 `WriteQueueManager::for_root`가 루트 식별자로 찾아 준 값이라, 같은 프로세스에서 같은 루트를 두 번 열면 두 핸들이 게이트를 공유한다([`crates/omnigraph/src/db/omnigraph.rs` L934-L936](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/db/omnigraph.rs#L934-L936)).

### 코드 분석: 엔진 계층 정책 검사

변경 진입점은 서버의 검사와 별도로 엔진 안에서 정책을 다시 확인한다. [`crates/omnigraph/src/db/omnigraph.rs` L1207-L1218](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/db/omnigraph.rs#L1207-L1218)

```rust
let Some(checker) = self.policy.as_ref() else {
    return Ok(());
};
let Some(actor) = actor else {
    return Err(OmniError::Policy(format!(
        "policy denied action '{action}': a policy is installed and the request names \
         no actor"
    )));
};
checker
    .check(action, scope, actor)
    .map_err(|err| OmniError::Policy(err.to_string()))
```

- 정책이 없으면 통과한다. 내장 개발 용도의 기본값이다
- 정책이 있는데 행위자가 없으면 거부한다. 서버를 거치지 않는 직접 쓰기도 정책을 우회하지 못한다
- 검사 단위는 동작, 범위, 행위자 세 값이다. 행 단위 검사를 여기에 넣지 말라는 주석이 필드 정의에 있다([`crates/omnigraph/src/db/omnigraph.rs` L490-L494](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/db/omnigraph.rs#L490-L494))

## Session과 설정 범위

쿼리와 변경은 `Omnigraph`가 아니라 `Session`의 메서드다. `Session`은 `Arc<Omnigraph>` 하나와 설정 묶음을 들고, 설정과 무관한 연산은 `Deref`로 핸들에 넘긴다([`crates/omnigraph/src/session.rs` L66-L72](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/session.rs#L66-L72)).

### 코드 분석: Session 구조

[`crates/omnigraph/src/session.rs` L28-L37](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/session.rs#L28-L37)

```rust
#[derive(Clone)]
pub struct Session {
    db: Arc<Omnigraph>,
    settings: SessionSettings,
    sources: Sources,
    baseline: SessionSettings,
    baseline_sources: Sources,
    #[cfg(feature = "test-util")]
    read_executor: Option<Arc<dyn ReadExecutor>>,
}
```

- 문서 주석에 따르면 CLI에서는 파일 하나, 서버에서는 요청 하나, 내장 호출자에게는 직접 만든 값 하나가 `Session`의 범위다. 저장되지 않고 요청 사이에서 공유되지 않는다
- `baseline`은 `reset`이 돌아갈 값이고, `sources`는 각 설정이 기본값, 환경 변수, 파일 중 어디서 왔는지 기록한다

### 코드 분석: 호출 단위 설정

쿼리 원본이 `set` 줄로 시작하면 그 호출에만 적용할 설정을 복사본에 만든다. [`crates/omnigraph/src/session.rs` L296-L309](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/session.rs#L296-L309)

```rust
pub fn effective(&self, source: &str) -> Result<SessionSettings> {
    if !has_settings_prefix(source) {
        return Ok(self.settings.clone());
    }
    let file = parse_query(source).map_err(OmniError::Compiler)?;
    let scoped = self.with_prefix(&file.settings).map_err(|error| {
        OmniError::Compiler(CompilerError::query(QueryDiagnostic::parse(
            Q003,
            error.to_string(),
            None,
        )))
    })?;
    Ok(scoped.settings)
}
```

- `has_settings_prefix`는 앞의 공백과 주석을 건너뛰고 첫 단어가 `set`이나 `reset`인지만 본다([`crates/omnigraph-compiler/src/query/parser.rs` L63-L70](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/query/parser.rs#L63-L70)). 접두부가 없으면 여기서 파싱하지 않아, 뒤의 컴파일 쿼리 캐시가 파싱 없이 맞을 수 있다
- 접두부가 있으면 `with_prefix`로 복사본에 적용하고 원래 `Session`은 그대로 둔다
- 설정 정의 표는 컴파일러 크레이트에 있고 04장에서 다룬다

## 런타임 캐시

핸들의 캐시는 정확성의 근거가 아니다. 버전과 식별자를 키에 넣어 낡은 항목이 맞지 않게 한다.

| 캐시 | 키 | 크기 | 비우는 시점 |
|---|---|---|---|
| 그래프 인덱스 | 엣지 테이블별 식별자, 버전, 고정 버전, e_tag | [8개](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/runtime_cache.rs#L258-L266) | 새로 고침, 스키마 적용, 최적화. 쓰기는 해당 엣지만 |
| 영속 인덱스 실패 기록 | 그래프 인덱스와 같음 | [32개](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/runtime_cache.rs#L258-L266) | 새 산출물 디코드 성공 |
| 컴파일 쿼리 | 원본 SHA-256, 쿼리 이름, 카탈로그 포인터 | [256개](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/runtime_cache.rs#L357-L359) | 비우지 않음 |

그래프 인덱스 캐시의 키는 스냅샷 id가 아니라 엣지 테이블의 물리 식별이다. 그래서 막 만든 브랜치처럼 엣지 테이블이 main과 물리적으로 같으면 main의 인덱스를 그대로 쓴다([`crates/omnigraph/src/runtime_cache.rs` L20-L29](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/runtime_cache.rs#L20-L29)). 캐시에 없으면 최적화 작업이 저장한 인덱스 산출물을 한 번만 디코드하고, 그것도 없으면 엣지 테이블을 스캔해 만든다([`crates/omnigraph/src/runtime_cache.rs` L149-L225](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/runtime_cache.rs#L149-L225)).

### 코드 분석: 컴파일 쿼리 캐시 조회

[`crates/omnigraph/src/runtime_cache.rs` L429-L440](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/runtime_cache.rs#L429-L440)

```rust
) -> Option<CompiledRead> {
    self.inner
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get(key)
        .filter(|entry| {
            entry
                .catalog
                .upgrade()
                .is_some_and(|held| Arc::ptr_eq(&held, catalog))
        })
        .map(|entry| entry.compiled.clone())
```

- 키에는 원본 대신 SHA-256 요약이 들어간다([`crates/omnigraph/src/runtime_cache.rs` L418-L423](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/runtime_cache.rs#L418-L423))
- 항목은 카탈로그를 `Weak`로 들고 있고, 조회 때 호출자의 `Arc`와 같은 객체인지 비교한다. 카탈로그가 다시 만들어지면 맞지 않는다

### 코드 분석: 코디네이터 교체

읽기가 잠금 없이 상태를 읽는 근거는 설치된 코디네이터를 포인터 교환으로만 바꾼다는 점이다. [`crates/omnigraph/src/db/omnigraph.rs` L2593-L2600](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/db/omnigraph.rs#L2593-L2600)

```rust
let ordered = Arc::ptr_eq(&installed, derived_from)
    || next
        .manifest_incarnation()
        .supersedes(&installed.manifest_incarnation());
if installed.current_branch() != next.current_branch() || !ordered {
    observed(&GATE_COORDINATOR_CAS_REFUSED);
    return false;
}
```

- 새 상태가 지금 설치된 상태에서 파생됐거나 매니페스트가 더 새로울 때만 바꾼다. 늦게 끝난 새로 고침이 새 상태를 덮어쓰지 못한다
- 묶인 브랜치가 다르면 거부한다
- `refresh`는 교체 뒤 스키마 뷰를 다시 읽고 읽기 캐시를 비운다([`crates/omnigraph/src/db/omnigraph.rs` L2624-L2632](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/db/omnigraph.rs#L2624-L2632))
