Omnigraph 의 데이터 적재는 세 가지 입력 형태(호환 JSONL, 엄격한 NDJSON, `load csv`)로 들어오지만 모두 같은 적재 트랜잭션 하나로 모인다. 이 장은 입력이 타입별 Arrow 배치로 바뀌는 과정과, 스키마에 선언된 제약(값, 고유성, 엣지 참조 무결성, `@card`)을 커밋된 그래프와 대조해 판정하는 `WriteJudge` 를 코드 수준에서 분석한다.

## 적재 표면과 입력 형태

적재 진입점은 `Session` 의 공개 메서드다. 이름은 여럿이지만 모두 비공개 함수 `load_input_as` 로 수렴하고, 차이는 입력을 읽는 형태(`LoadInputShape`)와 결과에 붙는 정보뿐이다. 원본 문서도 "All surfaces converge on the same writer" 라고 적는다([ingestion.md L36-L46](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/docs/dev/ingestion.md#L36-L46)).

| 진입점 | 입력 형태 | 없는 브랜치 처리 |
|---|---|---|
| `load_as` | 호환 JSONL (`LoaderCompatible`) | `base` 가 있을 때만 fork |
| `load_graph_batch_as` | 엄격한 NDJSON (`StrictGraphBatch`) | `base` 가 있을 때만 fork |
| `load_csv_as_with_receipt` | `load csv` 문장 (`Csv`) | fork 하지 않음 |
| `ingest_as` (deprecated) | 호환 JSONL | `from` 이 없으면 `main` 에서 fork |
| `load_file_as` | 파일을 읽어 `load_as` 에 위임 | `load_as` 와 같음 |

`ingest_as` 는 `from.unwrap_or("main")` 으로 base 를 채워 `load_as` 를 부르므로, 예전 `ingest` 는 오타 난 브랜치 이름으로도 새 브랜치를 만들었다. `load_as` 는 base 가 없으면 없는 브랜치를 오류로 처리한다. 문서는 이 규칙을 "a typo never forks implicitly" 라고 적는다.

### 코드 분석: `LoadMode`

파일: `crates/omnigraph/src/loader/mod.rs`, [mod.rs L167-L177](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/loader/mod.rs#L167-L177)

모드 세 가지는 테이블 단위의 의미를 정한다.

```rust
pub enum LoadMode {
    /// Overwrite existing data.
    Overwrite,
    /// Append to existing data.
    Append,
    /// Merge by `id` key (upsert).
    Merge,
}
```

`Merge` 는 정확한 `id` 기준 upsert, `Append` 는 없는 `id` 만 넣는 엄격한 삽입, `Overwrite` 는 적재에 등장한 테이블의 내용만 교체한다([ingestion.md L50-L54](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/docs/dev/ingestion.md#L50-L54)). 적재에 나오지 않은 테이블은 overwrite 에서도 그대로 남는다. 입력 형태 `LoadInputShape` 의 `Csv` 변형은 `load csv` 요청, 세션의 `load_source_max_bytes`, 쓰기 연산자가 실행 동안 보유할 `Arc<Omnigraph>` 를 함께 포함한다.

### 코드 분석: 정책 검사와 범위 정규화

파일: `crates/omnigraph/src/loader/mod.rs`, [mod.rs L497-L537](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/loader/mod.rs#L497-L537)

`load_input_as` 는 어떤 할당보다 먼저 정책 검사를 한다.

```rust
        self.enforce(
            omnigraph_policy::PolicyAction::Change,
            &omnigraph_policy::ResourceScope::Branch(branch.to_string()),
            actor_id,
        )?;
        let (requested, base_branch) = Self::normalize_load_scope(branch, base)?;
```

`Change` 동작을 `Branch(branch)` 범위로 검사하고, `normalize_load_scope` 가 `main` 을 `None` 으로 바꾼다. fork 가 필요하면 `branch_create_from_as` 가 `BranchCreate` 를 따로 검사하므로, 새 브랜치 적재는 두 권한을 모두 요구한다. 모든 쓰기 `_as` 진입점이 정책을 적용해야 한다는 것은 불변식 10번이다([invariants.md L111-L115](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/docs/dev/invariants.md#L111-L115)). 정책 자체는 14장에서 다룬다.

## 정리

- 호환 JSONL, 엄격한 NDJSON, `load csv` 세 입력은 `load_input_as` 하나로 모이고, 정책 검사, `WriteTxn` 권한 캡처, 판정, 스테이징, manifest 한 번의 게시를 공유한다.
- 없는 브랜치는 명시적 base 가 있을 때만 fork 되며, fork 는 독립 게시라서 이후 실패에 `after_earlier_publication` 이 붙는다. `load csv` 는 브랜치를 만들지 않는다.
- 물리 `id` 는 `@key` 가 있으면 저장될 Arrow 값의 정규 표기(`canonical_key_id`)에서, 없으면 명시 `id` 또는 ULID 에서 나오고, 같은 정규화 함수가 `@unique` 튜플도 만든다.
- append, merge 는 테이블당 8,192행과 `write_max_bytes` 의 여러 독립 예산으로 파싱 전에 제한되고, overwrite 는 대량 입력으로 다루되 제거 ID 예산과 Arrow 사전 계산으로 제한된다.
- 컴파일러의 `checks_for` 가 검사 목록과 거부 우선순위를, 플래너가 `Probe` 와 `Scan` 접근 경로를 정하고, `WriteJudge` 는 한 번의 sweep 으로 사실을 뽑아 값, 고유성, 엣지 참조, `@card`, 남은 엣지를 판정한다.
- 인덱스 커버리지, 읽기 재사용, 정수 키 집합은 비용만 바꾸고 판정 결과는 바꾸지 않는다.
