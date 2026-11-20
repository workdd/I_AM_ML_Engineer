Blob 셀의 세 상태와 읽기 제한을 본다.

## Blob 저장과 조회

Blob 은 기존 노드나 엣지의 속성 셀 하나다([blob.md L1-L4](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/docs/dev/blob.md#L1-L4)). 물리 배치(inline, packed, dedicated)는 Lance Blob-v2 가 정하고, Omnigraph 는 논리 식별, 스냅샷 선택, 외부 원본 허용, 범위 제한 전달을 맡는다. 셀 쓰기(`put_blob_at_as`)는 9장에서 다룬다.

### 코드 분석: BlobDescriptor

파일: `crates/omnigraph/src/blob.rs`, [blob.rs L986-L997](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/blob.rs#L986-L997)

저장된 descriptor 에서 해석한 논리 상태다.

```rust
pub(crate) enum BlobDescriptor {
    Null,
    Managed {
        length: u64,
    },
    External {
        uri: String,
        offset: u64,
        length: Option<u64>,
    },
}
```

`Null` 은 Arrow validity 로만 판정하고, 길이 0 의 `Managed` 도 유효한 값이다. `External` 은 호출자가 소유한 절대 URI 와 선택적 범위다.

`BlobDescriptorDecoder::classify`([blob.rs L1077-L1105](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/blob.rs#L1077-L1105))는 부모 struct 의 null 여부를 다른 어떤 자식 검사보다 먼저 본다. Lance 는 null 부모의 자식에 의미 없는 값을 채우기 때문이다. 반대로 null 이 아닌 부모에 null 자식이 있으면 손상으로 보고 거부해, 자식 null 이 또 하나의 null 표현이 되지 않게 한다. 물리 종류마다 무시되는 필드도 정해진 값인지 확인해, 손상된 상태가 그럴듯한 바이트로 바뀌지 않게 한다.

### 코드 분석: BlobReader::read_range

파일: `crates/omnigraph/src/blob.rs`, [blob.rs L200-L217](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/blob.rs#L200-L217)

관리형 Blob 의 반열린 범위를 읽기 전에 범위와 상한을 확인한다.

```rust
    pub async fn read_range(&self, range: Range<u64>) -> Result<Bytes> {
        if range.start > range.end || range.end > self.length {
            return Err(OmniError::BlobRangeNotSatisfiable {
                start: range.start,
                end: range.end,
                length: self.length,
            });
        }

        let requested = range.end - range.start;
        if requested > BLOB_READ_RANGE_MAX_BYTES {
            return Err(OmniError::resource_limit(
                BLOB_READ_RANGE_BYTES_RESOURCE,
                BLOB_READ_RANGE_MAX_BYTES,
                requested,
            ));
        }
```

`BlobReader`([blob.rs L175-L179](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/blob.rs#L175-L179))는 연 시점의 불변 dataset 버전을 쥐고 있어 이후 브랜치가 움직여도 따라가지 않는다. 주석은 이 고정이 영속 리스가 아니어서, 정리 작업이 버전을 회수하면 캐시되지 않은 이후 범위 읽기가 명시적으로 실패한다고 적는다. 전체를 한 번에 읽는 공개 API 는 없고 큰 값은 상한 이하의 범위 읽기를 반복한다.

| 상수 | 값 | 위치 |
|---|---|---|
| `BLOB_READ_RANGE_MAX_BYTES` | 4 MiB | [blob.rs L44](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/blob.rs#L44-L44) |
| `BLOB_WRITE_MAX_BYTES` | `KEYED_WRITE_MAX_BYTES` 와 같음 | [blob.rs L50](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/blob.rs#L50-L50) |
| `KEYED_WRITE_MAX_ROWS`, `KEYED_WRITE_MAX_BYTES` | 8,192 행, 32 MiB | [storage_layer.rs L71-L72](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/storage_layer.rs#L71-L72) |

### ETag 와 식별자 검사

관리형 셀의 강한 ETag 는 내용 해시가 아니다. `blob_etag`([blob.rs L1777-L1808](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/blob.rs#L1777-L1808))는 안정 테이블 ID, incarnation ID, 안정 속성 ID, 테이블 버전, stable row ID, 트랜잭션 파일 이름을 SHA-256 에 넣고 앞 16 바이트를 16진수로 큰따옴표 안에 담는다. "어느 논리 셀의 어느 불변 버전인가" 를 식별하므로 이름을 바꿔도 유지되고, 다른 쓰기가 행을 바꾸면 버전과 트랜잭션 파일이 달라져 바뀐다. 셀 읽기 진입점 `read_blob_at`([blob.rs L1372-L1376](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/blob.rs#L1372-L1376))은 선택한 스냅샷의 테이블 식별자가 카탈로그가 해석한 식별자와 같은지 먼저 확인하고, 다르면 Blob 무결성 오류를 낸다. 외부 값은 descriptor 만 돌려주고 대상 객체를 열지 않는다.

### 외부 Blob 정책

`ExternalBlobPolicy`([blob.rs L424-L431](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/blob.rs#L424-L431))는 기본이 `Deny` 이고, `Allow` 는 기준 경로 목록을 갖는다. 빈 목록과 서로 겹치는 기준 경로는 검증에서 거부한다. `authorize`([blob.rs L519-L558](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/blob.rs#L519-L558))는 두 단계로 판정한다. 먼저 문자열 정규화로 어느 기준 경로 안에 있는지 보고, `file://` 이면 실제 파일 경로로 정규화한 뒤 다시 확인해 심볼릭 링크로 기준 밖을 가리키는 경우를 막는다. 서버 실행 문맥은 `server_safe_only` 로 내장 전용 기준 경로를 뺀다. 이미 저장된 외부 참조는 현재 정책이 새로 허용하지 않더라도 읽기 시 분류할 수 있다. `load csv` 의 원본 파일도 같은 정책을 거친다. `TableStore::external_source`([table_store.rs L2332-L2352](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/table_store.rs#L2332-L2352))가 정책 허용과 크기 확인을 하고, 이후 읽기는 확인 때의 ETag 에 고정된다. 적재 흐름은 10장에서 다룬다.

### Blob 배치 조회와 압축 배치 크기

물질화가 필요한 재작성은 한 배치의 관리형 셀을 `managed_blob_payloads`([table_store.rs L3025-L3050](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/table_store.rs#L3025-L3050))로 읽는다. 행마다 `BlobFile::read` 를 부르지 않고 `Dataset::read_blobs` 한 번에 관리형 row ID 만 넘긴다. 외부 행을 넘기면 Lance 가 참조 객체를 직접 열어 읽기 때문이다. IO 버퍼를 8 MiB 로 명시하는 이유는 Lance 기본값이 32 MiB 에 저장소 IO 병렬도를 곱한 값이기 때문이다.

### 코드 분석: compaction_blob_batch_rows

파일: `crates/omnigraph/src/table_store.rs`, [table_store.rs L8294-L8301](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/table_store.rs#L8294-L8301)

Blob 테이블 압축 배치의 행 수를 정한다.

```rust
pub(crate) fn compaction_blob_batch_rows(max_row_blob_bytes: u64) -> usize {
    if max_row_blob_bytes == 0 {
        return COMPACTION_MAX_BATCH_ROWS;
    }
    usize::try_from(COMPACTION_BLOB_BATCH_BYTES / max_row_blob_bytes)
        .unwrap_or(1)
        .clamp(1, COMPACTION_MAX_BATCH_ROWS)
}
```

Lance 13 은 Blob 열이 있는 테이블을 바이너리 복사로 압축하지 않고 스캐너로 다시 쓰며, 그 스캐너는 배치마다 관리형 페이로드를 모두 읽는다. 그래서 각 압축 작업의 fragment 를 descriptor 만 읽어 가장 큰 행의 관리형 Blob 바이트를 구하고, `COMPACTION_BLOB_BATCH_BYTES`(`KEYED_WRITE_MAX_BYTES` 와 같은 32 MiB)를 그 값으로 나눈 행 수를 배치 크기로 정한다. 행 하나가 16 MiB 를 넘으면 배치는 1 행이 된다. 외부 descriptor 는 읽지 않고 옮기므로 0 으로 센다. 이 상한은 배치당 페이로드의 상한이지 힙 전체의 상한이 아니라고 [blob.md L67](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/docs/dev/blob.md#L67-L67) 이 밝힌다. 압축 실행 전체는 13장에서 다룬다.

### 쓰기 예산

Blob 과 일반 행 데이터의 쓰기 한도는 `WriteBudget`([storage_layer.rs L79-L97](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/storage_layer.rs#L79-L97))이 정한다. 연산마다 세션 설정 `write_max_bytes` 를 한 번 캡처하는 불변 값이고, 기본값은 33,554,432 바이트(32 MiB)다([settings.rs L123-L130](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-compiler/src/settings.rs#L123-L130)). 행 데이터, 물질화된 Blob 페이로드, 파서, 삭제 ID 수집이 각각 이 한도를 독립적으로 가진다. 엄격한 insert 와 덮어쓰기는 이 값을 Lance 파일 작성기의 누적 목표로도 쓰지만(`budgeted_file_writer_options`, [table_store.rs L120-L126](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/table_store.rs#L120-L126)), 주석은 이것이 전체 메모리 상한이 아니라고 밝힌다. 예산이 쓰기 단계마다 어떻게 차감되는지는 9장에서 다룬다.
