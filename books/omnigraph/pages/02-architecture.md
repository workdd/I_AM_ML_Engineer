Omnigraph는 Cargo 워크스페이스의 20개 크레이트로 나뉘어 있고, 엔진 크레이트 `omnigraph-engine`이 컴파일러, 계획기, 카탈로그, 코어, 저장 크레이트를 묶어 `Omnigraph` 핸들과 `Session`으로 공개한다. 이 장은 크레이트 사이의 의존 방향을 Cargo.toml 기준으로 정리하고, 쿼리 하나와 변경 하나가 어떤 함수를 거쳐 어느 크레이트로 넘어가는지 따라간다. 각 단계의 내부는 뒤의 장에서 다루고, 여기서는 경계와 넘겨지는 값만 본다.

## 워크스페이스 구성

루트 Cargo.toml은 `crates/` 아래 20개 크레이트와 `skills`, `tools/omnigraph-vocabulary-guard`를 멤버로 둔다. 인자 없이 `cargo build`를 실행하면 `default-members`의 세 크레이트만 빌드된다.

### 코드 분석: 기본 빌드 대상

멤버 목록의 끝과 기본 빌드 대상이다. [`Cargo.toml` L24-L31](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/Cargo.toml#L24-L31)

```toml
    "skills",
    "tools/omnigraph-vocabulary-guard",
]
default-members = [
    "crates/omnigraph",
    "crates/omnigraph-cli",
    "crates/omnigraph-server",
]
```

- 컴파일러나 계획기가 목록에 없어도 엔진이 의존하므로 함께 빌드된다
- 엔진의 `dst` 기능 주석은 `--workspace` 빌드에서는 이 시험용 기능이 dev 크레이트를 통해 켜지고, 릴리스 산출물은 `-p`로 빌드한다고 밝힌다([`crates/omnigraph/Cargo.toml` L17-L27](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/Cargo.toml#L17-L27))
- Lance 계열 의존성은 `=13.0.0`으로 고정되어 있다. 주석은 Lance 크레이트 버전을 함께 맞추고(lockstep) 호환성 요구 사항은 `docs/dev/lance.md`에 둔다고 적는다([`Cargo.toml` L51-L53](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/Cargo.toml#L51-L53))

크레이트는 역할에 따라 네 무리로 나뉜다. 역할 설명은 각 Cargo.toml의 `description`과 [아키텍처 문서의 계층 표](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/docs/dev/architecture.md#L96-L105)를 따랐다. 이 책은 첫 무리를 깊게 다루고, 둘째 무리는 14장에서 개요만 본다.

| 무리 | 크레이트 | 역할 | 장 |
|---|---|---|---|
| 핵심 엔진 | `omnigraph-compiler` | `.pg`, `.gq` 파싱, 카탈로그, 타입 검사, 마이그레이션 계획, IR 하강 | 03, 04 |
| 핵심 엔진 | `omnigraph-planner` | 논리 계획, 물리 계획, 최적화 패스, 비용, explain 문서 | 05 |
| 핵심 엔진 | `omnigraph-engine` (라이브러리 이름 `omnigraph`) | `Omnigraph`, `Session`, `Snapshot`, 실행, 쓰기, 검증, 복구 | 06~13 |
| 핵심 엔진 | `omnigraph-catalog` | `__manifest` 게시, 브랜치 등록과 계보, 보존, 커밋 그래프 | 11 |
| 핵심 엔진 | `omnigraph-core` | `OmniError`, Lance dataset 접근과 계측, 네이티브 ref 제어, dataset 주소 | 08, 11 |
| 핵심 엔진 | `omnigraph-storage` | 로컬, S3, Azure 제어 객체 접근 | 08 |
| 운영 표면 | `omnigraph-cli`, `omnigraph-server`, `omnigraph-cluster`, `omnigraph-policy`, `omnigraph-mcp`, `omnigraph-api-types`, `omnigraph-azure-admission` | 명령행, HTTP, 배포 상태, Cedar 정책, MCP, 전송 DTO, Azure 임대 | 14 |
| 시험 도구 | `omnigraph-seams`, `omnigraph-dst`, `omnigraph-gqt`, `omnigraph-gqt-core`, `omnigraph-gqt-served`, `omnigraph-reference-engine` | 시험용 대체 지점, deterministic 시뮬레이션, `.gqt` 논리 시험, 고정된 v1 참조 실행기 | 인용만 |
| 측정 | `omnigraph-bench` | 선언형 벤치마크 계획과 실행 | 인용만 |

`omnigraph-seams`는 시험 도구지만 엔진, 코어, 카탈로그가 모두 의존한다. 시험이 이름 있는 지점을 채우고 엔진이 그 지점을 조회하는 타입 하나를 제공하며, `install` 기능 없이 빌드하면 조회만 할 수 있고 채울 수는 없다([`crates/omnigraph-seams/Cargo.toml` L5-L18](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-seams/Cargo.toml#L5-L18)).

## 정리

- 워크스페이스는 20개 크레이트이고 핵심 엔진은 compiler, planner, engine, catalog, core, storage 여섯 개다. compiler와 planner는 Lance에 의존하지 않고, storage는 내부 의존이 없는 가장 아래층이다
- 엔진 크레이트(`omnigraph-engine`, 코드 이름 `omnigraph`)가 나머지를 감싸 `Omnigraph`, `Session`, `Snapshot`과 몇 타입만 공개한다. catalog와 core는 호환성 약속이 없는 내부 크레이트다
- 쿼리는 설정 확정, `capture_read_view`의 스냅샷과 카탈로그 포착, 컴파일러 세 함수의 IR 생성, `QuerySource::gather`와 계획기 수락, 실행 엔진 순서로 흐른다
- 변경은 쓰기 트랜잭션 포착, 컴파일, 행 누적, 검증, 분리 버전 스테이징과 커밋, 카탈로그 게시자의 매니페스트 CAS 한 번으로 끝난다. 프로세스 안의 게이트는 순서만 정하고 정확성은 CAS가 보장한다
- 캐시는 버전과 식별자를 키로 써서 낡은 항목이 맞지 않게 하며, 설치된 상태는 `ArcSwap` 포인터 교환으로만 바뀐다
- 오류는 코어의 `OmniError` 하나로 모이고, 쓰기 실패에는 그래프가 바뀌었는지를 나타내는 완료 증거가 붙는다
