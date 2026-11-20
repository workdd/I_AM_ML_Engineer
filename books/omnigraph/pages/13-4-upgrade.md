오프라인 업그레이드 절차와 복구 모델을 본다.

## 저장 형식 업그레이드와 복구

현재 바이너리가 쓰는 내부 저장 형식은 17 이고, 정상 open 이 받는 최소 형식은 14 다 ([migrations.rs L172-L181](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-catalog/src/migrations.rs#L172-L181)). 14 에서 16 사이 그래프는 그대로 읽고, 8, 9, 13 형식의 독립 루트는 오프라인 `omnigraph upgrade` 로만 17 이 된다 ([migrations.rs L540-L546](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-catalog/src/migrations.rs#L540-L546)).

| 출발 형식 | 경로 이름 | 처리 |
|---|---|---|
| 8, 9 | `history-lance-files-v8-to-v17`, `history-lance-files-v9-to-v17` | 그래프 루트 파일에서 스키마 계약을 읽어 변환 |
| 13 | `history-lance-files-v13-to-v17` | `__manifest` 행에서 스키마 계약을 읽어 변환 |
| 14, 15, 16 | `stamp-v14-to-v17` | 살아 있는 ref 에 형식 stamp 만 기록 |
| 그 밖 | 없음 | export 와 load 로 재구성 |

경로 이름은 `handler` 와 `stamp_handler` 가 만든다 ([upgrade.rs L63-L80](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/db/upgrade.rs#L63-L80)).

### 코드 분석: upgrade 의 브랜치 변환

파일: `crates/omnigraph/src/db/upgrade.rs`, [upgrade.rs L836-L844](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/db/upgrade.rs#L836-L844)

fence 이후 브랜치마다 변환을 게시하는 반복을 본다.

```rust
        if branch_completed(&current, branch, &intent)? {
            continue;
        }
        verify_source_head(&current, branch, native.is_none(), source)?;
        let head = converted_head(root, &session, branch, census.as_ref()).await?;
        publish_conversion(root, &session, reader, current, branch, &intent, &head).await?;
        report.last_durable_completed_boundary =
            Some(format!("converted:{}", native.unwrap_or("main")));
        fail(&UPGRADE_AFTER_BRANCH)?;
```

업그레이드 전체는 `__manifest` 의 모든 ref 를 고정 버전에서 읽고, main 에 intent 를 fence 로 기록하고, 원본 그래프 커밋을 `__history/legacy/` 아래 불변 객체로 쓰고, 각 ref 를 head 만 변환하고, 검증한 뒤 main 을 활성화하는 순서다 ([upgrade.rs L1-L9](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/db/upgrade.rs#L1-L9)). 위 반복은 끝난 ref 를 `branch_completed` 로 건너뛰므로 중간에 멈춘 실행을 다시 돌리면 이어서 진행한다. `verify_source_head` 는 fence 이후 이전 빌드가 이름 있는 ref 에 새로 게시했는지 확인하고, 그런 ref 는 받아들이지 않고 거절한다. 모든 ref 를 다시 검증한 다음 `publish_activation` 이 main 을 활성화한다 ([upgrade.rs L877-L884](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/db/upgrade.rs#L877-L884)).

`UpgradeOptions::check` 는 아무것도 쓰지 않고 변환 가능 여부만 보고하며, 결과는 `UpgradeOutcome` 여섯 가지 중 하나다 ([upgrade.rs L118-L141](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/db/upgrade.rs#L118-L141)). 한 head 의 행 수는 1,000,000, 메타데이터는 64 MiB 로 제한된다.

### 복구 모델

현재 엔진의 일반 쓰기는 복구 sidecar, roll-forward, rollback 을 남기지 않는다. [recovery.md](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/docs/dev/recovery.md) 는 그래프 쓰기가 브랜치 `__manifest` 게시 시점에 보이고, 게시 전 중단은 참조되지 않은 detached 버전만 남기며, 그것은 정리 수집기가 권한이 사라졌을 때 회수한다고 정리한다. 남는 예외는 업그레이드다. `UPGRADE_PENDING_KEY` 가 main 에 걸려 있는 동안 일반 open 은 읽기 전용까지 거절하고, 같은 `omnigraph upgrade` 를 다시 돌려야 한다.

| 멈춘 지점 | 다음 실행 |
|---|---|
| fence 이전 | 일반 실행 |
| `UPGRADE_AFTER_FENCE` | 고정 버전에서 census 를 다시 하고 디렉터리 해시 비교 |
| `UPGRADE_AFTER_LEGACY` | census 없이 디렉터리를 계획으로 사용 |
| `UPGRADE_AFTER_BRANCH` | 변환된 ref 건너뜀 |
| `UPGRADE_BEFORE_ACTIVATION` | 다시 검증한 뒤 활성화 |
| `UPGRADE_AFTER_ACTIVATION` | `already_current` |

복구 권한을 별도 내구 레코드로 두자는 [RFC 0034](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/docs/rfcs/0034-durable-recovery-authority.md) 는 2026-09-29 에 서버 런타임 RFC 로 대체되었고 구현되지 않았다.
