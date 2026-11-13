예약 수집은 외부 폴더를 주기적으로 조회해 바뀐 파일을 수집 큐에 넣는다.

## 예약 수집

### 설정과 타이머

예약 수집은 외부 폴더(예: 다운로드 폴더, 동기화 폴더)를 주기적으로 확인해 새 파일이나 바뀐 파일을 위키 원본으로 가져온다. 설정은 네 필드다([wiki-store.ts#L207-L212](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/stores/wiki-store.ts#L207-L212)).

| 필드 | 의미 | 기본값 |
| --- | --- | --- |
| `enabled` | 켜짐 여부 | `false` |
| `path` | 감시할 폴더 경로 | `""` |
| `interval` | 스캔 간격 (분), 설정 화면에서 1~1440 | 60 |
| `lastScan` | 마지막 스캔 시각 (ms) | `null` |

기본값은 앱 시작 코드([App.tsx#L120-L125](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/App.tsx#L120-L125)), 입력 범위는 설정 화면([scheduled-import-section.tsx#L153-L156](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/components/settings/sections/scheduled-import-section.tsx#L153-L156))에 있다.

`startScheduledImport`는 프로젝트별 간격과 상관없이 60초마다 한 번 깨어난다. 깨어날 때마다 현재 프로젝트와 최근 프로젝트 목록 전체를 돌며 각자의 설정을 읽고, 간격이 지난 프로젝트만 스캔한다.

무엇을 보는지: 60초 타이머와 프로젝트별 due 판정.
[src/lib/scheduled-import.ts#L437-L445](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/scheduled-import.ts#L437-L445), [#L641-L657](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/scheduled-import.ts#L641-L657)

```ts
export function isScheduledImportDue(config: ScheduledImportConfig, now = Date.now()): boolean {
  if (!config.enabled || !config.path || config.interval <= 0) return false
  if (!config.lastScan) return true
  const intervalMs = Math.max(1, Math.min(1440, config.interval)) * 60 * 1000
  return now - config.lastScan >= intervalMs
}

export function startScheduledImport(project: WikiProject, config: ScheduledImportConfig): void {
  stopScheduledImport()
  const runId = ++activeRunId
  void runScheduledImportSweep(project, config, runId, true).catch((err) => {
    console.error("Scheduled import sweep failed:", err)
  })
  scanTimer = setInterval(() => {
    void runScheduledImportSweep(project, config, runId, false).catch((err) => {
      console.error("Scheduled import sweep failed:", err)
    })
  }, 60 * 1000)
}
```

해설:

- 시작 직후 한 번은 `forceActive = true`로 현재 프로젝트를 간격과 상관없이 스캔한다.
- 열려 있지 않은 최근 프로젝트도 자기 설정이 켜져 있으면 스캔된다. 앱 시작 코드의 주석이 "recently opened projects may still have active folder monitors"라고 이유를 적는다([App.tsx#L132-L134](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/App.tsx#L132-L134)).
- `activeRunId`는 세대 번호다. 설정을 바꿔 `startScheduledImport`를 다시 부르면 번호가 올라가고, 이전 세대의 스캔은 다음 검사 지점에서 스스로 멈춘다.

### 한 번의 스캔: scanAndImport

한 번의 스캔은 다음 순서로 진행된다.

1. 감시 경로가 프로젝트 내부면 건너뛰고 활동 패널에 오류를 한 번 남긴다.
2. 파일마다 제외 규칙(`.llm-wiki`, `wiki/`, 숨김 파일, 설정 파일 확장자, 수집 불가 확장자, 크기 상한)을 적용한다.
3. 크기와 수정 시각이 기록과 같으면 기록된 해시를 재사용하고, 다르면 MD5를 계산한다.
4. 해시가 바뀐 파일만 `raw/sources/scheduled-import/`로 복사하고 수집 큐에 등록한다.
5. 감시 폴더에서 사라진 파일을 정리하고, DB와 `lastScan`을 저장한다.

자기 참조 경로 차단이 먼저다. 감시 경로가 프로젝트 루트이거나, 프로젝트를 포함하거나, `raw/`, `raw/sources/`, `wiki/`, `.llm-wiki/` 안에 있으면 스캔하지 않는다([scheduled-import.ts#L150-L164](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/scheduled-import.ts#L150-L164)). 위키가 만든 파일을 다시 원본으로 가져오는 고리를 막는 장치다. 활동 패널 메시지는 프로젝트 원본은 "source folder monitoring"이 처리한다고 안내한다. 원본 폴더 감시는 5장에서 다룬다.

무엇을 보는지: stat 캐시 fast path와 해시 비교.
[src/lib/scheduled-import.ts#L523-L543](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/scheduled-import.ts#L523-L543)

```ts
const cachedStat = db.statCache?.[key]
const mtime = await getFileModifiedTime(sourcePath)
if (cachedStat && db.files[key] && cachedStat.size === size && cachedStat.mtime === mtime) {
  nextDb.files[key] = db.files[key]
  nextStatCache[key] = cachedStat
  continue
}

const md5 = await getFileMd5(sourcePath)
nextStatCache[key] = { size, mtime }

if (db.files[key] === md5) {
  nextDb.files[key] = md5
  continue
}
// 여기까지 오면 내용이 바뀐 파일: raw/sources/scheduled-import/ 로 복사 후 changedFiles 에 추가
```

`touch`처럼 시각만 바뀐 파일은 해시가 같으므로 다시 수집하지 않는다. 해시 기록은 `.llm-wiki/scheduled-import-db.json`에 원자적 쓰기로 저장된다([scheduled-import.ts#L410-L422](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/scheduled-import.ts#L410-L422)). `statCache`는 나중에 추가된 필드라서, 이전 버전이 쓴 DB에는 없고 첫 스캔에서 한 번 해시를 계산한다(주석, [scheduled-import.ts#L42-L48](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/scheduled-import.ts#L42-L48)).

| 상수 / 규칙 | 값 | 출처 |
| --- | --- | --- |
| `MAX_SCHEDULED_IMPORT_BYTES` | 100 MiB | [L82](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/scheduled-import.ts#L82-L82) |
| 실제 크기 상한 | 위 값과 원본 감시 설정 `maxFileSizeMb` 중 작은 쪽 | [L507-L514](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/scheduled-import.ts#L507-L514) |
| 제외 확장자 | `json`, `yaml`, `yml`, `xml` | [L83](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/scheduled-import.ts#L83-L83) |
| 복사 위치 | `raw/sources/scheduled-import/<상대경로>` | [L275-L297](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/scheduled-import.ts#L275-L297) |
| 대상 경로 정리 | Windows 금지 문자 치환, `con`, `nul` 등 예약 이름 앞에 `_` | [L188-L231](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/scheduled-import.ts#L188-L231) |

경로 정리 과정에서 이름이 바뀌면 원래 경로의 FNV 해시 앞 6글자를 파일명에 붙인다. 서로 다른 원본이 정리 후 같은 이름이 되어 덮어쓰는 것을 막는다. Windows 드라이브 경로와 UNC 경로는 비교할 때만 소문자로 접고, 실제로 쓰는 경로는 그대로 둔다([scheduled-import.ts#L121-L127](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/scheduled-import.ts#L121-L127)).

### 등록 실패와 삭제 처리

수집 큐 등록이 실패한 파일은 DB에 해시를 기록하지 않는다([scheduled-import.ts#L595-L601](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/scheduled-import.ts#L595-L601)). `enqueueSourceIngest`는 수집용 LLM이 설정되지 않았으면 빈 배열을 돌려주는데([source-lifecycle.ts#L258-L268](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/source-lifecycle.ts#L258-L268)), 이때 해시를 기록하면 나중에 LLM을 설정해도 그 파일은 "이미 가져온 파일"로 남는다. 기록하지 않으므로 다음 스캔에서 다시 시도된다.

감시 폴더에서 사라졌거나 제외 규칙에 새로 걸린 파일은 `cleanupRemovedScheduledImports`가 정리한다. 수집 큐에 남은 작업을 버리고, 복사본과 그 복사본에서 생성된 위키 페이지를 `deleteSourceFile`로 지운다([scheduled-import.ts#L323-L361](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/scheduled-import.ts#L323-L361)). 삭제가 실패하면 옛 기록을 다시 넣어 다음 스캔이 재시도하게 한다. 이 정리는 활성 프로젝트에서만 하고, 열려 있지 않은 프로젝트는 옛 기록을 보존했다가 다음에 열릴 때 처리한다([scheduled-import.ts#L566-L572](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/scheduled-import.ts#L566-L572)). 지우는 작업은 파일을 되돌릴 수 없으므로 사용자가 보고 있는 프로젝트에서만 한다는 선택이다.

> **주의**: 감시 폴더에서 원본 파일을 지우면 그 파일에서 만든 위키 페이지까지 지워진다. 감시 폴더를 "임시 받은 편지함"처럼 비우는 용도로 쓰면 위키 내용도 함께 사라진다(코드 동작으로 추론, 실제 사용 흐름은 미검증).

`scanning` 플래그는 모듈 전역 하나라서 프로젝트가 여럿이어도 스캔은 한 번에 하나씩만 돈다([scheduled-import.ts#L466-L468](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/scheduled-import.ts#L466-L468)).
