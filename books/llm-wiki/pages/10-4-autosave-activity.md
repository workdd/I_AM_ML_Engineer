앱 상태의 자동 저장과 에이전트의 파일 변경 추적을 본다.

## 상태 저장: auto-save

리뷰, 린트, 채팅 상태는 Zustand 저장소에 있고, `setupAutoSave`가 저장소를 구독해 변경이 생기면 디바운스 후 `.llm-wiki/` 아래 JSON으로 쓴다([auto-save.ts#L83-L130](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/auto-save.ts#L83-L130)).

| 저장소 | 디바운스 | 저장 파일 | 예외 |
| --- | --- | --- | --- |
| review-store | 1초 | `.llm-wiki/review.json` | 없음 |
| lint-store | 1초 | `.llm-wiki/lint.json` | 없음 |
| chat-store | 2초 | 채팅 기록 + 채팅 설정 | 스트리밍 중에는 건너뜀 |

이 구조에는 프로젝트를 바꿀 때 생기는 함정이 있다. 프로젝트 전환은 `resetProjectState()`로 모든 저장소를 비우는데, 이 변경도 구독자에게 전달된다. 그대로 두면 디바운스 콜백이 빈 배열을 이전 프로젝트의 `review.json`에 써서 남아 있던 리뷰가 모두 지워진다.

무엇을 보는지: 전환 직전에 실제 데이터를 먼저 쓰고 저장을 멈추는 함수.
[src/lib/auto-save.ts#L30-L53](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/auto-save.ts#L30-L53)

```ts
export async function flushAndSuspendAutoSave(): Promise<void> {
  suspended = true
  clearTimers()
  const projectPath = useWikiStore.getState().project?.path
  if (!projectPath) return
  const review = useReviewStore.getState().items
  const lint = useLintStore.getState().items
  const chat = useChatStore.getState()
  await Promise.allSettled([
    saveReviewItems(projectPath, review),
    saveLintItems(projectPath, lint),
    saveChatPreferences(projectPath, { /* ... */ }),
    chat.isStreaming
      ? Promise.resolve()
      : saveChatHistory(projectPath, chat.conversations, chat.messages),
  ])
}
```

순서는 세 단계다. 먼저 `suspended = true`로 구독 콜백을 막고 대기 중인 타이머를 모두 취소한다. 그다음 지금 저장소에 있는 실제 데이터를 이전 프로젝트에 바로 쓴다. 마지막으로 새 프로젝트를 다 불러온 뒤 `resumeAutoSave()`로 다시 연다. `runWithSuspendedAutoSave`는 이 순서를 감싼 도우미다. 작업이 실패하면 `onFailure`로 반쯤 불러온 프로젝트 경로를 지운 다음에 저장을 재개한다([auto-save.ts#L64-L81](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/auto-save.ts#L64-L81)). 순서가 반대면 실패한 전환의 빈 상태가 엉뚱한 경로에 저장된다.

구독 콜백은 저장소가 바뀐 시점의 `projectPath`를 읽어 타이머에 넘긴다. 타이머가 터질 때 프로젝트가 바뀌었어도 변경이 일어난 프로젝트에 쓴다.

## 에이전트 파일 변경 추적: agent-file-activity

채팅 에이전트(11장)는 위키 파일을 직접 만들고 고칠 수 있다. 원문 llm-wiki.md는 "위키는 마크다운 파일로 된 git 저장소일 뿐이라 버전 기록이 저절로 생긴다"고 쓰는데([llm-wiki.md#L62](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/llm-wiki.md#L62-L62)), 앱은 git을 강제하지 않는다. 대신 에이전트 실행 한 번이 어떤 파일을 어떻게 바꿨는지 채팅 기록에 남기고, 바로 되돌릴 수 있게 한다.

`summarizeAgentFileChange`는 변경 전후 내용을 받아 화면용 diff를 만든다.

무엇을 보는지: 공통 접두부와 공통 접미부를 잘라 낸 단일 hunk 계산.
[src/lib/agent-file-activity.ts#L25-L46](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/agent-file-activity.ts#L25-L46)

```ts
const before = splitLines(input.beforeContent ?? "")
const after = splitLines(input.afterContent)
let prefix = 0
while (prefix < before.length && prefix < after.length && before[prefix] === after[prefix]) {
  prefix += 1
}
let suffix = 0
while (
  suffix < before.length - prefix
  && suffix < after.length - prefix
  && before[before.length - 1 - suffix] === after[after.length - 1 - suffix]
) {
  suffix += 1
}

const removed = before.slice(prefix, before.length - suffix)
const added = after.slice(prefix, after.length - suffix)
const diffLines = [
  `@@ -${prefix + 1},${removed.length} +${prefix + 1},${added.length} @@`,
```

LCS 기반 diff 대신 이 방법을 쓴 이유를 주석이 설명한다. 에이전트의 쓰기는 대부분 파일 전체 교체나 끝에 덧붙이기여서, 앞뒤 공통 부분만 잘라도 충분히 읽을 만한 diff가 나온다. LCS는 두 파일 길이의 곱만큼 메모리가 들지만 이 방법은 선형 시간이다([agent-file-activity.ts#L11-L16](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/agent-file-activity.ts#L11-L16)). 대신 파일 중간 두 곳을 고치면 그 사이 전체가 하나의 hunk로 보인다.

| 상수 / 필드 | 값 | 의미 |
| --- | --- | --- |
| `MAX_DIFF_LINES` | 240 | diff 표시 줄 수 상한 |
| `MAX_DIFF_CHARS` | 48,000 | diff 문자 수 상한 |
| `operation` | `beforeContent === null`이면 `created` | 새 파일과 수정 구분 |
| `beforeContent`, `afterContent` | 실행 중에만 보관 | 되돌리기용 스냅숏. `persist.ts`가 저장 전에 지운다 |

([agent-file-activity.ts#L3-L4](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/agent-file-activity.ts#L3-L4), [#L60-L63](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/agent-file-activity.ts#L60-L63))

스냅숏을 디스크에 저장하지 않으므로 앱을 다시 열면 변경 기록은 남지만 되돌리기 버튼은 비활성화된다. 채팅 기록에 파일 전문이 두 벌씩 쌓이는 것을 피한 선택이다([persist.ts#L66-L73](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/persist.ts#L66-L73)).

`mergeAgentFileChange`는 같은 파일을 여러 번 고친 경우 처음 상태(`beforeContent`, `operation`)와 마지막 상태(`afterContent`)를 합친다([agent-file-activity.ts#L67-L79](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/agent-file-activity.ts#L67-L79)). 되돌리면 실행 전 상태로 한 번에 돌아간다. `groupAgentFileChanges`는 파일별로 묶고 추가, 삭제 줄 수를 합산하되, 처음 본 순서를 유지해 실행 순서와 화면 순서가 같게 한다.

되돌리기는 안전 검사를 한 번 거친다. 현재 파일 내용이 에이전트가 쓴 직후 내용(`afterContent`)과 다르면 그 사이 사람이 고친 것으로 보고 되돌리지 않는다([agent-file-activity.tsx#L31-L52](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/components/chat/agent-file-activity.tsx#L31-L52)). 새로 만든 파일이면 지우고, 고친 파일이면 이전 내용을 다시 쓴다.

변경 전 내용을 확실히 모르는 경우에는 되돌리기를 아예 막는다. 셸 도구가 만든 파일은 쓰기 전 스냅숏을 얻을 수 없으므로 `beforeContent`를 `undefined`로 두고 diff 자리에 "스냅숏 없음" 안내를 넣는다([chat-panel.tsx#L873-L894](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/components/chat/chat-panel.tsx#L873-L894)). 주석은 관찰된 파일을 새 파일로 짐작하면 되돌리기가 사용자 데이터를 지울 수 있다고 경고한다. 같은 함수는 편집기의 파일 기록 패널에서도 과거 버전과 현재 내용을 비교하는 데 쓰인다([file-history-panel.tsx#L27-L29](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/components/editor/file-history-panel.tsx#L27-L29)).
