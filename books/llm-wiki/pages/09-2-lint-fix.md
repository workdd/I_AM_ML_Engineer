린트가 찾은 문제를 단건 또는 일괄로 고치는 경로를 본다.

## 린트 결과 수정

### 항목 하나 고치기: handleFix

린트 화면의 "고치기" 버튼은 유형별로 다른 수정을 한다.

무엇을 보는지: 깨진 링크 처리. 교정 후보가 있으면 대상을 바꾸고, 없으면 stub 페이지를 만든다.
[src/components/lint/lint-view.tsx#L276-L290](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/components/lint/lint-view.tsx#L276-L290)

```ts
case "broken-link": {
  const pagePath = `${pp}/wiki/${item.page}`
  if (item.brokenTarget && item.suggestedTarget) {
    const content = await readFile(pagePath)
    await writeFile(pagePath, rewriteWikilinkTarget(content, item.brokenTarget, item.suggestedTarget))
  } else if (item.brokenTarget) {
    const content = await readFile(pagePath)
    const stub = await ensureBrokenLinkStub(pp, item.brokenTarget)
    await writeFile(pagePath, rewriteWikilinkTarget(content, item.brokenTarget, stub.relativePath))
  } else {
    addLintItemToReview(item)
  }
  useLintStore.getState().removeItem(item.id)
  break
}
```

| 유형 | 추천 있음 | 추천 없음 |
| --- | --- | --- |
| `orphan` | `suggestedSource` 페이지에 이 페이지로 가는 링크 추가 | 리뷰 큐로 보냄 |
| `no-outlinks` | 이 페이지에 `suggestedTarget`으로 가는 링크 추가 | 리뷰 큐로 보냄 |
| `broken-link` | 위키링크 대상을 `suggestedTarget`으로 교체 | stub 페이지를 만들고 링크를 그쪽으로 교체 |
| `semantic` | 해당 없음 | 항상 리뷰 큐로 보냄 |

수정 함수 세 개는 `lint-fixes.ts`에 순수 문자열 변환으로 모여 있다.

- `appendWikilink`: 이미 같은 대상의 링크가 있으면 원문을 그대로 돌려준다. 없으면 `## Related` 제목 바로 아래에 `- [[target]]` 줄을 넣고, 제목이 없으면 문서 끝에 제목째 만든다([lint-fixes.ts#L23-L33](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/lint-fixes.ts#L23-L33)). 여러 번 눌러도 링크가 중복되지 않는다.
- `rewriteWikilinkTarget`: 대상만 바꾸고 `|alias` 부분은 그대로 둔다([lint-fixes.ts#L35-L49](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/lint-fixes.ts#L35-L49)). 화면에 보이는 링크 글자가 바뀌지 않는다.
- `ensureBrokenLinkStub`: 깨진 대상에 폴더가 있으면 그 경로를, 없으면 `queries/` 아래를 쓴다. 이미 파일이 있으면 만들지 않는다. 만든 stub에는 `tags: [stub, lint]`와 "Created by Wiki Lint as a placeholder" 문장이 들어간다([lint-fixes.ts#L69-L102](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/lint-fixes.ts#L69-L102)). 페이지 유형은 경로에서 추론하고, 추론하지 못하면 `query`가 된다. frontmatter 구조는 5장에서 다룬다.

### 다중 항목 일괄 수정

여러 결과가 같은 페이지를 고치려 할 때, 하나씩 읽고 쓰면 앞의 수정이 뒤의 수정에 덮여 사라질 수 있다. 일괄 수정은 파일별로 변환 함수를 모은 뒤 한 번 읽고, 변환을 차례로 적용하고, 한 번 쓴다.

무엇을 보는지: 파일 경로별 변환 큐와 reduce 적용.
[src/components/lint/lint-view.tsx#L435-L445](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/components/lint/lint-view.tsx#L435-L445)

```ts
// Multiple findings can target the same page. Apply their transforms in
// memory and write that page once, avoiding lost updates and N full reads.
for (const [path, pending] of edits) {
  const original = await readFile(path)
  const updated = pending.reduce((content, edit) => edit.apply(content), original)
  if (updated !== original) {
    await writeFile(path, updated)
    filesystemChanged = true
  }
  removeLintItems(pending.map((edit) => edit.id))
}
```

중간에 실패해도 `finally`에서 파일 트리 갱신을 한 번 돌린다. 앞에서 이미 쓴 파일이나 stub이 화면에 반영되게 하기 위해서다([lint-view.tsx#L450-L461](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/components/lint/lint-view.tsx#L450-L461)). `lint-fixes`의 변환 함수가 모두 같은 입력에 같은 결과를 내는 순수 함수라서 이렇게 합성할 수 있다.

### 고아 페이지 삭제

고아 페이지에는 "삭제" 버튼도 있다. 확인 대화상자를 거친 뒤 `cascadeDeleteWikiPagesWithRefs`로 지운다([lint-view.tsx#L326-L355](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/components/lint/lint-view.tsx#L326-L355)). 파일만 지우지 않고 임베딩 청크, 본문 위키링크, `index.md` 항목, frontmatter `related:` 배열까지 함께 정리한다. 고아 판정이 본문 링크만 보기 때문에 `related:`나 `index.md`에는 이 페이지를 가리키는 참조가 남아 있을 수 있어서다. 연쇄 삭제의 세부는 5장의 원본 수명주기 절에서 다룬다.

### lint-store

`useLintStore`는 결과를 `LintItem`으로 감싸 `lint-N` 형태의 id를 붙인다([lint-store.ts#L17-L30](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/stores/lint-store.ts#L17-L30)). 디스크에서 불러올 때(`setItems`)는 저장된 id 중 가장 큰 번호로 카운터를 맞춘다([lint-store.ts#L32-L41](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/stores/lint-store.ts#L32-L41)). 그러지 않으면 재시작 후 새 결과가 이전 결과와 같은 id를 받아 선택이나 삭제가 엉뚱한 항목에 적용된다. 린트를 다시 돌리면 `clearItems()`로 이전 결과를 모두 지우고 새로 채우므로, 린트 결과는 리뷰와 달리 "현재 스냅숏" 성격이다.
