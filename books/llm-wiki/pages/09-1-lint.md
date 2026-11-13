린트는 코드만으로 판정하는 구조 린트와 LLM이 판단하는 의미 린트로 나뉜다.

## 린트 결과 형식

모든 린트 결과는 `LintResult` 하나의 타입으로 표현된다.

무엇을 보는지: 결과 타입과 선택 필드.
[src/lib/lint.ts#L16-L25](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/lint.ts#L16-L25)

```ts
export interface LintResult {
  type: "orphan" | "broken-link" | "no-outlinks" | "semantic"
  severity: "warning" | "info"
  page: string
  detail: string
  affectedPages?: string[]
  brokenTarget?: string
  suggestedTarget?: string
  suggestedSource?: string
}
```

선택 필드 세 개가 자동 수정의 재료다. 어떤 필드가 채워지는지는 유형마다 다르다.

| 유형 | 심각도 | `page` 의미 | 자동 수정 재료 |
| --- | --- | --- | --- |
| `orphan` | info | 고아 페이지의 wiki 상대 경로 | `suggestedSource`: 이 페이지로 링크를 걸 만한 다른 페이지 |
| `no-outlinks` | info | 나가는 링크가 없는 페이지 | `suggestedTarget`: 이 페이지가 링크할 만한 다른 페이지 |
| `broken-link` | warning | 깨진 링크가 들어 있는 페이지 | `brokenTarget`: 깨진 링크 대상, `suggestedTarget`: 철자가 비슷한 실제 페이지 |
| `semantic` | LLM이 정한 값 | LLM이 붙인 짧은 제목 | 없음. `affectedPages`만 채워진다 |

`semantic` 결과는 자동 수정 재료가 없어서 "고치기"를 누르면 항상 리뷰 큐로 간다. 코드가 확정할 수 없는 일은 사람에게 넘긴다는 원칙이 타입 수준에서 드러난다.

## 구조 린트

### 페이지 수집: runStructuralLint

구조 린트는 `wiki/` 아래의 모든 `.md`를 읽어 페이지마다 경로, 제목, 나가는 위키링크, 토큰 집합을 만든다. `index.md`와 `log.md`는 처음부터 제외한다. 두 파일은 모든 페이지를 나열하거나 기록하는 용도라서 고아 판정과 나가는 링크 판정에 넣으면 결과가 왜곡된다.

무엇을 보는지: 페이지 하나를 `StructuralLintPage`로 바꾸는 루프.
[src/lib/lint.ts#L169-L193](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/lint.ts#L169-L193)

```ts
const wikiFiles = flattenMdFiles(tree)
// Exclude index.md and log.md from orphan checks
const contentFiles = wikiFiles.filter(
  (f) => f.name !== "index.md" && f.name !== "log.md"
)
const pages: StructuralLintPage[] = []
for (let index = 0; index < contentFiles.length; index += 1) {
  if (options.signal?.aborted) throw new DOMException("Structural lint cancelled", "AbortError")
  const f = contentFiles[index]
  try {
    const content = await readFile(f.path)
    const shortName = getRelativePath(f.path, wikiRoot)
    const slug = relativeToSlug(shortName)
    const title = extractTitle(content, shortName)
    const outlinks = extractWikilinks(content)
    const slugName = getFileName(slug)
    const tokens = Array.from(tokenizeForSuggestion(`${title}\n${slugName}\n${content.slice(0, SUGGESTION_TOKEN_WINDOW)}`))
    pages.push({ shortName, slug, title, outlinks, tokens })
```

- 위키링크 추출 정규식은 `[[target|alias]]`에서 `target`만 꺼낸다([lint.ts#L43-L51](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/lint.ts#L43-L51)). 별칭은 판정에 쓰지 않는다.
- 제목은 frontmatter의 `title:`, 첫 `#` 제목, 파일명 순으로 찾는다([lint.ts#L82-L93](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/lint.ts#L82-L93)).
- 토큰은 제목, 파일명, 본문 앞 `SUGGESTION_TOKEN_WINDOW = 4000`자에서만 뽑는다([lint.ts#L27](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/lint.ts#L27-L27)). 토큰화는 NFKC 정규화 후 Unicode 글자와 숫자 연속을 단어로 보고, 한자 범위(`㐀-鿿`)는 글자 하나씩도 토큰으로 넣는다([lint.ts#L95-L106](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/lint.ts#L95-L106)). 띄어쓰기가 없는 중국어 본문을 위한 처리다. 한글은 이 범위에 들어가지 않으므로 어절 단위 토큰만 만들어진다.

진행률은 파일 읽기에 절반, 계산에 절반을 배정한다. 읽기 루프는 `index + 1`을 `contentFiles.length * 2`에 대해 보고하고, 워커 진행률은 그 뒤에 이어 붙인다.

### Web Worker 기반 계산

페이지가 수천 개가 되면 후보 추천 계산이 UI 스레드를 막는다. 그래서 계산 함수 `computeStructuralLint`는 순수 함수로 분리되어 있고([lint-structural-core.ts](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/lint-structural-core.ts#L102-L224)), 브라우저 환경에서는 모듈 워커로 돌린다.

워커가 없는 환경(테스트)에서는 같은 함수를 직접 부르는 분기가 있다([lint.ts#L116-L124](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/lint.ts#L116-L124)). 워커 쪽 코드는 17줄짜리 어댑터다([lint-structural.worker.ts](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/lint-structural.worker.ts#L1-L17)). `progress`와 `done` 두 종류 메시지만 보낸다. 취소는 `AbortSignal`이 오면 `worker.terminate()`로 즉시 끊고 `AbortError`를 던진다([lint.ts#L125-L133](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/lint.ts#L125-L133)). 계산이 순수 함수라서 테스트는 워커 없이 같은 함수를 검증할 수 있다. `lint-structural-core.test.ts`에는 5,000페이지에서 후보 확장이 이차로 늘지 않는지 보는 테스트가 있다([lint-structural-core.test.ts#L24-L35](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/lint-structural-core.test.ts#L24-L35)).

### 세 가지 판정: computeStructuralLint

계산은 세 개의 색인을 먼저 만든다.

| 색인 | 키 | 값 | 용도 |
| --- | --- | --- | --- |
| `slugMap` | 정규화한 slug, 정규화한 파일명 | 페이지 번호 | 위키링크가 실제 페이지를 가리키는지 확인 |
| `tokenIndex` | 토큰 | 그 토큰을 가진 페이지 번호 목록 | 고아 / 나가는 링크 없음의 관련 페이지 추천 |
| `fragmentIndex` | 2글자 조각(bigram) | 페이지 번호 목록 | 깨진 링크의 철자 교정 후보 |

정규화는 역슬래시를 슬래시로 바꾸고, 앞의 `wiki/`와 뒤의 `.md`를 떼고, 소문자로 만든다([lint-structural-core.ts#L41-L47](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/lint-structural-core.ts#L41-L47)). `slugMap`에 전체 slug와 파일명을 둘 다 넣기 때문에 `[[entities/foo]]`와 `[[foo]]`가 모두 같은 페이지로 연결된다.

무엇을 보는지: 들어오는 링크 수를 세는 부분과 세 판정.
[src/lib/lint-structural-core.ts#L122-L129](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/lint-structural-core.ts#L122-L129), [#L189-L218](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/lint-structural-core.ts#L189-L218)

```ts
const inboundCounts = new Map<number, number>()
for (const page of pages) {
  for (const link of page.outlinks) {
    const target = slugMap.get(normalizeTarget(link))
      ?? slugMap.get(normalizeTarget(fileName(link).replace(/\.md$/i, "")))
    if (target !== undefined) inboundCounts.set(target, (inboundCounts.get(target) ?? 0) + 1)
  }
}
// ...
if (!ignored && !config.ignoreOrphan && !inboundCounts.has(pageIndex)) {
  results.push({ type: "orphan", severity: "info", page: page.shortName,
    detail: "No other pages link to this page.",
    suggestedSource: relatedCandidate(pageIndex, "source")?.shortName })
}
```

- 고아 판정은 본문 위키링크만 본다. 자기 자신을 가리키는 링크도 들어오는 링크로 세므로, 자기 링크만 있는 페이지는 고아로 잡히지 않는다(코드상 자기 참조를 제외하는 분기가 없다).
- `index.md`는 페이지 목록에서 빠져 있으므로, `index.md`에만 등재된 페이지도 고아로 잡힌다. 고아 삭제 버튼의 주석이 이 점을 직접 언급한다([lint-view.tsx#L337-L342](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/components/lint/lint-view.tsx#L337-L342)).
- 깨진 링크는 `slugMap`에 전체 경로도 파일명도 없을 때만 보고하고, 심각도가 `warning`인 유일한 구조 결함이다([lint-structural-core.ts#L207-L218](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/lint-structural-core.ts#L207-L218)).

### 추천 후보 계산

린트가 결함만 알리고 끝나면 사람이 하나씩 링크를 달아야 한다. 이 구현은 결함마다 "어디에 연결하면 되는지"를 함께 계산해서 버튼 하나로 고칠 수 있게 한다.

관련 페이지 추천(`relatedCandidate`)은 토큰 겹침을 두 집합 크기의 기하평균으로 나눈 값에 같은 폴더 가산점을 더한다. 페이지 $p$와 후보 $c$의 토큰 집합을 $T_p$, $T_c$라고 하면 다음과 같다.

$$
\text{score}(p, c) = \frac{\sum_{t \in T_p \cap T_c} w(t)}{\sqrt{\max(1, |T_p|) \cdot \max(1, |T_c|)}} + b_{\text{folder}}
$$

$w(t)$는 두 글자 이상 토큰이면 1, 한 글자 한자 토큰이면 0.35다. $b_{\text{folder}}$는 두 페이지의 최상위 폴더가 같을 때 0.08이다. 문서 전체의 25%(최소 20개)보다 많은 페이지에 나오는 토큰은 점수 계산에서 아예 빠진다. 흔한 단어가 후보를 늘려 계산이 이차로 커지는 것을 막기 위해서다([lint-structural-core.ts#L134-L141](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/lint-structural-core.ts#L134-L141)).

깨진 링크의 교정 후보(`brokenCandidate`)는 다른 방식이다. 2글자 조각 색인으로 후보를 모은 뒤 문자열 유사도를 계산한다([lint-structural-core.ts#L66-L77](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/lint-structural-core.ts#L66-L77)). 정확히 같으면 1, 파일명이 같으면 0.96, 한쪽이 다른 쪽을 포함하면 0.82, 그 밖에는 Levenshtein 거리로 $1 - d/\max(|a|,|b|)$를 쓴다. 파일명이 5글자 미만이면 편집 거리 비교를 하지 않고 0으로 둔다. 짧은 이름끼리는 한두 글자 차이로도 엉뚱한 페이지가 걸리기 때문이다.

상수는 모두 파일 맨 위에 모여 있다([lint-structural-core.ts#L29-L35](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/lint-structural-core.ts#L29-L35)).

| 상수 | 값 | 의미 |
| --- | --- | --- |
| `BROKEN_LINK_SUGGESTION_MIN_SCORE` | 0.74 | 깨진 링크 교정 후보로 인정하는 최소 유사도 |
| `RELATED_PAGE_SUGGESTION_MIN_SCORE` | 0.08 | 관련 페이지 추천의 최소 점수 |
| `SAME_FOLDER_SCORE_BONUS` | 0.08 | 같은 최상위 폴더 가산점 |
| `SINGLE_CJK_TOKEN_WEIGHT` | 0.35 | 한 글자 한자 토큰 가중치 |
| `SAME_BASENAME_SCORE` | 0.96 | 경로는 다르고 파일명이 같을 때 유사도 |
| `CONTAINS_TARGET_SCORE` | 0.82 | 한쪽이 다른 쪽을 포함할 때 유사도 |
| `MAX_SUGGESTION_CANDIDATES` | 64 | 점수 계산까지 가는 후보 수 상한 |

`no-outlinks`의 추천 대상을 고를 때는 이미 링크한 페이지를 건너뛴다(`direction === "target"` 분기). 링크가 0개인 페이지에서는 실질적인 의미가 없지만, 같은 함수를 공유하기 위한 처리다.

### 무시 규칙: lint-config

위키에 일부러 고립시킨 페이지(예: 초안, 보관 페이지)가 있으면 매번 같은 경고가 뜬다. 린트 화면 좌측 톱니 버튼으로 규칙을 바꿀 수 있고, 프로젝트별로 `.llm-wiki/lint-config.json`에 저장된다([lint-config.ts#L27-L49](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/lint-config.ts#L27-L49)).

| 필드 | 기본값 | 효과 |
| --- | --- | --- |
| `ignoreOrphan` | `false` | 고아 판정 전체를 끈다 |
| `ignoreNoOutlinks` | `false` | 나가는 링크 없음 판정 전체를 끈다 |
| `ignorePages` | `[]` | 목록의 페이지는 세 판정을 모두 건너뛴다. 쉼표, 전각 쉼표, 줄바꿈으로 구분한다 |

무시된 페이지도 다른 페이지의 링크 대상으로는 계속 인정된다. `slugMap`은 무시 여부와 상관없이 모든 페이지로 만들어지기 때문이다. 또 무시 목록은 slug와 경로로만 비교하고 파일명만으로는 비교하지 않는다. 주석이 설명하듯 최상위 `foo`를 무시해도 `archive/foo`는 숨겨지지 않는다([lint-structural-core.ts#L184-L188](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/lint-structural-core.ts#L184-L188)).

## 의미 린트

의미 린트는 위키 전체를 한 번의 LLM 호출로 점검한다. 페이지마다 앞 500자(frontmatter 포함)만 잘라 `### 경로` 아래에 붙인 요약 묶음을 만들고, 이것을 프롬프트에 넣는다([lint.ts#L233-L252](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/lint.ts#L233-L252)).

무엇을 보는지: LLM에게 주는 출력 형식과 유형 정의.
[src/lib/lint.ts#L270-L288](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/lint.ts#L270-L288)

```ts
"For each issue, output exactly this format:",
"",
"---LINT: type | severity | Short title---",
"Description of the issue.",
"PAGES: page1.md, page2.md",
"---END LINT---",
"",
"Types:",
"- contradiction: two or more pages make conflicting claims",
"- stale: information that appears outdated or superseded",
"- missing-page: an important concept is heavily referenced but has no dedicated page",
"- suggestion: a question or source worth adding to the wiki",
"For missing-page findings, Short title must be only the exact missing concept or entity name, without explanatory prefixes or suffixes.",
```

해설:

- 출력은 JSON이 아니라 구분자 블록이다. 4장의 수집 파이프라인이 쓰는 `---FILE: ...---` 블록, 리뷰의 `---REVIEW: ...---` 블록과 같은 방식이고, 정규식 `LINT_BLOCK_REGEX` 하나로 파싱한다([lint.ts#L202-L203](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/lint.ts#L202-L203)). 블록 형식을 지키지 않은 출력은 조용히 버려진다.
- 원문의 여섯 항목 가운데 LLM 몫인 모순, 오래된 주장, 빠진 개념, 조사할 질문을 그대로 네 유형으로 옮겼다.
- 언어 지시문(`buildLanguageDirective`)은 요약 앞 2,000자를 표본으로 써서 위키의 언어에 맞춘다. 한국어 위키면 결과도 한국어로 나오도록 하는 장치다.

### missing-page 오탐 필터링

LLM은 페이지 목록을 정확히 대조하지 못해서 이미 있는 개념을 "페이지 없음"으로 보고하는 일이 잦다. 코드는 파일명과 frontmatter 제목을 모두 정규화해 집합으로 만든 뒤, `missing-page` 결과의 제목이 이 집합에 정확히 있으면 버린다([lint.ts#L327-L332](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/lint.ts#L327-L332)).

정규화는 리뷰 제목 정규화(`normalizeReviewTitle`)에 NFKC와 소문자를 더한 것이다([lint.ts#L62-L64](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/lint.ts#L62-L64)). 부분 문자열 비교는 일부러 쓰지 않는다. 주석에 따르면 짧은 페이지 제목이 다른 문장 속 평범한 단어와 겹쳐 진짜 빠진 페이지까지 지워지기 때문이다([lint.ts#L66-L80](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/lint.ts#L66-L80)). 프롬프트에서 "제목에는 개념 이름만 쓰라"고 요구하는 것도 이 정확 일치 필터가 통하게 하기 위해서다.

> **주의**: 의미 린트는 페이지당 500자만 본다. 모순되는 주장이 페이지 뒷부분에 있으면 LLM이 볼 수 없고, 페이지가 많으면 프롬프트 전체가 모델의 컨텍스트 한도를 넘을 수 있다. 코드에는 페이지 수나 전체 길이 상한이 없다(미검증: 실제 대형 위키에서 어떻게 실패하는지는 확인하지 않았다).

결과는 모두 `type: "semantic"`으로 저장되고, 원래 유형은 `detail` 앞에 `[contradiction]`처럼 붙는다([lint.ts#L341-L347](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/lint.ts#L341-L347)). 화면에서는 네 유형이 하나의 아이콘으로 묶인다.
