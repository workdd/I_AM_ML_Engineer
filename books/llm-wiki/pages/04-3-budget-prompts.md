원본을 모델 문맥 예산에 맞춰 나눈 뒤, 분석과 생성 두 번의 LLM 호출로 위키 페이지를 만든다.

## 예산 계산과 긴 원본 분할

설정의 `maxContextSize` 는 토큰이 아니라 문자 수다([context-budget.ts#L4-L7](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/context-budget.ts#L4-L7)). 값이 없으면 204,800 이고 응답 예약분은 15% 다.

코드 분석: 원본 본문에 쓸 수 있는 문자 수를 정하는 함수다. [src/lib/ingest.ts#L2844-L2854](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/ingest.ts#L2844-L2854)

```ts
export function computeIngestSourceBudget(
  maxContextSize: number | undefined,
  stableContextLength: number,
): number {
  const { maxCtx, responseReserve } = computeContextBudget(maxContextSize)
  const stableReserve = Math.min(Math.floor(maxCtx * 0.25), Math.max(12_000, stableContextLength))
  const instructionReserve = Math.max(12_000, Math.floor(maxCtx * 0.08))
  const available = maxCtx - responseReserve - stableReserve - instructionReserve
  const upper = Math.min(LONG_SOURCE_MAX_SINGLE_PASS_BUDGET, Math.max(LONG_SOURCE_MIN_BUDGET, Math.floor(maxCtx * 0.6)))
  return clampNumber(Math.floor(available), LONG_SOURCE_MIN_BUDGET, upper)
}
```

`stableContextLength` 는 `schema.md`, `purpose.md`, `wiki/index.md`, `wiki/overview.md` 길이의 합이다. 기본값 204,800 에 새 프로젝트(안정 문맥 12,000자 미만)를 넣어 코드대로 계산하면 다음과 같다. 아래 수치는 코드 상수로 계산한 값이며 실측이 아니다.

$$
\text{available} = 204{,}800 - 30{,}720 - 12{,}000 - 16{,}384 = 145{,}696
$$

상한이 $\min(300{,}000,\ 0.6 \times 204{,}800) = 122{,}880$ 이므로 원본 예산은 122,880자다. 이보다 긴 원본은 청크 분석으로 넘어간다. 같은 조건에서 출력 토큰 상한은 아래처럼 정해진다([#L2856-L2878](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/ingest.ts#L2856-L2878)).

| 호출 | 상한 규칙 | 기본 문맥에서의 값 |
|---|---|---|
| 생성 | 문맥 128K 미만 8,192 · 128K 이상 16,384 · 256K 이상 24,576 · 512K 이상 32,768 | 16,384 |
| 분석 | 응답 예약분 ÷ 2 를 4,096 ~ 생성 상한으로 고정, 추론 예산 별도 가산 | 15,360 + 추론 |
| 리뷰 제안 | 생성 상한 ÷ 2 를 4,096 ~ 16,384 로 고정 | 8,192 |
| 잘림 복구 | 생성 상한과 같음 | 16,384 |

### 의미 단위 청크와 전역 다이제스트

원본이 예산을 넘으면 `analyzeLongSourceInChunks` 가 원본을 청크로 나눠 하나씩 분석한다. 청크 목표 크기는 예산의 55% 를 12,000 ~ 60,000 자로 고정한 값이고, 겹침은 목표의 8% 를 800 ~ 3,000 자로 고정한 값이다([ingest.ts#L3254-L3255](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/ingest.ts#L3254-L3255)). 위 예시에서는 목표 60,000자, 겹침 3,000자다.

`splitSourceIntoSemanticChunks` 는 Markdown 머리글을 스택으로 따라가며 `## A > ### B` 같은 heading path 를 기록하고, 빈 줄 단위 문단을 쌓다가 목표를 넘으면 자른다. 문단 하나가 목표의 1.25배를 넘으면 문장 부호(`.!?。！？`) 단위로 다시 쪼갠다([#L2880-L3002](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/ingest.ts#L2880-L3002)). 앞 청크 끝부분은 문단이나 문장 경계에 맞춰 `overlapBefore` 로 붙는다.

각 청크 호출은 두 절을 돌려줘야 한다. `## Chunk Analysis` 는 이 청크만의 분석이고, `## Updated Global Digest` 는 이전 다이제스트에 이 청크를 반영한 문서 전체 요약이다. 다음 청크는 갱신된 다이제스트(최대 15,000자)를 받아 이어 간다. 마지막에 다이제스트와 청크별 분석을 묶어 1단계 분석 결과로 쓰고, 2단계에는 원본 대신 이 묶음이 "Long Source Context" 로 들어간다([#L3354-L3377](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/ingest.ts#L3354-L3377)).

청크 하나가 끝날 때마다 `.llm-wiki/ingest-progress/<slug>-<해시>.json` 에 체크포인트를 쓴다. 원본 해시, 길이, 예산, 목표 크기, 겹침, 청크 수가 모두 같을 때만 이어서 시작한다([#L3110-L3134](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/ingest.ts#L3110-L3134)). 해시는 64비트 FNV-1a 이며 주석에 보안용이 아니라 안정성 키라고 적혀 있다.

청크 분석 프롬프트는 엔터티 판단을 미룬다. 한 청크만 보고 독립 페이지를 만들지 정하지 말고, 다이제스트에 등장 청크와 근거를 누적해 `Page-worthy: yes/no` 를 매기라고 지시한다([#L2424-L2428](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/ingest.ts#L2424-L2428)).

> **주의**: `src/lib/text-chunker.ts` 도 청크를 만들지만 용도가 다르다. 이쪽은 임베딩용 청커로 기본 목표 1,000자, 최대 1,500자, 최소 200자, 겹침 200자이며 코드 블록과 표를 자르지 않는다([text-chunker.ts#L66-L71](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/text-chunker.ts#L66-L71)). 수집 단계에서는 쓰이지 않고 `embedding.ts` 가 가져다 쓴다. 자세한 내용은 6장에서 다룬다.

| 구분 | 수집용 `splitSourceIntoSemanticChunks` | 임베딩용 `chunkMarkdown` |
|---|---|---|
| 목적 | LLM 분석 호출 하나에 들어갈 크기 | 벡터 하나에 담길 크기 |
| 크기 | 12,000 ~ 60,000자 | 목표 1,000자 |
| 경계 | 머리글 · 문단 · 문장 | 머리글 · 문단 · 줄 · 문장 · 공백 · 강제 자르기 |
| 코드 블록 · 표 보호 | 없음 | 있음 |
| 위치 | `ingest.ts` | `text-chunker.ts` (Rust 쪽 `page_embedding.rs` 와 동작 일치) |

## 1단계: 분석 프롬프트

분석 호출은 `temperature: 0.1` 로 원본을 읽고 구조화된 분석을 낸다. 시스템 프롬프트는 `buildAnalysisPrompt` 가 만들고, 사용자 메시지에 원본 식별자 · 폴더 문맥 · 본문이 들어간다.

코드 분석: 분석 출력의 분량 상한이다. [src/lib/ingest.ts#L2408-L2415](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/ingest.ts#L2408-L2415)

```ts
const ANALYSIS_OUTPUT_LIMITS = [
  "HARD OUTPUT LIMITS: a response exceeding these limits is invalid.",
  "- Key Entities: at most 25 candidates, one compact line per candidate, no table.",
  "- Key Concepts: at most 25 entries.",
  "- Main Arguments & Findings: at most 15 entries.",
  "- Connections, Contradictions, and Recommendations: at most 8 entries per section.",
  "- Total response: at most 16,000 characters. Prefer high-value reusable knowledge over exhaustive enumeration.",
] as const
```

프롬프트 본문은 여섯 절로 나뉜다([#L2434-L2496](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/ingest.ts#L2434-L2496)). 각 절이 무엇을 요구하는지 정리하면 다음과 같다.

| 절 | 요구 내용 |
|---|---|
| Key Entities | 사람 · 조직 · 제품 · 데이터셋 · 도구 후보, 위키에 이미 있는지, 후보마다 `Page-worthy: yes/no` 와 근거 한 줄 |
| Key Concepts | 이론 · 방법 · 현상, 짧은 정의, 위키에 이미 있는지 |
| Main Arguments & Findings | 핵심 주장과 근거의 강도, 주장이 어느 대상에 대한 것인지 구분, SQL DDL · API 시그니처 · 표는 원문 그대로 보존 |
| Connections to Existing Wiki | 기존 페이지 중 관련된 것, 강화 · 반박 · 확장 여부 |
| Contradictions & Tensions | 기존 위키와의 충돌, 원본 내부의 긴장 |
| Recommendations | 만들거나 갱신할 페이지, schema 에 정의된 사용자 정의 타입(goal, habit 등)이 실제로 있을 때만 추천 |

이 표가 llm-wiki.md 의 "discusses key takeaways" 를 대신한다. 사람과의 대화 대신 LLM 스스로 쓴 분석을 2단계의 문맥으로 쓴다. 엔터티 규칙에는 "지나가는 언급, 이름 없는 인물, 장 · 부 제목은 독립 페이지로 만들지 말라", "기존 위키 페이지 제목을 먼저 재사용하고, 이름을 음역하거나 지어내지 말라" 가 들어 있다([#L2401-L2406](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/ingest.ts#L2401-L2406)). 끝에 schema, purpose, index 가 붙는데 index 는 `boundWikiIndexForPrompt` 가 32,000자로 자른다. 앞 60% 와 뒤를 남기고 가운데를 생략 표시로 바꾼다([#L3009-L3024](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/ingest.ts#L3009-L3024)).

분석 응답이 출력 상한에 걸려 잘리면 생성 단계로 넘어가지 않고 예외를 던진다([#L1094-L1100](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/ingest.ts#L1094-L1100)). 불완전한 분석으로 페이지를 만들면 빠진 엔터티가 조용히 사라지기 때문이다.

## 2단계: 생성 프롬프트

생성 호출이 실제 위키 파일을 만든다. 출력 형식은 `---FILE: 경로---` 와 `---END FILE---` 로 감싼 블록의 연속이고, 그 뒤에 선택적으로 `---REVIEW: 종류 | 제목---` 블록이 온다.

코드 분석: 무엇을 만들지 지시하는 절이다. [src/lib/ingest.ts#L2535-L2544](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/ingest.ts#L2535-L2544)

```ts
    "## What to generate",
    "",
    `1. A source summary page at **${summaryPath}** (MUST use this exact path)`,
    "Emit exactly one source summary FILE block. Do not create additional wiki/sources/ pages for works merely cited by the source; keep those citations in the summary or link to existing pages.",
    "2. Entity or schema-defined typed pages for key named things identified in the analysis. Prefer schema-defined directories when present; otherwise use wiki/entities/.",
    "3. Concept or schema-defined typed pages for key ideas, methods, techniques, and abstractions. Prefer schema-defined directories when present; otherwise use wiki/concepts/.",
    "4. A log entry for wiki/log.md (just the new entry to append, format: ## [YYYY-MM-DD] ingest | Title)",
    ...ENTITY_PAGE_RULES,
    "For consolidated long-document analyses, the Final Global Digest is authoritative for `Page-worthy` decisions; per-chunk mention notes are evidence, not page-creation instructions.",
    "Do not generate wiki/index.md or wiki/overview.md. The application maintains aggregate navigation separately so large wikis are never rewritten through model output.",
```

요약 · 개체 · 개념 · log 의 네 종류가 llm-wiki.md 의 ingest 묘사와 정확히 맞는다. 요약 페이지 경로는 LLM 이 정하지 않고 코드가 원본 식별자에서 계산해 넘긴다(5장). 마지막 줄은 index 와 overview 를 LLM 이 쓰지 못하게 한다. 위키가 커지면 index 전체를 모델 출력으로 다시 쓰는 일이 비싸고 위험하기 때문이다.

프롬프트의 나머지 구성은 다음 순서다([#L2498-L2671](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/ingest.ts#L2498-L2671)).

1. 역할과 언어 규칙
2. 원본 파일 이름, 오늘 날짜(모든 `created` · `updated` 에 쓰라는 지시)
3. schema 가 있으면 "AUTHORITATIVE" 로 표시한 라우팅 규칙. 사용자 정의 폴더를 `wiki/entities/` 보다 우선
4. 위의 "What to generate"
5. frontmatter 규칙과 예시 페이지(5장에서 자세히 다룬다)
6. 파일 이름 규칙: 라틴 문자는 kebab-case, 한중일 제목은 로마자로 바꾸지 말 것, OpenAI · GPT-5 · PyTorch 같은 고유명은 원형 유지
7. 대상 경계 규칙: 한 원본이 여러 모델 · 제품을 다룰 때 평가 · 한계를 정확한 대상에 붙이고, 같은 용어를 공유한다는 이유로 섞지 말 것
8. REVIEW 블록 종류(`contradiction` · `duplicate` · `missing-page` · `suggestion`)와 `SEARCH:` 줄에 검색어 2~3개
9. purpose, index, overview
10. 출력 형식과 엄격 요구 사항
11. 언어 규칙 반복

출력 형식 절 앞에는 "OUTPUT FORMAT MUST BE THE LAST SECTION, models weight recent instructions highest" 라는 주석이 있고, 그 뒤에 언어 규칙을 한 번 더 둔다. 주석에 따르면 중소형 모델이 개별 페이지에서 학습 데이터의 언어로 돌아가는 일을 막으려는 것이다([#L2662-L2670](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/ingest.ts#L2662-L2670)).

> **주의**: 생성 프롬프트에는 여전히 `## Current Overview (update this to reflect the new source)` 절이 들어간다([#L2628](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/ingest.ts#L2628)). 같은 프롬프트가 overview 를 생성하지 말라고 하므로, 이 절은 문맥 제공 이상의 의미가 없다. README 의 "Updated index.md, log.md, overview.md" 설명도 이 커밋의 동작과 다르다.

사용자 메시지는 1단계 분석을 "context only, do not repeat" 로 감싸 넣고, 첫 글자를 `---FILE:` 로 시작하라고 다시 못 박는다. 이전 시도에서 이미 쓴 파일이 있으면 "Resume checkpoint" 절에 그 경로를 나열하고, 모두 끝났으면 `---RESUME COMPLETE---` 만 돌려달라고 한다([#L1136-L1170](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/ingest.ts#L1136-L1170)).

### 리뷰 제안 전용 호출

생성 출력이 10,000자 이상이거나 FILE 블록이 4개 이상이거나 REVIEW 블록이 시작된 흔적이 있으면, 리뷰 블록만 따로 뽑는 호출을 한 번 더 한다([#L2395-L2399](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/ingest.ts#L2395-L2399)). 출력이 큰 원본일수록 생성 호출이 FILE 블록에 토큰을 다 쓰고 REVIEW 를 빠뜨리기 쉽기 때문이다(추정). 이 호출은 "1~5개의 가치 있는 리뷰만, 없으면 아무것도 출력하지 말 것" 을 요구한다. 리뷰 항목이 어떻게 쓰이는지는 9장에서 다룬다.
