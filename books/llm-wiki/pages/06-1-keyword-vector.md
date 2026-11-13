검색은 가중 키워드 점수와 선택적인 벡터 검색 두 순위로 구성된다.

## 진입점: searchWiki와 search_project

프론트엔드 검색 패널은 `searchWiki`를 부르고, 이 함수는 Tauri command `search_project`를 그대로 호출한다. 질의 임베딩은 프론트에서 만들지 않고 `queryEmbedding: null`과 임베딩 설정만 넘긴다.

코드 분석: 프론트 진입점이 백엔드에 넘기는 인자를 본다. [src/lib/search.ts#L62-L83](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/search.ts#L62-L83)

```ts
  const response = await invoke<BackendSearchResponse>("search_project", {
    projectPath: pp,
    query,
    topK: 20,
    includeContent: false,
    queryEmbedding: null,
    embeddingConfig: embCfg,
  })
```

`topK`는 20으로 고정이고, 본문 전체(`includeContent`)는 받지 않는다. 백엔드 쪽 `search_project`는 `resolve_query_embedding`으로 질의 벡터를 먼저 얻은 다음 `search_project_inner`에 넘긴다([search.rs#L107-L130](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/commands/search.rs#L107-L130)). 질의 벡터를 프론트에서 만들지 않는 이유는 `BackendSearchResponse` 주석이 밝힌다. API와 WebView 검색이 "the same retrieval contract"를 공유하도록 하기 위해서다([search.ts#L20-L28](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/search.ts#L20-L28)).

검색 동작을 정하는 상수는 파일 맨 위에 모여 있다([search.rs#L12-L26](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/commands/search.rs#L12-L26)).

| 상수 | 값 | 쓰이는 곳 |
| --- | --- | --- |
| `DEFAULT_RESULTS` | 20 | `top_k` 미지정 시 기본 결과 수 |
| `MAX_RESULTS` | 50 | `top_k`를 1..50으로 clamp |
| `RRF_K` | 60.0 | RRF 분모의 상수 |
| `FILENAME_EXACT_BONUS` | 200.0 | 파일명 stem이 질의 구문과 같을 때 |
| `PHRASE_IN_TITLE_BONUS` | 50.0 | 제목에 질의 구문 전체가 있을 때 |
| `PHRASE_IN_CONTENT_PER_OCC` | 20.0 | 본문에 질의 구문이 나온 횟수당 |
| `MAX_PHRASE_OCC_COUNTED` | 10 | 본문 구문 횟수 상한 |
| `TITLE_TOKEN_WEIGHT` | 5.0 | 제목에 들어 있는 토큰 하나당 |
| `CONTENT_TOKEN_WEIGHT` | 1.0 | 본문에 들어 있는 토큰 하나당 |
| `SEARCH_EMBEDDING_TIMEOUT_SECS` | 8 | 임베딩 HTTP 요청 시간 제한 |
| `MAX_SEARCH_FILES` | 10,000 | 한 번에 읽는 markdown 파일 상한 |
| `MIN_GRAPH_RESULT_RATIO` / `MAX_GRAPH_RESULT_RATIO` | 0.15 / 0.30 | 그래프 이웃에 배정하는 자리 비율 |
| `MAX_GRAPH_SEEDS` | 20 | 그래프 확장의 시작점 수 상한 |

## 키워드 점수: BM25가 아닌 가중 포함 개수

`search_project_inner`는 인덱스를 미리 만들어 두지 않는다. 질의가 올 때마다 `wiki/` 아래 `.md` 파일을 `WalkDir`로 전부 읽고, 파일 하나마다 `score_file`을 부른다([search.rs#L349-L413](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/commands/search.rs#L349-L413)). 같은 순회에서 그래프 확장에 쓸 링크 목록도 함께 모은다.

코드 분석: 한 파일의 키워드 점수를 어떻게 합산하는지 본다. [search.rs#L960-L970](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/commands/search.rs#L960-L970)

```rust
    let score = (if filename_exact {
        FILENAME_EXACT_BONUS
    } else {
        0.0
    }) + (if title_has_phrase {
        PHRASE_IN_TITLE_BONUS
    } else {
        0.0
    }) + content_phrase_occ as f64 * PHRASE_IN_CONTENT_PER_OCC
        + title_token_score as f64 * TITLE_TOKEN_WEIGHT
        + content_token_score as f64 * CONTENT_TOKEN_WEIGHT;
```

`title_token_score`와 `content_token_score`는 `token_match_score`가 돌려주는 값인데, 이 함수는 "토큰이 텍스트에 포함되어 있는가"를 토큰마다 한 번씩 세는 것이 전부다([search.rs#L1103-L1109](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/commands/search.rs#L1103-L1109)). 정리하면 한 페이지의 점수는 다음과 같다.

$$
\text{score} = 200\cdot[\text{stem}=q] + 50\cdot[q \subset \text{title}] + 20\cdot\min(\text{occ}(q),10) + 5\cdot|T\cap\text{title}| + 1\cdot|T\cap\text{body}|
$$

여기서 $q$는 구두점을 걷어낸 소문자 질의 구문, $T$는 토큰 집합이다. 문서 빈도(IDF)도, 문서 길이 정규화도, 토큰 반복 횟수도 들어가지 않으므로 BM25가 아니다. `embedding.ts` 주석에는 벡터 검색이 실패하면 "fell back to BM25"라는 표현이 남아 있지만([embedding.ts#L35-L40](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/embedding.ts#L35-L40)), 실제 대체 경로는 위의 가중 키워드 점수다. README의 "Title match bonus (+10 score)" 설명([README.md#L204-L208](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/README.md#L204-L208))도 현재 상수(구문 50, 토큰 5)와 맞지 않는다.

> **주의**: README는 Phase 1 검색이 "Searches both wiki/ and raw/sources/"라고 적지만, `search_project_inner`가 순회하는 경로는 `project_path/wiki` 하나다([search.rs#L350-L351](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/commands/search.rs#L350-L351)). 원본 문서 검색은 에이전트의 `source.search` 도구가 따로 맡는다(11장).

### 토큰화와 CJK bigram

`tokenize_query`는 소문자로 바꾼 뒤 공백, ASCII 구두점, 중국어 전각 구두점으로 자르고, 한 글자 토큰과 불용어를 버린다. 토큰에 CJK 문자가 있고 세 글자 이상이면 인접 두 글자(bigram), 낱글자, 원 토큰을 모두 넣는다([search.rs#L995-L1027](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/commands/search.rs#L995-L1027)).

```rust
        let has_cjk = chars.iter().any(|c| ('\u{3400}'..='\u{9fff}').contains(c));
        if has_cjk && chars.len() > 2 {
            for pair in chars.windows(2) {
                out.push(pair.iter().collect());
            }
```

CJK 판정 범위가 `U+3400..U+9FFF`(한자)라서 한글 음절(`U+AC00..U+D7A3`)은 이 분기에 들어가지 않는다. 한국어 질의는 공백 단위 토큰 그대로 부분 문자열 포함 여부를 보게 된다. `token_match_score`가 `contains`로 비교하므로 "임베딩을"처럼 조사가 붙은 토큰은 본문의 "임베딩"과 맞지 않는다(코드로 확인한 동작이며 실측은 미검증). 불용어 목록도 중국어와 영어만 있다.

프론트의 `tokenizeQuery`([search.ts#L38-L60](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/search.ts#L38-L60))는 같은 규칙의 TypeScript 사본이다. 검색 패널이 결과 snippet에서 일치 구간을 강조할 때만 쓴다.

## 벡터 검색: chunk 검색과 페이지 단위 집계

질의 임베딩은 `resolve_query_embedding`이 만든다. 호출자가 벡터를 직접 넘기면 비어 있지 않은지, 모든 값이 유한한지만 검사하고 쓴다. 설정이 꺼져 있거나 endpoint, model이 비어 있으면 `None`을 돌려 키워드 검색만 한다. 임베딩 호출이 실패해도 오류를 올리지 않고 로그만 남긴 채 `None`이 된다([search.rs#L294-L315](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/commands/search.rs#L294-L315)). 질의 쪽은 재시도 횟수 0으로 호출하므로, 입력이 길다는 오류가 와도 절반으로 줄여 다시 보내지 않는다.

코드 분석: LanceDB가 돌려준 chunk를 페이지 점수로 바꾸는 부분을 본다. [search.rs#L826-L836](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/commands/search.rs#L826-L836)

```rust
    for (id, mut chunks) in by_page {
        chunks.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.chunk_index.cmp(&b.chunk_index))
        });
        let top_chunk = chunks[0].clone();
        let top = top_chunk.score;
        let tail: f32 = chunks.iter().skip(1).map(|chunk| chunk.score).sum();
        let blended = top + (tail * 0.3).min((1.0 - top).max(0.0));
```

`search_by_embedding`은 원하는 페이지 수의 세 배(최소 30)만큼 chunk를 가져와 `page_id`로 묶는다([search.rs#L802-L852](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/commands/search.rs#L802-L852)). 페이지 점수는 가장 가까운 chunk 점수에 나머지 chunk 점수 합의 30%를 더하되, 더하는 양이 $1 - s_{top}$을 넘지 않게 자른다.

$$
s_{page} = s_{top} + \min\left(0.3\sum_{i\ge 2} s_i,\ \max(0,\ 1 - s_{top})\right)
$$

chunk 하나의 점수는 LanceDB가 준 거리 $d$를 $1/(1+d)$로 바꾼 값이다([vectorstore.rs#L660-L669](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/commands/vectorstore.rs#L660-L669)). 꼬리 가산에 상한을 둔 이유는 TypeScript 쪽 같은 계산의 주석이 설명한다. 약한 chunk가 많은 페이지가 강한 chunk 하나를 가진 페이지를 덮지 못하게 하려는 것이고, 0.3은 "empirical; adjust later with real data"라고 적혀 있다([embedding.ts#L817-L823](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/embedding.ts#L817-L823)).

키워드 단계에서 잡히지 않았지만 벡터로만 찾은 페이지는 `materialize_vector_only_results`가 파일을 다시 읽어 결과 목록에 더한다. 이때 snippet은 가장 가까운 chunk의 앞부분 160자에 heading 경로를 붙인 문자열이다([search.rs#L854-L909](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/commands/search.rs#L854-L909)).
