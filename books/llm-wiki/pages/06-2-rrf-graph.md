두 순위는 RRF로 합치고, 마지막에 상위 결과의 위키링크 이웃을 결과에 추가한다.

## 두 순위의 결합: RRF

벡터 결과가 하나라도 있으면 키워드 점수를 버리고 순위만 남겨 Reciprocal Rank Fusion으로 다시 매긴다.

코드 분석: 한 결과의 최종 점수를 계산하는 부분을 본다. [search.rs#L519-L537](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/commands/search.rs#L519-L537)

```rust
        let mut rrf = 0.0;
        if let Some(rank) = token {
            rrf += 1.0 / (RRF_K + rank as f64);
        }
        if let Some(rank) = vector {
            rrf += 1.0 / (RRF_K + rank as f64);
        }
```

$$
\text{RRF}(p) = \frac{1}{60 + r_{tok}(p)} + \frac{1}{60 + r_{vec}(p)}
$$

한쪽 순위에만 있는 페이지는 해당 항 하나만 받는다. 순위로만 합치므로 키워드 점수의 절댓값(파일명 일치 보너스 200 등)은 벡터가 켜진 순간 의미를 잃는다. 파일명이 질의와 정확히 같은 페이지는 키워드 1위가 되어 $1/61$을 받지만, 벡터 순위가 낮으면 양쪽 모두 중간쯤인 페이지에 밀릴 수 있다. 벡터 순위를 찾을 때는 경로 기반 page id(`entities/foo`)를 먼저 보고, 예전 형식인 파일명 stem 행은 같은 stem을 가진 파일이 하나뿐일 때만 인정한다([search.rs#L497-L539](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/commands/search.rs#L497-L539)).

## 그래프 확장: 상위 결과의 1홉 이웃 추가

정렬이 끝나면 `blend_graph_results`가 결과 창의 일부를 위키링크 이웃에게 할당한다. 이웃 관계는 같은 순회에서 모은 본문 `[[...]]`와 프론트매터 `related:` 값으로 만들고([search.rs#L707-L727](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/commands/search.rs#L707-L727)), 경로, `wiki/` 뺀 경로, 파일명 stem, 제목을 모두 별칭으로 등록해 링크 대상을 찾는다. 방향은 무시하고 양방향 인접 목록으로 만든다.

코드 분석: 그래프 이웃에게 몇 자리를 줄지 정하는 함수를 본다. [search.rs#L541-L553](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/commands/search.rs#L541-L553)

```rust
fn graph_result_quota(limit: usize, vector_hits: usize) -> usize {
    if limit < 2 {
        return 0;
    }
    let vector_coverage = vector_hits.min(limit) as f64 / limit as f64;
    let ratio = MAX_GRAPH_RESULT_RATIO
        - (MAX_GRAPH_RESULT_RATIO - MIN_GRAPH_RESULT_RATIO) * vector_coverage;
    ((limit as f64 * ratio).ceil() as usize).clamp(1, limit - 1)
}
```

벡터 결과가 창을 가득 채우면 15%, 벡터가 없으면 30%를 그래프에 준다. 테스트가 `limit=20`일 때 벡터 0개면 6자리, 10개면 5자리, 20개면 3자리라고 고정해 둔다([search.rs#L2117-L2123](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/commands/search.rs#L2117-L2123)). 벡터 검색이 약할수록 링크 구조에 더 기대는 설계다.

후보 점수는 상위 결과(최대 20개)를 씨앗으로 삼아, 씨앗 순위 $r$에서 연결되는 이웃마다 $1/(r+1)$을 더한 값이다. 이미 씨앗인 페이지는 후보에서 빠진다([search.rs#L609-L625](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/commands/search.rs#L609-L625)). 뽑힌 이웃은 원래 결과 뒤에 붙고, 키워드나 벡터로 이미 찾았던 페이지라면 그 결과에 `graph_related_to`만 채운다. 새로 들어온 이웃의 snippet은 "Graph neighbor of ..." 문자열이고 점수는 $\text{graph\_score}/61$이다. 그래프 이웃 후보가 없으면 그 자리는 키워드와 벡터 결과에 돌아간다.

`search_mode`가 응답에 적는 모드는 다음 규칙을 따른다([search.rs#L782-L794](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/commands/search.rs#L782-L794)).

| 조건 | `mode` |
| --- | --- |
| 그래프 이웃이 하나라도 들어감 | `hybrid` |
| 벡터 결과 0개 | `keyword` |
| 벡터 결과는 있고 키워드 결과 0개 | `vector` |
| 벡터와 키워드 결과가 모두 있음 | `hybrid` |

결국 질문 "검색은 BM25/벡터/하이브리드 중 무엇인가"에 대한 답은 이렇다. 임베딩을 켜지 않으면 가중 키워드 점수에 그래프 확장을 더한 방식이고, 켜면 키워드 순위와 chunk 벡터 순위를 RRF로 합친 뒤 그래프 확장을 더한 방식이다. 두 경우 모두 그래프 이웃이 들어가면 응답 모드는 `hybrid`로 표시된다.
