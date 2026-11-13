그래프 화면의 필터 · 검색과 sigma.js 기반 렌더링을 본다.

## 필터와 그래프 안 검색

화면에 그리기 전에 `applyGraphFilters`와 `applyGraphSearch`를 차례로 적용한다([graph-view.tsx#L925-L931](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/components/graph/graph-view.tsx#L925-L931)). 숨긴 노드에 연결된 엣지는 함께 사라진다.

| 필터 필드 | 기본값 | 숨기는 노드 |
| --- | --- | --- |
| `hiddenTypes` | 빈 Set | type이 Set에 있는 노드 |
| `hiddenNodeIds` | 빈 Set | 사용자가 하나씩 숨긴 노드 |
| `hideStructural` | `true` | id가 `index`, `overview`, `log`, `schema`, `purpose`이거나 type이 `overview`이거나 해당 경로로 끝나는 노드 |
| `hideIsolated` | `false` | `linkCount ≤ 0` |
| `minLinks` / `maxLinks` | 없음 | 연결 수가 범위 밖인 노드 |

값은 [graph-filters.ts#L19-L43](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/graph-filters.ts#L19-L43)에 있다. 구조 페이지를 기본으로 숨기는 이유는 인사이트에서 빼는 이유와 같다. `index.md`는 모든 페이지를 링크하므로, 그리면 모든 노드가 한 점으로 끌려가 레이아웃이 의미를 잃는다.

그래프 안 검색은 6장의 검색기와 무관한 단순 필터다. 질의를 공백으로 나눠, 모든 토큰이 노드의 제목, id, type, 경로를 이은 문자열에 들어 있는 노드만 남긴다(AND 조건)([graph-search.ts#L9-L43](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/graph-search.ts#L9-L43)). 일치한 노드끼리의 엣지만 남기므로 검색 중에는 이웃이 함께 보이지 않는다.

## 렌더링: sigma.js, graphology, ForceAtlas2

`GraphLoader`는 결과를 graphology `Graph`로 옮기고 sigma.js(`@react-sigma/core`)로 그린다. 노드 크기와 엣지 모양은 데이터에서 계산한다.

코드 분석: 노드 크기 계산을 본다. [graph-view.tsx#L228-L238](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/components/graph/graph-view.tsx#L228-L238)

```tsx
function graphDensityScale(nodeCount: number): number {
  if (nodeCount <= 150) return 1
  return Math.max(0.35, Math.sqrt(150 / nodeCount))
}

function nodeSize(linkCount: number, maxLinks: number, nodeCount: number, userScale: number): number {
  if (maxLinks === 0) return BASE_NODE_SIZE
  const ratio = linkCount / maxLinks
  const size = BASE_NODE_SIZE + Math.sqrt(ratio) * (MAX_NODE_SIZE - BASE_NODE_SIZE)
  return size * graphDensityScale(nodeCount) * userScale
}
```

노드 크기는 8에서 28 사이에서 `linkCount` 비율의 제곱근을 따른다. 제곱근을 쓰면 허브 하나가 화면을 덮지 않고, 연결이 적은 노드끼리의 차이도 보인다. 노드가 150개를 넘으면 전체 크기를 줄이되 0.35배 아래로는 줄이지 않는다. 엣지는 weight를 최댓값으로 나눈 비율로 굵기 0.5..4, 불투명도 40..220/255를 정한다([graph-view.tsx#L350-L373](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/components/graph/graph-view.tsx#L350-L373)). README의 "green=strong, gray=weak" 설명([README.md#L156](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/README.md#L156))과 달리 현재 코드는 slate 한 색에 투명도만 바꾼다.

노드 수에 따라 바뀌는 설정은 다음과 같다([graph-view.tsx#L240-L267](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/components/graph/graph-view.tsx#L240-L267)).

| 노드 수 | ForceAtlas2 반복 | 약한 엣지 기준 (정규화 weight) | 라벨 크기 하한 | 라벨 밀도 |
| --- | --- | --- | --- | --- |
| 250 이하 | 140 | 0 (모두 표시) | 6 | 0.4 |
| 251..600 | 90 | 0 | 6 | 0.4 |
| 601..700 | 65 | 0 | 10 | 0.24 |
| 701..1200 | 65 | 0.05 | 10 | 0.24 |
| 1201..2500 | 40 | 0.1 | 14 | 0.14 |
| 2500 초과 | 28 | 0.16 | 18 | 0.08 |

약한 엣지 기준보다 낮은 엣지는 `lowPriority`로 표시해 덜 중요하게 그린다. 레이아웃은 ForceAtlas2에 `strongGravityMode`, 노드 50개 초과 시 Barnes-Hut 근사를 켠다. 노드가 220개 이상이면 `graph-layout-worker.ts`의 Web Worker에서 계산하고 끝나면 좌표만 받아 반영한다([graph-layout-worker.ts#L12-L45](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/components/graph/graph-layout-worker.ts#L12-L45)). 계산된 좌표는 모듈 수준 `positionCache`에 남겨, 데이터가 조금 바뀌어도 기존 노드가 제자리에 있게 한다. 레이아웃은 노드 id, 엣지, weight, 간격 설정으로 만든 키가 바뀔 때만 다시 돌린다.

마우스를 노드에 올리면 sigma 그래프의 `neighbors`로 이웃 Set을 만들어 이웃만 밝게 남기고, 클릭하면 오른쪽 미리보기 패널에서 페이지를 연다([graph-view.tsx#L536-L573](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/components/graph/graph-view.tsx#L536-L573)). 색은 type 모드(entity 파랑, concept 보라, source 주황 등 11종)와 커뮤니티 모드(12색 순환) 사이를 오간다([graph-view.tsx#L28-L66](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/components/graph/graph-view.tsx#L28-L66)).
