그래프는 위키링크와 frontmatter에서 만들고, 엣지 weight는 네 가지 신호를 합친 관련도로 정한다.

## 위키링크 기반 그래프 구성: buildWikiGraph

`buildWikiGraph`는 같은 프로젝트와 `dataVersion` 조합이면 캐시를 돌려주고, 같은 조합의 빌드가 진행 중이면 그 Promise를 공유한다. 캐시는 프로젝트 2개까지만 유지하고, 늦게 끝난 오래된 빌드가 새 결과를 덮지 않게 버전을 비교한다([wiki-graph.ts#L170-L204](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/wiki-graph.ts#L170-L204)).

실제 계산은 `buildWikiGraphUncached`가 한다. 파일마다 제목(프론트매터 `title`, 없으면 첫 `# ` heading, 그것도 없으면 파일명), type(없으면 `other`), 링크 목록을 뽑는다. 링크 목록은 본문 위키링크와 프론트매터 `related` 값을 합쳐 중복을 지운 것이다. 위키링크 정규식은 `[[대상|별칭]]`에서 대상만 잡는다([wiki-graph.ts#L74](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/wiki-graph.ts#L74)). `related` 값은 `[[...]]`, `|별칭`, `#앵커`, 디렉터리, `.md`를 모두 걷어내 파일명 stem만 남긴다([wiki-graph.ts#L114-L123](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/wiki-graph.ts#L114-L123)).

코드 분석: 링크를 엣지로 바꾸고 연결 수를 세는 루프를 본다. [wiki-graph.ts#L269-L281](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/wiki-graph.ts#L269-L281)

```ts
  for (const [sourceId, nodeData] of nodeMap) {
    for (const targetRaw of nodeData.links) {
      // Normalize target: try matching by id (case-insensitive, hyphen/space)
      const targetId = resolveTarget(targetRaw, targetIndex)
      if (targetId === null) continue
      if (targetId === sourceId) continue

      rawEdges.push({ source: sourceId, target: targetId, weight: 1 })

      linkCounts.set(sourceId, (linkCounts.get(sourceId) ?? 0) + 1)
      linkCounts.set(targetId, (linkCounts.get(targetId) ?? 0) + 1)
    }
  }
```

대상 해석은 정확한 id를 먼저 보고, 없으면 소문자, 소문자에 공백을 하이픈으로 바꾼 별칭 순으로 찾는다. 대소문자를 구분하는 파일 시스템에서 `Foo.md`와 `foo.md`가 따로 있으면 `[[foo]]`는 정확히 `foo`로 간다([wiki-graph.ts#L129-L147](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/wiki-graph.ts#L129-L147)). 대상을 찾지 못한 링크(아직 없는 페이지를 가리키는 링크)는 조용히 버린다. 그래프에 "빈 노드"를 그리지 않는다는 뜻이고, 깨진 링크 진단은 린트(9장)의 몫이다.

이 루프를 읽을 때 놓치기 쉬운 세 가지가 있다.

| 지점 | 동작 | 영향 |
| --- | --- | --- |
| 노드 id | 파일명 stem만 쓴다 (`fileNameToId`) | `entities/foo.md`와 `concepts/foo.md`가 있으면 `nodeMap.set`이 뒤의 것으로 덮는다. 벡터 인덱스(6장)의 디렉터리 포함 page id와 규칙이 다르다 |
| `linkCount` | 중복 제거 전 `rawEdges` 기준으로 센다 | A와 B가 서로 링크하면 각자 2를 받는다. 노드 크기와 고립 판정이 "링크 표기 수"를 따른다 |
| `query` type | 노드에서 제거한다 | 저장한 채팅 답(8장)과 리서치 결과는 그래프에 나오지 않는다 |

`query` 제거 이유는 주석에 적혀 있다. 질의 페이지는 "intermediate artifacts, not knowledge structure"이고, auto-ingest로 거기서 뽑아낸 entity와 concept가 그래프에 들어가야 한다는 판단이다([wiki-graph.ts#L250-L258](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/wiki-graph.ts#L250-L258)). 엣지는 방향을 무시하고 `A:::B`와 `B:::A`를 같은 것으로 보아 하나만 남긴다.

## 엣지 weight: 4신호 관련도 모델

중복 제거된 엣지마다 `calculateRelevance`로 weight를 매긴다. 이를 위해 `graph-relevance.ts`가 같은 파일들을 한 번 더 읽어 별도의 `RetrievalGraph`를 만든다([graph-relevance.ts#L202-L297](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/graph-relevance.ts#L202-L297)). 이 그래프는 방향 있는 `outLinks`, `inLinks`와 프론트매터 `sources` 목록을 노드에 담는다. `query` 노드를 빼지 않으므로 공통 이웃 계산에는 질의 페이지도 들어간다.

| 신호 | 가중치 | 계산 |
| --- | --- | --- |
| 직접 링크 | 3.0 | A→B, B→A 각각 1. 양방향이면 6.0 |
| 원본 겹침 | 4.0 | 두 페이지 `sources[]`에 같이 있는 원본 수 |
| 공통 이웃 (Adamic-Adar) | 1.5 | 공통 이웃 $z$마다 $1/\ln(\max(\deg z, 2))$의 합 |
| type 친화도 | 1.0 | `TYPE_AFFINITY[A.type][B.type]`, 표에 없으면 0.5 |

가중치와 type 친화도 표는 [graph-relevance.ts#L33-L46](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/graph-relevance.ts#L33-L46)에 있다.

$$
w(A,B) = 3\,(l_{AB}+l_{BA}) + 4\,|S_A \cap S_B| + 1.5\sum_{z \in N(A)\cap N(B)} \frac{1}{\ln \max(\deg z, 2)} + \tau(A,B)
$$

코드 분석: 공통 이웃 신호의 계산을 본다. [graph-relevance.ts#L328-L346](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/graph-relevance.ts#L328-L346)

```ts
  // Signal 3: Common neighbors - Adamic-Adar (weight 1.5)
  const neighborsA = getNeighbors(nodeA)
  const neighborsB = getNeighbors(nodeB)
  let adamicAdar = 0
  for (const neighborId of neighborsA) {
    if (neighborsB.has(neighborId)) {
      const neighbor = graph.nodes.get(neighborId)
      if (neighbor) {
        const degree = getNodeDegree(neighbor)
        adamicAdar += 1 / Math.log(Math.max(degree, 2))
      }
    }
  }
  const commonNeighborScore = adamicAdar * WEIGHTS.commonNeighbor

  // Signal 4: Type affinity (weight 1.0)
  const affinityMap = TYPE_AFFINITY[nodeA.type]
  const typeAffinityScore = (affinityMap?.[nodeB.type] ?? 0.5) * WEIGHTS.typeAffinity
```

Adamic-Adar는 링크 예측에서 쓰는 지표로, 연결이 많은 허브를 공유하는 것보다 연결이 적은 페이지를 공유하는 것을 더 강한 신호로 친다. 차수 하한을 2로 잡아 $\ln 1 = 0$으로 나누는 일을 막는다. 원본 겹침에 가장 큰 가중치(4.0)를 준 것은, 같은 원본에서 수집된 페이지는 링크가 없어도 내용이 가깝다는 가정이다. 이 가정은 원본 수명주기(5장)의 `sources[]` 프론트매터가 정확할 때만 성립한다.

type 친화도는 항상 0.5 이상이라 `calculateRelevance`는 0보다 큰 값을 돌려준다. 이 모듈의 `getRelatedNodes`는 "relevance > 0"인 노드를 모두 후보로 삼는데, 그러면 실질적으로 모든 노드가 후보가 된다. 다만 `getRelatedNodes`는 테스트 밖에서 호출하는 곳이 없다. README가 질의 파이프라인 Phase 2로 설명하는 "4-signal relevance model finds related pages, 2-hop traversal with decay"([README.md#L216-L219](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/README.md#L216-L219))는 현재 코드에서는 이 TypeScript 모델이 아니라 Rust 검색의 1홉 확장(6장)이 맡는다. 현재 이 관련도 모델의 실제 쓰임은 그래프 뷰 엣지 weight 하나다.

> **주의**: `RetrievalGraph`의 프론트매터 파서는 정규식으로 `sources:` 블록 목록과 한 줄 배열만 읽는다([graph-relevance.ts#L91-L151](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/graph-relevance.ts#L91-L151)). 같은 파일의 `related`는 공용 `parseFrontmatter`로 읽는다. YAML 표기가 이 두 형태를 벗어나면 원본 겹침 신호가 0이 될 수 있다(추정).
