물리 계획이 엔진 연산자로 연결되고 검증을 거쳐 수용되는 과정을 본다.

## Lower 트레이트와 엔진 연결

물리 계획에서 연산자로 가는 순회는 계획기 크레이트가 소유하고, 엔진은 노드 종류마다 메서드 하나를 구현한다.

### 코드 분석: Lower 트레이트

`crates/omnigraph-planner/src/lower.rs`, [lower.rs L125-L148](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-planner/src/lower.rs#L125-L148)

```rust
pub trait Lower {
    type Op;
    type Error: From<PlanError>;

    fn metadata_count(
        &mut self,
        id: NodeId,
        spec: &ScanSpec,
        return_exprs: &[IRProjection],
    ) -> Result<Self::Op, Self::Error>;
```

`Op`가 연관 타입이라 계획기 크레이트는 DataFusion에 의존하지 않는다. 메서드에는 기본 구현이 없고 순회 함수 `lower_node`의 매치에도 와일드카드 팔이 없다. 물리 노드를 추가하면 순회와 모든 `impl Lower`가 컴파일되지 않는다. 테스트 `physical_node_matches.rs`는 크레이트 소스를 파싱해 노드 종류에 대한 매치가 허용된 위치 밖에 생기지 않았는지 검사한다 ([docs/dev/execution.md L683-L687](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/docs/dev/execution.md#L683-L687)).

진입점 `PhysicalPlan::lower`는 `check_total`, 루트부터의 `lower_node`, `finish` 세 호출로 끝난다 ([lower.rs L320-L324](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-planner/src/lower.rs#L320-L324)). `check_total`은 한 노드가 두 소비자의 입력인 경우와 살아 있는 노드가 루트에서 도달되지 않는 경우를 거부한다 ([lower.rs L333-L358](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-planner/src/lower.rs#L333-L358)). 그래서 계획은 항상 트리이고 노드 하나가 연산자 하나가 된다. 예외적인 호출은 둘이다. `AntiJoin`은 외부 트리와 내부 트리 사이에 `anti_join_outer`를 부르고 ([lower.rs L533-L543](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-planner/src/lower.rs#L533-L543)), `finish`는 루트에 한 번 불린다. 엔진의 구현체는 `engine/lower.rs`의 `Walk`이며 ([engine/lower.rs L554](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/engine/lower.rs#L554)), 각 메서드가 만드는 연산자는 6장에서 다룬다.

## 계획 수용과 재생

물리 계획이 만들어졌다고 바로 실행되지는 않는다. `validate::accept`만이 `AcceptedPlan`을 만들고, 그 필드는 비공개이며 역직렬화로도 만들 수 없다 ([validate/mod.rs L148-L155](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-planner/src/validate/mod.rs#L148-L155)). 엔진의 `bind`와 `execute`는 이 타입만 받는다. 새 계획이 검사에 실패하면 "a planner defect"이지 언어 수준의 거부가 아니다 ([validate/mod.rs L1-L13](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-planner/src/validate/mod.rs#L1-L13)).

### 코드 분석: check의 세 단계

`crates/omnigraph-planner/src/validate/mod.rs`, [validate/mod.rs L241-L262](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-planner/src/validate/mod.rs#L241-L262)

```rust
    budget.visit(u64::try_from(plan.live().count()).unwrap_or(u64::MAX))?;
    let formed = |check: &'static str, result: Result<(), crate::error::PlanError>| {
        result.map_err(|error| ValidationError::violated(check, error.to_string()))
    };
    formed(
        "aggregate spec",
        crate::aggregate::validate_aggregate_specs(plan),
    )?;
    formed(
        "output schema",
        crate::output::validate_output_schemas(plan),
    )?;
    formed(
        "fusion row keys",
        crate::optimizer::validate_rank_fuse_row_tiebreaks(plan),
    )?;
    formed("catalogue source", source::validate_catalogue_sources(plan))?;
    let requirements = Requirements::derive(input, budget)?;
    requirements.check(plan, input, budget)?;
    let Some(member) = subset::membership(input) else {
        return Ok((ValidationScope::InvariantsOnly, None));
    };
```

첫 단계는 계획이 스스로 일관적인지 본다. 둘째 단계는 타입 검사된 선언에서 요구사항을 뽑아 계획이 지키는지 본다 ([validate/requirements.rs L1-L12](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-planner/src/validate/requirements.rs#L1-L12)). 검사 순서는 바인딩, 적격성 필터, 상관 블록, 검색 정체성, 행 절단, 반환, 지연 조회, 순서와 절단, 정책이다 ([validate/invariants.rs L260-L294](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-planner/src/validate/invariants.rs#L260-L294)). 셋째 단계는 쿼리가 정확 부분집합(exact fragment)에 속할 때만 실행된다.

| 범위 | 판정 조건 | 보장 수준 |
|---|---|---|
| `invariants_only` | 모든 계획의 기본 | 요구사항 보존. 행 선택의 동치까지는 증명하지 않음 |
| `exact_subset` | 단일 바인딩, 단순 필터, 선택적 선두 `bm25()`, 속성 정렬, limit, 투영 | 정규형에서 계획까지의 재작성을 재구성해 노드 단위로 일치 확인 |

두 범위의 정의는 [validate/mod.rs L40-L52](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-planner/src/validate/mod.rs#L40-L52)에 있다.

### 코드 분석: 정확 부분집합의 재작성 규칙

`crates/omnigraph-planner/src/validate/subset.rs`, [validate/subset.rs L372-L398](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-planner/src/validate/subset.rs#L372-L398)

정확 부분집합 쿼리는 정규 체인 `Limit(Sort(Project([Search](Filter(Scan)))))`에서 출발한다 ([validate/subset.rs L1-L18](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-planner/src/validate/subset.rs#L1-L18)). 최적화기는 적용한 규칙을 `Tracer`에 기록하고, 검증기는 전제 조건을 확인하며 규칙을 직접 다시 적용한다.

```rust
pub enum Rule {
    /// `Filter(p ∧ c, Scan(C, f))` → `Filter(p, Scan(C, f ∧ c))`, `at`
    /// `[filter, scan]`; `c` must be one of the filter's conjuncts and
    /// within the fragment's predicate forms, which the scan evaluates with
    /// the same typed semantics.
    AbsorbScanFilter { conjunct: ExprMirror },
    /// The scan reads `columns`, which must hold every column the nodes
    /// above it read and only columns of its table; `at` `[scan]`.
    PruneScanColumns { columns: Vec<String> },
    /// A logical node becomes its physical operator unchanged; `at` `[node]`.
    Lower,
```

규칙은 여섯 개로 닫혀 있고 앞서 본 패스와 대응한다. `AbsorbScanFilter`는 필터 배치, `PruneScanColumns`는 투영 pushdown, `RankBm25Scan`은 BM25 순위 스캔, `HydrateReturnColumns`는 지연 조회, `NarrowKey`는 키 좁히기다. 검증기는 최적화기의 후속 노드를 믿지 않고 직접 재구성한 결과가 후보 계획과 같은지 비교한다. 규칙 목록 버전 `RULES_VERSION`은 2다 ([validate/replay.rs L27-L32](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-planner/src/validate/replay.rs#L27-L32)). 검증은 예산 안에서 돈다. `ValidationLimits::DEFAULT`는 증거 1 MiB, 유도 노드 4096개, 규칙 적용 1024회, 방문 $2^{20}$회이며 ([validate/budget.rs L25-L34](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-planner/src/validate/budget.rs#L25-L34)), 소진은 계획 오류가 아니라 `ValidationExhausted`다.

### 바인딩과 재생

수용된 계획은 `bind`로 실행 값을 결합한다 ([validate/mod.rs L176-L186](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-planner/src/validate/mod.rs#L176-L186)). `ValueTable`은 해석된 파라미터와 `nearest` 스캔별 쿼리 벡터를 담는다 ([bound.rs L37-L44](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-planner/src/bound.rs#L37-L44)). 바인딩은 계획 노드를 바꾸지 않으므로 다시 검증하지 않는다.

`BoundPlan`은 직렬화할 수 있다. 컴파일러의 `IRExpr`과 Arrow schema는 `Serialize`를 구현하지 않으므로 `mirror.rs`가 계획기 소유의 대응 타입을 둔다. 두 물리 계획의 동등성도 이 mirror로 비교한다. 저장 형식 버전은 `BOUND_PLAN_VERSION = 11`이며 버전별 추가 필드가 주석에 누적된다 ([bound.rs L17-L28](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-planner/src/bound.rs#L17-L28)).

저장된 계획은 `ReplayEnvelope`에 쿼리 원문, 수용 범위, schema 다이제스트, 바인딩된 계획, 유도 이력을 담아 다시 들어온다 ([validate/replay.rs L44-L59](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-planner/src/validate/replay.rs#L44-L59)). 재생은 바이트 한도와 버전을 먼저 확인하고, 다른 schema나 고정되지 않은 dataset을 충돌로 거부하며, 전문 검색 커버리지를 고정 snapshot에서 다시 도출해 기록과 비교한 뒤 같은 `check`를 통과해야 한다 ([docs/dev/execution.md L984-L992](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/docs/dev/execution.md#L984-L992)). 저장된 계획 파일이 스스로를 검증하는 경로는 없다.

> **주의**: 계획기 크레이트 문서는 자신을 "Planner and optimizer for Omnigraph operations (RFC 0068)"라고 소개한다 ([lib.rs L1](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-planner/src/lib.rs#L1)). 그러나 기준 커밋의 `docs/rfcs/0068-graph-commit-record.md`는 "Graph commit record" 제목의 초안 RFC로 주제가 다르다. 수용 모듈은 RFC 0047을 인용하며, 이 장의 수용 설명은 코드와 `docs/dev/execution.md`를 근거로 한다.
