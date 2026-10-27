# Open Ontologies: 온톨로지 변경 계획과 검증 가능한 추론 증명서

- **레포**: [fabio-rovai/open-ontologies](https://github.com/fabio-rovai/open-ontologies) · [브라우저 체험 페이지](https://open-ontologies-try.vercel.app/) · 논문 [arXiv:2605.09184](https://arxiv.org/abs/2605.09184) (미검증, 본문 읽지 않음)
- **제작**: Fabio Rovai (Tesseract Semantics). 커밋 대부분이 한 사람이다 (878건, 2위 기여자 55건 · 2026-10-27 GitHub API 기준)
- **공개**: 2026-03-09 저장소 생성일 (Rust, MIT, 별 1,499개 · 2026-10-27 기준)
- **읽은 날짜**: 2026-10-27 (커밋 `9e73d95`, 최신 릴리즈 `v2.1.0` 2026-10-23)
- **형태**: Rust 단일 바이너리. CLI이자 MCP 서버(`serve`). 증명 검사기는 Lean 4로 따로 빌드한다
- **태그**: #Ontology #OWL #SHACL #KnowledgeGraph #MCP #FormalVerification #GraphRAG

## 한 줄 요약

RDF/OWL 온톨로지를 운영 환경에서 바꿀 때 쓰는 **Terraform식 변경 관리 도구**다. `plan`으로 바뀌는 클래스·트리플과 위험도를 보고, `apply`·`drift`·`rollback`으로 적용과 복구를 한다. 다른 도구와 갈리는 지점은 두 가지다. 추론 결과마다 **증명서 파일**(전제와 규칙을 한 줄씩 적은 TSV)을 남기고 Lean 4로 짠 검사기가 그 파일을 다시 확인한다. 또 RAG용으로 잘라 낸 부분 그래프가 **답에 필요한 결론을 실제로 유지하는지** 주장 단위로 판정한다.

## 구성

| 구성 요소 | 내용 |
|---|---|
| 저장소와 질의 | Oxigraph 0.5 위의 RDF 저장소, SPARQL 1.1. 상태·계보는 SQLite |
| 추론 | RDFS, OWL 2 RL 일부(78개 규칙 중 29개), SHIQ 타블로. SWRL·RIF Core·Horn 규칙표를 사용자 규칙으로 받는다 |
| 검증 | SHACL(W3C 테스트 스위트로 측정), `onto_enforce` 설계 패턴 검사(고아 클래스, 라벨 누락, 사용자 SPARQL 규칙) |
| 변경 관리 | `plan` → `enforce` → `apply`(safe / migrate) → `monitor` → `drift` → `rollback`. 잠금 IRI로 실수 삭제 방지 |
| 데이터 적재 | CSV, JSON, XML, YAML, XLSX, Parquet, PostgreSQL·DuckDB SELECT 결과를 매핑 설정으로 RDF 변환 |
| 증명서 검사기 | Lean 4(`oo-cert`, `oo-horn`). Mathlib 없이 core Lean만. Isabelle/HOL로 같은 바이트를 두 번째로 검사 |
| 외부 증명기 내보내기 | TPTP, CLIF, SMT-LIB, LADR. 옮기지 못한 항목 수를 함께 보고 |
| MCP 도구 | `onto_*` 116개. 4개는 `embeddings`, 2개는 `plugins`, 2개는 `postgres`/`duckdb` 기능 플래그가 있어야 동작하고 배포 바이너리에는 빠져 있다 |
| 기타 | 표준 온톨로지 33종 마켓플레이스, 임상 용어 매핑, Tauri 데스크톱 Studio, 순수 Python 엔진 `open-ontologies-lite` |

## 핵심 아이디어

### 모양 diff와 의미 diff는 다르다

README 첫 예시는 `ex:hasParent rdfs:domain ex:Person` 한 줄을 추가하는 변경이다. 클래스 추가·삭제 0, 영향 트리플 0, 위험도 low로 나오지만 **새 결론이 901개** 생긴다. 이 속성을 가진 모든 개체가 `Person` 타입을 새로 얻기 때문이다. `plan --check_conservativity`는 변경 전후 그래프를 각각 끝까지 추론해, 기존 이름들에 대해 결론이 달라졌는지(보존 확장인지)를 규칙과 전제와 함께 보고한다. 비싼 연산이라 선택 옵션이고, 꺼져 있으면 `ran: false`를 명시해 "검사 안 함"과 "문제 없음"이 같아 보이지 않게 한다 ([docs/lifecycle.md](https://github.com/fabio-rovai/open-ontologies/blob/main/docs/lifecycle.md)).

### 추론 결과에 증명서를 붙인다

`reason --certificate <dir>`은 `asserted.tsv`(입력 트리플)와 `derivations.tsv`(규칙, 결론, 전제를 한 줄씩)를 쓴다. 검사기는 각 줄의 결론이 그 규칙과 전제에서 나오는지, 전제가 입력이거나 앞줄의 결론인지 확인한다. 결론 하나만 위조해도 exit 1과 함께 줄 번호·규칙 이름을 낸다. 엔진 코드를 믿을 필요 없이 파일 두 개와 Lean 빌드만 있으면 몇 달 뒤 감사자가 다시 확인할 수 있다는 것이 주장이다.

판정 단어를 섞지 않는 규율이 눈에 띈다. 표준 규칙만 썼으면 `entailed`, 사용자 규칙이 끼면 `entailed_under_supplied_rules`로 단어가 바뀐다. "모든 공급사는 적합하다" 같은 규칙을 넣으면 증명서는 영원히 통과하므로, 그 규칙이 가정이라는 사실을 판정에 싣는다. 불만족(inconsistent) 판정에는 보증이 없다고 표에 적어 둔다.

### 검색된 부분 그래프는 결론을 유지하거나 잃는다

[decision 0007](https://github.com/fabio-rovai/open-ontologies/blob/main/docs/decisions/0007-a-slice-preserves-a-conclusion-or-it-does-not.md)이 GraphRAG 관점에서 가장 쓸모 있다. 원본 그래프 G에서 잘라 낸 슬라이스 P를 "몇 %의 트리플을 담았나"(coverage)로 재면 함정에 빠진다.

- 피자 온톨로지에서 `NamedPizza rdfs:subClassOf Pizza` 한 줄만 뺀 슬라이스는 coverage 1.0, `ok: true`로 나오지만 `Veneziana rdfs:subClassOf Food` 결론을 잃는다
- 반대로 트리플 3개짜리 슬라이스는 coverage 0.00128인데 물어본 주장을 모두 유지한다
- coverage는 슬라이스가 클수록 오르므로, 이 지표로 튜닝한 검색기는 "맞는 것"이 아니라 "많이" 가져오도록 배운다

대신 답이 기대는 주장 q마다 G와 P가 각각 q를 함의하는지 보고 네 칸으로 나눈다.

| G가 q를 함의 | P가 q를 함의 | 판정 | 고칠 쪽 |
|---|---|---|---|
| 예 | 예 | preserved (증명서 첨부) | 없음 |
| 예 | 아니오 | 슬라이스가 잃음 | 검색기 |
| 아니오 | 예 | P가 G의 부분집합이 아니거나 엔진 버그 | 파이프라인 / 엔진 |
| 아니오 | 아니오 | `ungrounded_in_source` | 생성기 (근거 없는 주장) |

마지막 두 칸을 "유지 안 됨" 하나로 묶으면 검색기 문제와 환각 문제를 같은 팀에 보내게 된다는 지적이 핵심이다. coverage는 지우지 않고 `is_a_warrant: false` 꼬리표를 붙여 남긴다.

## 읽을 때 감안할 것

- 사실상 1인 프로젝트이고 기능 범위가 매우 넓다(MCP 도구 116개, 인과 추론, PDDL 계획, 임상 매핑까지). README도 "기능 목록이 논거가 아니다"라며 변경 관리 루프만 앞에 내세운다. 주변 기능의 성숙도는 확인하지 않았다
- "증명"의 범위가 좁다. Lean 검사기가 보증하는 것은 **증명서에 적힌 도출 단계가 규칙대로라는 것**이다. 엔진이 결론을 빠뜨리지 않았다는 완전성이나, 불만족 판정은 보증하지 않는다고 스스로 밝힌다
- OWL 2 RL 규칙 78개 중 29개만 다루므로 `ungrounded_in_source`는 "이 규칙 범위에서 도출 안 됨"이지 거짓이라는 뜻이 아니다
- 기본 저장 모드가 메모리라서 `load`와 `reason`을 따로 실행하면 빈 저장소에서 추론한다. README가 직접 경고한다
- README와 문서 수치는 레포 안 테스트가 다시 계산한다고 적혀 있지만, 이 노트의 수치(901, 0.00128 등)는 문서에서 옮겼고 직접 실행하지 않았다
- 이 노트는 Claude가 레포와 문서를 읽고 썼다

## 가져갈 지점

- **coverage는 보증이 아니다.** 검색 결과 평가에서 "얼마나 담았나" 대신 "답이 기대는 주장을 담았나"를 주장 단위로 본다. [GraphRAG-Bench 정리](../papers/[20260929]%20GraphRAG-Bench_When_to_use_Graphs_in_RAG.md)의 검색 평가와 함께 볼 만하다
- **실패를 고칠 쪽 기준으로 나눈다.** 검색기가 잃은 것과 생성기가 지어낸 것은 처방이 반대이므로 한 칸에 넣지 않는다
- **판정 단어에 가정을 싣는다.** 사용자가 넣은 규칙·임계값으로 통과한 결과는 다른 단어로 보고한다
- **꺼진 검사는 꺼졌다고 출력한다.** `ran: false`와 이유를 결과에 남긴다
- 같은 온톨로지·MCP 계열인 [OpenCrab 정리]([20260322]%20AlexAI-MCP_OpenCrab_MCP로_붙이는_온톨로지_공장.md)는 온톨로지를 "만드는" 쪽이고, 이 도구는 이미 있는 온톨로지를 "바꾸고 검증하는" 쪽이다

## 결론

RDF/OWL 온톨로지를 운영하면서 변경 이력과 추론 결과를 감사해야 하는 환경이라면 plan/apply와 Lean 증명서가 바로 쓸모 있다. 추론 단계가 없는 속성 그래프에서는 도구 자체보다 규율 쪽이 가져갈 만하다. coverage 대신 주장 단위 4칸 판정, 가정을 실은 판정 단어, `ran: false`로 남기는 꺼진 검사는 어떤 검색 평가에도 옮겨 쓸 수 있다. 1인 프로젝트에 기능 범위가 넓으니 변경 관리 루프와 증명서 검사기 외의 기능은 직접 확인한 뒤 쓰는 편이 낫다.
