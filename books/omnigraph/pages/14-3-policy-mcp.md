정책 컴파일과 판정, MCP 표면을 본다.

## Cedar 정책

`omnigraph-policy` 는 YAML 규칙을 Cedar 정책으로 컴파일하고 판정한다. 같은 판정을 서버 핸들러(`authorize_request`)와 엔진(`enforce`)이 각자 부른다.

### 행동과 범위

`PolicyAction` 은 열여섯 개 행동이다 ([lib.rs L16-L58](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-policy/src/lib.rs#L16-L58)). 행동마다 브랜치 범위를 쓰는지, 대상 브랜치 범위를 쓰는지, Cedar 의 어느 리소스에 걸리는지가 정해져 있다 ([lib.rs L83-L127](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-policy/src/lib.rs#L83-L127)).

| 행동 | 범위 | 리소스 | 사용처 예 |
|---|---|---|---|
| `read`, `export`, `change` | 브랜치 | Graph | 질의, 기준선, mutate |
| `maintain` | 브랜치 | Graph | optimize, 색인 빌드 |
| `reclaim` | 없음 | Graph | cleanup |
| `schema_apply`, `branch_delete` | 대상 브랜치 | Graph | 스키마 적용, 브랜치 삭제 |
| `branch_create`, `branch_merge` | 브랜치와 대상 브랜치 | Graph | 분기, 병합 |
| `graph_list` | 없음 | Server | 그래프 목록 |
| `config_manage` | 없음 | Cluster | 클러스터 배포 |

엔진 쪽 범위는 `ResourceScope` 네 형태가 `(branch, target_branch)` 쌍으로 내려가 HTTP 층과 같은 판정을 받는다 ([lib.rs L1040-L1084](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-policy/src/lib.rs#L1040-L1084)). 규칙 파일은 `version`, 행위자 `groups`, `protected_branches`, `rules` 로 구성된다 ([lib.rs L228-L238](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-policy/src/lib.rs#L228-L238)).

### 코드 분석: compile_policy_source

파일: `crates/omnigraph-policy/src/lib.rs`, [lib.rs L902-L912](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-policy/src/lib.rs#L902-L912)

YAML 규칙 하나가 Cedar `permit` 문이 되는 모양을 본다.

```rust
    format!(
        r#"permit (
    principal in Omnigraph::Group::{group},
    action == Omnigraph::Action::{action},
    resource == {resource_literal}
){when};"#,
        group = cedar_literal(&rule.allow.actors.group),
        action = cedar_literal(action.as_str()),
        when = when,
        resource_literal = resource_literal,
    )
```

규칙의 행동마다 `permit` 문이 하나씩 생긴다. `branch_scope` 가 `protected` 면 `when { context.has_branch && context.branch_is_protected }` 조건이 붙는다 ([lib.rs L915-L925](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-policy/src/lib.rs#L915-L925)). 컴파일은 정책 스키마를 읽고 엔티티와 정책을 만든 뒤 Cedar `Validator` 로 strict 검증을 통과해야 끝난다 ([lib.rs L503-L539](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-policy/src/lib.rs#L503-L539)). 판정은 모든 `permit` 이 없으면 거부되는 Cedar 기본 동작을 따른다.

### 코드 분석: authorize 의 요청 구성

파일: `crates/omnigraph-policy/src/lib.rs`, [lib.rs L628-L644](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-policy/src/lib.rs#L628-L644)

Cedar 요청의 리소스와 context 를 만드는 부분을 본다.

```rust
        let resource = match request.action.resource_kind() {
            PolicyResourceKind::Server => entity_uid("Server", SERVER_RESOURCE_ID)?,
            PolicyResourceKind::Graph => entity_uid("Graph", &self.graph_id)?,
            PolicyResourceKind::Cluster => entity_uid("Cluster", CLUSTER_RESOURCE_ID)?,
        };
        let context_value = json!({
            "has_branch": request.branch.is_some(),
            "branch": request.branch.clone().unwrap_or_default(),
            "has_target_branch": request.target_branch.is_some(),
            "target_branch": request.target_branch.clone().unwrap_or_default(),
            "branch_is_protected": request.branch.as_ref().is_some_and(|branch| self.protected_branches.contains(branch)),
            "target_branch_is_protected": request.target_branch.as_ref().is_some_and(|branch| self.protected_branches.contains(branch)),
        });
        let context = Context::from_json_value(context_value, Some((&self.schema, &action)))?;
        let cedar_request = Request::new(principal, action, resource, context, Some(&self.schema))?;
        let response =
            Authorizer::new().is_authorized(&cedar_request, &self.policies, &self.entities);
```

보호 브랜치 여부는 Cedar 엔티티가 아니라 context 의 불리언으로 전달된다. 그 전에 브랜치 이름이 정규형인지 확인하고, 규칙 파일의 그룹 어디에도 없는 행위자는 Cedar 를 부르지 않고 바로 거부한다 ([lib.rs L602-L620](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-policy/src/lib.rs#L602-L620)). 허용된 경우 근거가 된 규칙 id 를 `PolicyDecision` 에 담는다 ([lib.rs L663-L671](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-policy/src/lib.rs#L663-L671)).

### 코드 분석: 엔진의 enforce

파일: `crates/omnigraph/src/db/omnigraph.rs`, [omnigraph.rs L1197-L1219](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph/src/db/omnigraph.rs#L1197-L1219)

엔진이 정책 크레이트를 부르는 단일 지점을 본다.

```rust
    pub(crate) fn enforce(
        &self,
        action: omnigraph_policy::PolicyAction,
        scope: &omnigraph_policy::ResourceScope,
        actor: Option<&str>,
    ) -> Result<()> {
        let (branch, target_branch) = scope.to_branch_pair();
        for name in [branch, target_branch].into_iter().flatten() {
            ensure_canonical_branch_name(name)?;
        }
        let Some(checker) = self.policy.as_ref() else {
            return Ok(());
        };
        let Some(actor) = actor else {
            return Err(OmniError::Policy(format!(
                "policy denied action '{action}': a policy is installed and the request names \
                 no actor"
            )));
        };
        checker
            .check(action, scope, actor)
            .map_err(|err| OmniError::Policy(err.to_string()))
    }
```

정책이 설치되지 않은 독립 그래프는 모든 행동을 허용한다. 정책이 있는데 행위자가 없으면 거부한다. 엔진은 Cedar 를 직접 알지 못하고 `PolicyChecker` 트레이트만 본다 ([lib.rs L1117-L1127](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-policy/src/lib.rs#L1117-L1127)). 13장의 `optimize_all_datasets` 가 `maintain` 을, `cleanup_all_datasets` 가 `reclaim` 을 부르는 곳이 이 함수다. 사용자 문서는 행위자가 스스로 선언한 값이므로 직접 저장소 접근에서는 저장소 권한이 신뢰 경계로 남는다고 적는다 ([maintenance.md](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/docs/user/operations/maintenance.md)).

## MCP 표면

`omnigraph-mcp` 는 리스너도, 그래프를 여는 코드도, 인증 권한도 없는 프로토콜 크레이트다. 머리 주석은 `Host` 가 인증된 요청 맥락을 공급하고 기존 그래프 핸들러로 일을 시작한다고 적는다 ([lib.rs L1-L6](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-mcp/src/lib.rs#L1-L6)). [RFC 0003](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/docs/rfcs/0003-mcp-server-surface.md) 은 모든 도구가 HTTP 경로와 같은 핸들러를 같은 행위자와 같은 Cedar 행동으로 부르며, 도구가 목록에 보이는 것이 권한을 뜻하지 않는다고 정한다. 이 RFC 의 상태는 draft, 구현은 partial 이다.

### 코드 분석: Host 트레이트

파일: `crates/omnigraph-mcp/src/lib.rs`, [lib.rs L90-L109](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-mcp/src/lib.rs#L90-L109)

MCP 크레이트가 서버에 요구하는 계약을 본다.

```rust
pub trait Host: Clone + Send + Sync + 'static {
    type Context: Send + 'static;
    /// Project a protocol failure through the same error contract as handlers.
    /// `write` preserves the need to reconcile an unfinished write or a write
    /// whose result cannot be delivered. The returned error must be bounded.
    fn error(&self, error: ProtocolError, write: bool) -> HostError;
    fn admit(&self, parts: Option<&Parts>) -> Result<Self::Context, AdmissionError>;
    fn start_tool(
        &self,
        context: Self::Context,
        request: ToolRequest,
        permit: OwnedSemaphorePermit,
    ) -> Pending<ToolResult>;
    fn start_resource(
        &self,
        context: Self::Context,
        request: ResourceRequest,
        permit: OwnedSemaphorePermit,
    ) -> Pending<ResourceOutput>;
}
```

`start_tool` 은 결과를 기다리는 future 가 아니라 이미 시작된 작업의 관찰자 `Pending` 을 돌려준다. 관찰자를 버려도 작업은 취소되지 않는다 ([lib.rs L58-L60](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-mcp/src/lib.rs#L58-L60)). `observe` 는 호출자의 대기만 기한으로 자르고, 호스트가 소유한 작업의 수명은 건드리지 않는다 ([lib.rs L219-L234](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-mcp/src/lib.rs#L219-L234)). 클라이언트가 끊겨도 쓰기가 중간에 끊기지 않게 하려는 구조다.

| 상수 | 값 | 의미 |
|---|---|---|
| `REQUEST_BYTES` | 64 KiB | MCP 요청 본문 상한 |
| `RESULT_BYTES` | 1 MiB | 도구 결과 인코딩 상한 |
| `READ_DEADLINE`, `WRITE_DEADLINE` | 30초 | 호출자 대기 기한 |
| 동시 도구 슬롯 | 16 | `Semaphore::new(16)` |

상수는 [lib.rs L48-L56](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-mcp/src/lib.rs#L48-L56), 슬롯은 [lib.rs L141-L145](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-mcp/src/lib.rs#L141-L145) 에 있다. 전송은 레거시 세션 모드를 끈 Streamable HTTP 이고 JSON 응답을 쓴다 ([lib.rs L115-L158](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-mcp/src/lib.rs#L115-L158)).

### 도구 카탈로그와 서버 쪽 호스트

도구는 `TOOLS` 상수 배열에 24개가 정의되어 있다. 읽기 도구는 `health`, `graphs`, `schema`, `queries`, `query`, `snapshot`, `access`, 브랜치와 커밋 조회, `diff`, `changes_poll`, `blobs_stat` 이고, 쓰기 도구는 `mutate`, `load`, `blobs_clear`, 브랜치 생성, 삭제, 병합, 커밋 게시와 폐기다 ([catalog.rs L40-L233](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-mcp/src/catalog.rs#L40-L233)). 쓰기 도구 설명에는 서버가 재시도하지 않으며 `outcome_unknown` 이면 대상을 읽어 확인한 뒤 다시 보내라는 계약 문구가 붙는다 ([catalog.rs L38](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-mcp/src/catalog.rs#L38)).

서버의 `mcp::router` 는 이 라우터를 HTTP 와 같은 `require_bearer_auth`, `require_expected_actor`, 입장 제어 아래에 마운트한다. OIDC 리소스가 없으면 브라우저 `Origin` 헤더가 있는 요청을 거절한다 ([mcp.rs L62-L99](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-server/src/mcp.rs#L62-L99)). `ServerHost::start_tool` 은 요청을 서버의 기존 실행 경로로 넘긴다 ([mcp.rs L223-L236](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-server/src/mcp.rs#L223-L236)). CLI 의 `omnigraph mcp` 는 서버를 띄우지 않고, 에이전트의 stdio MCP 를 기존 서버의 `/mcp` 로 중계하기만 한다 ([cli.rs L234-L242](https://github.com/ModernRelay/omnigraph/blob/00b8cca842f2bf009bebeefcf765672b5ff64b26/crates/omnigraph-cli/src/cli.rs#L234-L242)).
