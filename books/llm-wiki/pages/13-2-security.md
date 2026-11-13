로컬 API의 보호 장치인 토큰 인증, 바인딩 주소와 CORS, 경로 제한을 본다.

## 인증

### 토큰의 출처와 비교

토큰은 환경 변수 `LLM_WIKI_API_TOKEN`이 있으면 그것을, 없으면 `app-state.json`의 `apiConfig.token`을 쓴다([`api_server.rs#L623-L637`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/api_server.rs#L623-L637)). 설정 화면의 생성 버튼은 12장의 `generateApiToken`으로 256비트 base64url 값을 만든다.

코드 분석: [`src-tauri/src/api_server.rs#L593-L621`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/api_server.rs#L593-L621) (발췌)

무엇을 보는지: 질의 문자열, 전용 헤더, Bearer 헤더 세 곳을 상수 시간 비교로 확인한다.

```rust
    let Some(token) = api_token(app) else {
        return false;
    };
    let params = parse_query(query);
    if params
        .get("token")
        .map(|v| constant_time_eq(v.as_bytes(), token.as_bytes()))
        .unwrap_or(false)
    {
        return true;
    }
    headers.iter().any(|(key, value)| {
        if key == "x-llm-wiki-token" {
            return constant_time_eq(value.as_bytes(), token.as_bytes());
        }
        if key == "authorization" {
            return value
                .strip_prefix("Bearer ")
                .map(|v| constant_time_eq(v.as_bytes(), token.as_bytes()))
```

토큰이 설정되지 않았으면 무조건 거짓이다. 그래서 새로 설치한 상태는 "API 켜짐 + 토큰 없음 = 401"이 되어 닫힌 쪽으로 실패한다고 `api_enabled` 주석이 설명한다([`api_server.rs#L687-L703`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/api_server.rs#L687-L703)). `constant_time_eq`는 길이가 달라도 긴 쪽 끝까지 XOR를 누적해 비교 시간이 일치 위치에 따라 달라지지 않게 한다([`api_server.rs#L716-L725`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/api_server.rs#L716-L725)).

> **주의**: `?token=` 질의 문자열도 허용된다. 브라우저 확장처럼 헤더를 붙이기 어려운 클라이언트를 위한 선택으로 보이지만(추정), URL에 실린 토큰은 셸 기록이나 프록시 로그에 남을 수 있다. MCP README는 토큰을 명령줄 인자가 아닌 환경 변수로 넘기라고 권한다([`mcp-server/README.md#L77`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/mcp-server/README.md#L77)).

### 설정 스위치

인증과 노출에 관한 설정은 모두 `app-state.json`의 `apiConfig` 아래에 있고, 서버는 이 파일을 5초 캐시로 읽는다([`api_server.rs#L727-L753`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/api_server.rs#L727-L753)). 설정을 저장하면 앱이 `invalidate_config_cache`를 불러 캐시를 비운다([`lib.rs#L58`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/lib.rs#L58)).

| 키 | 설정 없을 때 | 효과 | 검사하는 곳 |
|---|---|---|---|
| `enabled` | 참 | 거짓이면 `/health` 말고 모두 503 | `api_enabled` |
| `token` | 없음 | 토큰 값. 환경 변수가 우선 | `api_token` |
| `allowUnauthenticated` | 거짓 | 참이면 토큰 필수 경로를 뺀 나머지를 토큰 없이 허용 | `api_allow_unauthenticated` |
| `allowLanAccess` | 거짓 | 참이면 재시작 후 `0.0.0.0`에 바인딩 | `server_bind.rs` |
| `mcpEnabled` | 거짓 | `/health` 응답에만 실린다. 차단은 MCP 서버가 한다 | `api_mcp_enabled` |

`enabled`의 기본값이 참인 이유는 주석에 있다. 끄기 스위치가 생기기 전부터 환경 변수 토큰만으로 쓰던 사용자의 설정을 깨지 않으려는 것이다.

> **주의**: `mcpEnabled`는 HTTP 서버가 강제하지 않는다. `api_mcp_enabled`의 호출처는 `/health` 응답 하나뿐이고([`api_server.rs#L322`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/api_server.rs#L322)), MCP 서버가 도구마다 `/health`를 읽어 꺼져 있으면 스스로 거절한다. 토큰을 가진 클라이언트가 HTTP를 직접 부르는 것은 이 스위치로 막히지 않는다.

## 바인딩 주소와 CORS

### 바인딩

기본 바인딩은 `127.0.0.1`이다. 바꾸는 방법은 두 가지다.

코드 분석: [`src-tauri/src/server_bind.rs#L6-L31`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/server_bind.rs#L6-L31) (발췌)

무엇을 보는지: 환경 변수가 먼저, 그다음 LAN 허용 스위치, 둘 다 없으면 루프백.

```rust
const DEFAULT_BIND_HOST: &str = "127.0.0.1";
const PUBLIC_BIND_HOST: &str = "0.0.0.0";
const BIND_HOST_ENV: &str = "LLM_WIKI_BIND_HOST";

pub fn configured_bind_host(app: &AppHandle) -> String {
    configured_env_bind_host()
        .or_else(|| configured_store_bind_host(app))
        .unwrap_or_else(|| DEFAULT_BIND_HOST.to_string())
}
```

`LLM_WIKI_BIND_HOST`는 영숫자와 `.`, `-`, `_`, `:`, `[`, `]`만 허용해 `0.0.0.0:19828/path`나 `127.0.0.1;rm` 같은 주입을 거절한다([`server_bind.rs#L41-L54`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/server_bind.rs#L41-L54)). IPv6 주소는 대괄호로 감싼다. 설정 화면의 LAN 허용 스위치 설명은 "앱을 재시작하면 API와 Clip 서버가 `0.0.0.0`에서 듣는다. 신뢰하는 네트워크에서만 쓰고, LAN을 완전히 믿지 않으면 토큰 인증을 유지하라"고 적는다([`api-server-section.tsx#L330-L338`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/components/settings/sections/api-server-section.tsx#L330-L338)).

> **주의**: 서버는 요청의 원격 주소를 검사하지 않는다. LAN 허용과 무인증 허용을 함께 켜면 같은 네트워크의 누구나 읽기 경로와 리뷰 변경 · 재스캔 경로를 토큰 없이 부를 수 있다. 두 스위치를 묶는 서버 쪽 검사는 코드에서 찾지 못했다.

### CORS

브라우저에서 오는 요청은 `Origin` 헤더가 허용 목록에 있을 때만 `Access-Control-Allow-Origin`을 돌려받는다.

코드 분석: [`src-tauri/src/cors.rs#L11-L23`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/cors.rs#L11-L23)

무엇을 보는지: 브라우저 확장, 로컬호스트, Tauri 웹뷰만 허용한다.

```rust
pub fn is_allowed_browser_origin(origin: &str) -> bool {
    origin.starts_with("chrome-extension://")
        || origin.starts_with("moz-extension://")
        || origin == "http://localhost"
        || origin.starts_with("http://localhost:")
        || origin == "http://127.0.0.1"
        || origin.starts_with("http://127.0.0.1:")
        || origin == "http://[::1]"
        || origin.starts_with("http://[::1]:")
        || origin == "tauri://localhost"
        || origin == "http://tauri.localhost"
        || origin == "https://tauri.localhost"
}
```

허용되면 `Vary: Origin`과 `Access-Control-Allow-Private-Network: true`도 붙인다([`cors.rs#L25-L41`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/cors.rs#L25-L41)). 임의의 웹사이트가 사용자 브라우저에서 로컬 API 응답을 읽는 것을 막는 장치다. 다만 CORS는 응답 읽기를 막을 뿐 요청 전송 자체를 막지는 않으므로, 상태를 바꾸는 경로를 지키는 것은 결국 토큰이다. `chrome-extension://` 전체를 허용하므로 설치된 어떤 확장이든 응답을 읽을 수 있다는 점도 함께 기억해 둘 만하다.

## 경로 제한

파일 읽기는 두 겹으로 막는다. 먼저 공개 경로인지 본다.

코드 분석: [`src-tauri/src/api_server.rs#L1031-L1044`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/api_server.rs#L1031-L1044)

무엇을 보는지: `purpose.md`, `schema.md`, `wiki/`, `raw/sources/`만 열고 숨김 구간은 모두 거절한다.

```rust
fn is_public_project_rel(rel: &str) -> bool {
    let rel = normalize_path(rel).trim_start_matches('/').to_string();
    if rel
        .split('/')
        .any(|part| part.is_empty() || part.starts_with('.'))
    {
        return false;
    }
    let lower = rel.to_lowercase();
    lower == "purpose.md"
        || lower == "schema.md"
        || lower.starts_with("wiki/")
        || lower.starts_with("raw/sources/")
}
```

`.llm-wiki/` 아래의 프로젝트 설정, 세션, 리뷰 원본 파일은 이 검사에서 걸러진다. 리뷰는 전용 엔드포인트를 통해서만, 그것도 필드를 골라 담은 사본으로 나간다([`api_server.rs#L1341-L1359`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/api_server.rs#L1341-L1359)). 다음으로 `safe_join`이 절대 경로와 `..`, 루트 구성 요소를 거절하고, 대상이 있으면 canonical 경로가 프로젝트 안인지, 없으면 부모의 canonical 경로가 프로젝트 안인지 확인한다([`api_server.rs#L989-L1029`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/api_server.rs#L989-L1029)). 심볼릭 링크로 프로젝트 밖을 가리키는 경우가 여기서 걸린다. 마지막으로 확장자가 `md`, `mdx`, `txt`, `csv`, `json`, `yaml`, `yml`, `xml`, `html`, `htm`, `log` 중 하나여야 한다([`api_server.rs#L1046-L1057`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/api_server.rs#L1046-L1057)).

이 공개 경로 규칙은 에이전트 도구의 `is_public_read_rel`과 같은 내용으로 따로 구현되어 있다([`tools.rs#L2699-L2708`](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/agent/tools.rs#L2699-L2708)).
