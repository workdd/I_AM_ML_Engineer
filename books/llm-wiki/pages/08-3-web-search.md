채팅과 딥 리서치가 함께 쓰는 웹 검색 공급자 구성을 본다.

## 웹 검색 공급자

채팅 입력창의 웹 검색 토글을 켜면 요청의 `tools.web`이 `true`가 되고, 에이전트는 `web.search` 도구를 쓸 수 있다. 라우터는 이 토글이 켜져 있으면 시스템 프롬프트에 "web.search is available" 힌트를 넣을 뿐이고([router.rs#L95](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/agent/router.rs#L95)), 실제로 검색할지는 모델이 루프에서 정한다. `faithful` 모드에서는 토글과 상관없이 등록하지 않는다.

공급자 선택은 설정 화면의 `SearchApiConfig.provider` 하나로 정한다. 타입이 허용하는 값은 `tavily`, `serpapi`, `searxng`, `ollama`, `brave`, `bocha`, `firecrawl`, `none`이다([wiki-store.ts#L65-L73](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/stores/wiki-store.ts#L65-L73)). README는 Tavily, SerpApi, SearXNG 세 가지를 소개하지만 코드에는 이후 추가된 공급자가 더 있다. `resolveSearchConfig`는 공급자별 설정(`providerConfigs`)에서 활성 공급자의 API 키, SerpApi 엔진, SearXNG URL과 카테고리를 꺼내 하나의 평면 설정으로 만든다([web-search.ts#L45-L108](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/web-search.ts#L45-L108)). 예전처럼 최상위 `apiKey` 하나만 있는 설정도 이 단계에서 공급자별 형태로 옮긴다.

| 공급자 | 필요한 설정 | 비고 |
| --- | --- | --- |
| `tavily`, `serpapi`, `brave`, `bocha` | API 키 | SerpApi는 Google, News, Scholar, Patents, Bing, DuckDuckGo 등 엔진 선택 |
| `searxng` | 인스턴스 URL | 카테고리 기본값 `general` |
| `ollama` | API 키 | 기본 URL `https://ollama.com` |
| `firecrawl` | 없음 (키 선택) | 기본 URL `https://api.firecrawl.dev` |

설정 검사는 프론트 `webSearch`가 먼저 하고([web-search.ts#L130-L162](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/web-search.ts#L130-L162)), 실제 HTTP 요청은 Rust가 한다. 프론트 `webSearch`는 Tauri command `web_search`를 부르고, 이 command와 에이전트의 `web.search` 도구가 같은 `run_web_search` 함수를 공유한다([external_search.rs#L19-L38](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/commands/external_search.rs#L19-L38)). 주석은 공급자와 네트워크 로직을 Rust로 옮기되, 딥 리서치(10장) 같은 기존 프론트 코드가 쓰던 `{ title, url, snippet, source }` 형태는 유지한다고 설명한다.

코드 분석: 공급자 문자열로 구현을 고르는 분기를 본다. [tools.rs#L1295-L1319](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/agent/tools.rs#L1295-L1319) 중 일부

```rust
    let config = config.resolved();
    let provider = config.provider.trim().to_ascii_lowercase();
    if provider.is_empty() || provider == "none" {
        return Err("Web search provider is not configured.".to_string());
    }
    let max_results = web_search_result_limit(&provider, top_k);
    let raw = match provider.as_str() {
        "firecrawl" => firecrawl_search(&client, query, &config, max_results).await?,
        "searxng" => searxng_search(&client, query, &config, max_results).await?,
        "tavily" => tavily_search(&client, query, &config, max_results).await?,
        "ollama" => ollama_search(&client, query, &config, max_results).await?,
        "brave" => brave_search(&client, query, &config, max_results).await?,
        "bocha" => bocha_search(&client, query, &config, max_results).await?,
        "serpapi" => serpapi_search(&client, query, &config, max_results).await?,
```

공급자는 한 번에 하나만 쓰고, 실패하면 다른 공급자로 넘어가지 않고 오류를 돌려준다. 결과 수는 Bocha만 1..50, 나머지는 1..20으로 clamp하고([tools.rs#L1322-L1327](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/agent/tools.rs#L1322-L1327)), 요청 timeout은 30초다. HTTP client는 앱의 프록시 설정을 따른다. 웹 결과는 `AgentReference`로 바뀌어 위키 참조와 같은 참고 자료 목록에 섞이지만, 위키에 저장되지는 않는다. 웹에서 찾은 내용을 위키로 쌓으려면 Save to Wiki로 답을 저장하거나 딥 리서치(10장)를 쓴다.
