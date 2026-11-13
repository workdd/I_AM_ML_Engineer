에이전트 런타임, LLM 공급자, MCP와 로컬 API를 다룬다.

| 장 | 핵심 |
|---|---|
| 11. 에이전트 런타임과 도구 | LLM Wiki의 채팅은 브라우저 안의 TypeScript 루프가 아니라 Rust 백엔드의 에이전트 런타임에서 돈다. |
| 12. LLM 공급자 연결: API와 CLI 전송 | LLM Wiki는 공급자 9종을 하나의 `LlmConfig`로 다룬다. |
| 13. MCP 서버와 로컬 API 서버 | LLM Wiki는 데스크톱 앱 안에 로컬 HTTP API 서버를 띄우고, 그 위에 얇은 MCP 서버 계층을 두어 Claude Code나 Codex 같은 외부 에이전트가 위키를 검색하고, 읽고, 쓰고, 백엔드 에이전트와 대화하게 한다. |
