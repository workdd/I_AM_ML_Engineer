# LLM Wiki 해부: 자동 정리형 지식 위키 앱 코드 분석

[nashsu/llm_wiki](https://github.com/nashsu/llm_wiki)는 문서를 넣으면 LLM이 요약 · 개체 · 개념 페이지를 만들고 서로 링크를 걸어, 질문할 때마다 다시 찾지 않고 계속 쌓이는 위키를 유지하는 데스크톱 앱이다. 이 책은 Tauri(Rust) 백엔드와 React 프론트엔드 코드를 따라가며 수집, 위키링크, 검색, 그래프, 린트, 에이전트 런타임, MCP 서버가 어떻게 맞물리는지 설명한다.

- 기준 커밋: [`e08cfce`](https://github.com/nashsu/llm_wiki/tree/e08cfce65aad8b76b13546a081c1a8d4422b70c3) (2026-10-21)
- 원본 라이선스: GPL-3.0. 코드 인용은 해설에 필요한 만큼 짧게 하고 원문 링크를 함께 건다
- 원고 저장소: [workdd/I_AM_ML_Engineer](https://github.com/workdd/I_AM_ML_Engineer) `books/llm-wiki`
