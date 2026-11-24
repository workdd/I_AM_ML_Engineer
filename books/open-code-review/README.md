# OpenCodeReview 해부: AI 코드 리뷰 CLI 코드 분석

Alibaba가 사내 코드 리뷰 도구를 공개한 [OpenCodeReview](https://github.com/alibaba/open-code-review)(`ocr`)를 코드 단위로 따라가는 책이다. Git diff 수집, 리뷰 대상 파일 선별과 묶음, 규칙 매칭, 도구를 쓰는 리뷰 에이전트, 코멘트 위치 보정과 반성 필터, LLM 프로바이더 계층, 전체 파일 스캔, 동작 영향 분석, 세션 저장과 뷰어, SARIF와 위임 모드까지 실제 소스 위치를 링크로 걸어 설명한다.

- 기준 커밋: [`e74f2c6`](https://github.com/alibaba/open-code-review/tree/e74f2c6bd4422bd1e12d257d1725bcce1d6b9ddd) (2026-11-22)
- 원본 라이선스: Apache-2.0. 코드 인용은 짧게 하고 원문 링크를 함께 건다
- 원고 저장소: [workdd/I_AM_ML_Engineer](https://github.com/workdd/I_AM_ML_Engineer) `books/open-code-review`
