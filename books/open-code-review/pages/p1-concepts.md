OpenCodeReview의 설계 원칙과 CLI 진입점을 다룬다.

| 장 | 핵심 |
|---|---|
| 01. OpenCodeReview 개요: deterministic 파이프라인과 에이전트의 결합 | OpenCodeReview는 리뷰 과정을 "틀리면 안 되는 단계"와 "상황에 따라 판단해야 하는 단계"로 나누고, 앞쪽은 Go 코드로, 뒤쪽은 도구를 쓰는 LLM 에이전트로 처리한다. |
| 02. CLI 구조와 첫 리뷰 실행 | `ocr` 실행 파일은 cobra 기반 명령 트리 하나로 리뷰, 전체 스캔, 세션 조회, 설정, 위임 모드를 모두 제공한다. |
