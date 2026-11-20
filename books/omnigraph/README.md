# Omnigraph 해부: 브랜치되는 타입 그래프 DB 코드 분석

[ModernRelay/omnigraph](https://github.com/ModernRelay/omnigraph)는 노드와 엣지 타입마다 Lance dataset 하나를 두고, `__manifest` 테이블 하나로 그래프 전체의 버전과 브랜치를 묶는 객체 저장소 기반 그래프 데이터베이스다. 이 책은 Rust 핵심 엔진 코드를 따라가며 스키마 컴파일, 쿼리 계획과 실행, 벡터와 전문 검색, 쓰기와 적재, 매니페스트 게시, 브랜치 병합, 변경 피드가 어떻게 맞물리는지 설명한다.

- 기준 커밋: [`00b8cca`](https://github.com/ModernRelay/omnigraph/tree/00b8cca842f2bf009bebeefcf765672b5ff64b26) (v0.16.0, 2026-11-19)
- 원본 라이선스: MIT. 코드 인용은 해설에 필요한 만큼 짧게 하고 원문 링크를 함께 건다
- 원고 저장소: [workdd/I_AM_ML_Engineer](https://github.com/workdd/I_AM_ML_Engineer) `books/omnigraph`
