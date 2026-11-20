Omnigraph의 기본 개념과 크레이트 구성을 다룬다.

| 장 | 핵심 |
|---|---|
| 01. Omnigraph 개념: 타입 그래프, 브랜치, Lance의 대응 | Omnigraph의 핵심은 그래프 하나를 타입별 Lance dataset 여러 개로 나눠 저장하고, 그 위에 `__manifest`라는 Lance 테이블 하나를 두어 "지금 그래프에서 보이는 각 테이블의 버전"을 한 번에 바꾸는 것이다. |
| 02. 크레이트 구성과 요청 경로 | Omnigraph는 Cargo 워크스페이스의 20개 크레이트로 나뉘어 있고, 엔진 크레이트 `omnigraph-engine`이 컴파일러, 계획기, 카탈로그, 코어, 저장 크레이트를 묶어 `Omnigraph` 핸들과 `Session`으로 공개한다. |
