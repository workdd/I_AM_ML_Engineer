# Context Engineering: AI 시대의 새로운 핵심 역량

- **출처**: [SK devocean](https://devocean.sk.com/blog/techBoardDetail.do?ID=167772&boardType=techBlog)
- **저자**: calm.ardent
- **게시일**: 2025-09-09
- **읽은 날짜**: 2025-12-30
- **태그**: #ContextEngineering #PromptEngineering #LLM #ContextWindow #AgenticAI

## 핵심 내용

### Context Engineering이란?
- **정의**: LLM을 위한 전체 정보 환경을 체계적으로 설계하는 학문 분야
- **배경**: 1,400개 이상의 연구 논문이 존재
- **핵심**: 단순 프롬프트 작성을 넘어 **정보 전달 전체 시스템**을 설계

### Prompt Engineering vs Context Engineering

| 구분 | Prompt Engineering | Context Engineering |
|------|-------------------|---------------------|
| **초점** | 단일 프롬프트 최적화 | 전체 정보 환경 설계 |
| **범위** | 하나의 입출력 쌍 | 다중 소스 통합 |
| **접근법** | 수동, 창의적 글쓰기 | 토큰 관리 포함 체계적 설계 |

### 3가지 핵심 구성요소

```
Context Engineering
├── 1. Context Window
│   └── LLM이 처리할 수 있는 최대 토큰 길이
│       - GPT-4: 128K tokens
│       - Claude: 1M tokens
│       - Gemini: 1M tokens
│
├── 2. System Architecture
│   └── 프롬프트 + 대화 기록 + 검색 문서의 통합 구조
│
└── 3. Dynamic Assembly
    └── 실시간 정보 전달 최적화
```

### "Lost in the Middle" 현상

| 정보 위치 | 성능 |
|----------|------|
| **시작 부분** | 높음 |
| **중간 부분** | 20-25% 성능 저하 |
| **끝 부분** | 높음 |

**시사점**: 관련 정보를 컨텍스트의 시작이나 끝에 배치하는 전략 필요

## 실제 구현 사례

### Cursor AI
- IDE 통합 코드베이스 인덱싱
- Context-aware 제안
- 프로젝트 전체를 이해하고 코드 제안

### Claude Code
- `CLAUDE.md` 파일로 지속적 메모리 관리
- Agentic execution 지원
- 프로젝트별 컨텍스트 설정

## 인상 깊은 부분

> Context Engineering은 단순히 좋은 프롬프트를 작성하는 것이 아니라, LLM이 최적의 성능을 발휘할 수 있도록 **정보 환경 전체를 설계**하는 것

## 실무 적용 포인트

### 1. 토큰 예산 관리
- Context Window 크기 인지
- 필요한 정보만 선별적으로 포함
- JSON DSL 등 압축 형식 활용 (컬리 사례 참고)

### 2. 정보 배치 전략
- "Lost in the Middle" 현상 고려
- 중요한 정보는 **시작 또는 끝**에 배치
- 덜 중요한 정보는 중간에

### 3. 다중 소스 통합
- 시스템 프롬프트 + 대화 기록 + RAG 검색 결과
- 각 소스의 역할과 우선순위 명확히 정의

### 4. 동적 컨텍스트 조립
- 상황에 따라 필요한 정보만 동적으로 구성
- 불필요한 정보로 토큰 낭비 방지

## 연관 글

- [컬리 OMS Claude AI 워크플로우](./[20251230]%20컬리_OMS_Claude_AI_워크플로우.md) - AI Context 설계의 실무 사례
- [AWS Strands 스킬 시스템](./[20251225]%20AWS_Strands_스킬시스템_Claude_Code_스타일.md) - Knowledge/Action Layer 구조

## 추가 학습 키워드

- Context Window 최적화
- RAG (Retrieval-Augmented Generation)
- Long-context LLM 활용
- Token budgeting
