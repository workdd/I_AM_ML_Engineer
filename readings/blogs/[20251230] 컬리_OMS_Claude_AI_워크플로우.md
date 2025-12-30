# OMS에서 Claude AI를 활용하여 변화된 업무 방식

- **출처**: [컬리 기술 블로그](https://helloworld.kurly.com/blog/oms-claude-ai-workflow/)
- **저자**: 이준환
- **게시일**: 2025-12-24
- **읽은 날짜**: 2025-12-30
- **태그**: #ClaudeAI #AgenticWorkflow #MSA #CleanArchitecture #팀생산성

## 핵심 내용

### 배경: 4명이 16명처럼 일하기
- **팀 구성**: PM 1명 + 엔지니어 3명 (총 4명)
- **관리 시스템**: 12개 마이크로서비스
- **핵심**: 체계적인 AI Context 설계로 소규모 팀이 대규모 조직처럼 운영

### Role-Based AI 아키텍처

| AI 페르소나 | 역할 | 활용 |
|------------|------|------|
| **TPM AI** | 아키텍처 분석, 요구사항 검토 | 설계 단계, 영향도 분석 |
| **MSA AI** | 구현, 개발 | 코드 작성, 배포 |

각 AI에 맞춤화된 Knowledge Context를 로드하여 도메인 전문가처럼 동작

### AI Context 구조

```
AI Context
├── Knowledge Layer (지식)
│   ├── domain-overview.md    # 도메인 개요
│   ├── data-model.md         # 데이터 모델
│   └── API specs (JSON DSL)  # API 스펙
│
└── Action Layer (행동)
    ├── Development skills    # 개발 스킬
    └── Deployment workflows  # 배포 워크플로우
```

**JSON DSL 선택 이유**: 산문(prose) 대비 **3배 압축** 효율 + 명확성 유지

### 아키텍처가 AI 활용에 미치는 영향

#### Clean Architecture > Layered Architecture
- **이유**: 비즈니스 로직이 단일 목적의 UseCase 클래스로 격리
- **효과**: AI가 처리해야 할 불필요한 코드(context noise) 최소화

#### MSA의 장점
- 서비스 경계가 명확 → AI로 병렬 개발 시 충돌 없음
- 도메인 중심 지식 관리 용이

## 워크플로우 변화

### Before AI
```
PM이 전체 팀 소집 → 용량 계획 회의
→ 엔지니어들이 개별 서비스 사일로 작업
→ 코드 리뷰 (구현 세부사항 수정)
```

### After AI
```
엔지니어가 TPM AI와 독립적으로 요구사항 분석
→ 팀이 아키텍처 결정만 함께 검증
→ 코드 리뷰가 "AI 멘토링" 방식으로 전환
```

## 실질적 성과

| 영역 | 개선 내용 |
|------|----------|
| **설계 변경** | AI가 처음부터 빠르게 재설계 → 아키텍처 재작업 불필요 |
| **배포 요청** | 일 7-8건 수동 양식 → 단일 명령으로 자동화 |
| **코드 일관성** | 분산 개발에도 일관성 향상 |
| **리뷰 초점** | 문법 교정 → 아키텍처 검증으로 전환 |

## 인상 깊은 부분

> "AI가 아무리 똑똑해도, AI 답변을 꼼꼼히 검증하고 최종 책임을 지는 것은 인간 엔지니어의 몫입니다."

AI는 도구이고, **검증과 책임**은 여전히 인간의 몫

## 향후 비전

다른 도메인 팀으로 확장 시:
- 50+ 마이크로서비스를 분석할 수 있는 **Cross-functional TPM AI** 구축 가능
- 조직 전체 이니셔티브의 영향도 분석

## 실무 적용 포인트

### 1. AI Context 설계가 핵심
- 무작정 AI를 쓰는 게 아니라, **체계적인 Knowledge/Action 구조** 설계 필요
- JSON DSL로 정보 압축 → 토큰 효율성 ↑

### 2. 아키텍처가 AI 활용 효율에 영향
- Clean Architecture + MSA 조합이 AI 활용에 유리
- UseCase 격리 → AI가 집중해야 할 맥락 최소화

### 3. Role-Based AI 페르소나
- 하나의 범용 AI보다 역할별 전문 AI가 효과적
- TPM AI (설계) + MSA AI (구현) 분리

### 4. 팀 워크플로우 재설계
- AI 도입 = 단순 도구 추가가 아닌 **프로세스 자체의 변화**
- 코드 리뷰 목적이 "수정"에서 "검증/멘토링"으로 전환

## 연관 글

- [AWS Strands 스킬 시스템](./[20251225]%20AWS_Strands_스킬시스템_Claude_Code_스타일.md) - AI Agent의 Skill/Workflow 구조
- [LangChain Subagents Supervisor 패턴](./[20251222]%20LangChain_Subagents_Supervisor_패턴.md) - Multi-Agent 오케스트레이션
