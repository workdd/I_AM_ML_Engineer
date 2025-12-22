# Subagents로 개인 비서 시스템 구축하기

- **출처**: [LangChain 공식 문서](https://docs.langchain.com/oss/python/langchain/multi-agent/subagents-personal-assistant)
- **저자**: LangChain 팀
- **읽은 날짜**: 2024-12-19
- **태그**: #LangChain #MultiAgent #Supervisor #에이전트

## 핵심 내용

**Supervisor 패턴**을 활용한 다중 에이전트 시스템 구축 가이드.

### 기존 방식의 문제점
> 단일 에이전트가 모든 도구에 접근 → 수백 개 도구 선택, 복잡한 API 형식 관리, 성능 저하

### Supervisor 패턴이란?
> "중앙 감독자가 전문화된 워커 에이전트들을 조율하는" 다중 에이전트 구조

## 시스템 아키텍처 (3계층)

```
┌─────────────────────────────────────┐
│         Supervisor Agent            │  ← 고수준 라우팅, 결과 통합
│    (schedule_event, manage_email)   │
└──────────────┬──────────────────────┘
               │
    ┌──────────┴──────────┐
    ▼                     ▼
┌─────────────┐    ┌─────────────┐
│  Calendar   │    │   Email     │  ← 자연어 → API 변환
│   Agent     │    │   Agent     │
└──────┬──────┘    └──────┬──────┘
       │                  │
       ▼                  ▼
┌─────────────┐    ┌─────────────┐
│ create_event│    │ send_email  │  ← 정확한 형식 요구
│ get_slots   │    │ draft_email │
└─────────────┘    └─────────────┘
```

| 계층 | 역할 | 특징 |
|------|------|------|
| 상위 | Supervisor | 고수준 라우팅, 결과 통합 |
| 중간 | Subagent | 자연어 → API 호출 변환 |
| 하위 | API Tools | 정확한 형식 요구 |

## 구현 단계

### 1. 저수준 API 도구 정의
```python
# 예시
create_calendar_event(title, start_time, end_time)
send_email(to, subject, body)
get_available_time_slots(date)
```

### 2. 특화된 서브 에이전트 구성
- **캘린더 에이전트**: 자연어 → ISO 형식 변환, 일정 생성
- **이메일 에이전트**: 자연어 → 전문적 이메일 작성

### 3. 서브 에이전트를 상위 도구로 래핑
```python
# 저수준 API가 아닌 고수준 도구로 노출
schedule_event = CalendarAgent.as_tool()
manage_email = EmailAgent.as_tool()
```

### 4. Supervisor 에이전트 생성
- 다중 도메인 요청 분해
- 적절한 도구 호출
- 결과 통합

## 활용 사례

### 단순 요청
```
User: "내일 오전 9시에 팀 스탠드업 일정 잡아줘"
→ Supervisor → schedule_event → Calendar Agent → API
```

### 복합 요청
```
User: "디자인팀과 화요일 2시 회의하고, 목업 검토 메일 보내줘"
→ Supervisor → schedule_event + manage_email → 각각 처리
```

## 보안: Human-in-the-Loop

```python
# middleware + checkpointer 활용
- 캘린더 이벤트 생성 전 승인 요청
- 이메일 발송 전 편집/거부 가능
```

## 인상 깊은 부분

### 관심사 분리의 장점
| 장점 | 설명 |
|------|------|
| 관심사 분리 | 각 에이전트가 고유 책임 담당 |
| 확장성 | 새 도메인 추가 용이 |
| 독립 테스트 | 각 계층 개별 검증 가능 |
| 성능 최적화 | 도메인별 정교한 프롬프트 적용 |

### 주의사항
> 서브 에이전트는 최종 응답에 **모든 관련 정보를 포함**해야 함
> Supervisor가 중간 과정을 볼 수 없으므로 완전한 확인 메시지 필수

## 내 생각 / 적용점

### 배운 점
1. **계층 분리**가 핵심 - Supervisor는 라우팅만, Subagent는 변환만
2. **도구 개수 폭발** 문제 해결 - 도메인별 에이전트로 분리
3. **Human-in-the-loop** 필수 - 중요 작업에 승인 단계

### 실무 연결
- 사내 챗봇에 적용 가능
  - HR Agent, IT Agent, Finance Agent 등 분리
  - Supervisor가 의도 파악 후 적절한 Agent 호출
- LangGraph로 구현 시 상태 관리 용이

### 다른 패턴과 비교
| 패턴 | 특징 |
|------|------|
| **Supervisor** | 중앙 집중 라우팅 (이 문서) |
| **Handoff** | 에이전트 간 직접 대화 |
| **Hierarchical** | 다단계 Supervisor |

### 후속 학습
- [ ] LangGraph로 Supervisor 패턴 구현 실습
- [ ] Handoff 패턴 학습
- [ ] CrewAI vs LangGraph 비교
