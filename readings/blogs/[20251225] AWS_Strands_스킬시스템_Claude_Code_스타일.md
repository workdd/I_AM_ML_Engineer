# Claude Code 스타일 스킬 시스템 (Strands Agent SDK)

- **출처**: [AWS Samples - Deep Insight Skill System](https://github.com/aws-samples/sample-deep-insight/blob/main/docs/features/skill-system.md)
- **저자**: AWS Samples
- **읽은 날짜**: 2025-12-25
- **태그**: #LLM #Agent #Skill-System #Claude-Code #Strands-SDK

## 핵심 내용

Strands Agent SDK를 위한 **동적 스킬 발견 및 로딩 메커니즘**. Claude Code의 스킬 아키텍처를 참고하여 에이전트가 필요할 때만 전문화된 지시사항을 로드할 수 있도록 설계됨.

### 핵심 장점

| 특징 | 설명 |
|------|------|
| **지연 로딩** | 시작 시점이 아닌 요청 시점에만 스킬 로드 |
| **자동 발견** | 디렉토리 구조에서 스킬 자동 감지 |
| **모듈식 설계** | 코드 수정 없이 새 스킬 추가 가능 |
| **토큰 효율성** | 활성 스킬만 컨텍스트 토큰 소비 |

### 작동 방식

**초기화 단계:**
1. `SkillDiscovery`가 `./skills/` 디렉토리 스캔 (메타데이터만)
2. 시스템 프롬프트에 `<available_skills>` 섹션 추가
3. 에이전트 준비 완료

**런타임 단계:**
1. 사용자 요청 분석
2. `skill_tool` 호출로 특정 스킬 로드
3. 로드된 `SKILL.md` 전체 내용 제공
4. 에이전트가 로드된 지시사항 따름

### 주요 구성요소

| 컴포넌트 | 역할 |
|----------|------|
| **SkillDiscovery** | 디렉토리 재귀 스캔, YAML 메타데이터 추출, 중복 감지 |
| **SkillLoader** | 지연 로드, 캐싱 없이 항상 최신 파일 읽기 |
| **SkillTool** | Strands SDK 호환 도구 래퍼 |
| **SkillUtils** | 초기화 조율 및 시스템 프롬프트 생성 |

### SKILL.md 파일 형식

```yaml
---
name: pdf
description: PDF 조작 도구 (텍스트/표 추출, 병합/분할, 양식 처리)
license: MIT
allowed-tools:
  - bash_tool
  - file_read
---

# PDF 처리 가이드

## 개요
...상세 지시사항...
```

**필수 필드:**
- `name`: 스킬 고유 식별자
- `description`: 스킬 목록에 표시되는 설명

**선택 필드:**
- `license`: 라이선스 정보
- `allowed-tools`: 사용 가능한 도구 목록

### 사용 예시

```python
from src.utils.skills.skill_utils import initialize_skills

# 1. 스킬 시스템 초기화
_, skill_prompt = initialize_skills(
    skill_dirs=["./skills"],
    verbose=True
)

# 2. 시스템 프롬프트 구성
system_prompt = base_prompt + skill_prompt

# 3. skill_tool 포함하여 에이전트 생성
agent = strands_utils.get_agent(
    agent_name="skill_agent",
    system_prompts=system_prompt,
    tools=[skill_tool, bash_tool, file_read, file_write],
    streaming=True
)
```

### 제공 스킬 예시

| 스킬 | 기능 |
|------|------|
| pdf | PDF 텍스트/표 추출, 문서 병합 분할 |
| docx | Word 문서 생성 및 처리 |
| xlsx | Excel 스프레드시트 분석 |
| pptx | PowerPoint 프레젠테이션 생성 |
| mcp-builder | MCP 서버 개발 |
| skill-creator | 새로운 스킬 작성 |

## 인상 깊은 부분

> "지연 로딩: 시작 시점이 아닌 요청 시점에만 스킬 로드"

토큰 효율성을 위해 필요한 스킬만 동적으로 로드하는 아키텍처가 인상적. LLM 에이전트의 컨텍스트 윈도우 한계를 고려한 실용적인 설계.

> "캐싱 없이 항상 최신 파일 읽기"

개발 중 스킬 수정 시 재시작 없이 바로 반영되는 점이 개발 경험(DX) 측면에서 좋음.

## 실무 적용 포인트

### 1. 사내 LLM 에이전트 스킬 시스템 구축 시 참고
- 도메인별 전문 스킬 정의 (예: 고객 응대, 데이터 분석, 코드 리뷰)
- SKILL.md 형식의 표준화된 스킬 정의 방식 도입

### 2. 토큰 최적화 전략
- 모든 지시사항을 시스템 프롬프트에 넣지 않고 필요 시 동적 로드
- 컨텍스트 윈도우 효율적 활용

### 3. 새 스킬 추가 워크플로우
1. `skills/my-skill/` 디렉토리 생성
2. `SKILL.md` 작성 (name, description 필수)
3. 상세 지시사항, 코드 예제, 모범 사례 추가
4. 에이전트 재시작 시 자동 발견

### 4. 스킬 작성 모범 사례
- 명확한 설명으로 사용 시점 판단 용이하게
- 구체적 코드 예제 포함 (에이전트 성능 향상)
- `##` 헤더로 구조화하여 네비게이션 용이
- 도메인별로 집중된 스킬 설계
