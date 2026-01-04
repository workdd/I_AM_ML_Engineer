# JSON vs TOON: LLM 입력 포맷만 바꿔도 토큰 효율은 달라질까

- **출처**: [NAVER Cloud Platform Forum](https://www.ncloud-forums.com/topic/594/)
- **저자**: CLOVA Studio 운영자
- **게시일**: 2026-01-02
- **읽은 날짜**: 2026-01-04
- **태그**: #TOON #JSON #토큰효율 #LLM #HyperCLOVA

## 핵심 내용

### 문제 제기
- JSON의 반복적인 key-value 구조가 과도한 토큰 소모
- 엔터프라이즈 환경에서 토큰 사용량 = 비용 + 사용자 경험에 직접 영향

### TOON (Token-Oriented Object Notation)이란?

**설계 철학**: 데이터 구조의 명확성과 이해 품질을 유지하면서 토큰 소비 감소

**핵심 아이디어**:
- 반복되는 객체 리스트를 **테이블 형태**로 표현
- 필드명을 **한 번만 선언**하고 값을 행 단위로 나열
- 들여쓰기로 계층 구조 표현
- 중복 키 반복 제거

### JSON vs TOON 예시

**JSON**:
```json
[
  {"name": "Alice", "age": 30, "city": "Seoul"},
  {"name": "Bob", "age": 25, "city": "Busan"},
  {"name": "Carol", "age": 35, "city": "Daegu"}
]
```

**TOON** (예상 형태):
```
name    age  city
Alice   30   Seoul
Bob     25   Busan
Carol   35   Daegu
```

## 벤치마크 결과

### GitHub 벤치마크
| 지표 | JSON | TOON |
|------|------|------|
| 토큰 사용량 | 기준 | **39.6% 감소** |
| 정확도 | 69.7% | **73.9%** |

### HyperCLOVA X 실험 결과

| 태스크 유형 | 결과 |
|------------|------|
| **단순 구조화 데이터** | 프롬프트 토큰 ~27.3% 감소 |
| **추론 태스크 (KMMLU)** | JSON이 더 안정적 |
| **RAG 기반 태스크** | TOON이 요약, 비교, 정보 추출에서 안정적 |
| **API/로그 분석** | 반복 데이터 패턴에서 TOON 효율적 |

## TOON이 효과적인 경우

```
✅ 단순하고 반복적인 데이터 구조
✅ 요약, 비교, 정보 추출 태스크
✅ API 응답, 로그 데이터 분석
✅ 토큰 비용이 중요한 대규모 서비스
```

## TOON이 덜 효과적인 경우

```
❌ 복잡한 추론이 필요한 태스크
❌ 중첩 구조가 깊은 데이터
❌ 스키마가 자주 변하는 데이터
```

## 인상 깊은 부분

> "포맷 최적화만으로도 특정 태스크 유형에서 토큰 효율성을 개선할 수 있다"

단순히 입력 형식을 바꾸는 것만으로 **39.6% 토큰 절감 + 정확도 향상**이 가능하다는 점이 인상적

## 실무 적용 포인트

### 1. 태스크별 포맷 선택
- 단순 데이터 처리: TOON 고려
- 복잡한 추론: JSON 유지

### 2. 비용 최적화 관점
- 대규모 서비스에서 토큰 39.6% 절감 = 상당한 비용 절감
- 특히 반복 호출이 많은 API에서 효과적

### 3. Context Engineering과 연결
- [Context Engineering](./[20251230]%20SKdevocean_Context_Engineering_핵심역량.md)에서 다룬 토큰 효율화의 구체적 방법
- [컬리 OMS](./[20251230]%20컬리_OMS_Claude_AI_워크플로우.md)에서 JSON DSL로 3배 압축한 것과 유사한 접근

### 4. 적용 시 고려사항
- 기존 시스템과의 호환성
- 파싱 로직 변경 필요성
- 태스크별 A/B 테스트 권장

## 연관 글

- [Context Engineering 핵심역량](./[20251230]%20SKdevocean_Context_Engineering_핵심역량.md) - 토큰 예산 관리
- [컬리 OMS Claude AI 워크플로우](./[20251230]%20컬리_OMS_Claude_AI_워크플로우.md) - JSON DSL 압축 사례
