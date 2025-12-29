# Visa Intelligent Commerce on AWS: Amazon Bedrock AgentCore로 구현하는 Agentic Commerce

- **출처**: [AWS Machine Learning Blog](https://aws.amazon.com/blogs/machine-learning/introducing-visa-intelligent-commerce-on-aws-enabling-agentic-commerce-with-amazon-bedrock-agentcore/)
- **저자**: Sangeetha Bharath, Seemal Zaman (Visa), Ankit Pathak, Daniela Vargas, Hardik Thakkar, Isaac Privitera, Ritambhara Chatterjee (AWS)
- **게시일**: 2025-12-23
- **읽은 날짜**: 2025-12-29
- **태그**: #AgenticAI #AmazonBedrock #AgentCore #Visa #MCP #MultiAgent #Payments

## 핵심 내용

### Agentic Commerce란?
- 기존: 사용자가 여러 앱/사이트를 오가며 수동으로 검색, 비교, 결제
- Agentic: AI 에이전트가 발견, 의사결정, 결제까지 자율적으로 처리
- 2000년대 초 이커머스 혁명과 유사한 패러다임 전환

### AWS + Visa 협업 배경
- **문제점**: AI 에이전트가 계획/비교까지는 가능했지만, 실제 결제는 사용자가 직접 해야 했음
- **해결책**: Visa Intelligent Commerce (2025년 4월 출시)
  - 자연어 명령으로 Visa 결제 네트워크에 직접 연결
  - Trusted Agent Protocol로 에이전트-머천트 간 안전한 통신

## Amazon Bedrock AgentCore 구성요소

| 컴포넌트 | 역할 |
|---------|------|
| **AgentCore Runtime** | 서버리스 호스팅 환경, micro-VM 샌드박스로 격리 실행 |
| **AgentCore Identity** | 인바운드 인증(사용자 로그인) + 아웃바운드 인증(외부 API 접근) |
| **AgentCore Gateway** | MCP 서버 및 도구에 대한 거버넌스된 접근 제어 |
| **AgentCore Memory** | 장기 세션 컨텍스트 유지 (다일 여행 계획 등) |
| **AgentCore Observability** | OpenTelemetry 기반 전체 워크플로우 추적, 감사 로그 |

## 아키텍처: 재사용 가능한 Supervisor 패턴

```
┌─────────────────────────────────────────┐
│           Supervisor Agent              │
│  - 요청 라우팅                           │
│  - 컨텍스트 유지 (AgentCore Memory)      │
│  - 응답 포맷팅                           │
└──────────────┬──────────────────────────┘
               │
    ┌──────────┼──────────┐
    ▼          ▼          ▼
┌───────┐  ┌───────┐  ┌───────┐
│Travel │  │Shopping│  │ Cart  │
│Agent  │  │Agent   │  │Manager│
└───────┘  └───────┘  └───────┘
```

**핵심**: 동일한 Supervisor를 여행/쇼핑 등 다양한 도메인에서 재사용 가능
- Sub-agent만 교체하면 새로운 유스케이스 적용
- 오케스트레이션 로직, 메모리 관리, 대화 처리는 동일

## 샘플 1: Travel Booking Agent

### 구성
1. **Supervisor**: 전체 오케스트레이션
2. **Travel Assistant**: 여행 계획, 항공/호텔 검색
3. **Cart Manager**: 장바구니, 결제 처리

### Travel Assistant 도구
- `get_weather(query)` - 날씨 정보
- `get_flight_offers_tool(origin, destination, departure_date, ...)` - 항공편 검색
- `get_hotel_data_tool(city_code, ratings, amenities, ...)` - 호텔 검색
- `google_places_tool(query)` - 로컬 장소 검색

### Cart Manager 도구
- `add_to_cart(user_id, items)` - 장바구니 추가
- `onboard_card(user_id, card_number, ...)` - 카드 등록
- `request_purchase_confirmation` - 결제 확인 요청 (Human-in-the-loop)
- `confirm_purchase` - 결제 실행

## 샘플 2: Shopping Assistant Agent

### 구성
1. **Supervisor**: 전체 오케스트레이션 (Travel과 동일)
2. **Shopping Assistant**: 상품 검색, 추천
3. **Cart Manager**: 장바구니, 결제 처리

### Shopping Assistant 도구
- `single_productsearch(user_id, question)` - 상품 검색
- `generate_packinglist(user_id, question)` - 패킹 리스트 생성

### 사용 시나리오
> "Find the best offer for Sony PlayStation 5 Pro, compare it across merchants for Black Friday promotions, check delivery dates, apply my rewards. My Budget is under $500."

에이전트가 자동으로:
1. 여러 머천트 사이트에서 상품 검색
2. 프로모션 포함 가격 비교
3. 배송 일정 확인
4. 리워드 적용
5. 결제 실행

## Human-in-the-loop 결제 흐름

```
1. 에이전트가 구매 준비 완료
2. request_purchase_confirmation → 사용자에게 명시적 승인 요청
3. 사용자 승인
4. confirm_purchase → Visa Intelligent Commerce API 호출
   - 결제 자격 증명 요청
   - 인증 트리거
   - 토큰화된 결제 실행
```

**보안**: 사용자가 명확한 파라미터와 지출 허가를 설정한 후에만 결제 진행

## 인상 깊은 부분

> "Unlike traditional AI systems that merely answer questions or provide suggestions, agentic AI introduces intelligent agents capable of reasoning, acting, collaborating with other agents, and completing multistep tasks on the user's behalf."

> "This modular approach reduces development overhead. Rather than building separate orchestration systems for each use case, developers can reuse the supervisor agent across multiple domains."

## 실무 적용 포인트

### 1. Supervisor 패턴의 재사용성
- 도메인별로 Sub-agent만 교체하면 새로운 서비스 구축 가능
- 핵심 오케스트레이션 로직은 한 번 구축 후 재사용

### 2. AgentCore의 엔터프라이즈 기능
- **Runtime**: micro-VM 격리로 보안 강화
- **Memory**: 장기 세션 지원 (여행 계획처럼 며칠 걸리는 작업)
- **Observability**: 금융 규제 준수를 위한 감사 로그

### 3. Human-in-the-loop 필수
- 결제처럼 민감한 작업은 반드시 사용자 확인 단계 포함
- `request_purchase_confirmation` → `confirm_purchase` 2단계 구조

### 4. MCP 서버 활용
- OTA(여행사), 리테일 등 외부 서비스를 MCP 서버로 연결
- AgentCore Gateway로 거버넌스된 접근 제어

## 관련 리소스

- [Travel Agent Sample - GitHub](https://github.com/awslabs/amazon-bedrock-agentcore-samples/tree/main/05-agent-collaboration/travel-agent)
- [Shopping Agent Sample - GitHub](https://github.com/awslabs/amazon-bedrock-agentcore-samples/tree/main/05-agent-collaboration/shopping-agent)
- [Visa Intelligent Commerce - AWS Marketplace](https://aws.amazon.com/marketplace)
