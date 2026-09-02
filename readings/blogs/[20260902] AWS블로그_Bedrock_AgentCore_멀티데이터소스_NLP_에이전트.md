# Bedrock AgentCore: LangGraph 코드를 그대로 두고 운영 레이어만 얹기

- **출처**: [Amazon Bedrock AgentCore로 멀티 데이터소스 자연어 질의 에이전트를 프로덕션에 배포하기](https://aws.amazon.com/ko/blogs/tech/amazon-bedrock-agentcore-multi-datasource-nlp-agent-production/)
- **저자**: Areum Lee, Hyewon Lee, Gonsoo Moon (AWS 솔루션즈 아키텍트)
- **발행**: 2026-08-31, AWS 기술 블로그
- **기반**: AWS Summit Seoul 세션 "200개국 삼성 스마트 TV 앱 데이터를 자연어로 묻다: 에이전틱 AI on AWS"
- **사례 주체**: 삼성전자 영상디스플레이 사업부 TV Apps Service 팀
- **읽은 날짜**: 2026-09-02
- **태그**: #AgentCore #Bedrock #LangGraph #MCP #ToolSelection #Observability #Evaluation #Cedar

## 한 줄 요약

LangGraph로 만든 PoC 에이전트를 **코드는 거의 손대지 않고** AgentCore의 Runtime, Gateway, Memory, Observability, Evaluations, Policy로 감싸 프로덕션에 올린 전환기. 프레임워크가 만들어주는 것은 에이전트의 판단이고 그 판단을 안전하게 굴리는 층은 따로 쌓아야 한다.

## 문제: 데이터가 없는 게 아니라 어디 있는지를 사람이 찾는다

삼성 Smart TV 앱 서비스는 200여 개국, 수천 종의 앱, 수만 종의 TV 모델과 펌웨어 버전을 다룬다. 앱마다 TV 환경에 따라 지원 여부와 설치되는 최상위 버전이 달라 조합의 경우의 수가 크다.

이 데이터를 담당자 수백 명이 매일 조회하는데, 정보가 검증계 DB, 운영계 DB, 사내 API, KPI 분석용 스토어로 흩어져 있다. 문의 하나에 담당자가 세 시스템을 각각 조회해야 했고 응답까지 수십 분이 걸렸다. 같은 질문이 여러 담당자에게 중복으로 도착하는 낭비도 있었다.

블로그가 문제를 규정한 문장이 정확하다. "데이터가 없다"가 아니라 "데이터가 어디에 있는지, 어떻게 조합해야 하는지를 사람이 매번 찾아야 한다".

팀은 Supervisor 아래 KPI 조회, 앱 지원 확인, 앱 검색 하위 에이전트를 두는 LangGraph 멀티 에이전트로 PoC를 만들었고 자동화 가능성은 입증됐다.

## PoC에서 프로덕션으로 갈 때 드러난 다섯 가지

프레임워크를 무엇으로 골랐든 공통으로 만나는 벽이라는 설명이 붙는다.

| 한계 | 구체적 증상 |
|------|-------------|
| 멀티 데이터소스 관리 복잡 | 검증계·운영계 DB, KPI 스토어 연결 코드를 따로 유지, 데이터소스 추가 때마다 에이전트 코드 수정 |
| 기능 동적 확장 불가 | 신규 기능 추가에 에이전트와 Tool 코드를 모두 고치고 각각 재배포 |
| 블랙박스 실행 | 응답이 느릴 때 어느 단계가 병목인지 식별 불가 |
| 인프라 운영 부담 | 컨테이너 실행 환경, 스케일링, 가용성을 팀이 직접 담당 |
| 메모리 관리 부담 | 멀티턴 세션용 상태 저장소를 따로 운영하고 세션 관리 코드를 직접 구현 |

프레임워크는 에이전트의 두뇌를 만드는 데 좋지만 그 두뇌를 안전하고 확장 가능하게 운영하는 층은 팀이 처음부터 쌓아야 한다. 다섯 항목 모두 PoC에서는 안 보이다가 프로덕션에 가까워질수록 비용이 되는 것들이다.

## AgentCore의 구성

라이프사이클 세 단계로 나뉜다. 집필 시점(2026년 8월) 기준이며 기능 추가가 빠른 서비스라고 블로그가 단서를 달았다.

| 단계 | 구성 요소 | 역할 |
|------|-----------|------|
| 빌드 | Gateway, Memory, Browser, Code Interpreter, Harness | 도구 사용, 기억, 웹 탐색, 코드 실행 |
| 배포 | Runtime, Identity, Policy, Payments | 안전한 대규모 배포 |
| 운영 | Observability, Evaluations, Optimization, AWS Agent Registry | 지속 모니터링과 품질 관리 |

핵심은 기존 에이전트를 대체하지 않는다는 설계다. LangGraph, Strands, CrewAI로 이미 만든 코드를 운영 레이어 위에 올린다. MCP, A2A, AG-UI, OpenTelemetry 같은 개방형 표준을 지원한다.

## 요청 처리 흐름

```text
사용자 자연어 질문 + Cognito JWT
  -> AgentCore Inbound authorizer 토큰 검증
  -> AgentCore Runtime 위의 기존 LangGraph 그래프 (Supervisor + 하위 에이전트)
  -> 데이터 필요 시 AgentCore Gateway 단일 엔드포인트만 호출
       도구가 많으면 시맨틱 검색으로 관련 도구 먼저 탐색
       Gateway가 검증계 DB / 운영계 DB / KPI 스토어 중 해당 target으로 전달
  -> 대화 맥락은 AgentCore Memory가 세션 단위로 유지
  -> 모든 실행 단계는 Observability를 통해 CloudWatch로 수집
  -> Policy(Cedar read-only 강제)가 런타임 보안, Evaluations가 배포 전 품질 게이트
```

에이전트의 핵심 로직은 그대로 두고 인증, 라우팅, 메모리, 관측, 품질 관리만 바깥에서 감싼 구조다.

## 컴포넌트별 적용 방식

| 컴포넌트 | 역할 | 삼성전자 적용 |
|----------|------|---------------|
| Runtime | LangGraph 실행 환경 | 기존 그래프를 코드 변경 없이 이관, microVM 세션 격리 |
| Gateway | MCP 단일 진입점 | 검증계·운영계 DB, KPI 스토어, 사내 API를 모두 MCP targets로 통합 |
| Memory | 멀티턴 세션 상태 | 자체 상태 저장소에서 관리형 Short-Term Memory로 전환 |
| Observability | 실행 trace 수집 | OTel 기반 CloudWatch 연동 |
| Evaluations | 품질 자동 평가 | CI/CD 통합, Tool 선택과 응답 품질 검증 |
| Identity | 인증·권한 | Cognito 연동 인바운드 인증 |
| Policy | 도구 호출 인가 | 자연어 규칙을 Cedar 정책으로 생성·검증해 read-only Tool만 허용 |

## 구현에서 눈여겨볼 대목

### Gateway: 도구 연결과 도구 선택은 다른 문제다

이전에는 에이전트가 검증계·운영계 endpoint를 각각 알고 호출했다. 환경이 바뀌거나 데이터소스가 늘면 에이전트 코드를 고쳐야 했다. Gateway 도입 후에는 에이전트가 Gateway 하나만 본다. 새 데이터소스는 target 등록만으로 붙고 에이전트 코드 변경이나 재배포 없이 즉시 노출된다.

target 카테고리는 세 가지다.

- **MCP targets**: 여러 백엔드를 하나의 가상 MCP 서버로 집계. 시맨틱 검색과 capability 동기화 지원. 하위 타입으로 Lambda 함수, API Gateway 스테이지, OpenAPI/Smithy 스키마, MCP 서버, 내장 커넥터가 있다.
- **HTTP targets**: 프로토콜 변환 없이 트래픽 직접 전달.
- **Inference targets**: 여러 모델 공급자로 라우팅.

삼성전자는 세 데이터소스를 모두 MCP targets로 등록해 하나의 `tools/list`로 노출되게 했다. 그래서 시맨틱 검색이 데이터소스 종류를 가리지 않고 동작한다.

연결 다음 문제가 선택이다. 서비스 하나에 도구가 수백 개씩 붙기도 하는데 모든 도구 정의를 컨텍스트에 넣으면 토큰도 비싸고 정확도도 떨어진다. Gateway 생성 시 시맨틱 검색을 켜면 등록된 도구가 자동 인덱싱된다. 에이전트는 `x_amz_bedrock_agentcore_search`를 자연어로 호출해 관련 도구 소수만 받아 그중 하나를 실행한다.

### Runtime: 세션마다 microVM

프레임워크 종속이 없어 기존 그래프를 그대로 옮긴다. 세션마다 전용 microVM(컴퓨팅, 메모리, 파일)을 할당하고 세션 종료 후 삭제해 세션 간 데이터 혼입과 권한 상승을 구조적으로 막는다. 자동 스케일링, 장기 실행 워크로드, 비동기 처리, 양방향 스트리밍, PrivateLink, VPC 보안을 제공하며 ECS, ALB, Auto Scaling을 직접 구성할 필요가 없다.

### Observability: OTel로 수집되는 것

- Supervisor가 어떤 하위 에이전트를 호출했는지 라우팅 흐름
- LLM 호출의 프롬프트, 응답, 토큰 사용량
- Tool 호출의 입력 파라미터, 출력 결과, 실행 시간
- 각 span의 duration

CloudWatch GenAI Observability 대시보드와 Transaction Search에서 바로 보이고 OTel 호환이라 Datadog 같은 기존 도구에도 붙는다.

### Evaluations: 판단이 필요한 지표와 그렇지 않은 지표를 가른다

평가기가 두 종류다. 품질이나 유용성처럼 판단이 필요한 지표는 LLM 심사자를 호출해 점수와 근거를 남긴다. `Builtin.TrajectoryExactOrderMatch`처럼 도구 호출 순서를 검사하는 평가기는 LLM 없이 결정론적으로 통과·실패를 판정한다. CI 하드 게이트에는 판정이 흔들리지 않는 후자를 쓰고 전자는 추세 관찰용으로 두라는 조합 권고가 실용적이다.

실행 방식은 세 가지다.

| | On-Demand (개발, 조사) | Online (프로덕션 모니터링) | Batch (회귀, 전후 비교) |
|---|---|---|---|
| 동작 | span/trace를 지정해 개별 평가 | 실시간 상호작용 모니터링, 트레이스 지속 샘플링 | CloudWatch Logs의 다수 세션을 비동기 일괄 평가 |
| 활용 | 빌드타임 테스트, 이슈 조사, 품질 게이트 | silent failure 포착, 품질 추세 추적 | baseline 측정, 프롬프트·모델 변경 전후 비교 |

삼성전자는 On-Demand를 CI/CD에 통합해 코드 변경이 올라오면 자동 평가가 돌고 기준 미달이면 배포가 차단되게 했다.

### Policy: 추론 루프 밖에서 결정론적으로 막는다

가장 눈여겨볼 설계다. 정책 엔진을 Gateway에 붙여 경유하는 모든 도구 호출을 가로채 평가한다. 정책은 Cedar로 표현되지만 개발자가 Cedar 문법을 쓸 필요는 없다. "읽기 도구만 쓰고 삭제·변경 도구는 호출할 수 없다"처럼 자연어로 규칙을 기술하면 서비스가 후보 정책을 만들고 도구 스키마에 비춰 유효성을 검증한 뒤 자동 추론으로 안전성까지 확인한다.

**평가 시점이 에이전트의 reasoning loop 밖**이라는 점이 중요하다. Gateway가 도구 호출을 전달하기 직전에 판정하므로 LLM이 어떻게 추론하든 허용·차단이 결정론적으로 적용된다. 프롬프트로 우회하려는 시도에도 안전하다는 근거가 여기서 나온다.

## 데모에서 건진 것: silent failure

CI에 통합한 Evaluation이 잡아낸 사례가 이 글에서 가장 실무적이다.

- **질문**: "특정 환경에서 지원되는 앱 리스트를 알려주세요"
- **관찰**: 에이전트가 파라미터를 정확히 추출해 Tool을 호출했고 지원 앱 리스트도 정상 수신했다
- **문제**: 그런데 **최종 응답에 Tool에서 받은 정보가 담기지 않았다**

Tool 선택은 성공했는데 응답 품질에서 실패한 경우다. 사람이 일일이 뜯어보지 않으면 발견하기 어렵고 도구 호출 로그만 보면 정상으로 보인다. 프롬프트를 고쳐 재수행하자 통과했다.

도구 호출 성공률과 응답 품질은 따로 재야 한다. 앞단만 보면 문제가 없어 보인다.

## 전환 효과

| 항목 | LangGraph PoC | AgentCore 전환 후 |
|------|---------------|-------------------|
| Gateway | 환경별 분기 로직이 코드에 포함, 데이터소스 추가 시 코드 수정 | Target 등록만으로 연결, 동적 라우팅 + 시맨틱 검색 |
| Observability | 실행 흐름 불투명, 로그 기반 디버깅 | 모든 trace 가시화, CloudWatch 실시간 모니터링 |
| Evaluations | 수동 테스트, 품질 기준 없음 | 자동 평가 + CI/CD 통합 |
| 문의 응답 시간 | 수십 분 (수동 조회) | 자연어 질문으로 즉시 응답 |
| 인프라 운영 | 컨테이너 실행 환경 + 상태 저장소 직접 관리 | 관리형 Runtime + Memory |

## 읽을 때 감안할 것

- **정량 지표가 없다.** 전환 효과 표가 전부 정성 서술이다. "수십 분에서 즉시"를 빼면 정확도, 지연시간, 비용, 동시 사용자 수, 도구 선택 정확도 같은 수치가 하나도 없다. AWS 기술 블로그의 성격상 당연하지만 도입 근거로 인용할 때는 이 점을 밝혀야 한다.
- **코드는 고객의 실제 코드가 아니다.** 블로그가 직접 밝힌다. Gateway target 등록과 Runtime 진입점 예제 모두 공식 문서 패턴 기반 참고용이다. 그대로 복사하기 전에 최신 스키마를 확인해야 한다.
- **CLI가 바뀌었다.** 기존 Python 기반 `bedrock-agentcore-starter-toolkit`(`agentcore configure`, `agentcore launch`)은 지원이 끝났고 `@aws/agentcore` npm CLI로 넘어갔다. 예전 튜토리얼을 따라가면 여기서 막힌다.
- **시점이 섞여 있다.** 프로젝트 구성은 2026년 4월 기준이고 집필은 8월이다. 당시 Evaluations는 On-Demand와 Online 두 가지뿐이라 배포 전 게이트로 On-Demand를 썼다고 블로그가 각주로 밝힌다. 지금 설계한다면 Batch까지 포함해 다시 고를 수 있다.
- **도구 이름에 prefix가 붙는다.** Gateway로 노출되는 도구는 `AthenaKpiTarget___get_app_install_kpi`처럼 target 이름이 앞에 붙는다. **Lambda 함수 안에서 이 prefix를 직접 떼고 분기해야 한다.** 처음 붙일 때 걸려 넘어지기 쉬운 지점이다.
- **종속의 방향이 바뀔 뿐이다.** 프레임워크 종속을 피한다는 설명은 맞지만 대신 AWS 관리형 서비스 종속이 생긴다. Gateway, Memory, Policy를 걷어내고 다른 클라우드로 옮기는 비용은 별도로 계산해야 한다.

## 내 작업과의 연결

1. **Gateway 시맨틱 검색은 tool retrieval 그 자체다**

   "도구가 많으면 전부 컨텍스트에 넣지 말고 검색해서 몇 개만 준다"는 구조는 [RAG-MCP 노트](../papers/%5B20260710%5D%20RAG-MCP_Prompt_Bloat_Tool_Selection.md)와 [How Many Tools Should an LLM Agent See?](../papers/%5B20260706%5D%20How_Many_Tools_Should_an_LLM_Agent_See.md)가 다루는 문제와 같다. 논문에서 제안하던 방식이 관리형 서비스의 기본 기능으로 들어온 사례다. 직접 구현할지 얹어 쓸지 판단할 때 이 글이 후자의 실물 예시가 된다.

2. **검증을 추론 루프 밖에 두는 설계**

   Policy가 LLM의 추론과 무관하게 Gateway 앞단에서 결정론적으로 판정한다는 부분은 우리 파이프라인에도 옮길 만하다. LLM에게 "이 도구를 써도 되나"를 묻는 대신 호출 직전에 코드로 막으면 프롬프트 우회가 통하지 않는다.

3. **결정론적 게이트와 LLM 판정의 역할 분리**

   CI 하드 게이트에는 결정론적 평가기를, 추세 관찰에는 LLM judge를 쓰라는 권고는 [ROGRAG 노트](../papers/%5B20260831%5D%20ROGRAG_Robustly_Optimized_GraphRAG.md)의 argument checking 논의와 같은 방향이다. 판정이 흔들리는 신호를 배포 차단 기준으로 삼으면 안 된다.

4. **도구 호출 성공과 응답 품질을 분리해 측정하기**

   silent failure 사례는 평가 지표를 설계할 때 바로 반영할 만하다. 검색이 성공했는지와 그 결과가 답변에 반영됐는지는 다른 지표다. 앞만 보면 정상으로 보이는 실패가 실제로 존재한다.

## 결론

새로운 기술 자체보다 **운영 레이어를 어디까지 직접 만들 것인가**를 다룬 사례로 읽힌다. 에이전트의 판단 로직은 프레임워크로 만들고 인증·라우팅·메모리·관측·품질·인가는 관리형 서비스에 넘긴 선택이다.

블로그가 권하는 첫걸음도 합리적이다. 여러 데이터소스를 Gateway 단일 진입점으로 통합하는 것부터 시작하라고 한다. 효과가 크고 기존 코드를 건드리지 않아 되돌리기도 쉽다.

다만 수치가 없는 벤더 사례라는 점은 감안해야 한다. "이렇게 하면 얼마나 좋아지는가"가 아니라 "이런 문제를 이런 층으로 나눠 풀었다"는 구조를 얻는 글이다.
