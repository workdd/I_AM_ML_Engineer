# 환각은 흔적을 남긴다: Future Context 기반 LLM 환각 검출

- **출처**: [NAVER CLOVA Tech Blog](https://clova.ai/tech-blog/%ED%99%98%EA%B0%81%EC%9D%80-%ED%9D%94%EC%A0%81%EC%9D%84-%EB%82%A8%EA%B8%B4%EB%8B%A4-future-context-%EA%B8%B0%EB%B0%98-llm-%ED%99%98%EA%B0%81-%EA%B2%80%EC%B6%9C)
- **저자**: 이주성, 박천복, 조휘열, 김정훈, 박준석 (NAVER Cloud · KAIST)
- **발표**: ACL 2026 (San Diego)
- **게시일**: 2026-06-11
- **읽은 날짜**: 2026-06-15
- **태그**: #환각검출 #Hallucination #BlackBox #FutureContext #SelfCheckGPT #ACL2026

## 핵심 문제의식

LLM이 만든 글에서 환각(존재하지 않는 사건, 잘못된 인과관계, 과거 문맥과 모순되는 주장)을 검출하는 문제. 어려움의 핵심:

- **블랙박스 환경**: 블로그·검색 응답처럼 최종 결과물만 공개되면 생성 모델의 내부 정보(logits)에 접근 불가
- **Snowball Effect**: 초기 오류가 이후 문맥을 오염시키며 번짐

### 기존 방법의 한계

| 방법 | 한계 |
|------|------|
| 불확실성 기반 (logits/token prob) | 모델 **내부 정보 필요** → API 모델·외부 콘텐츠엔 적용 불가 |
| 샘플링 기반 (SelfCheckGPT) | 여러 응답 생성 → 추가 비용 |
| 검색 기반 | 검색 비용 + 검색 결과 자체의 신뢰성 보장 어려움 |

## 핵심 아이디어 — Future Context

> **"현재 문장이 환각이라면, 이후에 이어지는 문장들 또한 환각일 가능성이 높다."**

거짓 전제("1969년 달이 지구 궤도에서 영구 제거되었다")가 주어지면, 모델은 그 뒤를 "그 이후 인공위성이 달의 역할을 대신해…"처럼 **자연스럽게 거짓을 확장**한다. 즉 환각은 한 문장에 고립되지 않고 **이후 문맥에 흔적을 남기며 전파**된다. 이 "흔적"을 거꾸로 환각의 단서로 쓴다.

## 방법론 동작 원리

```
1. 생성된 응답에서 검증 대상 문장 선택
2. 그 문장 뒤에 이어질 '미래 문장'을 Detector LLM으로 샘플링
3. 생성된 미래 문맥을 기존 환각 검출기의 입력에 추가
4. 현재 문장의 사실성 판단
```

기존 검출기에 **얹는(plug-in)** 방식으로, 세 가지로 결합:

- **Direct + Future Context**: 현재 문장 사실 여부를 직접 묻되, 미래 문맥과 얼마나 자연스럽게 이어지는지 함께 고려
- **SelfCheckGPT + Future Context**: 대체 응답 간 일관성 확인에 미래 문맥 방향성까지 활용
- **Self-Contradiction + Future Context**: 원문-대체문 논리 모순 분석에 미래 문맥 추가

## 실험 결과

- Detector LLM: LLaMA 3.1, Gemma 3, Qwen 2.5
- 데이터셋: SelfCheckGPT, SC-ChatGPT/GPT4/LLaMA/Vicuna, True-False

| 방식 | 기존 AUROC | +Future Context |
|------|-----------|-----------------|
| Direct | 68.9 | **71.1** |
| Self-Contradiction | 65.7 | **70.8** |

- 미래 문장을 **많이 샘플링할수록** 검출 성능 향상 경향 (그림 3)
- 현재 문장과 미래 문맥이 **둘 다 환각이거나 둘 다 사실**일수록 도움이 큼

## 결론·시사점

- 생성 모델 내부 정보 **없이** 동작하는 블랙박스 환각 검출
- 기존 샘플링 방식과 쉽게 결합, 적은 비용으로 의미 있는 향상
- 관점 전환: "이 문장이 사실인가?" → **"이 문장이 이후 문맥에 어떤 영향을 남기는가?"**

---

## 메모 — 내 작업(logprob confidence 게이팅)과의 대비

이 논문은 내가 설계 중인 **logprob 기반 confidence 검증과 정반대 진영**이다. 좋은 보완점이 있다.

| 구분 | logprob 기반 (내 설계) | Future Context (이 논문) |
|------|----------------------|--------------------------|
| 전제 | 모델 내부 logits 접근 가능 | **블랙박스** (내부 정보 불필요) |
| 신호원 | 토큰 확률 | 이후 문맥의 일관성 |
| 약점 | RL 모델에서 logprob 이진화·과신 | 미래 문맥 샘플링 비용 |
| 강점 | 무료(생성 시 동반), 즉시 | API 모델·외부 콘텐츠에도 적용 |

**적용 포인트**: 내 멀티 에이전트 최종 답변 검증에서 — logprob을 못 쓰거나(외부 API 에이전트) 신뢰 못 할 때(RL 과신), Future Context가 대안 신호가 된다. 특히 Snowball Effect 관찰은 내가 우려한 "고신뢰 오답이 하류로 전파"와 같은 현상을 검출 측면에서 뒤집어 쓴 것 — **전파된 흔적으로 원천 환각을 역추적**. 검증 강도 배분(③)에서 logprob 신호와 Future Context 신호를 **multi-scoring으로 결합**하면 단일 방법 한계(arXiv:2407.21424)를 보완할 수 있다.

## 연관 자료

- [readings/papers/llm-uq-rag-confidence/](../papers/llm-uq-rag-confidence/) — UQ·confidence 논문 모음
- [practical_tips/logprob_confidence_gating_설계.md](../../practical_tips/logprob_confidence_gating_설계.md) — logprob 게이팅 설계안
