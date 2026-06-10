# Open Problems — LLM UQ & RAG Confidence

> 이 리서치 세션에서 확인되지 않은 오픈 문제들.
> 후속 리서치 또는 실험으로 해결할 필요가 있음.

---

## Problem 1: 독점 모델(GPT-4급)에서 Margin/Variance 신호 검증 미비

### 현재 상태
- Margin > Entropy 주장은 Llama-3.1-8B, Qwen2.5-7B (open-weight, white-box)에서 검증됨
- 독점 모델(GPT-4, Claude, Gemini)은 logit 접근 불가 → 직접 검증 불가능

### 문제의 구체적 질문들
1. GPT-4 계열 모델의 내부 entropy 분포가 instruction-tuned 오픈 모델과 유사하게 압축되어 있는가?
2. API를 통해 노출되는 `logprob` (top-5 토큰) 만으로 margin signal을 근사할 수 있는가?
3. GPT-4의 강화 RLHF가 entropy 압축을 더 심화시키는가, 아니면 reasoning 능력과 함께 emergence가 일어나는가?

### 잠재적 접근
- OpenAI API의 `logprobs=True` 파라미터로 top-5 토큰 확률 접근 가능 → margin 근사 가능성
- Sampling 기반 consistency measure로 logit 접근 없이 uncertainty 측정 (SelfCheckGPT 방식)
- Model-agnostic proxy: verbalized confidence와 실제 margin의 correlation 측정

---

## Problem 2: RLHF 방식(PPO vs DPO vs GRPO)별 Calibration 저하 메커니즘

### 현재 상태
- "Instruction tuning이 entropy를 줄이고 ECE를 높인다"는 것은 확인됨 (2511.11966)
- "Advanced post-training (RLHF/GRPO)에서 reasoning model은 entropy emergence를 보인다" (2510.08146)
- 하지만 **어떤 RLHF variant가 어떤 calibration 특성을 만드는지** 체계적 비교 없음

### 문제의 구체적 질문들
1. **PPO** (online RL, reward model 기반): 보상 최대화가 entropy를 어떻게 왜곡하는가?
2. **DPO** (offline RL, preference pair 기반): PPO 대비 calibration 손상이 적은가?
3. **GRPO** (group relative policy optimization, reasoning 최적화): entropy emergence 조건이 무엇인가?
4. Calibration 손상이 주로 "sharp answer" 선호 학습에서 오는가, 아니면 reward hacking에서 오는가?

### 왜 중요한가
- 실용적 함의: RAG gating signal 선택이 모델의 학습 방식에 따라 달라질 수 있음
- Margin이 reasoning model에서도 best signal인지, 아니면 entropy emergence가 있는 모델에선 entropy가 더 나은지 불명확

### 잠재적 접근
- 동일 base model에서 PPO/DPO/GRPO로 각각 파인튜닝 → ECE, margin AUROC, entropy 분포 비교
- 기존 오픈 모델들 비교: Tulu 3 (PPO/DPO), Qwen2.5-Math (GRPO), DeepSeek-R1 (GRPO)

---

## Problem 3: Dynamic RAG에서 Conformal Prediction Exchangeability 위반 시 Coverage 저하 정도

### 현재 상태
- Exchangeability 위반 시 coverage가 "catastrophically fail"할 수 있다는 것은 이론적으로 알려짐
- 하지만 **저하 정도의 정량적 측정**은 없음 (arXiv 2510.05566, 2603.27403에서 연구 중)

### 문제의 구체적 질문들
1. 문서 풀이 N% 바뀔 때 conformal prediction의 coverage가 얼마나 떨어지는가?
2. 시간에 따라 변하는 문서 풀(뉴스, 실시간 데이터)에서 재캘리브레이션 주기는 얼마가 적절한가?
3. Distribution shift가 marginal coverage와 conditional coverage 보장에 다르게 영향을 주는가?

### Dynamic RAG의 구체적 위반 패턴
```
캘리브레이션 시점: 2025-01 문서 풀 → threshold τ 설정
배포 시점:        2026-01 문서 풀 → 완전히 다른 분포

→ 이전에 설정한 τ가 새 분포에서 과도하게 retrieval하거나 과도하게 suppression
```

### 잠재적 접근
- **Sliding window calibration**: 최근 N 쿼리로 주기적 재캘리브레이션
- **Online conformal prediction**: 데이터 도착 시마다 threshold 업데이트
- **Coverage monitoring**: 실제 deployment에서 empirical coverage 지속 측정 → drift 감지 시 재캘리브레이션 트리거
- 참고 연구: arXiv 2510.05566 (domain-shift-aware conformal), arXiv 2603.27403 (conditional factuality control)

---

## 우선순위 판단

| 문제 | 실용적 중요도 | 연구 난이도 |
|------|-------------|-----------|
| Problem 1 (독점 모델) | 높음 (production 직결) | 중간 (API 근사 가능) |
| Problem 2 (RLHF variant) | 중간 (모델 선택 기준) | 높음 (컴퓨팅 필요) |
| Problem 3 (Dynamic RAG) | 높음 (실서비스 필수) | 중간 (온라인 캘리브레이션) |
