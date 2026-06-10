# LLM Uncertainty Quantification & RAG Confidence — 리딩 노트

> 연구 날짜: 2026-06-09  
> 주제: LLM의 불확실성 정량화(UQ), RAG 신뢰도 추정, Conformal Prediction 적용

## 이 폴더의 목적

LLM 소프트맥스 확률을 신뢰도 점수로 직접 쓸 수 있냐는 질문에서 시작한 심층 리서치 세션.
핵심 논문들을 adversarial fact-checking 방식으로 검토해 **실제로 작동하는 것과 문헌이 주장하는 것 사이의 간격**을 정리했다.

---

## 커버된 논문 목록

| 파일 | 논문 | 베뉴 |
|------|------|------|
| [2503.15850-llm-uq-survey.md](./2503.15850-llm-uq-survey.md) | A Survey on Uncertainty Quantification of LLMs | KDD 2025 |
| [2306.10193-conformal-lm.md](./2306.10193-conformal-lm.md) | Conformal Language Modeling | ICLR 2024 |
| [2511.09803-targ.md](./2511.09803-targ.md) | TARG: Training-Free Adaptive Retrieval Gating | arXiv Nov 2025 |
| [entropy-calibration-instruct.md](./entropy-calibration-instruct.md) | Entropy degradation in instruction-tuned LLMs | arXiv 2510, 2511 |
| [rag-hallucination.md](./rag-hallucination.md) | RAG hallucination detection & P(True) | arXiv 2407, 2603 |

---

## 핵심 발견 요약

### 1. "전통 UQ 방법이 LLM에 부적합하다"는 주장 — 과장됨
서베이 논문들(2503.15850)이 반복하는 수사지만, 실증 근거는 "계산 비용이 높다"는 엔지니어링 제약에 가깝다.
Temperature scaling, conformal prediction, token entropy 모두 production에서 작동 중.

### 2. Entropy는 instruction-tuned 모델에서 신호를 잃는다
RLHF/SFT 이후 entropy가 압축되어 판별력이 사라진다. **Margin(top-1 − top-2 logit gap)이 대안.**
단, margin도 OOD shift 앞에선 취약하다.

### 3. Conformal prediction의 coverage 보장은 marginal이다
`P(Y ∈ C_α) ≥ 1−α`는 **평균** 보장이지, 쿼리별 조건부 보장이 아니다.
쉬운 쿼리 over-cover / 어려운 쿼리 under-cover 가능.

### 4. 교환가능성(exchangeability) 가정 — Dynamic RAG에서 위반됨
Conformal prediction의 전제 조건. 캘리브레이션 분포 ≠ 배포 분포이면 커버리지 보장이 무너진다.
Dynamic RAG(문서 풀이 실시간 변동)에서 이 가정은 구조적으로 위반된다.

---

## 오픈 문제 (미해결)

→ [`open-problems.md`](./open-problems.md) 참조

---

## 관련 선행 개념

- Aleatoric vs Epistemic uncertainty (Hüllermeier & Waegeman 2021)
- Split conformal prediction (Vovk et al.)
- Semantic Entropy (Kuhn et al., Nature 2024)
- RLHF calibration tax (alignment 과정에서 calibration 손상)
