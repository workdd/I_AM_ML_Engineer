# RAG Hallucination Detection & Confidence Methods

> 관련 논문:
> - arXiv 2407.21424 — Multi-scoring ensemble hallucination detection (KDD 2024 GenAI Workshop)
> - arXiv 2603.21172 — "Confidently wrong" failure mode
> - arXiv 2603.20975 — DiscoUQ (multi-LLM ensemble)

---

## P(True) 방법

### 동작 원리

```python
# LLM에게 "이 답이 맞나요?" yes/no 질문 → 첫 토큰 logit 추출
P(True | z^(1)) = p(w_True | z^(1)) / [p(w_True | z^(1)) + p(w_False | z^(1))]
```

- Logit 접근 가능한 white-box 모델 필요
- Black-box API에서는 sampling으로 근사 → 비용 증가

### 성능 (arXiv 2407.21424)

| 데이터셋 | 모델 | F1 |
|---------|------|-----|
| HaluEval | Mistral-7B | 0.7595 |
| HaluEval | Falcon-7B | 0.2794 |
| TriviaQA | — | 0.8267 |

⚠️ **주의**: Falcon-7B에서 F1이 0.28로 급락 → 모델 의존성 매우 높음  
⚠️ **출판 베뉴**: KDD 2024 GenAI Evaluation **Workshop** (메인 컨퍼런스 아님, peer-review 기준 낮음)

---

## Multi-Scoring Ensemble

여러 detection method를 logistic regression으로 앙상블.

**TriviaQA 결과** (arXiv 2407.21424):

| Method | F1 | Brier | Accuracy |
|--------|-----|-------|----------|
| Multi-score | **0.9106** | 0.1105 | 0.8593 |
| SelfCheckGPT-NLI (best single) | 0.8614 | 0.1434 | 0.8011 |

→ 앙상블이 최고 단일 방법보다 5.7% F1 향상

**비용 효율**: C=5 예산 multi-scoring이 SelfCheckGPT (C=9 generations)와 동등

⚠️ **한계**: 2024년 이후 신규 방법들이 이 baseline 초과함  
- ANAH-v2: HaluEval accuracy 81.54%  
- HaluCheck 3B: F1 0.753  

---

## RAG 환경에서 Confidence Miscalibration

### 주요 문제 패턴

1. **Overconfidence in high-score queries**  
   표준 RAG는 retrieval score 높은 쿼리에서 과신뢰 → 실제 성능은 낮아도 자신감 유지

2. **Noise context → confident wrong answer**  
   무관 문서 포함 시: accuracy ↓ + confidence ↑ (동시 발생)

3. **All-negative context**  
   정답이 없는 맥락이 모두 주어질 때 → confidence 점수가 오히려 높게 분산

4. **Context override (>60% 비율)**  
   LLM이 잘못된 retrieved context를 올바른 prior 지식보다 우선함

---

## RAG Hallucination Benchmark의 구조적 문제

**Faithfulness vs Factuality 혼동**:
- 현재 벤치마크 다수가 "문서에 없는 내용 = hallucination"으로 처리
- 문서에 없지만 사실적으로 맞는 답을 오분류 (false positive)
- 즉, faithfulness(문서 충실도) ≠ factuality(사실 정확성)

---

## Citation Chain Error (실제 발견)

한 popular claim: "RAG에서 retrieved context token의 logit이 inflate됨"  
→ fact-check 결과: 직접 출처 논문이 이 주장을 하지 않음 (인용 체인 오류)  
→ RAG 문헌에서 이런 무검증 인용이 생각보다 많음, 출처 직접 확인 필요

---

## 실무 정리

| 목적 | 권장 방법 |
|------|----------|
| 빠른 단일 방법 | P(True) (white-box) |
| 정확도 최대화 | Multi-scoring ensemble |
| Black-box API | SelfCheckGPT (sampling-based) |
| RAG 신뢰도 모니터링 | 단일 logit signal 불충분, OOD 탐지 병행 권장 |
