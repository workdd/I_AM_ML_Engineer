# Logits as Confidence: LLM·VLM에서 AI 엔지니어가 꼭 알아야 할 숨겨진 힘

- **출처**: [Medium - Dineshkumar Anandan](https://medium.com/@adkananthi/logits-as-confidence-the-hidden-power-ai-engineers-need-to-unlock-in-llms-and-vlms-194d512c31f2)
- **저자**: Dineshkumar Anandan
- **게시일**: 2025-01-12
- **읽은 날짜**: 2026-06-09
- **태그**: #Logits #Confidence #LLM #VLM #Softmax #vLLM #Reranker #LogProbs

## 핵심 주장

소프트맥스 이전의 원시 점수인 **logit**을 신뢰도 지표로 활용하면, AI 엔지니어는 모델의 의사결정 과정을 투명하게 들여다보고 출력 품질을 정밀하게 제어할 수 있다.

---

## Logit이란?

모델이 다음 토큰 후보 각각에 부여하는 **정규화 전 원시 점수(raw unnormalized score)**.

- 소프트맥스를 거치면 0~1 사이 확률로 변환됨
- **높은 logit** = 모델이 해당 토큰을 강하게 확신
- **낮은 logit** = 불확실, 여러 후보 사이에서 갈팡질팡

```
토큰 후보들의 logit → softmax → 확률 분포 → 가장 높은 확률의 토큰 선택
```

---

## 실용적 활용법

### 1. Transformers 라이브러리 — `output_scores=True`

```python
# generate 시 각 스텝의 logit 추출
outputs = model.generate(..., output_scores=True, return_dict_in_generate=True)
scores = outputs.scores  # 각 생성 스텝별 logit 텐서 리스트

confidence_scores = []
for i, logits in enumerate(scores):
    probabilities = softmax(logits, dim=-1)
    token_id = outputs.sequences[0, i + 1]
    token = tokenizer.decode([token_id])
    confidence = probabilities[0, token_id].item()
    confidence_scores.append((token, confidence))
```

### 2. vLLM — `logprobs` 파라미터

```python
sampling_params = SamplingParams(
    temperature=0.1,
    logprobs=1,       # 각 스텝 상위 N개 토큰의 로그 확률 출력
    max_tokens=8192
)
outputs = llm.generate(prompts, sampling_params)
```

로그 확률(log probability)이므로 `exp(logprob)`으로 실제 확률 복원.

### 3. Reranker (Cross-Encoder) — 직접 관련성 점수

```python
# BAAI/bge-reranker-v2-m3 등 cross-encoder 모델
outputs = model.score(query, candidate_texts)
for text, output in zip(candidate_texts, outputs):
    score = output.outputs.score  # 시그모이드 거친 0~1 점수
```

RAG 파이프라인에서 1차 검색 결과를 재정렬할 때 사용.

---

## 신뢰도 점수를 쓰면 뭐가 좋나?

| 활용 포인트 | 설명 |
|-------------|------|
| **신뢰성 필터링** | 임계값 이상의 예측만 사용자에게 전달 |
| **불확실 구간 탐지** | 낮은 confidence → 파인튜닝 대상 데이터 발굴 |
| **투명한 UX** | "이 답변의 확신도 87%" 형태로 신뢰도 노출 |
| **품질 제어** | 자동화 파이프라인에서 재시도/에스컬레이션 트리거 |
| **의사결정 최적화** | 여러 후보 답변 중 가장 확신 높은 것 선택 |

---

## Logit vs Log Probability

| | Logit | Log Probability |
|---|---|---|
| 범위 | -∞ ~ +∞ | -∞ ~ 0 |
| 변환 | softmax 이전 | softmax 이후 log 적용 |
| vLLM 파라미터 | — | `logprobs=N` |
| 해석 | 상대적 선호도 | 실제 확률의 log 값 |

---

## 인상 깊은 부분

> "Logits are essential because they represent the raw, unnormalized scores assigned by the model to each potential token — before any probability transformation."

> "Confidence scores are essential for enhancing the performance and reliability of LLMs and VLMs."

모델이 "모른다"는 신호를 보낼 때 그걸 그냥 출력하지 않고 잡아낼 수 있다는 점이 핵심. 특히 RAG + Reranker 파이프라인에서 관련성 점수 임계값 설정에 직접 활용 가능.

---

## 연관 글

- [JSON vs TOON 토큰효율](./[20260104]%20네이버클라우드_JSON_vs_TOON_토큰효율.md) — 토큰 수준 제어의 다른 관점
- [LLM 서빙 성능최적화](./[20251219]%20네이버클로바_LLM서빙_성능최적화.md) — KV Cache, Goodput 등 서빙 레이어
- [Speculative Decoding 적용기](./[20251219]%20네이버클로바_Speculative_Decoding_적용기.md) — 디코딩 전략과 확률 분포의 관계
