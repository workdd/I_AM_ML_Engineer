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

> "Logits are essential because they represent the raw, unnormalized scores assigned by the model to each potential token — before any probability transformation."

- 범위는 -∞ ~ +∞로 제한 없음
- **높은 logit** = 모델이 해당 토큰을 강하게 선호 → 높은 신뢰도
- **낮은 logit** = 여러 후보 사이에서 불확실 → 낮은 신뢰도

```
raw logits (예: [2.1, 0.3, -1.5, ...])
    → softmax →
확률 분포 (예: [0.72, 0.11, 0.04, ...])
    → argmax →
최종 토큰 선택
```

---

## Logit → 신뢰도 변환 원리

### Softmax (LLM 토큰 생성)

모든 후보 토큰에 대한 logit을 확률로 정규화한다. 실제 선택된 토큰의 확률값이 곧 해당 스텝의 신뢰도.

```
P(token_i) = exp(logit_i) / Σ exp(logit_j)
```

- softmax는 가장 큰 logit을 가진 토큰의 확률을 크게 증폭시킴
- 결과값 합이 1 → 확률로 직접 해석 가능
- 값이 0.9에 가까우면 거의 확정적, 0.3 이하면 불확실

### Sigmoid (Reranker, 이진 관련성 판단)

query-document 쌍의 관련성을 0~1로 압축할 때 사용.

```
score = 1 / (1 + exp(-logit))
```

Softmax와 달리 다른 후보들과 무관하게 독립적으로 점수를 계산.

---

## 신뢰도를 실제로 꺼내는 방법

### 방법 1. Transformers — `output_scores=True`

토큰 생성 시 각 스텝의 logit 텐서를 함께 반환받아, 실제 선택된 토큰의 확률을 추출한다.

```python
from torch.nn.functional import softmax

def generate_with_confidence(model, processor, inputs, max_new_tokens=2048):
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        output_scores=True,              # 핵심: logit 텐서 반환
        return_dict_in_generate=True
    )
    return outputs

def compute_confidence_scores(scores, generated_token_ids, processor):
    confidence_scores = []
    for i, logits in enumerate(scores):
        probabilities = softmax(logits, dim=-1)          # logit → 확률
        token_id = generated_token_ids[0, i].item()
        token = processor.tokenizer.decode([token_id])
        confidence = probabilities[0, token_id].item()   # 해당 토큰의 확률 = 신뢰도
        confidence_scores.append((token, confidence))
    return confidence_scores
```

VLM(`MllamaForConditionalGeneration` 등)도 동일한 방식으로 적용 가능.

### 방법 2. vLLM — `logprobs` 파라미터

추론 서버 환경에서 log probability로 꺼낸다. `exp(logprob)`으로 실제 확률 복원.

```python
from vllm import LLM, SamplingParams

sampling_params = SamplingParams(
    temperature=0.1,
    top_p=0.001,
    repetition_penalty=1.05,
    max_tokens=8192,
    logprobs=1          # 각 스텝 상위 N개 토큰의 log probability 반환
)

llm = LLM(model="your-model")
outputs = llm.generate(prompts, sampling_params)

# log probability → 확률 복원
import math
for output in outputs:
    for token_logprob in output.outputs[0].logprobs:
        prob = math.exp(token_logprob)  # exp(log P) = P
```

### 방법 3. Reranker — 직접 관련성 점수

> "Unlike traditional models that generate logits for each possible token, rerankers take a query and a set of candidate responses and output a relevance score."

토큰 단위가 아닌 query-document 쌍 전체의 관련성을 sigmoid로 0~1 점수로 출력한다.

```python
from vllm import LLM

model = LLM(
    model="BAAI/bge-reranker-v2-m3",
    task="score",
    enforce_eager=True,
)

outputs = model.score(query, candidate_texts)
for text, output in zip(candidate_texts, outputs):
    score = output.outputs.score    # sigmoid 거친 0~1 관련성 점수
    print(f"Score: {score:.4f} | {text[:50]}")
```

RAG 파이프라인에서 1차 검색 결과를 재정렬할 때 바로 활용.

---

## 세 방법 비교

| | Transformers | vLLM | Reranker |
|---|---|---|---|
| **신뢰도 단위** | 토큰별 확률 | 토큰별 log probability | query-doc 쌍 점수 |
| **변환 함수** | softmax | softmax → log | sigmoid |
| **주요 파라미터** | `output_scores=True` | `logprobs=N` | `task="score"` |
| **적합한 용도** | 생성 품질 모니터링 | 서빙 환경 로깅 | RAG 재순위 |

---

## 신뢰도 점수를 어디에 쓰나?

| 활용 포인트 | 구체적 방법 |
|---|---|
| **출력 필터링** | confidence < 0.5인 토큰 포함 응답은 재생성 or 거부 |
| **불확실 구간 탐지** | 낮은 confidence 구간 → 파인튜닝 학습 데이터 후보 |
| **투명한 UX** | 응답과 함께 신뢰도 수치 노출 ("확신도 87%") |
| **자동화 파이프라인** | 임계값 미달 시 에스컬레이션·재시도 트리거 |
| **RAG 품질 제어** | Reranker 점수로 관련 없는 청크 자동 탈락 |

> "By understanding which predictions the model is confident about, engineers can make better decisions about which outputs to trust."

---

## Logit vs Log Probability

| | Logit | Log Probability |
|---|---|---|
| 범위 | -∞ ~ +∞ | -∞ ~ 0 |
| 변환 | softmax 이전 원시 점수 | softmax 이후 log 적용 |
| vLLM 파라미터 | — | `logprobs=N` |
| 확률 복원 | softmax 적용 | `math.exp(logprob)` |

---

## 인상 깊은 부분

모델이 "모른다"는 신호를 내부적으로 logit 분포의 평탄함(entropy)으로 드러낸다는 점이 핵심. 특정 토큰에 logit이 집중될수록 softmax 후 확률도 뾰족하게 올라가고, 분산될수록 낮고 넓게 퍼진다. 이걸 읽어내면 모델의 확신 정도를 숫자로 다룰 수 있다.

---

## 연관 글

- [JSON vs TOON 토큰효율](./[20260104]%20네이버클라우드_JSON_vs_TOON_토큰효율.md) — 토큰 수준 제어의 다른 관점
- [LLM 서빙 성능최적화](./[20251219]%20네이버클로바_LLM서빙_성능최적화.md) — KV Cache, Goodput 등 서빙 레이어
- [Speculative Decoding 적용기](./[20251219]%20네이버클로바_Speculative_Decoding_적용기.md) — 디코딩 전략과 확률 분포의 관계
