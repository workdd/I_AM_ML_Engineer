# NCP-ArchPreview: 토큰 다음에 "개념"을 예측하게 만들기

- **논문**: [NCP-ArchPreview Technical Report: Moving towards Latent Space Language Models through Next Concept Prediction](https://arxiv.org/abs/2609.10715)
- **저자**: The Intern-NCP Team (29명)
- **arXiv**: 2609.10715 [cs.CL], 2026-09-09 제출, 2026-09-14 갱신
- **분량**: 43쪽 (본문 24쪽 + 부록)
- **읽은 날짜**: 2026-09-14
- **태그**: #LatentSpaceLM #NextConceptPrediction #VectorQuantization #Pretraining #OLMo #ScalingLaw

## 한 줄 요약

Next Token Prediction 옆에 **Next Concept Prediction**을 붙인다. 은닉 상태에서 직접 만든 이산 개념 어휘를 두고, 여러 토큰에 걸친 다음 개념을 예측하게 해 **같은 데이터로 OLMo-3-7B의 최종 손실을 토큰 51.3%만에 도달**한다.

## 읽는 방식

**원문 목차를 그대로 따라간다.** 절 번호와 제목이 원문과 같다. 마지막 두 절(읽을 때 감안할 것, 가져갈 지점)만 읽는 쪽에서 붙였다.

---

## 1. Introduction

문제 제기가 명확하다. 모델의 은닉 상태에 추상적 구조가 생긴다는 것은 알려져 있는데, **표준 NTP 아래에서 그 추상은 간접적 부산물로만 생긴다.** 지도 신호가 토큰 단위에 갇혀 있어서, 의미 구조가 여러 토큰에 걸쳐 어떻게 펼쳐지는지를 이끄는 목적함수가 없다.

선행 연구를 두 축으로 가른다.

| 축 | 방식 | 사례 |
|----|------|------|
| **잠재 표현을 어떻게 만드나** | 고정 계층 구조. 미리 정한 다중 해상도나 바이트 패치 | Hourglass Transformer, ContextLM, MegaByte |
| | 입력 적응형 청킹. 경계나 패치 크기가 입력에 따라 변함 | Byte Latent Transformer, DLCM, H-Net |
| **무엇을 예측하게 하나** | 새로운 잠재 예측 대상이나 학습 목표를 도입 | (본문에서 별도 분류) |

NCP-ArchPreview는 **NTP와 NCP를 함께 사전학습**해 잠재 공간 언어 모델링을 5.73T 토큰까지 끌어올렸다. 이 규모에서 아키텍처가 성립하고 확장된다는 것을 보이는 것이 목표다.

## 2. Model Architecture

### 2.1 Overview

백본이 세 모듈이다.

```text
Token Encoder  ->  Concept Module  ->  Token Decoder
  토큰 표현         다음 개념 예측       다음 토큰 예측
                  (압축된 시퀀스)      (예측 개념 + 토큰 표현)
```

흐름은 이렇다. Token Encoder가 토큰 단위 은닉 상태 `h(1:T)`를 만든다. 압축 계수를 `k`라 할 때 **연속된 `k`개 상태를 평균 풀링**해 개념 시퀀스 `c(1:M)`을 만든다. `M = ⌊T/k⌋`이므로 시퀀스가 `k`배 짧아진다.

Vector Quantization으로 유한한 개념 공간, 곧 **개념 어휘**를 학습한다. Concept Module은 코드북 항목의 가중 조합으로 **미분 가능한 다음 개념 표현**을 예측한다. 이 예측 개념 시퀀스를 토큰 해상도로 되풀이하고 **인과성을 지키도록 shift**해 Token Decoder에 주입한다.

여기에 **계층적 잔차**가 모듈 안과 모듈 사이를 잇는다.

실제 구성은 Token Encoder ×4, Concept Module ×2, Token Decoder이며 FULL과 SWA 어텐션을 섞는다.

### 2.2 Learning the Discrete Concept Vocabulary with Vector Quantization

VQ 목적함수는 **코드북 항목이 연속 개념 표현의 분포를 덮도록** 유도한다. 그래야 예측 대상이 되는 구조화된 공간이 생긴다.

Token Encoder의 은닉 상태는 세 곳에 쓰인다.

1. Token Decoder가 소비
2. 개념 단위 표현으로 압축되어 Concept Module로
3. 개념 코드북 학습에

큰 단일 코드북을 쓰지 않고 **product quantization**으로 이산 공간의 용량을 키운다. 코드북을 여러 개로 쪼개 조합 수를 늘리는 방식이다.

### 2.3 Predicting the Next Concept with the Learned Vocabulary

Concept Module이 **각 코드북에 대한 확률 분포**를 내놓고, 그 가중 조합이 미분 가능한 개념 예측이 된다. 이산 코드북을 쓰면서도 역전파가 끊기지 않게 하는 장치다.

### 2.4 Injecting Predicted Concepts into the Token Stream

예측된 개념을 토큰 해상도로 되풀이하고 shift해 Token Decoder에 넣는다. **예측 개념이 이후 생성을 안내**하되 표준 자기회귀 생성은 그대로 유지된다.

### 2.5 Hierarchical Residual Connections

Token Encoder, Concept Module, Token Decoder를 잇는다.

- **2.5.1 Intra-Module**: 모듈 안 레이어 사이
- **2.5.2 Cross-Module**: 세 모듈 사이

학습된 대각 스케일링으로 대상 스트림에 더한다. 정보가 깊이와 계층 양쪽으로 흐르게 하는 것이 목적이다.

## 3. Training

### 3.5 Joint Objective and Gradient Flow

세 목적함수를 함께 최적화한다.

```text
L_total = L_NTP + α · L_NCP + β · L_VQ
```

| 항 | 무엇을 학습시키나 |
|----|-------------------|
| `L_NTP` | 다음 토큰 예측 |
| `L_NCP` | Concept Module과 Token Encoder의 다음 개념 예측 |
| `L_VQ` | 코드북 항목을 연속 개념 표현에 맞춤 |

`α`와 `β`가 두 보조 목적의 가중치다. **세 목적이 처음부터 끝까지 함께 최적화된다.**

나머지 하위 절은 3.1 End-to-End Training Overview, 3.2 Learning the Discrete Concept Vocabulary, 3.3 Next Concept Prediction, 3.4 Next Token Prediction, 3.6 Optimization이다.

## 4. Experiments

### 4.1 Experimental Setup

**OLMo-3-7B를 백본으로 쓰고 OLMo-3의 단계별 데이터 커리큘럼을 그대로 따른다.** Stage-1은 Dolma 3 Mix, Stage-2는 Dolma 3 Dolmino다. 평가는 OLMo-Core의 프로토콜로 30개 벤치마크 계열을 돈다.

설정은 OLMo-3-7B를 유지한다. hidden size 4,096, FFN 11,008, 어텐션 헤드 32다.

**같은 데이터, 같은 평가 프로토콜로 비교한다는 점이 이 실험의 강점이다.**

### 4.2 Main Results

#### 4.2.2 Training Loss Performance

| 단계 | 수렴 속도 | 손실 차이 |
|------|-----------|-----------|
| Stage-1 (5.73T 토큰) | **1.95배 빠름** | 최종 구간에서 0.091 낮음 |
| Stage-2 (약 90B 토큰) | **1.51배 빠름** | 변동이 적고 계속 낮음 |

초록의 핵심 수치가 여기서 나온다. **전체 학습 토큰의 51.3%만 쓰고 OLMo-3-7B의 최종 사전학습 손실에 도달한다.**

#### 4.2.3 Downstream Performance

| 단계 | macro-average 차이 | 두드러진 영역 |
|------|-------------------|---------------|
| Stage-1 | **+2.45점** | MATH(최대 18% 개선), Code, MC-Non-STEM |
| Stage-2 | +0.59점 | MATH, MC-Non-STEM, GenQA |

GSM8K에서 **5.99점** 올랐다.

Stage-2에서 개선 폭이 줄어든 이유를 저자들이 설명한다. **Stage-2에는 코드 데이터가 약 10%뿐이라, 전체 혼합을 더 잘 맞추는 것이 평균은 올리면서 비중이 작은 영역은 약화시킬 수 있다**는 것이다.

### 4.3 Ablation Studies

#### 4.3.1 Matched Model Size and Computation

개선이 아키텍처 덕인지 파라미터나 연산이 늘어서인지를 가른다. OLMo-3 기반 베이스라인 셋과 비교한다. Vanilla, Vanilla size-aligned, Vanilla computation-aligned다.

계산 구조가 깔끔하다. Concept Module은 토큰 은닉 상태 네 개를 개념 하나로 묶으므로 **시퀀스 길이가 약 1/4**이 된다. 따라서 Concept Module 블록 하나는 **표준 블록과 비슷한 파라미터를 쓰면서 연산은 1/4 미만**이다.

결과는 **파라미터 정합 8.9B 40레이어 Transformer의 학습 손실에 연산의 85%만으로 근접**한다.

### 4.5 Scaling Law

여러 FLOPs 예산에서 하이퍼파라미터와 모델 크기 대 데이터 배분을 탐색해 각 예산의 최적 검증 손실을 보고한다. 결과는 **compute-optimal 학습에서 OLMo-3 대비 1.74배 연산 효율**이다.

### 4.6 Model Analysis

#### 4.6.1 Numerical Stability of the OLMo-3-7B Configuration

실무에 쓸모 있는 관찰이다. OLMo-3의 레이어별 Q/K 정규화를 그대로 쓰면 **어텐션 로짓이 계속 커진다.** Q/K 헤드 블록의 행렬 노름 불균형이 함께 나타나고, **소수의 이상치 헤드가 pre-softmax 점수를 지배해** 어텐션 분포가 극단적으로 쏠린다. 커진 Q/K 상태는 불균등한 그래디언트 흐름, 비정상적인 Q/K 그래디언트 노름, 결국 전역 그래디언트 노름 스파이크로 이어진다.

## 5. Additional Results

### 5.1 VQ Training

Stage-1 체크포인트를 코드(Magicoder), 수학(Orca-Math), 지식(TriviaQA-RC)에 각각 이어 학습해 도메인 적응을 본다. **VQ 파라미터 17M만으로 도메인 적응이 된다**는 것이 초록의 주장이다.

### 5.3 Predicted Concept Improves Block-Parallel Speculative Drafting

예측된 개념을 speculative decoding에 쓴다. DFlash2 기반 drafter의 **평균 수용 길이가 4.17% 늘고 오버헤드는 무시할 수준**이다.

## 6. Related Work

6.1 Abstract-Level Prediction, 6.2 Hierarchical and Latent-Space Language Modeling, 6.3 Residual Connections in Deep Transformers로 나뉜다.

## 7. Limitations

저자들이 스스로 적은 한계가 둘이다.

- **장문 컨텍스트 학습이 빠져 있다.** 이 보고서는 표준 컨텍스트 길이에서 5.73T 사전학습과 100B 중간학습까지만 다룬다. **개념 경로가 압축된 시퀀스에서 동작하므로 긴 컨텍스트가 오히려 유리할 수 있다**고 보지만 확인하지 않았다.
- **손실 이점이 다운스트림으로 얼마나 옮겨가는지는 조건에 달렸다.** 손실 우위는 두 단계 모두에서 일관되지만, 다운스트림 개선은 사전학습 뒤 2.45점, 중간학습 뒤 0.59점으로 줄고 벤치마크마다 다르다. 저자들은 이를 **"교차 엔트로피에서 다운스트림 능력으로의 전환이 학습 분포와 평가 도메인에 달려 있다"** 고 정리한다.

## 8. Conclusion

**"언어 모델이 학습한 잠재 표현이 조 단위 토큰 규모에서 일급 예측 대상이 될 수 있다"** 는 것이 이 연구의 주장이다.

| 항목 | 수치 |
|------|------|
| 파라미터 | 8.9B |
| 사전학습 토큰 | 5.73T (Dolma-3) |
| OLMo-3-7B 최종 손실 도달 | 토큰 **51.3%** |
| 다운스트림 macro-average | **+2.45점** |
| GSM8K | **+5.99점** |
| 파라미터 정합 베이스라인 대비 연산 | **85%** |
| 스케일링 법칙 연산 효율 | **1.74배** |
| 도메인 적응 VQ 파라미터 | 17M |
| speculative drafting 평균 수용 길이 | +4.17% |

---

## 읽을 때 감안할 것

- **기술 보고서이고 아직 아키텍처 미리보기다.** 제목의 `ArchPreview`가 그 뜻이다. 전체 학습 레시피가 아니라 아키텍처가 성립하는지를 보인 단계다. 장문 컨텍스트가 빠졌다고 저자들도 명시한다.
- **비교 대상이 OLMo-3-7B 하나다.** 같은 데이터와 평가 프로토콜을 쓴 점은 좋지만, 다른 계열 모델과의 비교가 없어 절대 위치를 알 수 없다.
- **Stage-2에서 개선이 1/4로 줄었다.** 2.45점에서 0.59점이다. 저자들은 데이터 혼합 탓으로 설명하는데, 달리 보면 **중간학습을 거치면 사전학습 단계의 이점이 상당 부분 희석된다**는 뜻이기도 하다. 실제 배포되는 모델은 후자 쪽이다.
- **손실 이점과 다운스트림 이점이 비례하지 않는다.** 저자들이 한계 절에서 직접 인정한다. "51.3% 토큰으로 같은 손실"이라는 수치를 곧바로 "학습 비용 절반"으로 읽으면 안 된다.
- **추론 비용이 정리되어 있지 않다.** Concept Module이 파라미터를 표준 블록만큼 더하고 연산은 1/4이라는 학습 쪽 계산은 있지만, **추론 지연과 메모리에 대한 측정은 본문에서 보이지 않는다.** speculative drafting 개선 4.17%가 유일한 추론 쪽 수치다.
- **개념이 실제로 무엇을 담는지에 대한 분석이 약하다.** 이산 개념 어휘를 학습한다는 것이 핵심 주장인데, 학습된 개념이 해석 가능한 단위에 대응하는지를 보이는 정성 분석이 본문에 거의 없다.
- **수치가 자체 보고다.** 코드나 체크포인트 공개 여부는 본문에서 확인되지 않는다.

## 가져갈 지점

1. **압축된 시퀀스에 별도 경로를 두는 발상**

   개념 경로가 토큰 시퀀스의 1/4 길이에서 도는 구조는, [DeepSeek-V4.1-Flash 노트](%5B20260911%5D%20DeepSeek-V4.1-Flash_KV_Cache_Compression.md)의 CSA2 압축률 `m`과 같은 축이다. **한쪽은 KV 저장을 줄이려고, 한쪽은 학습 신호를 추가하려고 같은 압축을 쓴다.** 긴 컨텍스트에서 둘이 만나는 지점이 있을 것 같다.

2. **보조 목적함수를 더할 때의 비용 계산법**

   `L_total = L_NTP + α·L_NCP + β·L_VQ` 구조에서, Concept Module이 **파라미터는 표준 블록만큼 쓰되 연산은 1/4**이라는 계산이 깔끔하다. 보조 목적을 붙일지 판단할 때 파라미터와 연산을 나눠 재는 방식으로 쓸 만하다.

3. **손실 개선을 성과로 곧바로 환산하지 않기**

   Stage-1에서 2.45점이던 개선이 Stage-2에서 0.59점으로 줄었다는 사실이 중요하다. **학습 손실 그래프가 예쁘다고 최종 제품이 그만큼 좋아지지 않는다.** 평가 설계에서 손실과 과제 성능을 별도 축으로 봐야 하는 근거다.

4. **Q/K 정규화의 수치 불안정**

   소수 이상치 헤드가 pre-softmax를 지배해 그래디언트 스파이크로 이어진다는 관찰은 학습을 돌릴 때 바로 확인할 항목이다. **어텐션 로짓 최대값을 학습 지표로 찍어두면** 터지기 전에 보인다.

## 결론

**"토큰만 예측하게 두면 추상은 부산물로만 생긴다"** 는 문제의식이 이 논문의 출발점이고, 그 해법으로 은닉 상태에서 직접 만든 이산 개념을 예측 대상으로 승격시킨다. 5.73T 토큰까지 밀어붙여 **이 방식이 프런티어 규모에서 성립한다**는 것을 보인 점이 값어치다.

다만 아키텍처 미리보기라는 성격을 놓치면 안 된다. 장문 컨텍스트가 없고, 비교 대상이 하나이며, **Stage-2에서 이점이 1/4로 줄어든다.** 51.3%라는 숫자는 사전학습 손실 기준이지 최종 성능 기준이 아니다.

가져갈 것은 수치보다 **구조**다. 압축된 잠재 경로를 병렬로 두고 세 목적을 함께 학습시키되, 파라미터와 연산을 나눠 계산해 이득을 귀속시키는 방식이다.
