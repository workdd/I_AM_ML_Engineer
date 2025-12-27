# Logistic Regression을 헷갈리게 하는 머신러닝

- **출처**: [Velog - Kanto](https://velog.io/@yonghyeokrhee/logistic-regression-%EC%9D%84-%ED%97%B7%EA%B0%88%EB%A6%AC%EA%B2%8C-%ED%95%98%EB%8A%94-%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D)
- **저자**: Kanto (칸토)
- **읽은 날짜**: 2025-12-27
- **태그**: #LogisticRegression #MLE #GLM #SGD #통계vsML

## 핵심 내용

기술 면접에서 통계학자와 ML 엔지니어가 같은 Logistic Regression을 두고 서로 다른 이야기를 하는 현상에 대한 분석.

### 통계학 관점 (MLE 기반)

Logistic Regression은 **일반화 선형 모델(GLM)**의 일종으로, **해석 가능성**에 중점:

| 요소 | 설명 |
|------|------|
| Link Function | logit(π) = log(π/(1-π)) = β₀ + β₁x |
| Sigmoid | p(x) = 1/(1+e^(-(β₀+β₁x))) |
| 추정 방법 | MLE (Maximum Likelihood Estimation) |
| 분포 가정 | Bernoulli Distribution |
| 목표 | 계수 해석을 통한 **"왜?"** 이해 |

### 머신러닝 관점 (SGD 기반)

**예측 정확도**에 중점, 해석보다는 성능:

| 요소 | 설명 |
|------|------|
| 최적화 | Gradient Descent (반복적 업데이트) |
| Loss Function | Cross-Entropy |
| 업데이트 | SGD로 데이터 스트리밍하며 weight 조정 |
| 핵심 장점 | sigmoid 미분 형태가 단순: σ(x)(1-σ(x)) → 효율적 backprop |
| 목표 | unseen data에 대한 **예측 성능** 최대화 |

### 핵심 차이점

```
통계학: "이 변수가 결과에 얼마나 영향을 미치는가?" (해석)
ML: "새로운 데이터에 대해 얼마나 정확하게 예측하는가?" (성능)
```

## 인상 깊은 부분

> 둘 다 **동일한 수학적 모델**을 사용하지만, **추정 방법론**과 **적용 맥락**에서 차이가 발생한다.
> - 통계학: 작은 데이터셋, 해석 가능성 중시
> - ML: 대규모 데이터셋, 예측력 중시

## 실무 적용 포인트

1. **면접 대비**: 같은 알고리즘도 관점에 따라 다르게 설명할 수 있음을 인지
2. **상황에 맞는 접근**:
   - 비즈니스 인사이트 필요 → 통계적 관점 (계수 해석)
   - 대규모 예측 서비스 → ML 관점 (SGD, Cross-Entropy)
3. **sigmoid 미분의 우아함**: σ'(x) = σ(x)(1-σ(x)) 형태가 backpropagation에서 계산 효율성 제공
