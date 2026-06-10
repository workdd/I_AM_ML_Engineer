# Multi-Agent 환경에서 logprob 기반 Confidence 활용 설계

> 코드 구현 LLM에 넘기기 위한 구현 명세서.
> 목적: 멀티 에이전트 파이프라인에서 답변을 그대로 쓸지 / 재생성할지 판단할 때,
> 항상 실행되는 fact verification의 **보조 근거**로 logprob 신호를 활용한다.

---

## 0. 핵심 설계 원칙 (먼저 읽을 것)

1. **logprob은 게이트가 아니라 진단 도구다.** 재시도 여부를 결정하는 주체는 항상 fact verification이다. logprob은 "왜 실패했는지"의 성격을 구분하는 데만 쓴다.
2. **logprob은 오답이 아니라 불확실성을 측정한다.** RLHF 정렬 모델은 구조적으로 과신(overconfidence)하므로, logprob이 높다고 정답이 아니다. 따라서 단독 신뢰 금지.
3. **신호는 두 개로 압축.** mean logprob + min logprob. hedging 키워드 매칭 같은 룰베이스는 도입하지 않는다.
4. **threshold는 고정 숫자로 박지 않는다.** 모델·도메인마다 logprob 분포가 다르므로 상대 기준으로 잡는다. 게이트가 아니라 진단용이라 기준이 다소 틀어져도 시스템이 무너지지 않는다.

---

## 1. 수집하는 신호

생성 시 토큰별 logprob을 받아 아래 두 값을 계산한다.

| 신호 | 정의 | 역할 |
|------|------|------|
| `mean_logprob` | 시퀀스 토큰 logprob의 평균 (length-normalized) | 답변 전체의 불확실성 |
| `min_logprob` | 시퀀스 토큰 logprob 중 최솟값 | 가장 불확실했던 국소 지점(특정 클레임) 포착 |

### 왜 length-normalize인가
- 긴 답변일수록 logprob 합(sum)이 자동으로 낮아진다 → 길이 편향.
- 합이 아니라 평균을 써서 길이 무관하게 비교 가능하게 만든다.
- (근거) length-normalized sequence likelihood는 confidence estimation의 표준 기법이며, raw log-likelihood 대비 캘리브레이션을 유의하게 개선한다. → [§참고문헌] R2, R3

### 왜 min을 같이 보는가
- mean만 보면 대부분 토큰이 확실한데 특정 클레임 하나만 불확실한 경우를 놓친다.
- min은 "어느 부분에서 모델이 망설였는지"를 잡아내, 집중 검증 대상을 알려준다.
- (직접 근거) **SelfCheckGPT (Manakul et al., 2023, EMNLP)** 가 문장 단위 hallucination score로 *average negative logprob* 과 *maximum negative logprob* 을 명시적으로 비교한다. 후자(= 문장에서 확률이 가장 낮은 토큰 = **min logprob**)가 hallucination을 효과적으로 포착한다는 결과를 보고했다. → [§참고문헌] R1
  - 주의: SelfCheckGPT는 흔히 self-consistency(N회 샘플링) 방법으로만 알려져 있으나, 논문 안에 **단일 응답의 토큰 확률만으로 점수 내는 변형**(avg/max logprob)이 별도로 존재한다. 이 변형은 추가 샘플링 비용이 없어 본 설계에 그대로 부합한다.
- (보조 맥락) 국소·margin 기반 신호가 Shannon entropy보다 신뢰할 수 있다는 흐름과도 일치한다. 지시 튜닝 모델에서 entropy는 출력 분포가 뾰족해져 판별력을 잃는다. → [§참고문헌] R4

### hedging을 별도 신호로 빼지 않는 이유
- 키워드 매칭은 룰베이스라 커버리지·오탐 관리가 부담.
- LLM 판단으로 하면 호출이 늘어난다.
- 어차피 fact verification 에이전트가 항상 돌고 있으므로, 언어적 망설임은 그 판단 과정에 자연스럽게 흡수시킨다.

---

## 2. 판단 로직 (핵심)

판단의 주체는 fact verification. logprob은 검증 결과와 **교차**해서 실패의 성격을 진단한다.

```
생성 → mean/min logprob 수집 → fact verification (항상 실행)
                                       ↓
                          (검증 결과 × logprob) 교차 진단
```

| fact verification | logprob 신호 | 진단 | 대응 |
|---|---|---|---|
| **통과** | — (무관) | 신뢰 가능 | 최종 응답으로 사용 |
| **실패** | 낮음 (불확실) | 불확실해서 틀림 | 재시도: 더 명확한 지시 + 추가 컨텍스트 |
| **실패** | 높음 (확신) | **자신있는 오답** (가장 위험) | 재시도: 실패 원인을 강하게 주입, 접근 자체를 전환 |

- 검증을 통과하면 logprob 값과 무관하게 답변을 채택한다.
- 검증 실패 시에만 logprob을 꺼내, 재시도 전략의 강도를 조절하는 데 쓴다.

---

## 3. 재시도 전략

- **재시도 시 실패 원인을 컨텍스트로 주입한다.** (같은 분포에서 재추첨하는 "복권 재시도" 방지)
- logprob 진단에 따라 주입 강도를 차등한다:
  - logprob 낮았던 실패 → 부족했던 정보·명확한 지시 보강 위주.
  - logprob 높았던 실패(자신있는 오답) → 실패 사실을 강하게 명시하고, 접근 방식 자체를 바꾸도록 유도.
- `max_attempts` 상한 필수. 무한 재시도 방지 + 비용/레이턴시 상한.
- 상한 도달 시 처리: 도메인 리스크에 따라 (a) 거부/에스컬레이션 (b) 불확실성 명시 후 best 후보 반환 중 선택.

---

## 4. 멀티 에이전트 파이프라인 주의

- 앞 에이전트의 출력이 다음 에이전트의 입력이 되면, **고신뢰 오답이 하류로 전파**된다. logprob 게이트만으로는 못 막는다 → 그래서 각 단계 fact verification이 1차 방어선.
- 파이프라인 전체 신뢰도는 **각 단계 신호의 최솟값(min)** 으로 본다 ("가장 약한 고리"). 한 단계라도 불확실하면 전체가 불안정.
- 중간 단계에서 검증 실패 + 재시도 상한 도달 시, 다음 에이전트로 오염된 컨텍스트를 넘기지 말고 파이프라인을 중단/에스컬레이션한다.

---

## 5. 운영상 주의

- logprob 절대 기준선은 모델·도메인마다 다르다. "낮음/높음"을 **실제 데이터 분포를 보고 상대적으로** 정한다 (예: 해당 도메인 샘플의 분위수 기준).
- 모델 교체·파인튜닝 시 기준선 재설정 필요.
- 진단용이라 기준이 다소 틀어져도 치명적이지 않다. 이것이 logprob을 게이트가 아닌 보조 근거로 두는 설계의 장점.

---

## 6. 의도적으로 배제한 것 (스코프 명시)

- **self-consistency (N회 샘플링 후 의미 일치 검증)**: "자신있는 오답"을 잡는 가장 효과적인 방법이지만 N배 비용이라 현 단계에서 배제. 추후 borderline 케이스에서 precision을 더 끌어올려야 할 때 선택적으로 도입 검토.
- **hedging 키워드 매칭**: 룰베이스 부담으로 배제. fact verification에 흡수.
- **temperature scaling 등 사전 캘리브레이션**: logprob을 게이트가 아니라 진단용으로만 쓰므로 필수 아님. 절대 임계값을 신뢰해야 하는 단계가 생기면 그때 도입.

---

## 7. 구현 시 정해야 할 파라미터 (TODO for 코드 LLM)

- [ ] logprob "낮음/높음" 상대 기준선 산정 방식 (분위수 / 도메인 샘플 기반)
- [ ] `max_attempts` 값
- [ ] 재시도 상한 도달 시 fallback 정책 (거부 / 불확실성 명시 후 best 반환)
- [ ] 파이프라인 단계별 신호 집계 방식 (min 채택 확정, 필요 시 곱 고려)
- [ ] 재시도 컨텍스트 주입 템플릿 (logprob 낮음용 / 높음용 분기)

---

## 참고문헌

| ID | 문헌 | 본 설계와의 연결 |
|----|------|-----------------|
| **R1** | SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection (Manakul et al., 2023, EMNLP) — [arXiv:2303.08896](https://arxiv.org/abs/2303.08896) | **min logprob의 직접 근거.** 문장 단위 avg/max negative logprob 비교, max-neg-logprob(=min prob 토큰)이 hallucination을 잘 포착. 토큰 확률만 쓰는 변형은 추가 샘플링 불필요. |
| **R2** | Contextualized Sequence Likelihood: Enhanced Confidence Scores for NLG (2024) — [arXiv:2406.01806](https://arxiv.org/pdf/2406.01806) | length-normalized sequence likelihood 기반 confidence 강화. mean logprob 사용의 근거. |
| **R3** | Factual Confidence of LLMs: on Reliability and Robustness of Current Estimators (2024) — [arXiv:2406.13415](https://arxiv.org/pdf/2406.13415) | logprob 기반 confidence estimator들의 신뢰성·견고성 비교. length normalization의 캘리브레이션 효과. |
| **R4** | (보조) 지시 튜닝 모델에서 Shannon entropy의 판별력 상실 — [arXiv:2511.09803](https://arxiv.org/html/2511.09803v2) | entropy 대신 margin·국소 신호가 더 신뢰할 수 있다는 흐름. min logprob 채택의 보조 맥락(직접 근거는 R1). |
| **R5** | (배경) RLHF 정렬 모델의 구조적 과신 — [arXiv:2404.02655](https://arxiv.org/pdf/2404.02655) | "logprob 높음 ≠ 정답"의 근거. logprob을 게이트가 아닌 진단 도구로 두는 설계 원칙(§0-2)의 배경. |

> 비고: R1만 본 설계 핵심 신호(min logprob)의 **직접 실험 근거**다. R2·R3은 mean logprob/length-normalization의 표준성 근거, R4·R5는 설계 철학을 뒷받침하는 보조·배경 문헌이다.
