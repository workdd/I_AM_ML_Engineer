# Entropy Calibration Degradation in Instruction-Tuned LLMs

> 관련 논문:
> - arXiv 2511.09803 (TARG) — entropy compression in instruction-tuned models
> - arXiv 2510.08146 — entropy-based calibration in standard vs reasoning models
> - arXiv 2511.11966 — instruction tuning reduces entropy / increases log loss
> - arXiv 2603.21172 — "confidently wrong" failure mode

---

## 핵심 발견

### 1. Instruction Tuning → Entropy 압축 → Calibration 손상

SFT/RLHF 이후 모델의 softmax 분포가 더 sharp해지면서:
- **Shannon entropy 감소** → 판별력 상실
- **Log loss 증가** → 캘리브레이션 오히려 악화
- 결과: entropy로는 "confident + correct"와 "confident + wrong"을 구분 못함

**구체적 데이터 (2511.11966)**:
```
pythia-7b (base):     ECE = 0.13, entropy = 1.32  ← well-calibrated
dolly-v2-7b (SFT):    ECE = 0.36, entropy ↓       ← 3× worse calibration
```

이를 "alignment tax"라 부름 — safety/helpfulness 최적화의 부작용.

---

### 2. Entropy의 "Confidently Wrong" Failure Mode

arXiv 2603.21172 데이터:

| 모델 | Confidently wrong rate |
|------|----------------------|
| 전체 평균 | 14.95% |
| LLaMA-3.1-8B-Instruct | 25.56% |
| LLaMA-3.2-3B-Instruct | 29.71% |

→ **최대 30%의 예측이 낮은 entropy지만 틀린 답**  
→ entropy만으로 selective prediction을 하면 위험

평가 범위: TriviaQA, BioASQ, MedicalQA (태스크 무관하게 발생)

---

### 3. Reasoning Model vs Standard Instruction-Tuned Model

arXiv 2510.08146의 핵심 구분:

```
Reasoning model (o1-style, RLHF/GRPO with CoT):
  → advanced post-training이 sequence-level entropy drop을 유발
  → 정답 지점에서 entropy가 명확히 낮아짐 (early-stopping 신호로 활용 가능)

Standard instruction-tuned model (Llama 3.3 70B 등):
  → entropy-based calibration이 "emergent property"가 아님
  → entropy-based early-stopping 작동 안 함
```

**직접 인용**: "entropy-based confidence calibration represents an emergent property of advanced post-training optimization present in modern reasoning models but notably **absent** in standard instruction-tuned and pre-trained models (Llama 3.3 70B)"

---

### 4. Token Probability는 discriminative하지만 calibration과 불일치

arXiv 2511.09803 등:
- Token probability AUROC: 0.71–0.87 (selective prediction용)
- ECE: 높음 (calibration 불량)
- → "구분은 되지만 확률 값을 신뢰할 수는 없음"

---

## Margin이 대안인 이유

Token top-1/top-2 logit gap (margin):
- Instruction-tuned 모델에서도 dynamic range 유지
- 진짜 ambiguity → gap 좁아짐 → margin score 상승 → retrieval 트리거
- Entropy처럼 압축 효과 없음

**직접 인용 (2511.09803)**: "As instruction-tuned models become more peaked, prefix entropies compress and lose ranking power"

---

## 실무 정리

| 상황 | 권장 signal |
|------|------------|
| Standard instruction-tuned (Llama, Qwen) | **Margin** |
| Reasoning model (o1-style) | Entropy (emergent) 또는 Margin |
| 예산 극히 제한 | Margin 기본, Variance fallback |
| OOD 위험이 높은 배포 환경 | 단일 logit signal 불충분 → ensemble 또는 별도 OOD 탐지 |

---

## 미검증 사항

- GPT-4급 독점 모델에서 margin/entropy 동작 방식 (logit 접근 불가로 인한 블랙박스)
- PPO vs DPO vs GRPO 학습 방식별 entropy 압축 정도 차이 (→ open-problems.md)
