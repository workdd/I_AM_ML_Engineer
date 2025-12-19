# Tensor Parallel - LLM 추론 최적화 심층 분석

- **출처**: [liyuan24 블로그](https://liyuan24.github.io/writings/2025_12_18_nanovllm_tensor_parallel_kernel_fusion.html)
- **저자**: liyuan24
- **읽은 날짜**: 2024-12-19
- **태그**: #LLM #TensorParallel #추론최적화 #vLLM #분산처리

## 핵심 내용

nano-vllm 프로젝트 기반으로 LLM 추론 최적화 기술 중 **Tensor Parallelism(TP)**을 상세히 설명.

### LLM 추론 최적화 분류
```
추론 최적화
├── 시스템 최적화
│   ├── Paged Attention
│   ├── Batching
│   └── Scheduler
└── 모델 최적화
    ├── Tensor Parallel ← 이 글의 주제
    ├── Kernel Fusion
    └── Weight Packing
```

### Tensor Parallelism이란?
> 모델 가중치를 여러 GPU에 분산시키고 KV 캐시도 함께 배포하여 **계산 속도와 처리량을 동시에 향상**시키는 기법

## 레이어별 TP 적용 방식

### 1. Embedding Layer
- **어휘 크기(vocab_size)** 차원으로 분할
- 각 GPU는 담당 범위 외 토큰 → 0 출력
- `all_reduce` 연산으로 동기화 (결과 합산)

### 2. Transformer Block

**Attention Layer:**
| 가중치 | 분할 방식 | 통신 필요 |
|--------|----------|----------|
| Q, K, V | 행(row) 분할 | X |
| Output Projection | 열(column) 분할 | O (all_reduce) |

**MLP Layer:**
| 가중치 | 분할 방식 | 통신 필요 |
|--------|----------|----------|
| 첫 번째 Linear | 행(row) 분할 | X |
| 두 번째 Linear | 열(column) 분할 | O (all_reduce) |

### 3. LM Head
- Embedding과 동일한 가중치 구조
- 어휘 크기 차원으로 분할
- `all_gather` 또는 `gather`로 동기화

## 인상 깊은 부분

> 행/열 차원 분할 전략으로 **GPU 간 통신을 최소화**하면서도 대규모 모델을 효율적으로 처리 가능

**핵심 인사이트:**
- Q, K, V를 행 분할하면 각 GPU가 독립적으로 Attention 계산 가능
- Output Projection만 all_reduce 필요 → 통신 오버헤드 최소화
- MLP도 동일 패턴 적용

## 내 생각 / 적용점

### 배운 점
1. **TP는 통신 최소화가 핵심** - 어디서 분할하느냐에 따라 통신 횟수가 달라짐
2. **행 분할 vs 열 분할** 선택이 중요
   - 행 분할: 독립 계산 가능, 통신 불필요
   - 열 분할: all_reduce 필요
3. Transformer 블록당 **2번의 all_reduce**만 필요 (Attention 후 1번, MLP 후 1번)

### 실무 연결
- vLLM, TGI 등 서빙 프레임워크가 내부적으로 TP 적용
- 멀티 GPU 환경에서 LLM 서빙 시 필수 지식
- NCCL 통신 최적화와 연결됨

### 후속 학습
- [ ] Pipeline Parallelism vs Tensor Parallelism 비교
- [ ] Megatron-LM의 TP 구현 분석
- [ ] Kernel Fusion 상세 학습
- [ ] all_reduce, all_gather 동작 원리
