# 메모리 아끼면서 Cross Entropy Loss 계산하기: Fused Linear Cross-Entropy 정리

- **출처**: [Trillion Labs Research - 메모리 아끼면서 Cross Entropy Loss 계산하기](https://trillion-labs.github.io/blog/posts/fused-linear-cross-entropy/)
- **저자**: 석주영
- **게시일**: 2026-07-04
- **읽은 날짜**: 2026-07-06
- **태그**: #CrossEntropy #FusedLinearCrossEntropy #LLMTraining #MemoryOptimization #CUDA #Systems

## 핵심 주장

긴 context와 큰 vocabulary로 LLM을 학습할 때, 마지막 `LM head + Cross Entropy`는 단순한 loss 계산이 아니라 **거대한 `(B, S, V)` logits와 dlogits를 materialize할지 말지의 systems problem**이 된다.

Fused Linear Cross-Entropy(FLCE)는 loss 수식을 바꾸지 않는다. 대신 logits를 chunk별로 만들자마자 loss와 gradient로 소비하고 버려서, 전체 logits/dlogits 텐서가 autograd graph에 오래 남지 않게 한다.

## 문제 상황

글의 출발점은 Gravity 16B를 128K context로 학습하던 중 발생한 OOM이다.

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 37.00 GiB.
```

batch size를 1까지 낮추고 gradient checkpointing을 켰는데도 마지막 cross entropy 계산에서 메모리가 터졌다. Memory Viz로 보면 대부분의 구간은 평탄한데, cross entropy 지점에서만 메모리가 수직으로 치솟는다.

핵심 원인은 마지막 hidden state를 vocabulary 전체로 projection하면서 생기는 logits다.

```text
hidden: (B, S, D)
LM head weight: (V, D)
logits: (B, S, V)
```

요즘 모델은 `V`가 100K를 훌쩍 넘고, `S`도 32K, 128K, 1M까지 커진다. 이 조합에서는 마지막 logits 하나가 모델 weight만큼 커질 수 있다.

## Cross Entropy를 다시 보면

한 토큰의 hidden state를 `h`, LM head weight를 `W`, 정답 token id를 `y`라고 하면:

```text
z = h W^T
loss = -z(y) + log(sum_k exp(z(k)))
```

logit에 대한 gradient는 familiar한 형태다.

```text
dL/dz(j) = softmax(z)(j) - 1[j = y]
```

여기서 중요한 점은 backward에도 logits 기반 확률 `softmax(z)`가 필요하다는 것이다. PyTorch autograd는 backward를 위해 logits 또는 관련 intermediate를 graph에 남겨둔다. 그리고 backward 시점에는 `(B, S, V)` 형태의 dlogits도 생긴다.

즉 long-context training에서는 다음 두 덩어리가 동시에 압박한다.

- forward에서 만든 logits
- backward에서 필요한 dlogits 또는 retained graph buffer

## 왜 logits가 특히 문제인가?

글에서는 `B=4`, `S=4096` 기준으로 여러 frontier 모델의 logits 크기를 비교한다. hidden state는 수백 MB 수준인데, logits는 4~9GB까지 커진다.

예를 들어 GPT-OSS 120B 설정에서는:

| 항목 | 크기 |
|------|------|
| hidden `(B, S, D)` | 0.09GB |
| logits `(B, S, V)` | 6.59GB |
| 비율 | 70배 |

context length가 더 길어지면 `S`에 비례해 선형으로 커진다. `B=1`, 128K context에서도 logits가 약 40GB에 가까워지고, 1M context에서는 수백 GB까지 커진다. 그래서 cross entropy는 마지막에 붙은 작은 연산처럼 보여도 실제로는 학습 가능 여부를 좌우한다.

## 어디를 쪼갤 수 있나?

문제는 `(B, S, V)`가 너무 크다는 것이므로, 어느 축으로든 나눠서 처리하는 방향을 생각할 수 있다.

| 축 | 가능성 | 설명 |
|----|--------|------|
| B | 낮음 | 이미 batch size 1에서도 터지는 상황이라 더 쪼갤 여지가 적음 |
| S | 높음 | token sequence를 chunk로 나누면 loss 합을 chunk별로 분해 가능 |
| V | 가능하지만 복잡 | vocab chunk별 partial sum은 가능하지만 log-sum-exp reduction 설계가 필요 |

글은 이 중 sequence/token axis를 쪼개는 방법에 집중한다.

## 1. Chunked Cross-Entropy

가장 직관적인 방법은 sequence를 여러 chunk로 나눠 각 chunk마다 logits를 만들고 loss를 계산하는 것이다.

```python
def chunked_cross_entropy(x, weight, target, n_chunks=8, ignore_index=-100):
    x = x.reshape(-1, x.shape[-1])
    target = target.reshape(-1)
    loss = x.new_zeros(())
    count = target.new_zeros((), dtype=torch.long)

    for x_i, y_i in zip(x.chunk(n_chunks, dim=0), target.chunk(n_chunks, dim=0)):
        loss = loss + F.cross_entropy(
            x_i @ weight.T,
            y_i,
            ignore_index=ignore_index,
            reduction="sum",
        )
        count = count + (y_i != ignore_index).sum()

    return loss / count.clamp_min(1)
```

forward 순간의 chunk logits 크기는 줄어든다. 문제는 training step peak memory다.

loop에서 chunk별 loss를 더하고 마지막에 한 번만 `backward()`를 부르면, 각 chunk의 CE graph가 최종 loss scalar에 연결된 채 backward 전까지 남아 있어야 한다. 그래서 chunk를 16개, 64개, 1024개로 쪼개도 step peak memory가 약 4.7GB 아래로 잘 내려가지 않는다.

글의 실험 조건은 Gravity-16B-A3B, `V=151,552`, `D=2048`, `B=1`, `S=8192`, bf16이다. 이때 logits 하나가 약 2.31GiB이고, forward logits와 backward buffer 성격의 메모리가 합쳐져 약 4.7GiB plateau가 생긴다.

## 2. Fused Linear Cross-Entropy

FLCE의 핵심은 chunked CE처럼 단순히 logits를 쪼개는 데서 끝나지 않는다. **각 chunk의 logits를 만든 즉시 loss와 gradient를 계산하고, 그 logits buffer를 버린다.**

각 token chunk에서 하는 일은 다음과 같다.

1. 현재 chunk만 LM head projection

```text
logits_chunk = x_chunk @ weight.T
```

2. cross entropy forward와 dlogits 생성

```text
dlogits_chunk = softmax(logits_chunk) - one_hot(target_chunk)
```

3. hidden gradient 계산

```text
dx_chunk = dlogits_chunk @ weight
```

4. weight gradient 누적

```text
dw += dlogits_chunk.T @ x_chunk
```

이렇게 하면 전체 `(B*S, V)` dlogits를 저장하지 않는다. chunk 단위 `(C, V)` buffer만 만들고, 그 buffer를 loss, dlogits, dx, dw 계산에 재사용한 뒤 다음 chunk로 넘어간다.

## Chunked CE와 FLCE의 차이

| 항목 | Chunked CE | FLCE |
|------|------------|------|
| logits 생성 | chunk별 생성 | chunk별 생성 |
| forward peak | 줄어듦 | 줄어듦 |
| training step peak | graph 누적으로 충분히 안 줄어듦 | chunk gradient를 즉시 계산해 줄어듦 |
| autograd 저장 대상 | chunk별 CE graph/intermediate | 미리 계산된 `dx`, `dw`, 일부 마지막 chunk 정보 |
| tradeoff | chunk 수 증가 시 latency만 커질 수 있음 | chunk 크기로 memory-latency 균형 조절 |

FLCE는 항상 무조건 빠른 기법이라기보다, full logits materialization을 피해서 memory와 latency 사이의 선택지를 만든다. chunk를 크게 잡으면 latency도 유리할 수 있고, chunk를 작게 잡으면 memory는 더 줄지만 작은 GEMM과 kernel launch 비용이 늘어난다.

## 구현 흐름

글은 QuACK의 `linear_cross_entropy.py`를 기준으로 구현을 설명한다.

### 1. Autograd shell 준비

입력 `x`와 `weight`를 적절한 dtype으로 변환하고, `x`를 `(B*S, D)`로 펼친다. 핵심 계산은 `chunked_linear_cross_entropy_fwd` 안에서 진행된다.

### 2. Token chunk stream

각 iteration에서는 현재 chunk 크기의 `(C, V)` logits buffer만 사용한다.

```python
torch.mm(x_chunk, weight.mT, out=logits_chunk)
```

### 3. Logits buffer를 dlogits로 재사용

`cross_entropy_fwd_out`은 loss를 계산하면서 같은 buffer를 `softmax(logits) - one_hot(target)`으로 덮어쓴다.

```python
dlogits_chunk = logits_chunk
```

이게 핵심이다. 별도의 거대한 dlogits 텐서를 만들지 않고, chunk logits buffer를 바로 gradient buffer로 바꾼다.

### 4. dx는 materialize하고, full dlogits는 만들지 않음

```python
torch.mm(dlogits_chunk, weight, out=dx_chunk)
```

결과적으로 전체 `dx`는 `(B*S, D)`라서 logits보다 훨씬 작고, 상위 layer로 gradient를 넘기는 데 필요하므로 저장한다.

### 5. dw 누적

chunk별 contribution을 누적한다.

```python
dw += dlogits_chunk.T @ x_chunk
```

마지막 chunk 일부 정보만 backward에서 scalar `dloss`와 함께 마무리할 수 있도록 저장한다.

### 6. Backward는 새 logits를 만들지 않음

backward에서는 이미 계산된 `dx`, `dw`를 upstream scalar gradient로 scale하고, 마지막 chunk의 weight gradient contribution만 정리한다. 큰 logits를 다시 만들지 않는다.

## 실무적으로 중요한 포인트

| 포인트 | 의미 |
|--------|------|
| loss 수식은 그대로 | numerical objective를 바꾸는 trick이 아니라 memory schedule을 바꾸는 최적화 |
| autograd retention이 병목 | forward buffer만 줄여도 backward 전 graph가 남으면 step peak는 잘 안 줄어듦 |
| logits는 activation보다 큼 | 긴 context와 큰 vocab에서는 마지막 projection이 hidden보다 훨씬 큼 |
| kernel fusion의 가치 | matmul, CE, dlogits, dx/dw 계산을 가까운 시점에 묶어 materialization을 피함 |
| chunk size tuning 필요 | 너무 작은 chunk는 memory는 줄이지만 latency가 증가 |

## 내 작업과의 연결

이 글은 LLM 학습/서빙 최적화에서 "수식상 같은 계산"과 "시스템상 같은 계산"이 다르다는 점을 잘 보여준다.

1. **Long-context 학습**

   context를 32K 이상으로 늘릴 때 attention memory만 보는 것은 부족하다. 마지막 LM head와 CE도 `S*V`로 커지므로, long-context 학습 병목 분석에 반드시 포함해야 한다.

2. **Loss 구현도 커널 설계 대상**

   cross entropy는 보통 framework 기본 함수를 그대로 쓰지만, 대규모 LLM에서는 loss 계산 자체가 custom kernel/fused op 대상이 된다. 특히 vocab이 큰 모델에서는 마지막 loss가 전체 step memory를 결정할 수 있다.

3. **Gradient checkpointing의 한계**

   checkpointing은 중간 activation 저장을 줄여주지만, CE에서 materialize되는 logits/dlogits 문제를 자동으로 없애주지는 않는다. FLCE는 checkpointing과 다른 층위의 최적화다.

4. **구현 검증 포인트**

   FLCE류 구현을 도입한다면 단순히 loss 값만 비교하면 부족하다. 다음을 같이 확인해야 한다.

   - naive CE와 loss 값 일치
   - `dx`, `dw` gradient 일치
   - `ignore_index`, reduction semantics 일치
   - bf16/fp32 accumulation 안정성
   - chunk size별 memory-latency curve

## 결론

FLCE의 핵심은 "cross entropy를 더 똑똑한 수식으로 바꾸는 것"이 아니라, **거대한 logits를 언제 만들고 언제 버릴지 다시 설계하는 것**이다.

긴 context와 큰 vocabulary가 표준이 될수록 마지막 loss 계산은 부차적인 구현 디테일이 아니다. full logits와 dlogits를 그대로 materialize하는 기본 구현은 학습 가능성을 막는 병목이 될 수 있고, FLCE 같은 memory-aware loss 계산은 long-context training의 필수 구성요소에 가까워진다.
