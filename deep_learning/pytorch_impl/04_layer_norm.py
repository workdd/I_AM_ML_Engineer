"""
Layer Normalization 구현

테스트: pytest deep_learning/pytorch_impl/tests/test_04_layer_norm.py -v

목표: Layer Normalization을 직접 구현합니다.

수식:
    LayerNorm(x) = γ * (x - μ) / √(σ² + ε) + β

    where:
        μ = mean(x)      (마지막 차원에 대해)
        σ² = var(x)      (마지막 차원에 대해)
        γ, β = 학습 가능한 파라미터

핵심 포인트:
- Batch Norm과 달리 **각 샘플 독립적**으로 정규화
- 마지막 차원(feature dimension)에 대해 평균/분산 계산
- ε(epsilon)은 분모가 0이 되는 것을 방지

Transformer에서의 위치:
- Pre-LN: LayerNorm -> Attention -> Residual
- Post-LN: Attention -> Residual -> LayerNorm (원래 논문)
- 현대 모델은 대부분 Pre-LN 사용

구현해야 할 것:
1. LayerNorm 클래스

금지 사항:
- torch.nn.LayerNorm 사용 금지
- F.layer_norm 사용 금지
"""

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    Layer Normalization

    Args:
        normalized_shape: 정규화할 shape (보통 d_model)
        eps: numerical stability를 위한 작은 값

    Forward:
        Input: (..., normalized_shape)
        Output: (..., normalized_shape)

    TODO: __init__과 forward를 구현하세요
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps

        # ============================================
        # 여기에 구현하세요
        # 필요한 것:
        # - gamma (weight): 학습 가능, 초기값 1
        # - beta (bias): 학습 가능, 초기값 0
        # ============================================
        raise NotImplementedError("LayerNorm.__init__을 구현하세요!")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., normalized_shape)

        Returns:
            output: (..., normalized_shape)

        구현 단계:
            1. 마지막 차원에 대해 mean 계산
            2. 마지막 차원에 대해 variance 계산
            3. 정규화: (x - mean) / sqrt(var + eps)
            4. scale & shift: gamma * normalized + beta
        """
        # ============================================
        # 여기에 구현하세요
        # ============================================
        raise NotImplementedError("LayerNorm.forward를 구현하세요!")


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (LLaMA 등에서 사용)

    수식:
        RMSNorm(x) = x / √(mean(x²) + ε) * γ

    LayerNorm과 차이:
        - mean을 빼지 않음 (centering 없음)
        - bias(β) 없음
        - 더 빠름

    이건 보너스입니다. LayerNorm 구현 후 도전해보세요!
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        # ============================================
        # 보너스: RMSNorm 구현
        # ============================================
        raise NotImplementedError("RMSNorm은 보너스입니다!")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("RMSNorm은 보너스입니다!")


if __name__ == "__main__":
    batch_size, seq_len, d_model = 2, 10, 64

    x = torch.randn(batch_size, seq_len, d_model)
    ln = LayerNorm(d_model)

    output = ln(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # 정규화 확인
    print(f"Output mean (should be ~0): {output.mean(dim=-1).mean():.6f}")
    print(f"Output std (should be ~1): {output.std(dim=-1).mean():.6f}")
