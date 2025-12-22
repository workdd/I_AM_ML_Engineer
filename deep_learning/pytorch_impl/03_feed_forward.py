"""
Position-wise Feed-Forward Network 구현

목표: Transformer의 FFN을 직접 구현합니다.

수식:
    FFN(x) = max(0, xW_1 + b_1)W_2 + b_2  (ReLU 버전)
    FFN(x) = GELU(xW_1 + b_1)W_2 + b_2    (GPT 스타일)

구조:
    Linear(d_model -> d_ff) -> Activation -> Linear(d_ff -> d_model)

핵심 포인트:
- d_ff는 보통 d_model의 4배 (예: d_model=512, d_ff=2048)
- "Position-wise" = 각 position에 독립적으로 적용 (같은 weight 공유)
- 활성화 함수: ReLU (원래 Transformer), GELU (GPT, BERT)

구현해야 할 것:
1. PositionWiseFeedForward 클래스
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionWiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network

    Args:
        d_model: 입력/출력 차원
        d_ff: hidden 차원 (보통 d_model * 4)
        dropout: dropout 확률
        activation: 'relu' 또는 'gelu'

    Forward:
        Input: (batch_size, seq_len, d_model)
        Output: (batch_size, seq_len, d_model)

    TODO: __init__과 forward를 구현하세요
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int = None,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        if d_ff is None:
            d_ff = d_model * 4

        self.d_model = d_model
        self.d_ff = d_ff

        # ============================================
        # 여기에 구현하세요
        # 필요한 것:
        # - fc1: Linear(d_model -> d_ff)
        # - fc2: Linear(d_ff -> d_model)
        # - dropout
        # - activation function
        # ============================================
        raise NotImplementedError("PositionWiseFeedForward.__init__을 구현하세요!")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)

        Returns:
            output: (batch_size, seq_len, d_model)
        """
        # ============================================
        # 여기에 구현하세요
        # x -> fc1 -> activation -> dropout -> fc2 -> dropout
        # ============================================
        raise NotImplementedError("PositionWiseFeedForward.forward를 구현하세요!")


class SwiGLU(nn.Module):
    """
    SwiGLU Activation (LLaMA, PaLM 등에서 사용)

    수식:
        SwiGLU(x) = (xW_1) * SiLU(xW_gate) * W_2

    이건 보너스입니다. 기본 FFN 구현 후 도전해보세요!
    """

    def __init__(self, d_model: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        if d_ff is None:
            d_ff = int(d_model * 4 * 2 / 3)  # SwiGLU는 보통 이 비율

        # ============================================
        # 보너스: SwiGLU 구현
        # ============================================
        raise NotImplementedError("SwiGLU는 보너스입니다!")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("SwiGLU는 보너스입니다!")


if __name__ == "__main__":
    batch_size, seq_len, d_model = 2, 10, 64

    x = torch.randn(batch_size, seq_len, d_model)
    ffn = PositionWiseFeedForward(d_model)

    output = ffn(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
