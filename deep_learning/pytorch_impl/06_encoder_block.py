"""
Transformer Encoder Block 구현

목표: 앞서 구현한 컴포넌트들을 조립하여 Encoder Block을 만듭니다.

구조 (Pre-LN, 현대적):
    x -> LayerNorm -> MultiHeadAttention -> + (residual) -> LayerNorm -> FFN -> + (residual)

구조 (Post-LN, 원래 논문):
    x -> MultiHeadAttention -> + (residual) -> LayerNorm -> FFN -> + (residual) -> LayerNorm

핵심 포인트:
- Residual Connection: 입력을 출력에 더함 (gradient flow 개선)
- Layer Normalization: 학습 안정화
- Pre-LN vs Post-LN: Pre-LN이 학습이 더 안정적 (현대 모델 대부분)

구현해야 할 것:
1. EncoderBlock 클래스
2. TransformerEncoder 클래스 (EncoderBlock N개 쌓기)
"""

import torch
import torch.nn as nn
from typing import Optional

# 이전에 구현한 모듈들을 import (구현 완료 후 주석 해제)
# from .layer_norm_04 import LayerNorm
# from .multihead_attention_02 import MultiHeadAttention
# from .feed_forward_03 import PositionWiseFeedForward


class EncoderBlock(nn.Module):
    """
    Transformer Encoder Block

    Args:
        d_model: 모델 차원
        num_heads: attention head 개수
        d_ff: FFN hidden 차원
        dropout: dropout 확률
        pre_norm: True면 Pre-LN, False면 Post-LN

    Forward:
        Input: (batch_size, seq_len, d_model)
        Output: (batch_size, seq_len, d_model)

    TODO: __init__과 forward를 구현하세요
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int = None,
        dropout: float = 0.1,
        pre_norm: bool = True,
    ):
        super().__init__()
        if d_ff is None:
            d_ff = d_model * 4

        self.pre_norm = pre_norm

        # ============================================
        # 여기에 구현하세요
        # 필요한 것:
        # - self_attn: MultiHeadAttention (또는 nn.MultiheadAttention)
        # - ffn: PositionWiseFeedForward
        # - norm1, norm2: LayerNorm (또는 nn.LayerNorm)
        # - dropout1, dropout2: Dropout
        #
        # 참고: 일단 nn.MultiheadAttention, nn.LayerNorm 사용해도 됨
        #       나중에 직접 구현한 것으로 교체
        # ============================================
        raise NotImplementedError("EncoderBlock.__init__을 구현하세요!")

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: attention mask (optional)

        Returns:
            output: (batch_size, seq_len, d_model)

        Pre-LN 구조:
            # Self-Attention sub-layer
            residual = x
            x = self.norm1(x)
            x = self.self_attn(x, x, x, mask)
            x = self.dropout1(x)
            x = residual + x

            # FFN sub-layer
            residual = x
            x = self.norm2(x)
            x = self.ffn(x)
            x = self.dropout2(x)
            x = residual + x

        Post-LN 구조:
            # Self-Attention sub-layer
            residual = x
            x = self.self_attn(x, x, x, mask)
            x = self.dropout1(x)
            x = self.norm1(residual + x)

            # FFN sub-layer
            residual = x
            x = self.ffn(x)
            x = self.dropout2(x)
            x = self.norm2(residual + x)
        """
        # ============================================
        # 여기에 구현하세요
        # ============================================
        raise NotImplementedError("EncoderBlock.forward를 구현하세요!")


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder (EncoderBlock N개 쌓기)

    Args:
        d_model: 모델 차원
        num_heads: attention head 개수
        num_layers: encoder block 개수
        d_ff: FFN hidden 차원
        dropout: dropout 확률
        pre_norm: Pre-LN 사용 여부

    Forward:
        Input: (batch_size, seq_len, d_model)
        Output: (batch_size, seq_len, d_model)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int = None,
        dropout: float = 0.1,
        pre_norm: bool = True,
    ):
        super().__init__()
        # ============================================
        # 여기에 구현하세요
        # nn.ModuleList로 EncoderBlock들을 쌓기
        # Pre-LN의 경우 마지막에 LayerNorm 추가
        # ============================================
        raise NotImplementedError("TransformerEncoder.__init__을 구현하세요!")

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: attention mask (optional)

        Returns:
            output: (batch_size, seq_len, d_model)
        """
        # ============================================
        # 여기에 구현하세요
        # ============================================
        raise NotImplementedError("TransformerEncoder.forward를 구현하세요!")


if __name__ == "__main__":
    batch_size, seq_len, d_model, num_heads, num_layers = 2, 10, 64, 8, 6

    x = torch.randn(batch_size, seq_len, d_model)

    # Single block
    block = EncoderBlock(d_model, num_heads)
    out_block = block(x)
    print(f"EncoderBlock output shape: {out_block.shape}")

    # Full encoder
    encoder = TransformerEncoder(d_model, num_heads, num_layers)
    out_encoder = encoder(x)
    print(f"TransformerEncoder output shape: {out_encoder.shape}")
