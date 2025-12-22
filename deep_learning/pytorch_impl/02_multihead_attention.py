"""
Multi-Head Attention 구현

목표: Multi-Head Attention을 직접 구현합니다.

수식:
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
    where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)

핵심 아이디어:
- 하나의 attention 대신 여러 개의 "head"로 병렬 처리
- 각 head는 서로 다른 관계 패턴을 학습
- d_model을 num_heads로 나눠서 d_k = d_model / num_heads

구현해야 할 것:
1. MultiHeadAttention 클래스

금지 사항:
- torch.nn.MultiheadAttention 사용 금지

허용 연산:
- 01_attention.py의 scaled_dot_product_attention 사용 가능
- torch.nn.Linear
- tensor reshape, transpose, view, split, chunk, cat
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math

# 01_attention에서 가져오기 (구현 완료 후 사용)
# from .attention_01 import scaled_dot_product_attention


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """01_attention.py에서 구현한 함수를 여기에 복사하거나 import하세요"""
    d_k = query.size(-1)
    scores = torch.bmm(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    output = torch.bmm(weights, value)
    return output, weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention Layer

    Args:
        d_model: 입력/출력 차원
        num_heads: attention head 개수
        dropout: dropout 확률 (optional)

    주의:
        - d_model은 num_heads로 나누어 떨어져야 함
        - d_k = d_v = d_model // num_heads

    Forward:
        Input: (batch_size, seq_len, d_model)
        Output: (batch_size, seq_len, d_model)

    구현 단계:
        1. Q, K, V projection (각각 d_model -> d_model)
        2. num_heads개로 split
        3. 각 head에 대해 attention 계산
        4. head들을 concat
        5. output projection

    TODO: __init__과 forward를 구현하세요
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # ============================================
        # 여기에 구현하세요
        # 필요한 것:
        # - W_q, W_k, W_v: (d_model -> d_model) Linear
        # - W_o: output projection (d_model -> d_model) Linear
        # - dropout (optional)
        # ============================================
        raise NotImplementedError("MultiHeadAttention.__init__을 구현하세요!")

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (batch_size, seq_len_q, d_model)
            key: (batch_size, seq_len_k, d_model)
            value: (batch_size, seq_len_k, d_model)
            mask: (batch_size, seq_len_q, seq_len_k) or broadcastable

        Returns:
            output: (batch_size, seq_len_q, d_model)
            attention_weights: (batch_size, num_heads, seq_len_q, seq_len_k)

        구현 힌트:
            1. Q, K, V projection
            2. reshape: (batch, seq, d_model) -> (batch, seq, num_heads, d_k)
            3. transpose: (batch, seq, num_heads, d_k) -> (batch, num_heads, seq, d_k)
            4. attention 계산 (batch*num_heads를 batch로 취급)
            5. transpose back & reshape
            6. output projection
        """
        # ============================================
        # 여기에 구현하세요
        # ============================================
        raise NotImplementedError("MultiHeadAttention.forward를 구현하세요!")


class MultiHeadSelfAttention(MultiHeadAttention):
    """
    Self-Attention 버전 (Q=K=V=X)

    편의를 위한 wrapper class
    """

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: optional attention mask

        Returns:
            output: (batch_size, seq_len, d_model)
            attention_weights: (batch_size, num_heads, seq_len, seq_len)
        """
        return super().forward(x, x, x, mask)


if __name__ == "__main__":
    batch_size, seq_len, d_model, num_heads = 2, 10, 64, 8

    x = torch.randn(batch_size, seq_len, d_model)
    mha = MultiHeadSelfAttention(d_model, num_heads)

    output, weights = mha(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
