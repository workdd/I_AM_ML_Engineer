"""
Self-Attention 구현

목표: Scaled Dot-Product Attention을 직접 구현합니다.

수식:
    Attention(Q, K, V) = softmax(QK^T / √d_k) V

구현해야 할 것:
1. scaled_dot_product_attention() - 핵심 attention 연산
2. SelfAttention 클래스 - Q, K, V projection 포함

금지 사항:
- torch.nn.MultiheadAttention 사용 금지
- torch.nn.functional.scaled_dot_product_attention 사용 금지

허용 연산:
- torch.matmul, torch.bmm
- torch.softmax (또는 F.softmax)
- torch.nn.Linear
- 기본 tensor 연산 (+, -, *, /)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Scaled Dot-Product Attention 계산

    Args:
        query: (batch_size, seq_len, d_k)
        key: (batch_size, seq_len, d_k)
        value: (batch_size, seq_len, d_v)
        mask: (batch_size, seq_len, seq_len) or (seq_len, seq_len), optional
              True인 위치는 attention에서 제외 (masked)

    Returns:
        output: (batch_size, seq_len, d_v)
        attention_weights: (batch_size, seq_len, seq_len)

    구현 단계:
        1. Q와 K^T 내적 계산
        2. √d_k로 스케일링
        3. mask 적용 (있는 경우)
        4. softmax로 attention weights 계산
        5. attention weights와 V 내적

    TODO: 아래 코드를 구현하세요
    """
    # ============================================
    # 여기에 구현하세요
    # ============================================
    raise NotImplementedError("scaled_dot_product_attention을 구현하세요!")


class SelfAttention(nn.Module):
    """
    Self-Attention Layer

    입력을 Q, K, V로 projection한 후 attention 계산

    Args:
        d_model: 입력 차원
        d_k: Query, Key 차원 (default: d_model)
        d_v: Value 차원 (default: d_model)

    Forward:
        Input: (batch_size, seq_len, d_model)
        Output: (batch_size, seq_len, d_v)

    TODO: __init__과 forward를 구현하세요
    """

    def __init__(self, d_model: int, d_k: int = None, d_v: int = None):
        super().__init__()
        # ============================================
        # 여기에 구현하세요
        # 필요한 것: W_q, W_k, W_v (Linear layers)
        # ============================================
        raise NotImplementedError("SelfAttention.__init__을 구현하세요!")

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: optional attention mask

        Returns:
            output: (batch_size, seq_len, d_v)
            attention_weights: (batch_size, seq_len, seq_len)
        """
        # ============================================
        # 여기에 구현하세요
        # 1. x를 Q, K, V로 projection
        # 2. scaled_dot_product_attention 호출
        # ============================================
        raise NotImplementedError("SelfAttention.forward를 구현하세요!")


# ============================================
# 디버깅용 헬퍼 (수정하지 마세요)
# ============================================
def print_shape(name: str, tensor: torch.Tensor):
    """텐서 shape 출력 헬퍼"""
    print(f"{name}: {tensor.shape}")


if __name__ == "__main__":
    # 간단한 테스트
    batch_size, seq_len, d_model = 2, 5, 64

    x = torch.randn(batch_size, seq_len, d_model)
    attention = SelfAttention(d_model)

    output, weights = attention(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
