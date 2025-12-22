"""
Multi-Head Attention 테스트

실행: pytest deep_learning/pytorch_impl/tests/test_02_multihead_attention.py -v
"""

import pytest
import torch
import torch.nn as nn
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import importlib.util

spec = importlib.util.spec_from_file_location(
    "multihead", Path(__file__).parent.parent / "02_multihead_attention.py"
)
mha_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mha_module)

MultiHeadAttention = mha_module.MultiHeadAttention
MultiHeadSelfAttention = mha_module.MultiHeadSelfAttention


@pytest.fixture
def d_model():
    return 64


@pytest.fixture
def num_heads():
    return 8


@pytest.fixture
def batch_size():
    return 2


@pytest.fixture
def seq_len():
    return 10


class TestMultiHeadAttention:
    """MultiHeadAttention 클래스 테스트"""

    def test_initialization(self, d_model, num_heads):
        """올바르게 초기화되는지 확인"""
        mha = MultiHeadAttention(d_model, num_heads)

        assert mha.d_model == d_model
        assert mha.num_heads == num_heads
        assert mha.d_k == d_model // num_heads

    def test_d_model_divisibility(self):
        """d_model이 num_heads로 나누어 떨어지지 않으면 에러"""
        with pytest.raises(AssertionError):
            MultiHeadAttention(d_model=64, num_heads=7)

    def test_output_shape(self, batch_size, seq_len, d_model, num_heads):
        """출력 shape 확인"""
        mha = MultiHeadAttention(d_model, num_heads)
        q = torch.randn(batch_size, seq_len, d_model)
        k = torch.randn(batch_size, seq_len, d_model)
        v = torch.randn(batch_size, seq_len, d_model)

        output, weights = mha(q, k, v)

        assert output.shape == (batch_size, seq_len, d_model)
        assert weights.shape == (batch_size, num_heads, seq_len, seq_len)

    def test_different_seq_lengths(self, batch_size, d_model, num_heads):
        """Q와 K/V의 시퀀스 길이가 다를 때"""
        mha = MultiHeadAttention(d_model, num_heads)
        seq_len_q, seq_len_k = 5, 10

        q = torch.randn(batch_size, seq_len_q, d_model)
        k = torch.randn(batch_size, seq_len_k, d_model)
        v = torch.randn(batch_size, seq_len_k, d_model)

        output, weights = mha(q, k, v)

        assert output.shape == (batch_size, seq_len_q, d_model)
        assert weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)

    def test_attention_weights_sum_to_one(self, batch_size, seq_len, d_model, num_heads):
        """각 head의 attention weights 합이 1인지"""
        mha = MultiHeadAttention(d_model, num_heads)
        x = torch.randn(batch_size, seq_len, d_model)

        _, weights = mha(x, x, x)

        # (batch, num_heads, seq_q, seq_k)의 마지막 dim 합이 1
        row_sums = weights.sum(dim=-1)
        expected = torch.ones_like(row_sums)

        assert torch.allclose(row_sums, expected, atol=1e-5)

    def test_with_mask(self, batch_size, seq_len, d_model, num_heads):
        """mask가 올바르게 적용되는지"""
        mha = MultiHeadAttention(d_model, num_heads)
        x = torch.randn(batch_size, seq_len, d_model)

        # Causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()

        output, weights = mha(x, x, x, mask=mask)

        # mask된 위치의 weight는 0에 가까워야 함
        mask_expanded = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, seq)
        masked_weights = weights[mask_expanded.expand_as(weights)]

        assert torch.allclose(masked_weights, torch.zeros_like(masked_weights), atol=1e-6)


class TestMultiHeadSelfAttention:
    """MultiHeadSelfAttention 테스트"""

    def test_output_shape(self, batch_size, seq_len, d_model, num_heads):
        """Self-attention wrapper 동작 확인"""
        mhsa = MultiHeadSelfAttention(d_model, num_heads)
        x = torch.randn(batch_size, seq_len, d_model)

        output, weights = mhsa(x)

        assert output.shape == x.shape
        assert weights.shape == (batch_size, num_heads, seq_len, seq_len)


class TestCompareWithPyTorch:
    """PyTorch nn.MultiheadAttention과 비교"""

    def test_compare_shapes(self, batch_size, seq_len, d_model, num_heads):
        """PyTorch 구현과 shape이 동일한지"""
        our_mha = MultiHeadAttention(d_model, num_heads)
        pytorch_mha = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

        x = torch.randn(batch_size, seq_len, d_model)

        our_output, our_weights = our_mha(x, x, x)
        pytorch_output, pytorch_weights = pytorch_mha(x, x, x)

        assert our_output.shape == pytorch_output.shape


# ============================================
# 힌트
# ============================================
"""
=== HINT LEVEL 1: 개념 ===

Multi-Head Attention = 여러 개의 Attention을 병렬로 수행

1. 입력 X를 Q, K, V로 projection
2. 각 Q, K, V를 num_heads개로 split
3. 각 head에서 독립적으로 attention 계산
4. 결과를 concat
5. output projection


=== HINT LEVEL 2: 핵심 연산 ===

__init__:
    self.W_q = nn.Linear(d_model, d_model)
    self.W_k = nn.Linear(d_model, d_model)
    self.W_v = nn.Linear(d_model, d_model)
    self.W_o = nn.Linear(d_model, d_model)

forward:
    # projection
    Q = self.W_q(query)  # (batch, seq, d_model)
    K = self.W_k(key)
    V = self.W_v(value)

    # reshape for multi-head
    # (batch, seq, d_model) -> (batch, seq, num_heads, d_k) -> (batch, num_heads, seq, d_k)
    Q = Q.view(batch, seq, num_heads, d_k).transpose(1, 2)

    # attention (각 head 독립적으로)
    # ...

    # concat & output projection


=== HINT LEVEL 3: 전체 구현 ===

def forward(self, query, key, value, mask=None):
    batch_size = query.size(0)
    seq_len_q = query.size(1)
    seq_len_k = key.size(1)

    # 1. Linear projection
    Q = self.W_q(query)  # (batch, seq_q, d_model)
    K = self.W_k(key)    # (batch, seq_k, d_model)
    V = self.W_v(value)  # (batch, seq_k, d_model)

    # 2. Split into heads
    # (batch, seq, d_model) -> (batch, seq, num_heads, d_k) -> (batch, num_heads, seq, d_k)
    Q = Q.view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)
    K = K.view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
    V = V.view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)

    # 3. Scaled dot-product attention
    # (batch, num_heads, seq_q, d_k) @ (batch, num_heads, d_k, seq_k) = (batch, num_heads, seq_q, seq_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

    if mask is not None:
        # mask shape을 맞춰줌
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask, float('-inf'))

    weights = F.softmax(scores, dim=-1)  # (batch, num_heads, seq_q, seq_k)

    # (batch, num_heads, seq_q, seq_k) @ (batch, num_heads, seq_k, d_k) = (batch, num_heads, seq_q, d_k)
    attn_output = torch.matmul(weights, V)

    # 4. Concat heads
    # (batch, num_heads, seq_q, d_k) -> (batch, seq_q, num_heads, d_k) -> (batch, seq_q, d_model)
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)

    # 5. Output projection
    output = self.W_o(attn_output)

    return output, weights
"""
