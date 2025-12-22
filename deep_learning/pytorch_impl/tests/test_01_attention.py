"""
Self-Attention 테스트

실행: pytest deep_learning/pytorch_impl/tests/test_01_attention.py -v

테스트 통과 조건:
1. scaled_dot_product_attention이 올바른 shape 반환
2. attention weights 합이 1 (softmax 검증)
3. mask가 올바르게 적용됨
4. SelfAttention 클래스가 올바르게 동작
5. (보너스) PyTorch 공식 구현과 결과 비교
"""

import pytest
import torch
import torch.nn.functional as F
import math
import sys
from pathlib import Path

# 상위 디렉토리를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

# 숫자로 시작하는 모듈 import
import importlib.util

spec = importlib.util.spec_from_file_location(
    "attention", Path(__file__).parent.parent / "01_attention.py"
)
attention_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(attention_module)

scaled_dot_product_attention = attention_module.scaled_dot_product_attention
SelfAttention = attention_module.SelfAttention


# ============================================
# 테스트 Fixtures
# ============================================
@pytest.fixture
def batch_size():
    return 2


@pytest.fixture
def seq_len():
    return 5


@pytest.fixture
def d_model():
    return 64


@pytest.fixture
def sample_qkv(batch_size, seq_len, d_model):
    """테스트용 Q, K, V 생성"""
    torch.manual_seed(42)
    q = torch.randn(batch_size, seq_len, d_model)
    k = torch.randn(batch_size, seq_len, d_model)
    v = torch.randn(batch_size, seq_len, d_model)
    return q, k, v


# ============================================
# Test: scaled_dot_product_attention
# ============================================
class TestScaledDotProductAttention:
    """scaled_dot_product_attention 함수 테스트"""

    def test_output_shape(self, sample_qkv):
        """출력 shape이 올바른지 확인"""
        q, k, v = sample_qkv
        output, weights = scaled_dot_product_attention(q, k, v)

        assert output.shape == v.shape, f"Output shape mismatch: {output.shape} != {v.shape}"
        assert weights.shape == (
            q.shape[0],
            q.shape[1],
            k.shape[1],
        ), f"Weights shape mismatch: {weights.shape}"

    def test_attention_weights_sum_to_one(self, sample_qkv):
        """attention weights의 각 row 합이 1인지 확인 (softmax 검증)"""
        q, k, v = sample_qkv
        _, weights = scaled_dot_product_attention(q, k, v)

        # 각 query position에서 weights 합이 1이어야 함
        row_sums = weights.sum(dim=-1)
        expected = torch.ones_like(row_sums)

        assert torch.allclose(
            row_sums, expected, atol=1e-5
        ), f"Attention weights don't sum to 1: {row_sums}"

    def test_scaling_factor(self, batch_size, seq_len, d_model):
        """√d_k 스케일링이 적용되는지 확인"""
        torch.manual_seed(42)
        q = torch.randn(batch_size, seq_len, d_model)
        k = torch.randn(batch_size, seq_len, d_model)
        v = torch.randn(batch_size, seq_len, d_model)

        output, weights = scaled_dot_product_attention(q, k, v)

        # 스케일링 없이 계산한 결과와 비교
        # 스케일링이 적용되면 weights가 더 균등해야 함
        unscaled_scores = torch.bmm(q, k.transpose(-2, -1))
        scaled_scores = unscaled_scores / math.sqrt(d_model)

        # scores의 variance 확인 (스케일링 후 더 작아야 함)
        assert scaled_scores.var() < unscaled_scores.var(), "Scaling should reduce variance"

    def test_mask_applied(self, sample_qkv):
        """mask가 올바르게 적용되는지 확인"""
        q, k, v = sample_qkv
        batch_size, seq_len, _ = q.shape

        # 상삼각 mask (causal mask)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)

        _, weights = scaled_dot_product_attention(q, k, v, mask=mask)

        # mask된 위치의 attention weight는 0에 가까워야 함
        masked_weights = weights[mask]
        assert torch.allclose(
            masked_weights, torch.zeros_like(masked_weights), atol=1e-6
        ), f"Masked positions should have ~0 weight: {masked_weights.max()}"

    def test_no_mask_vs_all_false_mask(self, sample_qkv):
        """mask=None과 mask=all_False가 동일한 결과를 내는지"""
        q, k, v = sample_qkv

        output1, weights1 = scaled_dot_product_attention(q, k, v, mask=None)

        all_false_mask = torch.zeros(q.shape[0], q.shape[1], k.shape[1]).bool()
        output2, weights2 = scaled_dot_product_attention(q, k, v, mask=all_false_mask)

        assert torch.allclose(output1, output2, atol=1e-5)
        assert torch.allclose(weights1, weights2, atol=1e-5)


# ============================================
# Test: SelfAttention Class
# ============================================
class TestSelfAttention:
    """SelfAttention 클래스 테스트"""

    def test_initialization(self, d_model):
        """클래스가 올바르게 초기화되는지 확인"""
        attention = SelfAttention(d_model)

        # Linear layers가 있는지 확인
        assert hasattr(attention, "W_q") or hasattr(
            attention, "query_proj"
        ), "Query projection layer missing"
        assert hasattr(attention, "W_k") or hasattr(
            attention, "key_proj"
        ), "Key projection layer missing"
        assert hasattr(attention, "W_v") or hasattr(
            attention, "value_proj"
        ), "Value projection layer missing"

    def test_forward_shape(self, batch_size, seq_len, d_model):
        """forward의 출력 shape 확인"""
        x = torch.randn(batch_size, seq_len, d_model)
        attention = SelfAttention(d_model)

        output, weights = attention(x)

        assert output.shape == x.shape, f"Output shape mismatch: {output.shape}"
        assert weights.shape == (
            batch_size,
            seq_len,
            seq_len,
        ), f"Weights shape mismatch: {weights.shape}"

    def test_different_dimensions(self):
        """d_k, d_v가 d_model과 다를 때도 동작하는지"""
        d_model, d_k, d_v = 64, 32, 48
        batch_size, seq_len = 2, 5

        attention = SelfAttention(d_model, d_k=d_k, d_v=d_v)
        x = torch.randn(batch_size, seq_len, d_model)

        output, weights = attention(x)

        assert output.shape == (batch_size, seq_len, d_v)
        assert weights.shape == (batch_size, seq_len, seq_len)

    def test_with_mask(self, batch_size, seq_len, d_model):
        """mask와 함께 동작하는지 확인"""
        x = torch.randn(batch_size, seq_len, d_model)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()

        attention = SelfAttention(d_model)
        output, weights = attention(x, mask=mask)

        assert output.shape == x.shape


# ============================================
# Bonus: PyTorch 공식 구현과 비교
# ============================================
class TestCompareWithPyTorch:
    """PyTorch 공식 구현과 결과 비교 (보너스)"""

    def test_compare_with_pytorch_sdpa(self, sample_qkv):
        """F.scaled_dot_product_attention과 결과 비교"""
        q, k, v = sample_qkv

        # 직접 구현
        our_output, _ = scaled_dot_product_attention(q, k, v)

        # PyTorch 공식 구현
        pytorch_output = F.scaled_dot_product_attention(q, k, v)

        assert torch.allclose(
            our_output, pytorch_output, atol=1e-5
        ), "Output differs from PyTorch implementation"


# ============================================
# 힌트 (막힐 때 참고하세요)
# ============================================
"""
=== HINT LEVEL 1: 수식 ===

Attention(Q, K, V) = softmax(QK^T / √d_k) V

1. scores = Q @ K^T          # (batch, seq, seq)
2. scores = scores / √d_k    # scaling
3. if mask: scores[mask] = -inf
4. weights = softmax(scores) # (batch, seq, seq)
5. output = weights @ V      # (batch, seq, d_v)


=== HINT LEVEL 2: 핵심 torch 함수 ===

- Q @ K^T: torch.bmm(q, k.transpose(-2, -1)) 또는 torch.matmul(q, k.transpose(-2, -1))
- √d_k: math.sqrt(d_k) 또는 d_k ** 0.5
- mask 적용: scores.masked_fill(mask, float('-inf'))
- softmax: F.softmax(scores, dim=-1)
- weights @ V: torch.bmm(weights, v)


=== HINT LEVEL 3: 전체 shape 흐름 ===

def scaled_dot_product_attention(query, key, value, mask=None):
    # query: (batch, seq_q, d_k)
    # key:   (batch, seq_k, d_k)
    # value: (batch, seq_k, d_v)

    d_k = query.size(-1)

    # Step 1: QK^T
    # (batch, seq_q, d_k) @ (batch, d_k, seq_k) = (batch, seq_q, seq_k)
    scores = torch.bmm(query, key.transpose(-2, -1))

    # Step 2: Scale
    scores = scores / math.sqrt(d_k)

    # Step 3: Mask (True인 위치를 -inf로)
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))

    # Step 4: Softmax (마지막 dim에 대해)
    weights = F.softmax(scores, dim=-1)

    # Step 5: Weighted sum with V
    # (batch, seq_q, seq_k) @ (batch, seq_k, d_v) = (batch, seq_q, d_v)
    output = torch.bmm(weights, value)

    return output, weights
"""
