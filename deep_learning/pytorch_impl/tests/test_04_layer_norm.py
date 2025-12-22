"""
Layer Normalization 테스트

실행: pytest deep_learning/pytorch_impl/tests/test_04_layer_norm.py -v
"""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import importlib.util

spec = importlib.util.spec_from_file_location(
    "layer_norm", Path(__file__).parent.parent / "04_layer_norm.py"
)
ln_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ln_module)

LayerNorm = ln_module.LayerNorm


@pytest.fixture
def d_model():
    return 64


@pytest.fixture
def batch_size():
    return 2


@pytest.fixture
def seq_len():
    return 10


class TestLayerNorm:
    """LayerNorm 테스트"""

    def test_initialization(self, d_model):
        """올바르게 초기화되는지"""
        ln = LayerNorm(d_model)

        assert ln.normalized_shape == d_model
        assert hasattr(ln, "weight") or hasattr(ln, "gamma")
        assert hasattr(ln, "bias") or hasattr(ln, "beta")

    def test_output_shape(self, batch_size, seq_len, d_model):
        """출력 shape이 입력과 동일한지"""
        ln = LayerNorm(d_model)
        x = torch.randn(batch_size, seq_len, d_model)

        output = ln(x)

        assert output.shape == x.shape

    def test_normalized_mean(self, batch_size, seq_len, d_model):
        """정규화 후 mean이 ~0인지"""
        ln = LayerNorm(d_model)

        # gamma=1, beta=0일 때 mean이 0에 가까워야 함
        # 초기화 직후 확인
        x = torch.randn(batch_size, seq_len, d_model)
        output = ln(x)

        # 각 position에서 마지막 차원의 mean
        means = output.mean(dim=-1)

        assert torch.allclose(means, torch.zeros_like(means), atol=1e-5)

    def test_normalized_std(self, batch_size, seq_len, d_model):
        """정규화 후 std가 ~1인지"""
        ln = LayerNorm(d_model)
        x = torch.randn(batch_size, seq_len, d_model)

        output = ln(x)

        # 각 position에서 마지막 차원의 std
        stds = output.std(dim=-1, unbiased=False)

        assert torch.allclose(stds, torch.ones_like(stds), atol=1e-4)

    def test_learnable_parameters(self, d_model):
        """gamma, beta가 학습 가능한지"""
        ln = LayerNorm(d_model)

        params = list(ln.parameters())
        assert len(params) == 2  # gamma and beta

        # 둘 다 d_model 크기
        for p in params:
            assert p.shape == (d_model,)
            assert p.requires_grad

    def test_different_shapes(self):
        """다양한 input shape에서 동작"""
        d_model = 32

        ln = LayerNorm(d_model)

        # 2D input
        x2d = torch.randn(10, d_model)
        out2d = ln(x2d)
        assert out2d.shape == x2d.shape

        # 3D input
        x3d = torch.randn(2, 10, d_model)
        out3d = ln(x3d)
        assert out3d.shape == x3d.shape

        # 4D input
        x4d = torch.randn(2, 4, 10, d_model)
        out4d = ln(x4d)
        assert out4d.shape == x4d.shape

    def test_compare_with_pytorch(self, batch_size, seq_len, d_model):
        """PyTorch nn.LayerNorm과 결과 비교"""
        our_ln = LayerNorm(d_model)
        pytorch_ln = nn.LayerNorm(d_model)

        # 같은 weight로 설정
        with torch.no_grad():
            if hasattr(our_ln, "weight"):
                pytorch_ln.weight.copy_(our_ln.weight)
                pytorch_ln.bias.copy_(our_ln.bias)
            else:
                pytorch_ln.weight.copy_(our_ln.gamma)
                pytorch_ln.bias.copy_(our_ln.beta)

        x = torch.randn(batch_size, seq_len, d_model)

        our_output = our_ln(x)
        pytorch_output = pytorch_ln(x)

        assert torch.allclose(our_output, pytorch_output, atol=1e-5)

    def test_gradient_flow(self, d_model):
        """gradient가 잘 흐르는지"""
        ln = LayerNorm(d_model)
        x = torch.randn(2, 5, d_model, requires_grad=True)

        output = ln(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


# ============================================
# 힌트
# ============================================
"""
=== HINT LEVEL 1: 수식 ===

LayerNorm(x) = γ * (x - μ) / √(σ² + ε) + β

1. μ = x.mean(dim=-1, keepdim=True)
2. σ² = x.var(dim=-1, keepdim=True)
3. x_norm = (x - μ) / √(σ² + ε)
4. output = γ * x_norm + β


=== HINT LEVEL 2: 핵심 코드 ===

__init__:
    self.gamma = nn.Parameter(torch.ones(normalized_shape))   # 또는 self.weight
    self.beta = nn.Parameter(torch.zeros(normalized_shape))   # 또는 self.bias

forward:
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    x_norm = (x - mean) / torch.sqrt(var + self.eps)
    return self.gamma * x_norm + self.beta


=== HINT LEVEL 3: 전체 구현 ===

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps

        # 학습 가능한 파라미터
        self.weight = nn.Parameter(torch.ones(normalized_shape))   # gamma
        self.bias = nn.Parameter(torch.zeros(normalized_shape))    # beta

    def forward(self, x):
        # x: (..., normalized_shape)

        # 마지막 차원에 대해 mean, var 계산
        mean = x.mean(dim=-1, keepdim=True)  # (..., 1)
        var = x.var(dim=-1, unbiased=False, keepdim=True)  # (..., 1)

        # 정규화
        x_norm = (x - mean) / torch.sqrt(var + self.eps)  # (..., normalized_shape)

        # scale and shift
        return self.weight * x_norm + self.bias


# 주의: var 계산 시 unbiased=False 사용!
# PyTorch의 기본 var는 unbiased=True (N-1로 나눔)
# LayerNorm은 biased variance를 사용 (N으로 나눔)
"""
