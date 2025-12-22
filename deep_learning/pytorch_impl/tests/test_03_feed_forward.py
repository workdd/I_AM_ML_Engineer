"""
Feed-Forward Network 테스트

실행: pytest deep_learning/pytorch_impl/tests/test_03_feed_forward.py -v
"""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import importlib.util

spec = importlib.util.spec_from_file_location(
    "ffn", Path(__file__).parent.parent / "03_feed_forward.py"
)
ffn_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ffn_module)

PositionWiseFeedForward = ffn_module.PositionWiseFeedForward


@pytest.fixture
def d_model():
    return 64


@pytest.fixture
def d_ff():
    return 256


@pytest.fixture
def batch_size():
    return 2


@pytest.fixture
def seq_len():
    return 10


class TestPositionWiseFeedForward:
    """PositionWiseFeedForward 테스트"""

    def test_initialization(self, d_model, d_ff):
        """올바르게 초기화되는지"""
        ffn = PositionWiseFeedForward(d_model, d_ff)

        assert ffn.d_model == d_model
        assert ffn.d_ff == d_ff

    def test_default_d_ff(self, d_model):
        """d_ff 기본값이 d_model * 4인지"""
        ffn = PositionWiseFeedForward(d_model)

        assert ffn.d_ff == d_model * 4

    def test_output_shape(self, batch_size, seq_len, d_model, d_ff):
        """출력 shape이 입력과 동일한지"""
        ffn = PositionWiseFeedForward(d_model, d_ff)
        x = torch.randn(batch_size, seq_len, d_model)

        output = ffn(x)

        assert output.shape == x.shape

    def test_relu_activation(self, batch_size, seq_len, d_model):
        """ReLU 활성화 함수 동작"""
        ffn = PositionWiseFeedForward(d_model, activation="relu")
        x = torch.randn(batch_size, seq_len, d_model)

        output = ffn(x)
        assert output.shape == x.shape

    def test_gelu_activation(self, batch_size, seq_len, d_model):
        """GELU 활성화 함수 동작"""
        ffn = PositionWiseFeedForward(d_model, activation="gelu")
        x = torch.randn(batch_size, seq_len, d_model)

        output = ffn(x)
        assert output.shape == x.shape

    def test_position_wise_independence(self, d_model):
        """각 position이 독립적으로 처리되는지 확인"""
        ffn = PositionWiseFeedForward(d_model)
        ffn.eval()  # dropout 비활성화

        # 단일 position 입력
        x1 = torch.randn(1, 1, d_model)
        # 여러 position 입력 (첫 번째 position은 x1과 동일)
        x2 = torch.randn(1, 5, d_model)
        x2[:, 0, :] = x1.squeeze(1)

        out1 = ffn(x1)
        out2 = ffn(x2)

        # 첫 번째 position 결과가 동일해야 함
        assert torch.allclose(out1.squeeze(1), out2[:, 0, :], atol=1e-5)

    def test_different_batch_sizes(self, seq_len, d_model):
        """다양한 batch size에서 동작"""
        ffn = PositionWiseFeedForward(d_model)

        for batch_size in [1, 4, 16]:
            x = torch.randn(batch_size, seq_len, d_model)
            output = ffn(x)
            assert output.shape == x.shape

    def test_parameter_count(self, d_model, d_ff):
        """파라미터 수가 올바른지"""
        ffn = PositionWiseFeedForward(d_model, d_ff)

        # fc1: d_model * d_ff + d_ff (weight + bias)
        # fc2: d_ff * d_model + d_model
        expected_params = (d_model * d_ff + d_ff) + (d_ff * d_model + d_model)

        actual_params = sum(p.numel() for p in ffn.parameters())

        assert actual_params == expected_params


# ============================================
# 힌트
# ============================================
"""
=== HINT LEVEL 1: 구조 ===

FFN = Linear -> Activation -> Dropout -> Linear -> Dropout

               d_model    d_ff       d_ff    d_model
Input(d_model) -----> Hidden(d_ff) ------> Output(d_model)


=== HINT LEVEL 2: 핵심 코드 ===

__init__:
    self.fc1 = nn.Linear(d_model, d_ff)
    self.fc2 = nn.Linear(d_ff, d_model)
    self.dropout = nn.Dropout(dropout)

    if activation == 'relu':
        self.activation = F.relu
    elif activation == 'gelu':
        self.activation = F.gelu

forward:
    x = self.fc1(x)
    x = self.activation(x)
    x = self.dropout(x)
    x = self.fc2(x)
    x = self.dropout(x)
    return x


=== HINT LEVEL 3: 전체 구현 ===

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff=None, dropout=0.1, activation='gelu'):
        super().__init__()
        if d_ff is None:
            d_ff = d_model * 4

        self.d_model = d_model
        self.d_ff = d_ff

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'gelu':
            self.activation = F.gelu
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x):
        # x: (batch, seq, d_model)
        x = self.fc1(x)          # (batch, seq, d_ff)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)          # (batch, seq, d_model)
        x = self.dropout(x)
        return x
"""
