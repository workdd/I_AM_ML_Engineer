"""
Positional Encoding 테스트

실행: pytest deep_learning/pytorch_impl/tests/test_05_positional_encoding.py -v
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
    "pe", Path(__file__).parent.parent / "05_positional_encoding.py"
)
pe_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pe_module)

SinusoidalPositionalEncoding = pe_module.SinusoidalPositionalEncoding
LearnedPositionalEncoding = pe_module.LearnedPositionalEncoding


@pytest.fixture
def d_model():
    return 64


@pytest.fixture
def max_len():
    return 100


@pytest.fixture
def batch_size():
    return 2


@pytest.fixture
def seq_len():
    return 10


class TestSinusoidalPositionalEncoding:
    """SinusoidalPositionalEncoding 테스트"""

    def test_output_shape(self, batch_size, seq_len, d_model, max_len):
        """출력 shape 확인"""
        pe = SinusoidalPositionalEncoding(d_model, max_len)
        x = torch.randn(batch_size, seq_len, d_model)

        output = pe(x)

        assert output.shape == x.shape

    def test_pe_buffer_shape(self, d_model, max_len):
        """PE buffer shape 확인"""
        pe = SinusoidalPositionalEncoding(d_model, max_len)

        assert hasattr(pe, "pe")
        assert pe.pe.shape[0] >= max_len
        assert pe.pe.shape[-1] == d_model

    def test_pe_values_range(self, d_model, max_len):
        """PE 값이 [-1, 1] 범위인지"""
        pe = SinusoidalPositionalEncoding(d_model, max_len)

        assert pe.pe.min() >= -1.0
        assert pe.pe.max() <= 1.0

    def test_different_positions_different_pe(self, d_model, max_len):
        """다른 position은 다른 PE 값을 가지는지"""
        pe = SinusoidalPositionalEncoding(d_model, max_len)

        # position 0과 position 1의 PE가 다른지
        assert not torch.allclose(pe.pe[0], pe.pe[1])

    def test_sin_cos_pattern(self, d_model, max_len):
        """짝수 인덱스는 sin, 홀수 인덱스는 cos인지 확인"""
        pe = SinusoidalPositionalEncoding(d_model, max_len)

        # position 0에서 확인
        pos = 0
        for i in range(0, d_model, 2):
            div_term = 10000 ** (i / d_model)
            expected_sin = math.sin(pos / div_term)
            expected_cos = math.cos(pos / div_term)

            # 짝수 인덱스: sin
            assert abs(pe.pe[pos, i].item() - expected_sin) < 1e-5
            # 홀수 인덱스: cos
            if i + 1 < d_model:
                assert abs(pe.pe[pos, i + 1].item() - expected_cos) < 1e-5

    def test_no_gradient(self, d_model, max_len):
        """PE는 학습되지 않아야 함"""
        pe = SinusoidalPositionalEncoding(d_model, max_len)

        # buffer는 requires_grad=False
        assert not pe.pe.requires_grad

    def test_longer_sequence_than_training(self, d_model):
        """학습 시보다 긴 시퀀스도 처리 가능한지"""
        max_len = 100
        pe = SinusoidalPositionalEncoding(d_model, max_len)

        # max_len보다 짧은 시퀀스
        x_short = torch.randn(1, 50, d_model)
        output_short = pe(x_short)
        assert output_short.shape == x_short.shape


class TestLearnedPositionalEncoding:
    """LearnedPositionalEncoding 테스트"""

    def test_output_shape(self, batch_size, seq_len, d_model, max_len):
        """출력 shape 확인"""
        pe = LearnedPositionalEncoding(d_model, max_len)
        x = torch.randn(batch_size, seq_len, d_model)

        output = pe(x)

        assert output.shape == x.shape

    def test_learnable_parameters(self, d_model, max_len):
        """학습 가능한 파라미터가 있는지"""
        pe = LearnedPositionalEncoding(d_model, max_len)

        params = list(pe.parameters())
        assert len(params) > 0

        # embedding 파라미터
        total_params = sum(p.numel() for p in params)
        assert total_params == max_len * d_model

    def test_gradient_flow(self, d_model, max_len):
        """gradient가 흐르는지"""
        pe = LearnedPositionalEncoding(d_model, max_len)
        x = torch.randn(2, 10, d_model)

        output = pe(x)
        loss = output.sum()
        loss.backward()

        # embedding에 gradient가 있어야 함
        for p in pe.parameters():
            assert p.grad is not None


# ============================================
# 힌트
# ============================================
"""
=== HINT LEVEL 1: 수식 ===

PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

여기서:
- pos: 위치 (0, 1, 2, ...)
- i: 차원 인덱스 (0, 1, 2, ..., d_model/2-1)
- 2i: 짝수 차원, 2i+1: 홀수 차원


=== HINT LEVEL 2: 핵심 코드 ===

# position 벡터 만들기
position = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)

# div_term 계산: 10000^(2i/d_model)
# = exp(2i * log(10000) / d_model)
# = exp(2i * (-log(10000) / d_model))  <- 이렇게 하면 더 안정적
div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

# PE 계산
pe = torch.zeros(max_len, d_model)
pe[:, 0::2] = torch.sin(position * div_term)  # 짝수
pe[:, 1::2] = torch.cos(position * div_term)  # 홀수

# buffer로 등록
self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)


=== HINT LEVEL 3: 전체 구현 ===

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        # PE 계산
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()  # (max_len, 1)

        # div_term = 10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)  # 짝수 인덱스
        pe[:, 1::2] = torch.cos(position * div_term)  # 홀수 인덱스

        # (max_len, d_model) -> (1, max_len, d_model)
        pe = pe.unsqueeze(0)

        # buffer로 등록 (학습 X, state_dict에 저장)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)

        # PE 더하기
        x = x + self.pe[:, :seq_len, :]

        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)  # (seq_len,)

        # 위치 임베딩 더하기
        x = x + self.embedding(positions)  # broadcasting

        return self.dropout(x)
"""
