"""
Encoder Block 테스트

실행: pytest deep_learning/pytorch_impl/tests/test_06_encoder_block.py -v
"""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import importlib.util

spec = importlib.util.spec_from_file_location(
    "encoder", Path(__file__).parent.parent / "06_encoder_block.py"
)
encoder_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(encoder_module)

EncoderBlock = encoder_module.EncoderBlock
TransformerEncoder = encoder_module.TransformerEncoder


@pytest.fixture
def d_model():
    return 64


@pytest.fixture
def num_heads():
    return 8


@pytest.fixture
def num_layers():
    return 6


@pytest.fixture
def batch_size():
    return 2


@pytest.fixture
def seq_len():
    return 10


class TestEncoderBlock:
    """EncoderBlock 테스트"""

    def test_output_shape(self, batch_size, seq_len, d_model, num_heads):
        """출력 shape 확인"""
        block = EncoderBlock(d_model, num_heads)
        x = torch.randn(batch_size, seq_len, d_model)

        output = block(x)

        assert output.shape == x.shape

    def test_with_mask(self, batch_size, seq_len, d_model, num_heads):
        """mask와 함께 동작"""
        block = EncoderBlock(d_model, num_heads)
        x = torch.randn(batch_size, seq_len, d_model)
        mask = torch.zeros(seq_len, seq_len).bool()

        output = block(x, mask=mask)

        assert output.shape == x.shape

    def test_pre_norm_vs_post_norm(self, batch_size, seq_len, d_model, num_heads):
        """Pre-LN과 Post-LN 결과가 다른지"""
        torch.manual_seed(42)
        block_pre = EncoderBlock(d_model, num_heads, pre_norm=True)

        torch.manual_seed(42)
        block_post = EncoderBlock(d_model, num_heads, pre_norm=False)

        x = torch.randn(batch_size, seq_len, d_model)

        out_pre = block_pre(x)
        out_post = block_post(x)

        # 구조가 다르므로 결과도 달라야 함
        assert not torch.allclose(out_pre, out_post)

    def test_residual_connection(self, d_model, num_heads):
        """Residual connection이 작동하는지"""
        block = EncoderBlock(d_model, num_heads, dropout=0.0)
        block.eval()

        x = torch.randn(1, 5, d_model)
        output = block(x)

        # residual이 있으므로 출력이 입력과 완전히 다르지 않아야 함
        # (물론 같지도 않음)
        diff = (output - x).abs().mean()
        assert diff > 0  # 변화가 있어야 함
        assert diff < x.abs().mean() * 10  # 너무 크게 변하면 안 됨

    def test_gradient_flow(self, d_model, num_heads):
        """gradient가 잘 흐르는지"""
        block = EncoderBlock(d_model, num_heads)
        x = torch.randn(2, 5, d_model, requires_grad=True)

        output = block(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestTransformerEncoder:
    """TransformerEncoder 테스트"""

    def test_output_shape(self, batch_size, seq_len, d_model, num_heads, num_layers):
        """출력 shape 확인"""
        encoder = TransformerEncoder(d_model, num_heads, num_layers)
        x = torch.randn(batch_size, seq_len, d_model)

        output = encoder(x)

        assert output.shape == x.shape

    def test_num_layers(self, d_model, num_heads):
        """지정한 수의 layer가 있는지"""
        for num_layers in [1, 3, 6, 12]:
            encoder = TransformerEncoder(d_model, num_heads, num_layers)

            # layers 속성 확인
            if hasattr(encoder, "layers"):
                assert len(encoder.layers) == num_layers
            elif hasattr(encoder, "blocks"):
                assert len(encoder.blocks) == num_layers

    def test_different_depths(self, batch_size, seq_len, d_model, num_heads):
        """다양한 depth에서 동작"""
        x = torch.randn(batch_size, seq_len, d_model)

        for num_layers in [1, 2, 4]:
            encoder = TransformerEncoder(d_model, num_heads, num_layers)
            output = encoder(x)
            assert output.shape == x.shape

    def test_with_mask(self, batch_size, seq_len, d_model, num_heads, num_layers):
        """mask와 함께 동작"""
        encoder = TransformerEncoder(d_model, num_heads, num_layers)
        x = torch.randn(batch_size, seq_len, d_model)
        mask = torch.zeros(seq_len, seq_len).bool()

        output = encoder(x, mask=mask)

        assert output.shape == x.shape


# ============================================
# 힌트
# ============================================
"""
=== HINT LEVEL 1: 구조 ===

Pre-LN Encoder Block:
    ┌─────────────┐
    │    Input    │
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │  LayerNorm  │
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │  Self-Attn  │
    └──────┬──────┘
           │
    ┌──────▼──────┐     ┌───────┐
    │   Dropout   │◄────│Residual│
    └──────┬──────┘     └───────┘
           │
    ┌──────▼──────┐
    │  LayerNorm  │
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │     FFN     │
    └──────┬──────┘
           │
    ┌──────▼──────┐     ┌───────┐
    │   Dropout   │◄────│Residual│
    └──────┬──────┘     └───────┘
           │
    ┌──────▼──────┐
    │   Output    │
    └─────────────┘


=== HINT LEVEL 2: 핵심 코드 ===

__init__:
    self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
    self.ffn = nn.Sequential(
        nn.Linear(d_model, d_ff),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(d_ff, d_model),
        nn.Dropout(dropout)
    )
    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)
    self.dropout1 = nn.Dropout(dropout)
    self.dropout2 = nn.Dropout(dropout)

forward (Pre-LN):
    # Self-Attention
    residual = x
    x = self.norm1(x)
    x, _ = self.self_attn(x, x, x, key_padding_mask=None, attn_mask=mask)
    x = self.dropout1(x)
    x = residual + x

    # FFN
    residual = x
    x = self.norm2(x)
    x = self.ffn(x)
    x = residual + x

    return x


=== HINT LEVEL 3: 전체 구현 ===

class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=None, dropout=0.1, pre_norm=True):
        super().__init__()
        if d_ff is None:
            d_ff = d_model * 4

        self.pre_norm = pre_norm

        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        if self.pre_norm:
            # Pre-LN
            residual = x
            x = self.norm1(x)
            x, _ = self.self_attn(x, x, x, attn_mask=mask)
            x = self.dropout1(x)
            x = residual + x

            residual = x
            x = self.norm2(x)
            x = self.ffn(x)
            x = self.dropout2(x)
            x = residual + x
        else:
            # Post-LN
            residual = x
            x, _ = self.self_attn(x, x, x, attn_mask=mask)
            x = self.dropout1(x)
            x = self.norm1(residual + x)

            residual = x
            x = self.ffn(x)
            x = self.dropout2(x)
            x = self.norm2(residual + x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, d_ff=None, dropout=0.1, pre_norm=True):
        super().__init__()

        self.layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff, dropout, pre_norm)
            for _ in range(num_layers)
        ])

        # Pre-LN의 경우 마지막에 LayerNorm 추가
        self.final_norm = nn.LayerNorm(d_model) if pre_norm else None

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)

        if self.final_norm is not None:
            x = self.final_norm(x)

        return x
"""
