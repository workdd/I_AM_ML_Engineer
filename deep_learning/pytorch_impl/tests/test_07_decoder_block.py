"""
Decoder Block 테스트

실행: pytest deep_learning/pytorch_impl/tests/test_07_decoder_block.py -v
"""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import importlib.util

spec = importlib.util.spec_from_file_location(
    "decoder", Path(__file__).parent.parent / "07_decoder_block.py"
)
decoder_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(decoder_module)

create_causal_mask = decoder_module.create_causal_mask
DecoderBlock = decoder_module.DecoderBlock
TransformerDecoder = decoder_module.TransformerDecoder


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
def vocab_size():
    return 1000


@pytest.fixture
def batch_size():
    return 2


@pytest.fixture
def seq_len():
    return 10


class TestCausalMask:
    """create_causal_mask 테스트"""

    def test_shape(self, seq_len):
        """mask shape 확인"""
        mask = create_causal_mask(seq_len)

        assert mask.shape == (seq_len, seq_len)

    def test_dtype(self, seq_len):
        """mask dtype이 bool인지"""
        mask = create_causal_mask(seq_len)

        assert mask.dtype == torch.bool

    def test_lower_triangular_false(self, seq_len):
        """하삼각 부분이 False인지 (attention 허용)"""
        mask = create_causal_mask(seq_len)

        # 대각선 포함 하삼각은 False
        for i in range(seq_len):
            for j in range(i + 1):
                assert mask[i, j] == False, f"Position ({i}, {j}) should be False"

    def test_upper_triangular_true(self, seq_len):
        """상삼각 부분이 True인지 (attention 차단)"""
        mask = create_causal_mask(seq_len)

        # 대각선 위는 True
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                assert mask[i, j] == True, f"Position ({i}, {j}) should be True"

    def test_device(self):
        """device 지정이 작동하는지"""
        if torch.cuda.is_available():
            mask = create_causal_mask(5, device=torch.device("cuda"))
            assert mask.device.type == "cuda"


class TestDecoderBlock:
    """DecoderBlock 테스트"""

    def test_output_shape(self, batch_size, seq_len, d_model, num_heads):
        """출력 shape 확인"""
        block = DecoderBlock(d_model, num_heads)
        x = torch.randn(batch_size, seq_len, d_model)

        output = block(x)

        assert output.shape == x.shape

    def test_causal_masking(self, d_model, num_heads):
        """causal masking이 적용되는지 확인"""
        block = DecoderBlock(d_model, num_heads)
        block.eval()

        # 시퀀스 생성
        x1 = torch.randn(1, 5, d_model)
        x2 = x1.clone()
        # x2의 마지막 토큰만 변경
        x2[:, -1, :] = torch.randn(d_model)

        out1 = block(x1)
        out2 = block(x2)

        # causal mask가 적용되면:
        # - 첫 4개 토큰의 출력은 동일해야 함 (마지막 토큰 변경에 영향 받지 않음)
        # - 마지막 토큰의 출력만 다름
        assert torch.allclose(
            out1[:, :-1, :], out2[:, :-1, :], atol=1e-5
        ), "Earlier positions should not be affected by later tokens"

    def test_gradient_flow(self, d_model, num_heads):
        """gradient가 흐르는지"""
        block = DecoderBlock(d_model, num_heads)
        x = torch.randn(2, 5, d_model, requires_grad=True)

        output = block(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None


class TestTransformerDecoder:
    """TransformerDecoder 테스트"""

    def test_output_shape(
        self, batch_size, seq_len, vocab_size, d_model, num_heads, num_layers
    ):
        """출력 shape 확인"""
        decoder = TransformerDecoder(vocab_size, d_model, num_heads, num_layers)
        x = torch.randint(0, vocab_size, (batch_size, seq_len))

        logits = decoder(x)

        assert logits.shape == (batch_size, seq_len, vocab_size)

    def test_autoregressive_property(self, vocab_size, d_model, num_heads, num_layers):
        """autoregressive 성질 확인"""
        decoder = TransformerDecoder(vocab_size, d_model, num_heads, num_layers)
        decoder.eval()

        # 같은 prefix, 다른 suffix
        x1 = torch.randint(0, vocab_size, (1, 10))
        x2 = x1.clone()
        x2[:, 5:] = torch.randint(0, vocab_size, (1, 5))  # 뒤 5개 변경

        logits1 = decoder(x1)
        logits2 = decoder(x2)

        # 앞 5개 position의 logits는 동일해야 함
        assert torch.allclose(logits1[:, :5, :], logits2[:, :5, :], atol=1e-5)

    def test_different_seq_lengths(self, vocab_size, d_model, num_heads, num_layers):
        """다양한 시퀀스 길이에서 동작"""
        decoder = TransformerDecoder(vocab_size, d_model, num_heads, num_layers)

        for seq_len in [1, 5, 20]:
            x = torch.randint(0, vocab_size, (2, seq_len))
            logits = decoder(x)
            assert logits.shape == (2, seq_len, vocab_size)


# ============================================
# 힌트
# ============================================
"""
=== HINT LEVEL 1: Causal Mask ===

Causal mask는 상삼각 행렬:

Position 0: [0, 1, 1, 1]  <- position 0은 자기 자신만 볼 수 있음
Position 1: [0, 0, 1, 1]  <- position 1은 0, 1만 볼 수 있음
Position 2: [0, 0, 0, 1]
Position 3: [0, 0, 0, 0]

torch.triu(diagonal=1) 사용


=== HINT LEVEL 2: 핵심 코드 ===

def create_causal_mask(seq_len, device=None):
    # 상삼각 행렬 생성 (대각선 위가 1)
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mask = mask.bool()
    if device is not None:
        mask = mask.to(device)
    return mask

class DecoderBlock:
    # EncoderBlock과 거의 동일
    # forward에서 causal mask를 항상 적용

    def forward(self, x, mask=None):
        seq_len = x.size(1)
        causal_mask = create_causal_mask(seq_len, device=x.device)

        if mask is not None:
            # 추가 mask와 결합
            causal_mask = causal_mask | mask

        # 나머지는 EncoderBlock과 동일
        ...


=== HINT LEVEL 3: 전체 구현 ===

def create_causal_mask(seq_len, device=None):
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    if device is not None:
        mask = mask.to(device)
    return mask


class DecoderBlock(nn.Module):
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
        seq_len = x.size(1)
        causal_mask = create_causal_mask(seq_len, device=x.device)

        if mask is not None:
            causal_mask = causal_mask | mask

        if self.pre_norm:
            residual = x
            x = self.norm1(x)
            x, _ = self.self_attn(x, x, x, attn_mask=causal_mask)
            x = self.dropout1(x)
            x = residual + x

            residual = x
            x = self.norm2(x)
            x = self.ffn(x)
            x = self.dropout2(x)
            x = residual + x
        else:
            # Post-LN...
            pass

        return x


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers,
                 d_ff=None, max_len=5000, dropout=0.1, pre_norm=True):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)

        self.layers = nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_ff, dropout, pre_norm)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model) if pre_norm else None
        self.lm_head = nn.Linear(d_model, vocab_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x: (batch, seq_len) token ids
        seq_len = x.size(1)

        # Embeddings
        tok_emb = self.token_embedding(x)  # (batch, seq_len, d_model)
        positions = torch.arange(seq_len, device=x.device)
        pos_emb = self.position_embedding(positions)  # (seq_len, d_model)

        x = self.dropout(tok_emb + pos_emb)

        # Decoder blocks
        for layer in self.layers:
            x = layer(x, mask)

        if self.final_norm is not None:
            x = self.final_norm(x)

        # LM head
        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)

        return logits
"""
