"""
Transformer Decoder Block 구현

테스트: pytest deep_learning/pytorch_impl/tests/test_07_decoder_block.py -v

목표: Decoder Block을 구현합니다. GPT 스타일 (Decoder-only)에 집중.

Encoder-Decoder vs Decoder-only:
- Encoder-Decoder (T5, BART): Cross-attention이 있음
- Decoder-only (GPT, LLaMA): Self-attention만, Causal Mask 사용

구조 (Decoder-only, Pre-LN):
    x -> LayerNorm -> Masked Self-Attention -> + -> LayerNorm -> FFN -> +

핵심 포인트:
- Causal Mask: 미래 토큰을 보지 못하게 함
- 상삼각 행렬로 mask 생성 (triu)

구현해야 할 것:
1. create_causal_mask() - causal mask 생성 함수
2. DecoderBlock - GPT 스타일 decoder block
3. TransformerDecoder - DecoderBlock N개 쌓기
"""

import torch
import torch.nn as nn
from typing import Optional


def create_causal_mask(seq_len: int, device: torch.device = None) -> torch.Tensor:
    """
    Causal (autoregressive) mask 생성

    Args:
        seq_len: 시퀀스 길이
        device: 텐서가 위치할 device

    Returns:
        mask: (seq_len, seq_len) bool tensor
              True인 위치는 attention에서 제외됨

    예시 (seq_len=4):
        [[False,  True,  True,  True],
         [False, False,  True,  True],
         [False, False, False,  True],
         [False, False, False, False]]

    TODO: 구현하세요
    """
    # ============================================
    # 여기에 구현하세요
    # torch.triu() 사용
    # ============================================
    raise NotImplementedError("create_causal_mask를 구현하세요!")


class DecoderBlock(nn.Module):
    """
    Transformer Decoder Block (GPT 스타일)

    Encoder Block과 거의 동일하지만:
    - Causal mask가 항상 적용됨
    - Cross-attention 없음 (Decoder-only)

    Args:
        d_model: 모델 차원
        num_heads: attention head 개수
        d_ff: FFN hidden 차원
        dropout: dropout 확률
        pre_norm: Pre-LN 사용 여부

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
        # EncoderBlock과 동일한 구조
        # ============================================
        raise NotImplementedError("DecoderBlock.__init__을 구현하세요!")

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: 추가 mask (optional, causal mask와 결합됨)

        Returns:
            output: (batch_size, seq_len, d_model)

        주의:
            - causal mask는 항상 적용
            - 추가 mask가 있으면 결합 (OR 연산)
        """
        # ============================================
        # 여기에 구현하세요
        # 1. causal mask 생성
        # 2. 추가 mask와 결합 (있는 경우)
        # 3. EncoderBlock과 동일한 forward
        # ============================================
        raise NotImplementedError("DecoderBlock.forward를 구현하세요!")


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder (GPT 스타일)

    DecoderBlock N개 + Positional Encoding

    Args:
        vocab_size: 어휘 크기
        d_model: 모델 차원
        num_heads: attention head 개수
        num_layers: decoder block 개수
        d_ff: FFN hidden 차원
        max_len: 최대 시퀀스 길이
        dropout: dropout 확률
        pre_norm: Pre-LN 사용 여부

    Forward:
        Input: (batch_size, seq_len) - token ids
        Output: (batch_size, seq_len, vocab_size) - logits
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int = None,
        max_len: int = 5000,
        dropout: float = 0.1,
        pre_norm: bool = True,
    ):
        super().__init__()
        # ============================================
        # 여기에 구현하세요
        # 필요한 것:
        # - token_embedding: nn.Embedding
        # - position_embedding: nn.Embedding 또는 SinusoidalPE
        # - layers: DecoderBlock들
        # - final_norm: LayerNorm (pre_norm인 경우)
        # - lm_head: Linear (d_model -> vocab_size)
        # ============================================
        raise NotImplementedError("TransformerDecoder.__init__을 구현하세요!")

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len) - token ids
            mask: 추가 mask (optional)

        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        # ============================================
        # 여기에 구현하세요
        # 1. token embedding
        # 2. position embedding 더하기
        # 3. dropout
        # 4. decoder blocks 통과
        # 5. final norm (pre_norm인 경우)
        # 6. lm_head로 logits 생성
        # ============================================
        raise NotImplementedError("TransformerDecoder.forward를 구현하세요!")


if __name__ == "__main__":
    batch_size, seq_len, d_model, num_heads, num_layers = 2, 10, 64, 8, 6
    vocab_size = 1000

    # Causal mask 테스트
    mask = create_causal_mask(seq_len)
    print(f"Causal mask shape: {mask.shape}")
    print(f"Causal mask:\n{mask.int()}")

    # Single block
    x = torch.randn(batch_size, seq_len, d_model)
    block = DecoderBlock(d_model, num_heads)
    out_block = block(x)
    print(f"\nDecoderBlock output shape: {out_block.shape}")

    # Full decoder
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    decoder = TransformerDecoder(vocab_size, d_model, num_heads, num_layers)
    logits = decoder(token_ids)
    print(f"TransformerDecoder output shape: {logits.shape}")
