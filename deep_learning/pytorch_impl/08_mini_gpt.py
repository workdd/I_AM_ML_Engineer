"""
Mini GPT 구현

목표: 지금까지 구현한 모든 것을 조립하여 동작하는 GPT를 만듭니다.

이 파일은:
1. 직접 구현한 컴포넌트들을 조립
2. 학습/추론 코드 포함
3. 간단한 텍스트 생성 데모

구현해야 할 것:
1. MiniGPT 클래스 - 전체 모델 조립
2. generate() - 텍스트 생성 (autoregressive decoding)
3. 학습 루프 (보너스)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class MiniGPTConfig:
    """GPT 설정"""

    def __init__(
        self,
        vocab_size: int = 50257,  # GPT-2 vocab size
        d_model: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        d_ff: int = None,  # default: d_model * 4
        max_len: int = 1024,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff if d_ff is not None else d_model * 4
        self.max_len = max_len
        self.dropout = dropout
        self.bias = bias


class MiniGPT(nn.Module):
    """
    Mini GPT (Decoder-only Transformer)

    앞서 구현한 컴포넌트들을 조립하여 완전한 GPT를 만듭니다.
    일단은 nn 모듈을 사용해도 되고, 직접 구현한 것으로 교체해도 됩니다.

    Args:
        config: MiniGPTConfig

    Forward:
        Input: (batch_size, seq_len) - token ids
        Output: (batch_size, seq_len, vocab_size) - logits

    TODO: __init__, forward, generate를 구현하세요
    """

    def __init__(self, config: MiniGPTConfig):
        super().__init__()
        self.config = config

        # ============================================
        # 여기에 구현하세요
        # 필요한 것:
        # - token_embedding: nn.Embedding(vocab_size, d_model)
        # - position_embedding: nn.Embedding(max_len, d_model)
        # - dropout
        # - blocks: DecoderBlock N개 (ModuleList)
        # - final_norm: LayerNorm
        # - lm_head: Linear(d_model, vocab_size)
        #
        # 팁: lm_head와 token_embedding의 weight를 tie할 수 있음 (선택)
        # ============================================
        raise NotImplementedError("MiniGPT.__init__을 구현하세요!")

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_ids: (batch_size, seq_len) - 입력 token ids
            targets: (batch_size, seq_len) - 타겟 token ids (학습 시)

        Returns:
            logits: (batch_size, seq_len, vocab_size)
            loss: cross entropy loss (targets가 주어진 경우)

        구현 단계:
            1. token embedding + position embedding
            2. dropout
            3. decoder blocks 통과
            4. final layer norm
            5. lm_head로 logits 생성
            6. loss 계산 (targets가 있는 경우)
        """
        # ============================================
        # 여기에 구현하세요
        # ============================================
        raise NotImplementedError("MiniGPT.forward를 구현하세요!")

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None,
    ) -> torch.Tensor:
        """
        Autoregressive 텍스트 생성

        Args:
            input_ids: (batch_size, seq_len) - 시작 토큰들
            max_new_tokens: 생성할 최대 토큰 수
            temperature: sampling temperature (높을수록 다양)
            top_k: top-k sampling (None이면 사용 안 함)
            top_p: nucleus sampling (None이면 사용 안 함)

        Returns:
            generated_ids: (batch_size, seq_len + max_new_tokens)

        Decoding 전략:
            - temperature=0 (또는 매우 작은 값): greedy decoding
            - temperature=1, top_k=None, top_p=None: pure sampling
            - top_k=50: top-k sampling
            - top_p=0.9: nucleus sampling

        TODO: 구현하세요
        """
        # ============================================
        # 여기에 구현하세요
        #
        # for _ in range(max_new_tokens):
        #     1. input_ids로 forward (마지막 position만 필요)
        #     2. logits에 temperature 적용
        #     3. top_k / top_p filtering (선택)
        #     4. softmax로 확률 변환
        #     5. sampling (또는 argmax for greedy)
        #     6. 새 토큰을 input_ids에 concat
        # ============================================
        raise NotImplementedError("MiniGPT.generate를 구현하세요!")


def count_parameters(model: nn.Module) -> int:
    """모델 파라미터 수 계산"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 작은 설정으로 테스트
    config = MiniGPTConfig(
        vocab_size=1000,
        d_model=64,
        num_heads=4,
        num_layers=2,
        max_len=128,
    )

    model = MiniGPT(config)
    print(f"Model parameters: {count_parameters(model):,}")

    # Forward test
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    logits, _ = model(input_ids)
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {logits.shape}")

    # Generation test
    prompt = torch.randint(0, config.vocab_size, (1, 5))
    generated = model.generate(prompt, max_new_tokens=20)
    print(f"Generated shape: {generated.shape}")
