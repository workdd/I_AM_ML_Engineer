"""
Positional Encoding 구현

테스트: pytest deep_learning/pytorch_impl/tests/test_05_positional_encoding.py -v

목표: Sinusoidal Positional Encoding을 직접 구현합니다.

수식 (원래 Transformer):
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

핵심 포인트:
- Transformer는 순서 정보가 없음 → PE로 위치 정보 주입
- sin/cos의 주기가 dimension마다 다름
- 학습 없이 고정된 값 사용 (original) vs 학습 가능 (BERT, GPT)

구현해야 할 것:
1. SinusoidalPositionalEncoding - 원래 Transformer 방식
2. LearnedPositionalEncoding - BERT/GPT 방식 (보너스)
"""

import torch
import torch.nn as nn
import math


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding (원래 Transformer)

    Args:
        d_model: 임베딩 차원
        max_len: 최대 시퀀스 길이
        dropout: dropout 확률

    Forward:
        Input: (batch_size, seq_len, d_model)
        Output: (batch_size, seq_len, d_model)

    특징:
        - 학습하지 않음 (buffer로 등록)
        - 임의의 길이에 대해 일반화 가능
        - 상대적 위치 정보 인코딩 (PE[pos+k]는 PE[pos]의 선형 변환)

    TODO: __init__과 forward를 구현하세요
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        # ============================================
        # 여기에 구현하세요
        #
        # 1. PE 행렬 계산 (max_len, d_model)
        # 2. self.register_buffer('pe', pe)로 등록
        #    - buffer는 학습되지 않지만 state_dict에 저장됨
        #
        # 힌트:
        #   position = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)
        #   div_term = torch.exp(...)  # 10000^(2i/d_model) 계산
        #   pe[:, 0::2] = sin(...)  # 짝수 인덱스
        #   pe[:, 1::2] = cos(...)  # 홀수 인덱스
        # ============================================
        raise NotImplementedError("SinusoidalPositionalEncoding.__init__을 구현하세요!")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)

        Returns:
            output: (batch_size, seq_len, d_model)

        구현:
            x + pe[:seq_len]
        """
        # ============================================
        # 여기에 구현하세요
        # ============================================
        raise NotImplementedError("SinusoidalPositionalEncoding.forward를 구현하세요!")


class LearnedPositionalEncoding(nn.Module):
    """
    Learned Positional Encoding (BERT, GPT 스타일)

    Args:
        d_model: 임베딩 차원
        max_len: 최대 시퀀스 길이
        dropout: dropout 확률

    특징:
        - nn.Embedding으로 위치별 임베딩 학습
        - 고정된 max_len까지만 처리 가능

    이건 간단하니까 바로 도전해보세요!
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        # ============================================
        # 여기에 구현하세요
        # nn.Embedding(max_len, d_model) 사용
        # ============================================
        raise NotImplementedError("LearnedPositionalEncoding.__init__을 구현하세요!")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)

        Returns:
            output: (batch_size, seq_len, d_model)
        """
        # ============================================
        # 여기에 구현하세요
        # positions = torch.arange(seq_len, device=x.device)
        # ============================================
        raise NotImplementedError("LearnedPositionalEncoding.forward를 구현하세요!")


if __name__ == "__main__":
    batch_size, seq_len, d_model = 2, 10, 64

    x = torch.randn(batch_size, seq_len, d_model)

    # Sinusoidal
    sin_pe = SinusoidalPositionalEncoding(d_model)
    output_sin = sin_pe(x)
    print(f"Sinusoidal PE output shape: {output_sin.shape}")

    # Learned
    learned_pe = LearnedPositionalEncoding(d_model)
    output_learned = learned_pe(x)
    print(f"Learned PE output shape: {output_learned.shape}")
