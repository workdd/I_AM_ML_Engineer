"""
Transformer 핵심 개념 시각화
1. sin/cos Positional Encoding 패턴
2. Self-Attention Heatmap
"""

import matplotlib.pyplot as plt
import numpy as np

# 한글 폰트 설정 (macOS)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120


def get_positional_encoding(max_len: int, d_model: int) -> np.ndarray:
    """
    sin/cos Positional Encoding 생성

    Args:
        max_len: 최대 시퀀스 길이
        d_model: 임베딩 차원

    Returns:
        (max_len, d_model) 크기의 PE 행렬
    """
    pe = np.zeros((max_len, d_model))
    position = np.arange(max_len)[:, np.newaxis]  # (max_len, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

    pe[:, 0::2] = np.sin(position * div_term)  # 짝수 인덱스: sin
    pe[:, 1::2] = np.cos(position * div_term)  # 홀수 인덱스: cos

    return pe


def visualize_pe_heatmap(max_len: int = 50, d_model: int = 64):
    """PE 전체를 heatmap으로 시각화"""
    pe = get_positional_encoding(max_len, d_model)

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(pe.T, aspect='auto', cmap='RdBu', vmin=-1, vmax=1)

    ax.set_xlabel('Position (위치)', fontsize=12)
    ax.set_ylabel('Dimension (차원)', fontsize=12)
    ax.set_title('Positional Encoding Heatmap\n(빨강=양수, 파랑=음수)', fontsize=14)

    plt.colorbar(im, ax=ax, label='값')
    plt.tight_layout()
    return fig


def visualize_pe_waves(max_len: int = 100, d_model: int = 64):
    """PE의 각 차원별 파형 시각화"""
    pe = get_positional_encoding(max_len, d_model)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # 선택할 차원들 (낮은 주파수 → 높은 주파수)
    dims = [0, 1, 20, 21]
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
    labels = ['dim 0 (sin, 낮은 주파수)',
              'dim 1 (cos, 낮은 주파수)',
              'dim 20 (sin, 높은 주파수)',
              'dim 21 (cos, 높은 주파수)']

    positions = np.arange(max_len)

    for idx, (ax, dim, color, label) in enumerate(zip(axes.flat, dims, colors, labels)):
        ax.plot(positions, pe[:, dim], color=color, linewidth=2)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Position')
        ax.set_ylabel('PE 값')
        ax.set_title(label)
        ax.set_ylim(-1.2, 1.2)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Positional Encoding: 차원별 파형\n(낮은 차원=긴 주기, 높은 차원=짧은 주기)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Softmax 함수"""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def compute_attention(Q: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Scaled Dot-Product Attention 계산"""
    d_k = K.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    return softmax(scores)


def visualize_attention_example():
    """예시 문장으로 Attention 시각화"""
    # 예시 토큰들
    tokens = ['나는', '오늘', '맛있는', '밥을', '먹었다']
    n = len(tokens)
    d_k = 8

    # 랜덤하게 Q, K 생성 (실제론 학습됨)
    np.random.seed(42)
    Q = np.random.randn(n, d_k)
    K = np.random.randn(n, d_k)

    # "나는"이 "밥을", "먹었다"에 더 집중하도록 조작
    K[3] = Q[0] * 0.8 + np.random.randn(d_k) * 0.2  # 밥을
    K[4] = Q[0] * 0.7 + np.random.randn(d_k) * 0.2  # 먹었다

    # "맛있는"이 "밥을"에 집중하도록
    K[3] = K[3] * 0.5 + Q[2] * 0.5

    attention_weights = compute_attention(Q, K)

    # 시각화
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(attention_weights, cmap='Blues', vmin=0, vmax=1)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(tokens, fontsize=11)
    ax.set_yticklabels(tokens, fontsize=11)

    ax.set_xlabel('Key (참조하는 토큰)', fontsize=12)
    ax.set_ylabel('Query (현재 토큰)', fontsize=12)
    ax.set_title('Self-Attention Weights\n(각 토큰이 어디를 주목하는가?)', fontsize=14)

    # 값 표시
    for i in range(n):
        for j in range(n):
            text = f'{attention_weights[i, j]:.2f}'
            color = 'white' if attention_weights[i, j] > 0.5 else 'black'
            ax.text(j, i, text, ha='center', va='center', color=color, fontsize=10)

    plt.colorbar(im, ax=ax, label='Attention Weight')
    plt.tight_layout()
    return fig


def visualize_attention_pattern():
    """다양한 Attention 패턴 시각화"""
    n = 8
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    patterns = []

    # 1. Diagonal (자기 자신 + 이전 토큰)
    diag = np.eye(n) * 0.5
    for i in range(1, n):
        diag[i, i-1] = 0.3
    diag = diag / diag.sum(axis=1, keepdims=True)
    patterns.append(('Local Attention\n(주변 토큰에 집중)', diag))

    # 2. First token (첫 토큰에 집중 - CLS 토큰처럼)
    first = np.ones((n, n)) * 0.1
    first[:, 0] = 0.7
    first = first / first.sum(axis=1, keepdims=True)
    patterns.append(('Global Attention\n(첫 토큰에 집중)', first))

    # 3. Uniform (균등 분포)
    uniform = np.ones((n, n)) / n
    patterns.append(('Uniform Attention\n(모든 토큰 균등)', uniform))

    for ax, (title, pattern) in zip(axes, patterns):
        im = ax.imshow(pattern, cmap='Blues', vmin=0, vmax=1)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('Key')
        ax.set_ylabel('Query')
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle('다양한 Attention 패턴', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


if __name__ == '__main__':
    import os
    os.makedirs('deep_learning/figures', exist_ok=True)

    # 1. PE Heatmap
    fig1 = visualize_pe_heatmap()
    fig1.savefig('deep_learning/figures/pe_heatmap.png', bbox_inches='tight', facecolor='white')
    print('Saved: pe_heatmap.png')

    # 2. PE Waves
    fig2 = visualize_pe_waves()
    fig2.savefig('deep_learning/figures/pe_waves.png', bbox_inches='tight', facecolor='white')
    print('Saved: pe_waves.png')

    # 3. Attention Example
    fig3 = visualize_attention_example()
    fig3.savefig('deep_learning/figures/attention_example.png', bbox_inches='tight', facecolor='white')
    print('Saved: attention_example.png')

    # 4. Attention Patterns
    fig4 = visualize_attention_pattern()
    fig4.savefig('deep_learning/figures/attention_patterns.png', bbox_inches='tight', facecolor='white')
    print('Saved: attention_patterns.png')

    plt.show()
