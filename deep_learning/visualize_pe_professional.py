"""
Positional Encoding 전문 시각화
1. sin/cos PE: PE 행렬 heatmap + 주파수별 패턴
2. RoPE: 2D 회전 + 상대 위치 내적
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150


# ============================================================
# sin/cos Positional Encoding
# ============================================================
def get_sincos_pe(seq_len: int, d_model: int) -> np.ndarray:
    """Transformer 원본 논문의 PE 계산"""
    pe = np.zeros((seq_len, d_model))
    position = np.arange(seq_len)[:, np.newaxis]

    div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe


def visualize_sincos_pe_professional():
    """sin/cos PE 전문 시각화"""
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.2, 1])

    seq_len, d_model = 64, 128
    pe = get_sincos_pe(seq_len, d_model)

    # (1) PE 행렬 Heatmap
    ax1 = fig.add_subplot(gs[0, :])
    im = ax1.imshow(pe.T, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    ax1.set_xlabel('Position (토큰 위치)', fontsize=12)
    ax1.set_ylabel('Embedding Dimension', fontsize=12)
    ax1.set_title('Positional Encoding Matrix\n(낮은 차원=저주파, 높은 차원=고주파)', fontsize=14, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax1, orientation='vertical', shrink=0.8)
    cbar.set_label('PE Value', fontsize=10)

    # (2) 저주파 vs 고주파 비교
    ax2 = fig.add_subplot(gs[1, 0])
    positions = np.arange(seq_len)

    # 저주파 (dim 0, 1)
    ax2.plot(positions, pe[:, 0], 'b-', linewidth=2, label='dim 0 (sin, 저주파)')
    ax2.plot(positions, pe[:, 1], 'b--', linewidth=2, label='dim 1 (cos, 저주파)')

    # 고주파 (dim 126, 127)
    ax2.plot(positions, pe[:, 126], 'r-', linewidth=2, label='dim 126 (sin, 고주파)')
    ax2.plot(positions, pe[:, 127], 'r--', linewidth=2, label='dim 127 (cos, 고주파)')

    ax2.set_xlabel('Position', fontsize=12)
    ax2.set_ylabel('PE Value', fontsize=12)
    ax2.set_title('저주파 vs 고주파 패턴', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-1.3, 1.3)

    # (3) 위치 간 유사도 (내적)
    ax3 = fig.add_subplot(gs[1, 1])

    # 위치 0, 16, 32를 기준으로 다른 위치와의 유사도
    ref_positions = [0, 16, 32]
    colors = ['#e74c3c', '#3498db', '#2ecc71']

    for ref_pos, color in zip(ref_positions, colors):
        similarities = []
        for p in positions:
            # 코사인 유사도
            sim = np.dot(pe[ref_pos], pe[p]) / (np.linalg.norm(pe[ref_pos]) * np.linalg.norm(pe[p]))
            similarities.append(sim)
        ax3.plot(positions, similarities, color=color, linewidth=2, label=f'기준: pos {ref_pos}')

    ax3.set_xlabel('Position', fontsize=12)
    ax3.set_ylabel('Cosine Similarity', fontsize=12)
    ax3.set_title('위치 간 PE 유사도\n(가까운 위치일수록 유사)', fontsize=13, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ============================================================
# RoPE (Rotary Position Embedding)
# ============================================================
def rope_2d_rotate(vec: np.ndarray, theta: float) -> np.ndarray:
    """2D 회전 행렬 적용"""
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([
        [cos_t, -sin_t],
        [sin_t, cos_t]
    ])
    return rotation_matrix @ vec


def visualize_rope_professional():
    """RoPE 전문 시각화"""
    fig = plt.figure(figsize=(14, 6))
    gs = GridSpec(1, 2, figure=fig)

    # (1) 2D 공간에서 회전 시각화
    ax1 = fig.add_subplot(gs[0, 0])

    # 원 배경
    theta_circle = np.linspace(0, 2*np.pi, 100)
    ax1.plot(np.cos(theta_circle), np.sin(theta_circle), 'lightgray', linewidth=1)
    ax1.plot(0.5*np.cos(theta_circle), 0.5*np.sin(theta_circle), 'lightgray', linewidth=0.5, linestyle='--')

    # Q 벡터 (위치 m에서)
    q_base = np.array([0.8, 0.3])
    m = 3  # Q의 위치
    theta_m = m * 0.3  # 위치에 비례한 회전각

    q_rotated = rope_2d_rotate(q_base, theta_m)

    # K 벡터 (여러 위치 n에서)
    k_base = np.array([0.6, 0.5])
    n_positions = [1, 3, 5, 7]
    k_colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']

    # Q 그리기
    ax1.annotate('', xy=q_rotated, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=3))
    ax1.text(q_rotated[0]+0.1, q_rotated[1]+0.1, f'Q (pos={m})',
            fontsize=11, color='#e74c3c', fontweight='bold')

    # K들 그리기
    for n, color in zip(n_positions, k_colors):
        theta_n = n * 0.3
        k_rotated = rope_2d_rotate(k_base, theta_n)
        ax1.annotate('', xy=k_rotated, xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2, alpha=0.7))
        ax1.text(k_rotated[0]+0.05, k_rotated[1]-0.15, f'K (pos={n})',
                fontsize=9, color=color)

    ax1.set_xlim(-1.2, 1.2)
    ax1.set_ylim(-1.2, 1.2)
    ax1.set_aspect('equal')
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax1.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title('RoPE: Q와 K 벡터의 회전\n(위치에 따라 다른 각도로 회전)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.2)

    # (2) Q·K 내적: 상대 위치에만 의존
    ax2 = fig.add_subplot(gs[0, 1])

    q_base = np.array([0.8, 0.3])
    k_base = np.array([0.6, 0.5])

    # Q 위치 고정 (m=10), K 위치 변화
    m = 10
    n_range = np.arange(0, 21)
    relative_positions = n_range - m  # 상대 위치

    dot_products = []
    for n in n_range:
        theta_m = m * 0.3
        theta_n = n * 0.3

        q_rot = rope_2d_rotate(q_base, theta_m)
        k_rot = rope_2d_rotate(k_base, theta_n)

        dot = np.dot(q_rot, k_rot)
        dot_products.append(dot)

    # 상대 위치로 x축 표시
    colors = ['#e74c3c' if rp == 0 else '#3498db' for rp in relative_positions]
    bars = ax2.bar(relative_positions, dot_products, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='같은 위치 (m=n)')
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)

    ax2.set_xlabel('상대 위치 (n - m)', fontsize=12)
    ax2.set_ylabel('Q · K (내적)', fontsize=12)
    ax2.set_title('RoPE의 핵심: Q·K는 상대 위치에만 의존!\n(절대 위치가 아닌 거리 차이만 중요)', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


if __name__ == '__main__':
    import os
    os.makedirs('deep_learning/figures', exist_ok=True)

    # 1. sin/cos PE
    fig1 = visualize_sincos_pe_professional()
    fig1.savefig('deep_learning/figures/sincos_pe_pro.png', bbox_inches='tight', facecolor='white')
    print('Saved: sincos_pe_pro.png')

    # 2. RoPE
    fig2 = visualize_rope_professional()
    fig2.savefig('deep_learning/figures/rope_pro.png', bbox_inches='tight', facecolor='white')
    print('Saved: rope_pro.png')

    plt.show()
