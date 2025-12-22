"""
PE 시각화 - 깔끔 버전 (설명 최소화)
"""

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150


def get_sincos_pe(seq_len: int, d_model: int) -> np.ndarray:
    pe = np.zeros((seq_len, d_model))
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe


def visualize_sincos_pe():
    """sin/cos PE heatmap"""
    fig, ax = plt.subplots(figsize=(12, 6))

    seq_len, d_model = 50, 64
    pe = get_sincos_pe(seq_len, d_model)

    im = ax.imshow(pe.T, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xlabel('Position', fontsize=12)
    ax.set_ylabel('Dimension', fontsize=12)
    ax.set_title('sin/cos Positional Encoding', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    return fig


def rope_2d_rotate(vec, theta):
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    return np.array([
        vec[0] * cos_t - vec[1] * sin_t,
        vec[0] * sin_t + vec[1] * cos_t
    ])


def visualize_rope():
    """RoPE 회전 + Q·K 내적"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # (1) 벡터 회전
    ax = axes[0]
    theta_circle = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta_circle), np.sin(theta_circle), 'lightgray', linewidth=1)

    base_vec = np.array([0.8, 0.2])
    positions = [0, 2, 4, 6, 8]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(positions)))

    for pos, color in zip(positions, colors):
        theta = pos * 0.25
        rotated = rope_2d_rotate(base_vec, theta)
        ax.annotate('', xy=rotated, xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2.5))
        ax.text(rotated[0]*1.15, rotated[1]*1.15, f'{pos}', fontsize=10, color=color, fontweight='bold')

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.set_title('RoPE: Vector Rotation', fontsize=14, fontweight='bold')
    ax.axis('off')

    # (2) Q·K 내적 (상대 위치)
    ax = axes[1]

    q_base = np.array([0.8, 0.3])
    k_base = np.array([0.6, 0.5])
    m = 10  # Q 위치 고정

    relative_pos = np.arange(-10, 11)
    dot_products = []

    for rel in relative_pos:
        n = m + rel
        q_rot = rope_2d_rotate(q_base, m * 0.25)
        k_rot = rope_2d_rotate(k_base, n * 0.25)
        dot_products.append(np.dot(q_rot, k_rot))

    colors = ['#e74c3c' if r == 0 else '#3498db' for r in relative_pos]
    ax.bar(relative_pos, dot_products, color=colors, alpha=0.8)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
    ax.set_xlabel('Relative Position (n - m)', fontsize=12)
    ax.set_ylabel('Q · K', fontsize=12)
    ax.set_title('RoPE: Q·K by Relative Position', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


if __name__ == '__main__':
    import os
    os.makedirs('deep_learning/figures', exist_ok=True)

    fig1 = visualize_sincos_pe()
    fig1.savefig('deep_learning/figures/sincos_pe_clean.png', bbox_inches='tight', facecolor='white')
    print('Saved: sincos_pe_clean.png')

    fig2 = visualize_rope()
    fig2.savefig('deep_learning/figures/rope_clean.png', bbox_inches='tight', facecolor='white')
    print('Saved: rope_clean.png')
