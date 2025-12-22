"""
심플한 PE 시각화 - 핵심만!
"""

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120


def simple_rope_rotation():
    """RoPE 핵심: 위치마다 벡터가 회전한다"""
    fig, ax = plt.subplots(figsize=(8, 8))

    # 원 그리기
    theta_circle = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta_circle), np.sin(theta_circle), 'gray', linestyle='--', alpha=0.3)

    # 기준 벡터 (모든 위치에서 같은 의미의 단어)
    base_angle = np.pi / 6  # 30도
    base_length = 0.9

    # 위치별로 회전
    positions = [0, 1, 2, 3, 4]
    colors = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#3498db']
    rotation_per_pos = np.pi / 8  # 위치당 회전 각도

    for pos, color in zip(positions, colors):
        angle = base_angle + pos * rotation_per_pos
        x = base_length * np.cos(angle)
        y = base_length * np.sin(angle)

        ax.annotate('', xy=(x, y), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color=color, lw=3))
        ax.plot(x, y, 'o', color=color, markersize=10)
        ax.text(x*1.15, y*1.15, f'pos {pos}', fontsize=12, ha='center', color=color, fontweight='bold')

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal')
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.2)
    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.2)
    ax.set_title('RoPE: 같은 단어도 위치마다 다르게 회전!\n(위치가 멀수록 각도 차이 큼)', fontsize=14, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    return fig


def simple_attention_by_distance():
    """상대 거리에 따른 Attention Score"""
    fig, ax = plt.subplots(figsize=(10, 5))

    # Q 위치 고정, K 위치 변화
    q_pos = 5
    k_positions = np.arange(0, 11)
    distances = k_positions - q_pos  # 상대 거리

    # 거리가 가까울수록 attention 높음 (예시)
    # 실제론 내용에 따라 다르지만, 위치 정보만 봤을 때
    attention_scores = np.exp(-0.3 * np.abs(distances))  # 거리 멀수록 감소

    colors = ['#3498db' if d != 0 else '#e74c3c' for d in distances]

    bars = ax.bar(k_positions, attention_scores, color=colors, alpha=0.8, edgecolor='black')

    # Q 위치 표시
    ax.axvline(x=q_pos, color='red', linestyle='--', linewidth=2)
    ax.text(q_pos, 1.05, 'Q 위치', ha='center', fontsize=12, color='red', fontweight='bold')

    # 거리 표시
    for i, (kp, score) in enumerate(zip(k_positions, attention_scores)):
        dist = kp - q_pos
        label = f'{dist:+d}' if dist != 0 else '0'
        ax.text(kp, score + 0.03, label, ha='center', fontsize=10)

    ax.set_xlabel('K 위치', fontsize=12)
    ax.set_ylabel('Attention Score', fontsize=12)
    ax.set_title('상대 위치와 Attention\n(가까운 토큰에 더 집중하는 경향)', fontsize=14, fontweight='bold')
    ax.set_xticks(k_positions)
    ax.set_ylim(0, 1.2)

    plt.tight_layout()
    return fig


def simple_sincos_pattern():
    """sin/cos PE: 위치마다 고유한 패턴"""
    fig, ax = plt.subplots(figsize=(10, 4))

    positions = np.arange(10)
    d_model = 4

    # 히트맵 데이터
    pe = np.zeros((len(positions), d_model))
    for pos in positions:
        for i in range(0, d_model, 2):
            div = 10000 ** (i / d_model)
            pe[pos, i] = np.sin(pos / div)
            if i + 1 < d_model:
                pe[pos, i + 1] = np.cos(pos / div)

    im = ax.imshow(pe, cmap='RdBu', vmin=-1, vmax=1, aspect='auto')

    ax.set_xticks(range(d_model))
    ax.set_xticklabels(['sin(저)', 'cos(저)', 'sin(고)', 'cos(고)'])
    ax.set_yticks(positions)
    ax.set_yticklabels([f'pos {p}' for p in positions])

    ax.set_xlabel('차원 (주파수)', fontsize=12)
    ax.set_ylabel('위치', fontsize=12)
    ax.set_title('sin/cos PE: 각 위치마다 고유한 패턴\n(빨강=+1, 파랑=-1)', fontsize=14, fontweight='bold')

    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    return fig


def compare_sincos_vs_rope():
    """sin/cos PE vs RoPE 핵심 차이"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- sin/cos PE ---
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('sin/cos PE', fontsize=14, fontweight='bold', color='coral')

    # 흐름도
    ax.text(5, 9, '입력 X', ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='lightyellow'))
    ax.text(5, 7.5, '+', ha='center', fontsize=16, fontweight='bold')
    ax.text(5, 6, 'PE (위치)', ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgreen'))
    ax.annotate('', xy=(5, 5), xytext=(5, 5.5), arrowprops=dict(arrowstyle='->', lw=2))
    ax.text(5, 4, 'X + PE\n(섞임)', ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='lightsalmon'))
    ax.annotate('', xy=(5, 3), xytext=(5, 3.5), arrowprops=dict(arrowstyle='->', lw=2))
    ax.text(5, 2, 'W_Q', ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray'))
    ax.annotate('', xy=(5, 1), xytext=(5, 1.5), arrowprops=dict(arrowstyle='->', lw=2))
    ax.text(5, 0.3, 'Q (위치 희석)', ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='lightcoral'))

    # --- RoPE ---
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('RoPE', fontsize=14, fontweight='bold', color='seagreen')

    # 흐름도 (분리)
    ax.text(3, 9, '입력 X', ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='lightyellow'))
    ax.text(7, 9, '위치', ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgreen'))

    ax.annotate('', xy=(3, 7), xytext=(3, 8.5), arrowprops=dict(arrowstyle='->', lw=2))
    ax.text(3, 6, 'W_Q\n(의미만)', ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='lightgray'))

    ax.annotate('', xy=(3, 4.5), xytext=(3, 5.5), arrowprops=dict(arrowstyle='->', lw=2))
    ax.text(3, 3.5, 'Q', ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='lightblue'))

    ax.annotate('', xy=(7, 4.5), xytext=(7, 8.5), arrowprops=dict(arrowstyle='->', lw=2))
    ax.text(7, 3.5, 'Rotate', ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgreen'))

    ax.annotate('', xy=(5, 1.5), xytext=(3.5, 3), arrowprops=dict(arrowstyle='->', lw=2))
    ax.annotate('', xy=(5, 1.5), xytext=(6.5, 3), arrowprops=dict(arrowstyle='->', lw=2))
    ax.text(5, 0.3, "Q' (정확한 위치)", ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='mediumaquamarine'))

    plt.suptitle('sin/cos PE vs RoPE: 핵심 차이', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig


if __name__ == '__main__':
    import os
    os.makedirs('deep_learning/figures', exist_ok=True)

    # 1. RoPE 회전
    fig1 = simple_rope_rotation()
    fig1.savefig('deep_learning/figures/simple_rope_rotation.png', bbox_inches='tight', facecolor='white')
    print('Saved: simple_rope_rotation.png')

    # 2. 거리별 Attention
    fig2 = simple_attention_by_distance()
    fig2.savefig('deep_learning/figures/simple_attention_distance.png', bbox_inches='tight', facecolor='white')
    print('Saved: simple_attention_distance.png')

    # 3. sin/cos 패턴
    fig3 = simple_sincos_pattern()
    fig3.savefig('deep_learning/figures/simple_sincos_pattern.png', bbox_inches='tight', facecolor='white')
    print('Saved: simple_sincos_pattern.png')

    # 4. 비교
    fig4 = compare_sincos_vs_rope()
    fig4.savefig('deep_learning/figures/simple_pe_comparison.png', bbox_inches='tight', facecolor='white')
    print('Saved: simple_pe_comparison.png')

    plt.show()
