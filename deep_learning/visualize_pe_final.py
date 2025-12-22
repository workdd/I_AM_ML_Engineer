"""
PE 시각화 - 딱 2개만!
1. sin/cos PE: 위치마다 다른 값이 더해짐
2. RoPE: 위치마다 벡터가 회전함
"""

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150


def visualize_sincos_pe():
    """sin/cos PE: 위치마다 다른 값이 더해진다"""
    fig, ax = plt.subplots(figsize=(10, 6))

    positions = np.arange(0, 20)

    # sin 값 (위치에 따라 변함)
    sin_values = np.sin(positions / 3)

    # 막대 그래프
    colors = plt.cm.coolwarm((sin_values + 1) / 2)  # -1~1 → 0~1
    bars = ax.bar(positions, sin_values, color=colors, edgecolor='black', linewidth=0.5)

    ax.axhline(y=0, color='gray', linestyle='-', linewidth=1)
    ax.set_xlabel('위치 (Position)', fontsize=14)
    ax.set_ylabel('더해지는 값', fontsize=14)
    ax.set_title('sin/cos PE\n각 위치마다 다른 값이 입력에 더해진다', fontsize=16, fontweight='bold')
    ax.set_ylim(-1.3, 1.3)

    # 설명 추가
    ax.text(10, -1.15, '파랑 = 음수값 더해짐 | 빨강 = 양수값 더해짐',
            ha='center', fontsize=11, style='italic')

    plt.tight_layout()
    return fig


def visualize_rope():
    """RoPE: 위치마다 벡터가 회전한다"""
    fig, ax = plt.subplots(figsize=(8, 8))

    # 원 그리기 (배경)
    theta_circle = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta_circle), np.sin(theta_circle), color='lightgray', linewidth=2)

    # 위치별 회전
    positions = [0, 2, 4, 6, 8]
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12']

    base_angle = 0.3  # 시작 각도
    rotation_step = 0.4  # 위치당 회전량

    for pos, color in zip(positions, colors):
        angle = base_angle + pos * rotation_step
        x = 0.85 * np.cos(angle)
        y = 0.85 * np.sin(angle)

        # 화살표
        ax.annotate('', xy=(x, y), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color=color, lw=4))

        # 점과 라벨
        ax.plot(x, y, 'o', color=color, markersize=12, zorder=5)
        ax.text(x * 1.2, y * 1.2, f'위치 {pos}', fontsize=13,
                ha='center', va='center', color=color, fontweight='bold')

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.set_title('RoPE\n같은 벡터도 위치마다 다르게 회전!', fontsize=16, fontweight='bold')

    # 설명
    ax.text(0, -1.35, '위치가 다르면 → 회전 각도가 다름 → Q·K 결과가 달라짐',
            ha='center', fontsize=11, style='italic')

    plt.tight_layout()
    return fig


if __name__ == '__main__':
    import os
    os.makedirs('deep_learning/figures', exist_ok=True)

    # 1. sin/cos PE
    fig1 = visualize_sincos_pe()
    fig1.savefig('deep_learning/figures/sincos_pe.png', bbox_inches='tight', facecolor='white')
    print('Saved: sincos_pe.png')

    # 2. RoPE
    fig2 = visualize_rope()
    fig2.savefig('deep_learning/figures/rope.png', bbox_inches='tight', facecolor='white')
    print('Saved: rope.png')

    plt.show()
