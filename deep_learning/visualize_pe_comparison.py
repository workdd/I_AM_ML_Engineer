"""
Positional Encoding 비교 시각화
- sin/cos PE vs RoPE의 데이터 흐름 차이
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# 한글 폰트 설정 (macOS)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150


def draw_box(ax, x, y, width, height, text, color='lightblue', fontsize=10):
    """박스와 텍스트를 그리는 헬퍼 함수"""
    box = FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.03,rounding_size=0.1",
        facecolor=color, edgecolor='black', linewidth=1.5
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, fontweight='bold')


def draw_arrow(ax, start, end, color='black'):
    """화살표를 그리는 헬퍼 함수"""
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle='->,head_width=0.15,head_length=0.1',
        color=color, linewidth=2,
        connectionstyle='arc3,rad=0'
    )
    ax.add_patch(arrow)


def visualize_sincos_pe():
    """sin/cos PE 데이터 흐름 시각화"""
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    ax.set_title('sin/cos Positional Encoding\n(기존 방식)', fontsize=14, fontweight='bold', pad=20)

    # 입력
    draw_box(ax, 3, 10.5, 2.5, 0.8, '입력 X\n(의미)', 'lightyellow')
    draw_box(ax, 7, 10.5, 2.5, 0.8, 'PE\n(위치)', 'lightgreen')

    # 합치기
    draw_arrow(ax, (3, 10.1), (5, 9.2))
    draw_arrow(ax, (7, 10.1), (5, 9.2))
    draw_box(ax, 5, 8.5, 3, 0.8, 'X + PE\n(의미+위치 섞임)', 'lightsalmon')

    # W_Q 변환
    draw_arrow(ax, (5, 8.1), (5, 7.2))
    draw_box(ax, 5, 6.5, 2, 0.8, 'W_Q', 'lightgray')

    # Q 결과
    draw_arrow(ax, (5, 6.1), (5, 5.2))
    draw_box(ax, 5, 4.5, 2.5, 0.8, 'Q\n(위치 희석됨)', 'lightcoral')

    # K도 동일
    ax.text(5, 3.5, '(K도 동일한 과정)', ha='center', fontsize=9, style='italic', color='gray')

    # Attention
    draw_arrow(ax, (5, 3.2), (5, 2.2))
    draw_box(ax, 5, 1.5, 3, 0.8, 'Q · K\n(간접적 위치 정보)', 'plum')

    # 문제점 표시
    ax.text(8.5, 8.5, '⚠️ 같은 W_Q가\n의미+위치 둘 다 처리',
            fontsize=8, color='red', ha='left')
    ax.text(8.5, 4.5, '⚠️ 위치 정보가\n변형/희석됨',
            fontsize=8, color='red', ha='left')

    plt.tight_layout()
    return fig


def visualize_rope():
    """RoPE 데이터 흐름 시각화"""
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    ax.set_title('RoPE (Rotary Position Embedding)\n(최신 방식)', fontsize=14, fontweight='bold', pad=20)

    # 입력
    draw_box(ax, 3, 10.5, 2.5, 0.8, '입력 X\n(의미만)', 'lightyellow')
    draw_box(ax, 7, 10.5, 2.5, 0.8, 'Position\n(위치)', 'lightgreen')

    # W_Q 변환 (의미만)
    draw_arrow(ax, (3, 10.1), (3, 9.2))
    draw_box(ax, 3, 8.5, 2, 0.8, 'W_Q', 'lightgray')

    # Q (순수 의미)
    draw_arrow(ax, (3, 8.1), (3, 7.2))
    draw_box(ax, 3, 6.5, 2.5, 0.8, 'Q\n(순수 의미)', 'lightblue')

    # Position → Rotation
    draw_arrow(ax, (7, 10.1), (7, 7.2))
    draw_box(ax, 7, 6.5, 2.5, 0.8, 'Rotate(θ)\n(회전 행렬)', 'lightgreen')

    # Q와 Rotation 합치기
    draw_arrow(ax, (3, 6.1), (5, 5.2))
    draw_arrow(ax, (7, 6.1), (5, 5.2))
    draw_box(ax, 5, 4.5, 2.5, 0.8, "Q'\n(의미+정확한 위치)", 'lightskyblue')

    # K도 동일
    ax.text(5, 3.5, "(K'도 동일한 과정)", ha='center', fontsize=9, style='italic', color='gray')

    # Attention
    draw_arrow(ax, (5, 3.2), (5, 2.2))
    draw_box(ax, 5, 1.5, 3, 0.8, "Q' · K'\n(상대위치 자동 계산!)", 'mediumaquamarine')

    # 장점 표시
    ax.text(0.5, 8.5, '✓ W_Q는 의미만 처리',
            fontsize=8, color='green', ha='left')
    ax.text(0.5, 6.5, '✓ 위치는 나중에\n   별도로 적용',
            fontsize=8, color='green', ha='left')
    ax.text(8.5, 1.5, '✓ m-n 상대위치\n   자동 계산',
            fontsize=8, color='green', ha='left')

    plt.tight_layout()
    return fig


def visualize_comparison():
    """두 방식 비교 시각화"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    for ax in axes:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')

    # --- 왼쪽: sin/cos PE ---
    ax = axes[0]
    ax.set_title('sin/cos PE (기존)', fontsize=14, fontweight='bold', color='coral')

    # 간단한 흐름
    draw_box(ax, 5, 9, 3.5, 0.8, '[의미] + [위치]', 'lightyellow')
    draw_arrow(ax, (5, 8.6), (5, 7.8))
    draw_box(ax, 5, 7.2, 2.5, 0.8, '섞인 상태', 'lightsalmon')
    draw_arrow(ax, (5, 6.8), (5, 6))
    draw_box(ax, 5, 5.4, 2, 0.8, 'W_Q', 'lightgray')
    draw_arrow(ax, (5, 5), (5, 4.2))
    draw_box(ax, 5, 3.6, 2.5, 0.8, 'Q (희석됨)', 'lightcoral')
    draw_arrow(ax, (5, 3.2), (5, 2.4))
    draw_box(ax, 5, 1.8, 3, 0.8, 'Attention', 'plum')

    ax.text(5, 0.8, '위치 정보: 간접적 ❌', ha='center', fontsize=11, color='red', fontweight='bold')

    # --- 오른쪽: RoPE ---
    ax = axes[1]
    ax.set_title('RoPE (최신)', fontsize=14, fontweight='bold', color='seagreen')

    # 분리된 흐름
    draw_box(ax, 3, 9, 2, 0.8, '[의미]', 'lightyellow')
    draw_box(ax, 7, 9, 2, 0.8, '[위치]', 'lightgreen')

    draw_arrow(ax, (3, 8.6), (3, 7.8))
    draw_box(ax, 3, 7.2, 2, 0.8, 'W_Q', 'lightgray')
    draw_arrow(ax, (3, 6.8), (3, 6))
    draw_box(ax, 3, 5.4, 2, 0.8, 'Q (순수)', 'lightblue')

    draw_arrow(ax, (7, 8.6), (7, 6))
    draw_box(ax, 7, 5.4, 2, 0.8, 'Rotate', 'lightgreen')

    draw_arrow(ax, (3, 5), (5, 4.2))
    draw_arrow(ax, (7, 5), (5, 4.2))
    draw_box(ax, 5, 3.6, 2.5, 0.8, "Q' (정확)", 'lightskyblue')

    draw_arrow(ax, (5, 3.2), (5, 2.4))
    draw_box(ax, 5, 1.8, 3, 0.8, 'Attention', 'mediumaquamarine')

    ax.text(5, 0.8, '위치 정보: 직접적 ✓', ha='center', fontsize=11, color='green', fontweight='bold')

    plt.suptitle('Positional Encoding 방식 비교', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    return fig


if __name__ == '__main__':
    # 개별 시각화 저장
    fig1 = visualize_sincos_pe()
    fig1.savefig('deep_learning/figures/sincos_pe_flow.png', bbox_inches='tight', facecolor='white')
    print('✓ sincos_pe_flow.png 저장 완료')

    fig2 = visualize_rope()
    fig2.savefig('deep_learning/figures/rope_flow.png', bbox_inches='tight', facecolor='white')
    print('✓ rope_flow.png 저장 완료')

    # 비교 시각화 저장
    fig3 = visualize_comparison()
    fig3.savefig('deep_learning/figures/pe_comparison.png', bbox_inches='tight', facecolor='white')
    print('✓ pe_comparison.png 저장 완료')

    plt.show()
