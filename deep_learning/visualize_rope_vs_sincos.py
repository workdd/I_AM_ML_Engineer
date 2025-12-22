"""
sin/cos PE vs RoPE 비교 시각화
B. 위치별 값 변화 비교
C. Q·K 내적 결과 (상대 위치에 따른 attention score)
"""

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120


# ============================================================
# sin/cos Positional Encoding
# ============================================================
def sincos_pe(position: int, d_model: int) -> np.ndarray:
    """sin/cos PE 계산"""
    pe = np.zeros(d_model)
    for i in range(0, d_model, 2):
        div_term = 10000 ** (i / d_model)
        pe[i] = np.sin(position / div_term)
        if i + 1 < d_model:
            pe[i + 1] = np.cos(position / div_term)
    return pe


# ============================================================
# RoPE (Rotary Position Embedding)
# ============================================================
def rope_rotate(x: np.ndarray, position: int, d_model: int) -> np.ndarray:
    """
    RoPE 회전 적용
    x: 입력 벡터 (d_model,)
    position: 위치
    """
    rotated = np.zeros_like(x)
    for i in range(0, d_model, 2):
        theta = position / (10000 ** (i / d_model))
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # 2D 회전 적용
        rotated[i] = x[i] * cos_theta - x[i + 1] * sin_theta
        rotated[i + 1] = x[i] * sin_theta + x[i + 1] * cos_theta

    return rotated


# ============================================================
# B. 위치별 값 변화 비교
# ============================================================
def visualize_position_values():
    """위치별로 PE 값이 어떻게 변하는지 비교"""
    d_model = 8
    max_pos = 20

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    positions = np.arange(max_pos)

    # --- sin/cos PE ---
    ax = axes[0, 0]
    pe_values = np.array([sincos_pe(p, d_model) for p in positions])

    for dim in range(min(4, d_model)):
        ax.plot(positions, pe_values[:, dim],
                marker='o', markersize=4, label=f'dim {dim}')

    ax.set_xlabel('Position')
    ax.set_ylabel('PE Value')
    ax.set_title('sin/cos PE: 위치별 값 변화')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # --- RoPE (고정 벡터에 회전 적용) ---
    ax = axes[0, 1]
    base_vector = np.ones(d_model)  # 기준 벡터

    rope_values = np.array([rope_rotate(base_vector, p, d_model) for p in positions])

    for dim in range(min(4, d_model)):
        ax.plot(positions, rope_values[:, dim],
                marker='s', markersize=4, label=f'dim {dim}')

    ax.set_xlabel('Position')
    ax.set_ylabel('Rotated Value')
    ax.set_title('RoPE: 위치별 회전 결과\n(기준 벡터 [1,1,1,...] 에 회전 적용)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # --- sin/cos PE: 위치 간 거리 (내적) ---
    ax = axes[1, 0]
    ref_pos = 5  # 기준 위치
    ref_pe = sincos_pe(ref_pos, d_model)

    similarities = []
    for p in positions:
        pe_p = sincos_pe(p, d_model)
        sim = np.dot(ref_pe, pe_p) / (np.linalg.norm(ref_pe) * np.linalg.norm(pe_p))
        similarities.append(sim)

    ax.bar(positions, similarities, color='steelblue', alpha=0.7)
    ax.axvline(x=ref_pos, color='red', linestyle='--', label=f'기준 위치 (pos={ref_pos})')
    ax.set_xlabel('Position')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title(f'sin/cos PE: Position {ref_pos}과의 유사도')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- RoPE: 위치 간 거리 (내적) ---
    ax = axes[1, 1]
    base_q = np.random.randn(d_model)
    base_k = base_q.copy()  # 같은 벡터로 시작

    q_rotated = rope_rotate(base_q, ref_pos, d_model)

    similarities = []
    for p in positions:
        k_rotated = rope_rotate(base_k, p, d_model)
        sim = np.dot(q_rotated, k_rotated) / (np.linalg.norm(q_rotated) * np.linalg.norm(k_rotated))
        similarities.append(sim)

    ax.bar(positions, similarities, color='seagreen', alpha=0.7)
    ax.axvline(x=ref_pos, color='red', linestyle='--', label=f'기준 위치 (pos={ref_pos})')
    ax.set_xlabel('Position')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title(f'RoPE: Position {ref_pos}과의 유사도\n(같은 벡터에 다른 위치 회전 적용)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('sin/cos PE vs RoPE: 위치별 값 변화 비교', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


# ============================================================
# C. Q·K 내적: 상대 위치에 따른 Attention Score
# ============================================================
def visualize_relative_position_attention():
    """상대 위치에 따라 Q·K 내적이 어떻게 변하는지"""
    d_model = 64
    max_distance = 30

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 랜덤 Q, K 벡터 생성
    np.random.seed(42)
    q_base = np.random.randn(d_model)
    k_base = np.random.randn(d_model)

    distances = np.arange(-max_distance, max_distance + 1)

    # --- sin/cos PE ---
    ax = axes[0]
    q_pos = 50  # Q의 위치 고정
    q_pe = sincos_pe(q_pos, d_model)
    q_with_pe = q_base + q_pe  # Q + PE

    scores_sincos = []
    for dist in distances:
        k_pos = q_pos + dist
        k_pe = sincos_pe(k_pos, d_model)
        k_with_pe = k_base + k_pe  # K + PE

        # Attention score (scaled dot product)
        score = np.dot(q_with_pe, k_with_pe) / np.sqrt(d_model)
        scores_sincos.append(score)

    ax.plot(distances, scores_sincos, 'b-', linewidth=2, marker='o', markersize=3)
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='같은 위치 (dist=0)')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('상대 위치 (K position - Q position)')
    ax.set_ylabel('Attention Score (Q·K / sqrt(d))')
    ax.set_title('sin/cos PE: 상대 위치에 따른 Attention Score')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- RoPE ---
    ax = axes[1]

    scores_rope = []
    for dist in distances:
        k_pos = q_pos + dist

        # RoPE: Q, K 각각 회전
        q_rotated = rope_rotate(q_base, q_pos, d_model)
        k_rotated = rope_rotate(k_base, k_pos, d_model)

        # Attention score
        score = np.dot(q_rotated, k_rotated) / np.sqrt(d_model)
        scores_rope.append(score)

    ax.plot(distances, scores_rope, 'g-', linewidth=2, marker='s', markersize=3)
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='같은 위치 (dist=0)')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('상대 위치 (K position - Q position)')
    ax.set_ylabel('Attention Score (Q·K / sqrt(d))')
    ax.set_title('RoPE: 상대 위치에 따른 Attention Score')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('상대 위치에 따른 Attention Score 변화\n(Q 위치 고정, K 위치 이동)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def visualize_rope_rotation_2d():
    """RoPE 회전을 2D에서 시각화"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 기준 벡터
    base_vec = np.array([1.0, 0.5])

    # 여러 위치에서 회전
    positions = [0, 2, 4, 6, 8, 10]
    colors = plt.cm.viridis(np.linspace(0, 1, len(positions)))

    ax = axes[0]
    for pos, color in zip(positions, colors):
        theta = pos / 10  # 간단한 회전 각도
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rotated = np.array([
            base_vec[0] * cos_t - base_vec[1] * sin_t,
            base_vec[0] * sin_t + base_vec[1] * cos_t
        ])
        ax.arrow(0, 0, rotated[0], rotated[1], head_width=0.05, head_length=0.03,
                 fc=color, ec=color, linewidth=2, label=f'pos={pos}')

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('RoPE: 위치에 따른 벡터 회전\n(같은 벡터가 위치마다 다르게 회전)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Q·K 내적 시각화 (같은 위치 vs 다른 위치)
    ax = axes[1]

    q_pos = 5
    theta_q = q_pos / 10
    q_rotated = np.array([
        base_vec[0] * np.cos(theta_q) - base_vec[1] * np.sin(theta_q),
        base_vec[0] * np.sin(theta_q) + base_vec[1] * np.cos(theta_q)
    ])

    k_positions = np.arange(0, 15)
    dot_products = []

    for k_pos in k_positions:
        theta_k = k_pos / 10
        k_rotated = np.array([
            base_vec[0] * np.cos(theta_k) - base_vec[1] * np.sin(theta_k),
            base_vec[0] * np.sin(theta_k) + base_vec[1] * np.cos(theta_k)
        ])
        dot_products.append(np.dot(q_rotated, k_rotated))

    ax.bar(k_positions, dot_products, color='seagreen', alpha=0.7)
    ax.axvline(x=q_pos, color='red', linestyle='--', linewidth=2, label=f'Q position={q_pos}')
    ax.set_xlabel('K Position')
    ax.set_ylabel('Q · K (내적)')
    ax.set_title('RoPE: Q·K 내적 결과\n(Q 고정, K 위치 변화)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('RoPE 회전 시각화', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


if __name__ == '__main__':
    import os
    os.makedirs('deep_learning/figures', exist_ok=True)

    # B. 위치별 값 변화 비교
    fig1 = visualize_position_values()
    fig1.savefig('deep_learning/figures/pe_position_values.png', bbox_inches='tight', facecolor='white')
    print('Saved: pe_position_values.png')

    # C. 상대 위치에 따른 Attention Score
    fig2 = visualize_relative_position_attention()
    fig2.savefig('deep_learning/figures/relative_position_attention.png', bbox_inches='tight', facecolor='white')
    print('Saved: relative_position_attention.png')

    # RoPE 2D 회전 시각화
    fig3 = visualize_rope_rotation_2d()
    fig3.savefig('deep_learning/figures/rope_rotation_2d.png', bbox_inches='tight', facecolor='white')
    print('Saved: rope_rotation_2d.png')

    plt.show()
