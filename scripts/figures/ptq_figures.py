#!/usr/bin/env python3
"""Why Does Post-Training Quantization Work? (arXiv:2609.11716) 노트의 그림을 생성한다.

    python scripts/figures/ptq_figures.py

readings/papers/assets/ptq_*.svg 를 다시 쓴다. SVG 는 손으로 고치지 않는다.
논문 그림·표의 수치를 아래 데이터 절에 옮겼고, 배율과 합계는 여기서 계산한다.
"""
from __future__ import annotations

import math
from pathlib import Path
from statistics import mean

from svgkit import fmt, make_doc, t

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "readings" / "papers" / "assets"
doc = make_doc("생성: python scripts/figures/ptq_figures.py · 손으로 고치지 말 것")
SRC = "출처: Why Does Post-Training Quantization Work? (arXiv:2609.11716)"

# ---------------------------------------------------------------- 데이터

# Fig. 1. 마지막 블록의 은닉 오차. (무작위 초기화 또는 step 0, 사전학습 완료)
HIDDEN_ERR = {
    "Qwen3-32B 절대 오차": (2683.0, 484.1),
    "Qwen3-32B 상대 오차": (1.109, 0.165),
    "OLMo3-7B 절대 오차": (254.9, 12.6),
    "OLMo3-7B 상대 오차": (0.498, 0.136),
}

# Sec. 3.2. 사전학습 Qwen3-32B, 블록 1~L 누적. T_add 대비 상쇄 비율(%)
CANCEL_INTER, CANCEL_ALIGN = 63.4, 18.4
CANCEL_RANDOM = 0.09

# Table 2. 이상 위치 필터 기준별 (유지 비율 %, T_inter 상쇄 비율 %)
FILTER = [("필터 없음", 100.00, 30.4), ("1.0", 99.92, 57.7), ("0.5 (기본)", 99.72, 50.4),
          ("0.25", 98.62, 50.2), ("0.1", 89.34, 51.8)]

# Fig. 3(A). 블록 17~48 개입 결과 (최종 상대 오차 R(L), KL)
INTERVENE = [("양자화 그대로", 0.165, 0.045), ("상쇄 제거", 0.485, 0.628), ("상쇄 반전", 1.387, 7.856)]

# Fig. 5(A). LM-head 입력 회전각과 어휘 평균 투영각 변화(도)
ROTATION = {"C4": (8.26, 0.109), "WikiText-103": (12.60, 0.158), "GSM8K": (16.54, 0.209)}
HIDDEN_DIM = 5120  # Qwen3-32B

# Fig. 4(A). 0-shot 정확도 (BF16, W4)
BENCH = {"ARC-Challenge": (61.09, 61.26), "ARC-Easy": (83.42, 83.33), "HellaSwag": (82.66, 82.45),
         "MMLU": (80.77, 80.34), "WinoGrande": (72.85, 70.48), "TruthfulQA MC1": (38.80, 39.17)}
BENCH_SD = {"ARC-Challenge": 1.42, "ARC-Easy": 0.76, "HellaSwag": 0.38, "MMLU": 0.34,
            "WinoGrande": 1.30, "TruthfulQA MC1": 1.71}  # W4 부트스트랩 SD


def mu_d(d: int) -> float:
    return math.exp(math.lgamma((d - 1) / 2) - math.lgamma(d / 2)) / math.sqrt(math.pi)


def bench_mean_drop() -> float:
    return mean(w - b for b, w in BENCH.values())


def attenuation_measured() -> float:
    return mean(r for r, _ in ROTATION.values()) / mean(a for _, a in ROTATION.values())


# ---------------------------------------------------------------- 그림


def fig_vectors():
    """상쇄의 기하. 값은 예시(입력 오차 1, 갱신 오차 0.8)이고 cos 만 바꾼다."""
    W, H = 860, 400
    body = []
    cases = [("상쇄 (cos = −0.25)", -0.25, "s1"), ("상쇄 없음 (cos = 0)", 0.0, "muted"), ("같은 방향 (cos = +0.5)", 0.5, "s2")]
    a, b = 1.0, 0.8
    scale = 120
    for i, (label, c, cls) in enumerate(cases):
        ox, oy = 60 + i * 270, 300
        th = math.acos(c)
        hx, hy = ox + a * scale, oy
        ux, uy = hx + b * scale * math.cos(th), oy - b * scale * math.sin(th)
        res = math.sqrt(a * a + b * b + 2 * a * b * c)
        color = {"s1": "var(--s1)", "s2": "var(--s2)", "muted": "var(--muted)"}[cls]
        body.append(f'<defs><marker id="m{i}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
                    f'<path d="M0,0 L10,5 L0,10 z" style="fill:{color}"/></marker>'
                    f'<marker id="k{i}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
                    f'<path d="M0,0 L10,5 L0,10 z" style="fill:var(--ink2)"/></marker></defs>')
        body.append(t(ox, 96, label, "b", size=13.5))
        body.append(f'<line x1="{ox}" y1="{oy}" x2="{hx:.1f}" y2="{hy:.1f}" style="stroke:var(--ink2);stroke-width:2" marker-end="url(#k{i})"/>')
        body.append(t((ox + hx) / 2, oy + 20, "Δh⁽ℓ⁻¹⁾ 물려받은 오차", "t2", "middle", size=11.5))
        body.append(f'<line x1="{hx:.1f}" y1="{hy:.1f}" x2="{ux:.1f}" y2="{uy:.1f}" style="stroke:{color};stroke-width:2" marker-end="url(#m{i})"/>')
        body.append(t(ux + 6, uy - 4, "Δu⁽ℓ⁾ 새 오차", "t2", size=11.5))
        body.append(f'<line x1="{ox}" y1="{oy}" x2="{ux:.1f}" y2="{uy:.1f}" style="stroke:{color};stroke-width:3;opacity:.85" marker-end="url(#m{i})"/>')
        body.append(t(ox, 128, f"합친 오차 길이 {res:.2f}", "b" if cls == "s1" else "t2", size=13))
        body.append(t(ox, 146, f"(상쇄 없을 때 {math.sqrt(a * a + b * b):.2f} 대비 {res / math.sqrt(a * a + b * b):.0%})", "mu"))
    body.append(t(24, 352, "‖Δh⁽ℓ⁾‖² = ‖Δh⁽ℓ⁻¹⁾‖² + ‖Δu⁽ℓ⁾‖² + 2⟨Δh⁽ℓ⁻¹⁾, Δu⁽ℓ⁾⟩. 마지막 항이 음수면 새 오차가 물려받은 오차를 일부 지운다.", "t2", size=12.5))
    return doc(W, H, "상쇄: 새 오차가 물려받은 오차의 반대쪽을 향한다",
               "같은 크기의 새 오차라도 방향에 따라 누적 오차가 달라진다. 길이는 예시 값, 식은 Prop. 1",
               body, SRC + " Prop. 1. 벡터 길이는 설명용 예시")


def fig_random_vs_pretrained():
    W = 820
    rows = list(HIDDEN_ERR.items())
    x0, x1 = 230, 640
    top, rowh = 100, 52
    body = []
    for i, (name, (rnd, pre)) in enumerate(rows):
        y = top + i * rowh
        vmax = rnd * 1.05
        sx = lambda v: x0 + (x1 - x0) * v / vmax
        body.append(t(x0 - 14, y + 14, name, "", "end"))
        body.append(f'<rect class="faintf" x="{x0}" y="{y}" width="{sx(rnd) - x0:.1f}" height="14" rx="3"/>')
        body.append(f'<rect class="s1f" x="{x0}" y="{y + 18}" width="{max(sx(pre) - x0, 2):.1f}" height="14" rx="3"/>')
        body.append(t(sx(rnd) + 8, y + 12, f"{rnd:g}", "t2", size=12))
        body.append(t(sx(pre) + 8, y + 30, f"{pre:g}", "b", size=12))
        body.append(t(x1 + 90, y + 22, f"{rnd / pre:.1f}배", "b", "middle", size=14))
    body.append(t(x1 + 90, top - 14, "차이", "t2", "middle", size=12))
    ly = 78
    body.append(f'<rect class="faintf" x="{x0}" y="{ly - 10}" width="14" height="12" rx="3"/>')
    body.append(t(x0 + 20, ly, "무작위 초기화 (OLMo3는 step 0)", size=12))
    body.append(f'<rect class="s1f" x="{x0 + 250}" y="{ly - 10}" width="14" height="12" rx="3"/>')
    body.append(t(x0 + 270, ly, "사전학습 완료", size=12))
    ny = top + rowh * len(rows) + 14
    body.append(t(24, ny, "가중치 복원 오차는 두 경우가 같습니다(cos 0.9955 대 0.9955, 상대 오차 9.48% 대 9.51%). 차이는 모델 쪽에서 생깁니다.", "t2", size=12.5))
    H = ny + 50
    return doc(W, H, "같은 양자화, 다른 결과: 사전학습이 오차 누적을 늦춘다",
               "NVFP4 가중치 양자화 후 마지막 블록의 은닉 오차. 막대 길이는 행마다 따로 맞춤",
               body, SRC + " Fig. 1, App. D.2 Table 1")


def fig_cancel_budget():
    W, H = 820, 350
    x0, x1 = 230, 720
    sx = lambda v: x0 + (x1 - x0) * v / 100
    body = []
    rows = [("새 오차 기여 T_add", 0, 100, "faintf", "100%"),
            ("오차 상호작용 T_inter", 100 - CANCEL_INTER, 100, "s1f", f"−{CANCEL_INTER}%"),
            ("은닉 상태 크기 변화 T_align", 100 - CANCEL_INTER - CANCEL_ALIGN, 100 - CANCEL_INTER, "s1f", f"−{CANCEL_ALIGN}%"),
            ("남은 누적 오차", 0, 100 - CANCEL_INTER - CANCEL_ALIGN, "s2f", f"{100 - CANCEL_INTER - CANCEL_ALIGN:.1f}%")]
    top, rowh = 96, 40
    for v in range(0, 101, 20):
        body.append(f'<line class="grid" x1="{sx(v):.1f}" y1="{top - 8}" x2="{sx(v):.1f}" y2="{top + rowh * 4 - 12}"/>')
        body.append(t(sx(v), top + rowh * 4 + 4, f"{v}%", "mu", "middle"))
    for i, (name, a, b, cls, lab) in enumerate(rows):
        y = top + i * rowh
        body.append(t(x0 - 14, y + 15, name, "b" if i == 3 else "", "end"))
        body.append(f'<rect class="{cls}" x="{sx(a):.1f}" y="{y}" width="{sx(b) - sx(a):.1f}" height="22" rx="4"/>')
        body.append(t(sx(b) + 8 if i != 0 else sx(b) - 8, y + 16, lab, "b", "start" if i != 0 else "end",
                      extra=' style="fill:var(--ink)"'))
    body.append(t(24, top + rowh * 4 + 36, f"무작위 초기화에서는 두 상쇄 항의 합이 T_add의 {CANCEL_RANDOM}%에 그칩니다.", "t2", size=12.5))
    return doc(W, H, "새로 생긴 오차의 82%가 누적 전에 지워진다",
               "사전학습 Qwen3-32B, 블록 1~64 누적. 상대 은닉 오차의 제곱이 커지는 양을 세 항으로 나눈 Thm. 1",
               body, SRC + " Sec. 3.2, Fig. 2")


def fig_filter():
    W, H = 760, 300
    x0, x1 = 170, 560
    sx = lambda v: x0 + (x1 - x0) * v / 70
    top, rowh = 96, 30
    body = []
    for v in range(0, 71, 10):
        body.append(f'<line class="grid" x1="{sx(v):.1f}" y1="{top - 8}" x2="{sx(v):.1f}" y2="{top + rowh * len(FILTER) - 8}"/>')
        body.append(t(sx(v), top + rowh * len(FILTER) + 6, f"{v}%", "mu", "middle"))
    body.append(t(x1 + 90, top - 14, "유지한 위치", "t2", "middle", size=12))
    for i, (name, kept, canc) in enumerate(FILTER):
        y = top + i * rowh
        hot = name == "필터 없음"
        body.append(t(x0 - 14, y + 14, name, "b" if hot or "기본" in name else "", "end"))
        body.append(f'<rect class="{"s2f" if hot else "s1f"}" x="{x0}" y="{y}" width="{sx(canc) - x0:.1f}" height="20" rx="4"/>')
        body.append(t(sx(canc) + 8, y + 15, f"{canc}%", "b"))
        body.append(t(x1 + 90, y + 15, f"{kept:.2f}%", "t2", "middle", size=12))
    return doc(W, H, "상쇄 비율은 0.3%의 이상 위치를 빼느냐에 달려 있다",
               "이상 위치 필터 기준(은닉 상태 크기 차이)별, T_inter가 지우는 T_add 비율. 모델·데이터 평균",
               body, SRC + " App. E.3 Table 2")


def fig_intervention():
    W, H = 870, 290
    body = []
    panels = [("최종 상대 오차 R(L)", 1, 1.5, [0, 0.5, 1, 1.5]), ("KL(원래 ‖ 양자화)", 2, 10, [0, 2, 4, 6, 8, 10])]
    pw = 220
    for pi, (title, idx, vmax, ticks) in enumerate(panels):
        ox = 190 + pi * (pw + 150)
        sx = lambda v: ox + pw * v / vmax
        top, rowh = 110, 40
        body.append(t(ox, top - 22, title, "b", size=13))
        for v in ticks:
            body.append(f'<line class="grid" x1="{sx(v):.1f}" y1="{top - 8}" x2="{sx(v):.1f}" y2="{top + rowh * 3 - 12}"/>')
            body.append(t(sx(v), top + rowh * 3 + 4, f"{v:g}", "mu", "middle"))
        base = INTERVENE[0][idx]
        for i, row in enumerate(INTERVENE):
            y = top + i * rowh
            v = row[idx]
            if pi == 0:
                body.append(t(ox - 14, y + 15, row[0], "b" if i == 0 else "", "end"))
            cls = "s1f" if i == 0 else "s2f"
            body.append(f'<rect class="{cls}" x="{ox}" y="{y}" width="{max(sx(v) - ox, 2):.1f}" height="22" rx="4"/>')
            lab = f"{v:g}" if i == 0 else f"{v:g} ({v / base:.1f}배)"
            body.append(t(sx(v) + 8, y + 16, lab, "b", size=12))
    return doc(W, H, "상쇄를 없애면 오차가 3배, 뒤집으면 8배",
               "Qwen3-32B, 블록 17~48에서 새 오차의 크기는 두고 방향만 바꾼 개입 실험",
               body, SRC + " Sec. 3.3, Fig. 3(A)")


def fig_rotation():
    W, H = 820, 330
    x0, x1 = 160, 700
    lo, hi = -1.2, 1.4
    sx = lambda v: x0 + (math.log10(v) - lo) / (hi - lo) * (x1 - x0)
    top, rowh = 112, 44
    body = []
    for v, lab in ((0.1, "0.1°"), (1, "1°"), (10, "10°")):
        body.append(f'<line class="grid" x1="{sx(v):.1f}" y1="{top - 10}" x2="{sx(v):.1f}" y2="{top + rowh * 3 - 14}"/>')
        body.append(t(sx(v), top + rowh * 3 + 2, lab, "mu", "middle"))
    for i, (name, (rot, ang)) in enumerate(ROTATION.items()):
        y = top + i * rowh
        body.append(t(x0 - 14, y + 4, name, "", "end"))
        body.append(f'<line x1="{sx(ang):.1f}" y1="{y}" x2="{sx(rot):.1f}" y2="{y}" style="stroke:var(--rule);stroke-width:2"/>')
        body.append(f'<circle class="s2f ring" cx="{sx(ang):.1f}" cy="{y}" r="6"/>')
        body.append(f'<circle class="s1f ring" cx="{sx(rot):.1f}" cy="{y}" r="6"/>')
        body.append(t(sx(ang) - 10, y + 4, f"{ang}°", "t2", "end", size=12))
        body.append(t(sx(rot) + 10, y + 4, f"{rot}° · {rot / ang:.0f}배", "b", size=12))
    ly = 82
    body.append(f'<circle class="s1f" cx="{x0}" cy="{ly - 4}" r="6"/>')
    body.append(t(x0 + 12, ly, "LM-head 입력이 돌아간 각도", size=12))
    body.append(f'<circle class="s2f" cx="{x0 + 220}" cy="{ly - 4}" r="6"/>')
    body.append(t(x0 + 232, ly, "어휘 각 행과의 각도가 바뀐 양 (평균)", size=12))
    pred = 1 / mu_d(HIDDEN_DIM)
    body.append(t(24, top + rowh * 3 + 34,
                  f"고차원 등방 회전 가정의 예측 감쇠 {pred:.1f}배 (d = {HIDDEN_DIM}), 측정 {attenuation_measured():.1f}배.", "t2", size=12.5))
    return doc(W, H, "은닉 상태가 12° 돌아도 각 토큰 점수는 0.16°만큼만 흔들린다",
               "Qwen3-32B W4. 고차원에서는 회전의 대부분이 특정 어휘 벡터 방향과 무관한 쪽으로 간다 (로그 눈금)",
               body, SRC + " Fig. 5(A), Thm. 2")


def fig_bench():
    W = 760
    x0, x1 = 190, 600
    vmin, vmax = -3, 1
    sx = lambda v: x0 + (x1 - x0) * (v - vmin) / (vmax - vmin)
    top, rowh = 96, 32
    rows = list(BENCH.items())
    body = []
    for v in range(vmin, vmax + 1):
        body.append(f'<line class="{"ref" if v == 0 else "grid"}" x1="{sx(v):.1f}" y1="{top - 10}" x2="{sx(v):.1f}" y2="{top + rowh * len(rows) - 10}"/>')
        body.append(t(sx(v), top + rowh * len(rows) + 6, f"{v:+d}" if v else "0", "mu", "middle"))
    for i, (name, (b, w)) in enumerate(rows):
        y = top + i * rowh
        d = w - b
        sd = BENCH_SD[name]
        body.append(t(x0 - 14, y + 4, name, "b" if name == "WinoGrande" else "", "end"))
        body.append(f'<line x1="{sx(max(vmin, d - sd)):.1f}" y1="{y}" x2="{sx(min(vmax, d + sd)):.1f}" y2="{y}" style="stroke:var(--rule);stroke-width:6;stroke-linecap:round"/>')
        body.append(f'<circle class="{"s2f" if d < 0 else "s1f"} ring" cx="{sx(d):.1f}" cy="{y}" r="6"/>')
        body.append(t(x1 + 20, y + 4, f"{d:+.2f}  ({b} → {w})", "b" if name == "WinoGrande" else "t2", size=12))
    ny = top + rowh * len(rows) + 30
    body.append(t(24, ny, f"평균 {bench_mean_drop():+.2f}%p. 회색 띠 = W4 점수의 부트스트랩 SD(차이의 신뢰구간이 아님). 하락의 대부분은 WinoGrande 하나입니다.", "t2", size=12))
    H = ny + 46
    return doc(W, H, "4비트로 반올림만 해도 정확도는 거의 그대로",
               "Qwen3-32B, NVFP4 가중치 RTN(보정 데이터 없음). 0-shot 정확도 W4 − BF16 (%p)",
               body, SRC + " Fig. 4(A)")


FIGURES = {
    "ptq_counteraction_vectors.svg": fig_vectors,
    "ptq_random_vs_pretrained.svg": fig_random_vs_pretrained,
    "ptq_cancel_budget.svg": fig_cancel_budget,
    "ptq_filter_sensitivity.svg": fig_filter,
    "ptq_intervention.svg": fig_intervention,
    "ptq_lmhead_rotation.svg": fig_rotation,
    "ptq_benchmarks.svg": fig_bench,
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in FIGURES.items():
        (OUT / name).write_text(fn(), encoding="utf-8")
        print("wrote", (OUT / name).relative_to(ROOT))


if __name__ == "__main__":
    main()
