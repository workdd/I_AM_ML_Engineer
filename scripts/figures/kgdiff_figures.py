"""iiwish/kgdiff 노트의 그림을 생성한다.

    python scripts/figures/kgdiff_figures.py              readings/repos/assets/kgdiff_*.svg 를 다시 쓴다
    python scripts/figures/kgdiff_figures.py --inline DIR  HTML 보고서용 (접두어·생성 메모 없이 DIR/<이름>.svg)

SVG 는 손으로 고치지 않는다. 가중치·임계값은 커밋 d502bc2 의 packages/core/src/match.ts(WEIGHTS, LABEL_FLOOR)와
options.ts(threshold 0.82, suggestThreshold 0.6) 값이고, score() 는 combineEvidence 를 그대로 옮긴 것이다.
원문에는 이 점수 곡선을 그린 그림이 없다.
"""
import sys
from pathlib import Path

FONT = "-apple-system,'Apple SD Gothic Neo','Noto Sans KR',sans-serif"
INK, SUB, GRID = "#1f2328", "#59636e", "#d8dee4"
GRAY, BLUE, ORANGE, GREEN, RED, PURPLE = "#8b949e", "#2563eb", "#ea580c", "#16a34a", "#dc2626", "#7c3aed"
SHOW_NOTE = True
NOTE = "생성: python scripts/figures/kgdiff_figures.py · 손으로 고치지 말 것"
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "readings" / "repos" / "assets"

WEIGHTS = {"label": 0.55, "neighbors": 0.25, "types": 0.1, "attrs": 0.1}
LABEL_FLOOR = 0.5
THRESHOLD, SUGGEST = 0.82, 0.6


def score(label, neighbors=None, types=None, attrs=None):
    """combineEvidence: 있는 신호만 가중 평균, 라벨 < 0.5 면 0.5 상한."""
    s, w = WEIGHTS["label"] * label, WEIGHTS["label"]
    for k, v in (("neighbors", neighbors), ("types", types), ("attrs", attrs)):
        if v is not None:
            s += WEIGHTS[k] * v
            w += WEIGHTS[k]
    out = s / w
    return round(min(out, LABEL_FLOOR) if label < LABEL_FLOOR else out, 6)


def min_label(neighbors=None, target=THRESHOLD):
    """라벨 외 신호가 이웃뿐일 때 target 점수에 닿는 최소 라벨 유사도. 닿지 못하면 None."""
    wn = 0.0 if neighbors is None else WEIGHTS["neighbors"]
    need = (target * (WEIGHTS["label"] + wn) - wn * (neighbors or 0)) / WEIGHTS["label"]
    return need if need <= 1 else None


# (이름, 이웃 신호, 색, 대시)
CURVES = [
    ("이웃 신호 없음 (라벨만)", None, INK, ""),
    ("이웃 1.0 (매칭된 이웃이 모두 일치)", 1.0, GREEN, ""),
    ("이웃 0.5", 0.5, BLUE, "6 3"),
    ("이웃 0 (매칭된 이웃이 하나도 안 맞음)", 0.0, RED, "2 3"),
]


def svg(w, h, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'role="img" aria-label="{title}" style="max-width:100%;height:auto;font-family:{FONT}">'
            f'<title>{title}</title><rect width="{w}" height="{h}" fill="#ffffff"/>{body}'
            + (f'<text x="{w-8}" y="{h-6}" font-size="9" fill="{GRAY}" text-anchor="end">{NOTE}</text>' if SHOW_NOTE else "")
            + '</svg>')


def t(x, y, s, size=12, fill=INK, anchor="start", weight="normal"):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}">{s}</text>')


def fig_score():
    w, h = 760, 490
    out = t(20, 28, "노드 매칭 점수: 라벨이 같아도 이웃이 하나도 안 맞으면 매칭되지 않는다", 14.5, INK, weight="bold")
    out += t(20, 46, "combineEvidence 를 그대로 계산. 타입·속성 신호가 없는 경우. 있는 신호만 평균하므로 이웃 신호가 생기면 분모가 커진다", 11.5, SUB)
    L, R, T, B = 70, 500, 92, 402
    X = lambda v: L + (R - L) * v
    Y = lambda v: B - (B - T) * v
    # 구간 배경: 매칭 / 제안 / 미매칭
    out += f'<rect x="{L}" y="{Y(1):.1f}" width="{R-L}" height="{Y(THRESHOLD)-Y(1):.1f}" fill="#dcfce7"/>'
    out += f'<rect x="{L}" y="{Y(THRESHOLD):.1f}" width="{R-L}" height="{Y(SUGGEST)-Y(THRESHOLD):.1f}" fill="#fef3c7"/>'
    out += f'<rect x="{L}" y="{Y(LABEL_FLOOR):.1f}" width="{X(LABEL_FLOOR)-L:.1f}" height="{B-Y(LABEL_FLOOR):.1f}" fill="#f3f4f6"/>'
    for v in (0, 0.2, 0.4, 0.6, 0.8, 1.0):
        out += f'<line x1="{X(v):.1f}" y1="{T}" x2="{X(v):.1f}" y2="{B}" stroke="{GRID}"/>'
        out += t(X(v), B + 16, f"{v:.1f}", 11, SUB, "middle")
        out += t(L - 8, Y(v) + 4, f"{v:.1f}", 11, SUB, "end")
    for v, label in ((THRESHOLD, f"매칭 기준 {THRESHOLD}"), (SUGGEST, f"제안 기준 {SUGGEST}")):
        out += f'<line x1="{L}" y1="{Y(v):.1f}" x2="{R}" y2="{Y(v):.1f}" stroke="{INK}" stroke-dasharray="4 3"/>'
        out += t(L + 6, Y(v) - 5, label, 10.5, INK, weight="bold")
    out += t(X(0.36), Y(0.09), "라벨 &lt; 0.5:", 10.5, GRAY, "middle")
    out += t(X(0.36), Y(0.09) + 14, "점수 상한 0.5", 10.5, GRAY, "middle")
    for name, nb, color, dash in CURVES:
        pts = " ".join(f"{X(i/100):.1f},{Y(score(i/100, nb)):.1f}" for i in range(0, 101))
        out += f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.2" stroke-dasharray="{dash}"/>'
        m = min_label(nb)
        if m is not None:
            out += f'<circle cx="{X(m):.1f}" cy="{Y(THRESHOLD):.1f}" r="4" fill="{color}"/>'
    out += f'<rect x="{L}" y="{T}" width="{R-L}" height="{B-T}" fill="none" stroke="{GRAY}"/>'
    out += t((L + R) / 2, B + 34, "라벨 유사도 (textSimilarity)", 11, SUB, "middle")
    out += t(L - 50, T - 10, "매칭 점수", 11, SUB)
    # 범례와 계산값
    lx = 530
    for i, (name, nb, color, dash) in enumerate(CURVES):
        y = T + 8 + 40 * i
        out += f'<line x1="{lx}" y1="{y}" x2="{lx+26}" y2="{y}" stroke="{color}" stroke-width="2.2" stroke-dasharray="{dash}"/>'
        out += t(lx + 34, y + 4, name, 10.5, INK)
        m = min_label(nb)
        note = f"라벨 ≥ {m:.3f} 이면 매칭" if m is not None else f"라벨 1.0 이어도 {score(1.0, nb):.4f} → 제안까지"
        out += t(lx + 34, y + 19, note, 10.5, color, weight="bold")
    out += t(lx, T + 182, "점 = 매칭 기준에 닿는 최소 라벨 유사도", 10.5, SUB)
    out += t(lx, T + 199, "가중치: 라벨 0.55 · 이웃 0.25", 10.5, SUB)
    out += t(lx, T + 216, "  · 타입 0.1 · 속성 0.1", 10.5, SUB)
    out += t(lx, T + 233, "경쟁 후보보다 0.05 이상 앞서야 수락", 10.5, SUB)
    out += t(20, h - 40, "이웃 신호는 이미 짝지어진 이웃이 있을 때만 생긴다. 첫 라운드에서 라벨만으로 짝을 찾고, 그 짝이 다음 라운드의 이웃 증거가 된다(최대 5라운드)", 11, SUB)
    out += t(20, h - 22, "타입·속성이 같으면 곡선이 위로, 다르면 아래로 움직인다. 이 그림은 둘 다 비어 있는 LLM 추출 그래프에 가까운 경우다", 11, SUB)
    return svg(w, h, out, "kgdiff 노드 매칭 점수와 임계값")


FIGS = [("score", fig_score)]


def main(argv):
    global SHOW_NOTE
    if argv[:1] == ["--inline"]:
        out, prefix, SHOW_NOTE = Path(argv[1]), "", False
    else:
        out, prefix = OUT, "kgdiff_"
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in FIGS:
        (out / f"{prefix}{name}.svg").write_text(fn(), "utf-8")
        print("wrote", out / f"{prefix}{name}.svg")


if __name__ == "__main__":
    main(sys.argv[1:])
