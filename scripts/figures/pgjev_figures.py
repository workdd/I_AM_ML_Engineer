"""realZachi/pg-jev 노트의 그림을 생성한다.

    python scripts/figures/pgjev_figures.py              readings/repos/assets/pgjev_*.svg 를 다시 쓴다
    python scripts/figures/pgjev_figures.py --inline DIR  HTML 보고서용 (접두어·생성 메모 없이 DIR/<이름>.svg)

SVG 는 손으로 고치지 않는다. 수치는 레포 v0.2.1 (커밋 e3450d4) 의 README.md,
.agents/skills/pgjev/references/how-it-works.md, CHANGELOG.md 0.2.0 절에서 옮긴 아래 상수만 쓴다.
원문에는 그래프가 없어, 흩어진 수치를 한 축에 모으고 토큰 어림값을 선형 모형으로 검산한다.
"""
import sys
from pathlib import Path

FONT = "-apple-system,'Apple SD Gothic Neo','Noto Sans KR',sans-serif"
INK, SUB, GRID = "#1f2328", "#59636e", "#d8dee4"
GRAY, BLUE, ORANGE, GREEN, LIGHT = "#8b949e", "#2563eb", "#ea580c", "#16a34a", "#93c5fd"
SHOW_NOTE = True
NOTE = "생성: python scripts/figures/pgjev_figures.py · 손으로 고치지 말 것"
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "readings" / "repos" / "assets"

# how-it-works.md: 요청 고정비 약 270토큰, 1행 단독 약 435토큰
OVERHEAD, SINGLE = 270, 435
PER_ROW = SINGLE - OVERHEAD          # 165. 노트가 세운 선형 모형의 행당 몫

# how-it-works.md: 정답 데이터 3종(각 400행)에서 묶음 크기별 정확도 (최소, 최대) %
ACCURACY = [("1–20행", 20, (100, 100)), ("40행", 40, (92, 98)), ("80행", 80, (77, 94))]

# README · CHANGELOG 0.2.0: 2,000행 테이블, 유럽 (API까지 약 190ms). (항목, v0.1.0 초, v0.2.x 초)
TIMINGS = [
    ("전체 스캔, 새 조건", 8.5, 3.5),
    ("LIMIT 3, 새 조건", 8.4, 0.6),
    ("연결이 살아 있는 세션의 새 조건", None, 2.3),
    ("같은 쿼리 재실행 (캐시)", None, 0.05),
]
TOKENS = (338_000, 296_000)          # 전체 스캔 입력 토큰 v0.1.0, v0.2.x


def tokens_per_row(n):
    """요청 하나에 n행을 묶을 때 행당 입력 토큰 (선형 모형)."""
    return (OVERHEAD + PER_ROW * n) / n


def svg(w, h, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'role="img" aria-label="{title}" style="max-width:100%;height:auto;font-family:{FONT}">'
            f'<title>{title}</title><rect width="{w}" height="{h}" fill="#ffffff"/>{body}'
            + (f'<text x="{w-8}" y="{h-6}" font-size="9" fill="{GRAY}" text-anchor="end">{NOTE}</text>' if SHOW_NOTE else "")
            + '</svg>')


def t(x, y, s, size=12, fill=INK, anchor="start", weight="normal"):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}">{s}</text>')


def fig_batch():
    w, h = 760, 400
    out = t(20, 28, "묶음 크기: 토큰은 20행 이후 거의 줄지 않고, 정확도는 40행부터 떨어진다", 14.5, INK, weight="bold")
    out += t(20, 46, "왼쪽은 원문 어림값(고정비 270, 단독 435)으로 세운 선형 모형, 오른쪽은 원문 실측 범위", 11.5, SUB)
    # 왼쪽: 행당 토큰
    L, R, T, B = 70, 370, 80, 300
    X = lambda n: L + (R - L) * (n - 1) / 79
    Y = lambda v: B - (B - T) * (v - 150) / (450 - 150)
    for v in (150, 250, 350, 450):
        out += f'<line x1="{L}" y1="{Y(v):.1f}" x2="{R}" y2="{Y(v):.1f}" stroke="{GRID}"/>' + t(L - 8, Y(v) + 4, v, 11, SUB, "end")
    for n in (1, 20, 40, 60, 80):
        out += t(X(n), B + 18, n, 11, SUB, "middle")
    out += t((L + R) / 2, B + 36, "요청 하나에 묶는 행 수", 11, SUB, "middle")
    out += t(L - 50, T - 18, "행당 입력 토큰", 11, SUB)
    pts = " ".join(f"{X(n):.1f},{Y(tokens_per_row(n)):.1f}" for n in range(1, 81))
    out += f'<polyline points="{pts}" fill="none" stroke="{BLUE}" stroke-width="2.4"/>'
    for n, dx, dy in ((1, 8, 4), (20, 4, -12), (40, 0, -12), (80, -4, -12)):
        v = tokens_per_row(n)
        out += f'<circle cx="{X(n):.1f}" cy="{Y(v):.1f}" r="3.5" fill="{BLUE}"/>'
        out += t(X(n) + dx, Y(v) + dy, f"{n}행: {v:.1f}", 10.5, BLUE, "end" if n == 80 else "start")
    ratio = (tokens_per_row(20) / tokens_per_row(40) - 1) * 100
    out += t(L + 40, T + 40, f"20행은 40행보다 {ratio:.1f}% 많다 (원문 4%)", 11, INK)
    # 오른쪽: 정확도 범위
    L2, R2 = 440, 730
    Y2 = lambda v: B - (B - T) * (v - 70) / 30
    for v in (70, 80, 90, 100):
        out += f'<line x1="{L2}" y1="{Y2(v):.1f}" x2="{R2}" y2="{Y2(v):.1f}" stroke="{GRID}"/>' + t(L2 - 8, Y2(v) + 4, f"{v}%", 11, SUB, "end")
    out += t(L2 - 40, T - 18, "정답 데이터 대비 정확도 (3종, 각 400행)", 11, SUB)
    step = (R2 - L2) / len(ACCURACY)
    for i, (label, _, (lo, hi)) in enumerate(ACCURACY):
        cx = L2 + step * (i + 0.5)
        color = GREEN if lo == 100 else ORANGE
        if lo == hi:
            out += f'<rect x="{cx-22}" y="{Y2(hi)-3:.1f}" width="44" height="6" rx="2" fill="{color}"/>'
            out += t(cx + 28, Y2(hi) + 4, f"{hi}%", 11, color, "start", "bold")
        else:
            out += f'<rect x="{cx-22}" y="{Y2(hi):.1f}" width="44" height="{Y2(lo)-Y2(hi):.1f}" rx="3" fill="{color}" fill-opacity="0.75"/>'
            out += t(cx, Y2(hi) - 6, f"{hi}%", 11, color, "middle", "bold")
            out += t(cx, Y2(lo) + 15, f"{lo}%", 11, color, "middle", "bold")
        out += t(cx, B + 18, label, 11, SUB, "middle")
    out += t(20, h - 40, f"선형 모형: 토큰 = 270 + {PER_ROW} × 행 수. 원문 \"20행 묶음이면 행당 약 175\" 와 178.5 로 맞는다", 11, SUB)
    out += t(20, h - 22, "정확도가 떨어지는 이유를 원문은 모델이 배열 안 rows[i] 를 위치로 찾기 때문이라고 설명한다", 11, SUB)
    return svg(w, h, out, "묶음 크기별 행당 토큰과 정확도")


def fig_versions():
    w, h = 760, 360
    out = t(20, 28, "v0.1.0 → v0.2.x: 2,000행 테이블 실측 시간 (유럽, API까지 약 190ms)", 14.5, INK, weight="bold")
    out += t(20, 46, "스트리밍 미리 읽기 · 영속 연결 · 20행 묶음 · 동시 요청 16개로 바꾼 결과", 11.5, SUB)
    L, R, T = 250, 690, 74
    vmax = 9
    X = lambda v: L + (R - L) * v / vmax
    for v in (0, 3, 6, 9):
        out += f'<line x1="{X(v):.1f}" y1="{T-6}" x2="{X(v):.1f}" y2="{T + 52 * len(TIMINGS)}" stroke="{GRID}"/>'
        out += t(X(v), T + 52 * len(TIMINGS) + 16, f"{v}초", 11, SUB, "middle")
    for i, (label, old, new) in enumerate(TIMINGS):
        y = T + 52 * i
        out += t(L - 10, y + 22, label, 12, INK, "end")
        if old is not None:
            out += f'<rect x="{L}" y="{y}" width="{X(old)-L:.1f}" height="16" rx="2" fill="{GRAY}"/>'
            out += t(X(old) + 6, y + 12, f"v0.1.0 {old}초", 11, SUB)
        bw = max(X(new) - L, 2)
        out += f'<rect x="{L}" y="{y+19}" width="{bw:.1f}" height="16" rx="2" fill="{BLUE}"/>'
        txt = f"v0.2.x {new}초" if new >= 0.1 else f"v0.2.x {new * 1000:.0f}ms"
        if old is not None:
            txt += f" ({old / new:.1f}배 빠름)"
        out += t(L + bw + 6, y + 31, txt, 11, BLUE)
    old_t, new_t = TOKENS
    cut = (1 - new_t / old_t) * 100
    out += t(20, h - 40, f"입력 토큰: 전체 스캔 {old_t // 1000}k → {new_t // 1000}k ({cut:.1f}% 감소). 같은 판에서 입력의 16%를 차지하던 범용 criteria 문장을 뺐고, 20행 묶음으로 4% 늘었다", 11, SUB)
    out += t(20, h - 22, "v0.1.0 은 연결 유지 · 캐시 재실행 수치를 따로 밝히지 않았다", 11, SUB)
    return svg(w, h, out, "pg-jev 버전별 실측 시간")


FIGS = [("batch", fig_batch), ("versions", fig_versions)]


def main(argv):
    global SHOW_NOTE
    if argv[:1] == ["--inline"]:
        out, prefix, SHOW_NOTE = Path(argv[1]), "", False
    else:
        out, prefix = OUT, "pgjev_"
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in FIGS:
        (out / f"{prefix}{name}.svg").write_text(fn(), "utf-8")
        print("wrote", out / f"{prefix}{name}.svg")


if __name__ == "__main__":
    main(sys.argv[1:])
