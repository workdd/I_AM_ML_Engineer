"""토스 채용 아티클 "AI-Ready Data의 최전선" 노트의 그림을 생성한다.

    python scripts/figures/tossai_figures.py              readings/blogs/assets/tossai_*.svg 를 다시 쓴다
    python scripts/figures/tossai_figures.py --inline DIR  HTML 보고서용 (접두어·생성 메모 없이 DIR/<이름>.svg)

SVG 는 손으로 고치지 않는다. 원문에는 그림이 없으므로 원문 수치를 다시 계산한 것만 그린다.
  errcut:  마이데이터 도메인 정답률 94% → 98% 를 오답률로 바꿔 본다
  goldenn: 골든셋 문항 수 N 에 따라 4pp · 10pp 변화가 몇 문항인지, 그리고 그 변화가 우연이 아니라고 말하려면
           (같은 세트 전후 대응 비교, 바뀐 문항이 모두 한 방향인 최선의 경우, exact McNemar 양측 5%) 몇 문항이 필요한지
"""
import sys
from pathlib import Path

FONT = "-apple-system,'Apple SD Gothic Neo','Noto Sans KR',sans-serif"
INK, SUB, GRID = "#1f2328", "#59636e", "#d8dee4"
GRAY, BLUE, ORANGE, GREEN, RED = "#8b949e", "#2563eb", "#ea580c", "#16a34a", "#dc2626"
SHOW_NOTE = True
NOTE = "생성: python scripts/figures/tossai_figures.py · 손으로 고치지 말 것"
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "readings" / "blogs" / "assets"

# 원문: 마이데이터 골든셋 정답률 90% 초과, 지식 보강 하나로 94% → 98%, 하룻밤 "두 자릿수 포인트 이상" 급락
FLOOR, BEFORE, AFTER = 90, 94, 98   # %
DROP_PP = 10                        # "두 자릿수" 의 최소
ALPHA = 0.05


def error_cut(before=BEFORE, after=AFTER):
    """정답률(%) 전후 → (전 오답률, 후 오답률, 오답 감소 비율)."""
    e0, e1 = 100 - before, 100 - after
    return e0, e1, 1 - e1 / e0


def mcnemar_p(b):
    """대응 비교에서 바뀐 문항 b 개가 모두 한 방향일 때 exact McNemar 양측 p."""
    return min(1.0, 2 * 0.5 ** b)


def min_flips(alpha=ALPHA):
    b = 1
    while mcnemar_p(b) >= alpha:
        b += 1
    return b


def min_n(pp, alpha=ALPHA):
    """pp 포인트 변화가 min_flips 문항 이상이 되는 최소 골든셋 크기."""
    need = min_flips(alpha)
    return -(-need * 100 // pp)


def svg(w, h, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'role="img" aria-label="{title}" style="max-width:100%;height:auto;font-family:{FONT}">'
            f'<title>{title}</title><rect width="{w}" height="{h}" fill="#ffffff"/>{body}'
            + (f'<text x="{w-8}" y="{h-6}" font-size="9" fill="{GRAY}" text-anchor="end">{NOTE}</text>' if SHOW_NOTE else "")
            + '</svg>')


def t(x, y, s, size=12, fill=INK, anchor="start", weight="normal"):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}">{s}</text>')


def fig_errcut():
    w, h = 680, 360
    e0, e1, cut = error_cut()
    ef = 100 - FLOOR
    out = t(20, 28, "정답률 94% → 98% 를 오답률로 보면 (원문 수치로 계산)", 14.5, INK, weight="bold")
    out += t(20, 46, "막대는 100 문항 중 틀리는 문항 수. 정답률은 4포인트 차이지만 오답은 세 개 중 두 개가 사라진다", 11.5, SUB)
    L, base, top = 230, 260, 80
    scale = (base - top) / 10
    rows = [(f"정답률 {FLOOR}% (원문: 90% 초과)", ef, GRAY), (f"보강 전 {BEFORE}%", e0, ORANGE), (f"보강 후 {AFTER}%", e1, GREEN)]
    bw, gap = 90, 60
    for i, (name, e, col) in enumerate(rows):
        x = L + i * (bw + gap)
        y = base - e * scale
        out += f'<rect x="{x}" y="{y:.1f}" width="{bw}" height="{e*scale:.1f}" fill="{col}" rx="3"/>'
        out += t(x + bw / 2, y - 8, f"{e}문항", 13, INK, "middle", "bold")
        out += t(x + bw / 2, base + 18, name, 11, SUB, "middle")
    out += f'<line x1="{L-20}" y1="{base}" x2="{L+3*bw+2*gap+20}" y2="{base}" stroke="{GRID}"/>'
    out += t(20, 140, "오답률", 12, INK, weight="bold")
    out += t(20, 160, f"{e0}% → {e1}%", 22, GREEN, weight="bold")
    out += t(20, 182, f"오답 {cut:.0%} 감소", 12, SUB)
    out += t(20, h - 42, "같은 4포인트라도 정답률이 높을수록 남은 오답 중 큰 몫이다. 90%대에서 개선 효과는 오답률로 읽는 편이 크기가 드러난다", 11, SUB)
    out += t(20, h - 24, "100 문항은 비율을 보이기 위한 단위다. 원문은 골든셋 문항 수를 밝히지 않았다", 11, SUB)
    return svg(w, h, out, "정답률 94%에서 98%로 오른 것을 오답률로 본 그림")


def fig_goldenn():
    w, h, L, R, T, B = 680, 430, 70, 600, 80, 300
    ns = list(range(25, 301, 25))
    nmax, ymax = 300, 30
    X = lambda n: L + (R - L) * n / nmax
    Y = lambda v: B - (B - T) * v / ymax
    need = min_flips()
    out = t(20, 28, "골든셋 문항 수에 따라 4포인트 · 10포인트 변화가 몇 문항인지 (노트에서 계산)", 14.5, INK, weight="bold")
    out += t(20, 46, f"빨간 선: 같은 세트 전후 비교에서 바뀐 문항이 모두 한 방향일 때 우연이 아니라고 말할 최소 문항 수 ({need}문항, 양측 5%)", 11.5, SUB)
    for v in range(0, ymax + 1, 5):
        out += f'<line x1="{L}" y1="{Y(v)}" x2="{R}" y2="{Y(v)}" stroke="{GRID}"/>' + t(L - 8, Y(v) + 4, f"{v}", 11, SUB, "end")
    for n in (0, 50, 100, 150, 200, 250, 300):
        out += t(X(n), B + 18, f"{n}", 11, SUB, "middle")
    out += t((L + R) / 2, B + 38, "골든셋 문항 수 N", 11, SUB, "middle")
    out += t(L - 40, T - 12, "바뀌는 문항 수", 11, SUB)
    out += f'<line x1="{L}" y1="{Y(need)}" x2="{R}" y2="{Y(need)}" stroke="{RED}" stroke-dasharray="4 4"/>'
    out += t(R - 4, Y(need) - 6, f"{need}문항: p = {mcnemar_p(need):.3f}", 11, RED, "end")
    for pp, col, label in ((DROP_PP, ORANGE, "10포인트 (하룻밤 급락의 최소폭)"), (AFTER - BEFORE, BLUE, "4포인트 (지식 보강 94%→98%)")):
        pts = " ".join(f"{X(n):.1f},{Y(n * pp / 100):.1f}" for n in [0] + ns)
        out += f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.4"/>'
        mn = min_n(pp)
        out += f'<circle cx="{X(mn):.1f}" cy="{Y(need):.1f}" r="4.5" fill="{col}"/>'
        out += t(X(mn) + 8, Y(need) + 18, f"N={mn}", 12, col, weight="bold")
        ye = min(ymax, nmax * pp / 100)
        xe = nmax if nmax * pp / 100 <= ymax else ymax * 100 / pp
        out += t(X(xe) - 4, Y(ye) - 8, label, 11, col, "end")
    out += t(20, h - 60, f"최선의 경우에도 4포인트 개선을 우연과 가르려면 골든셋이 {min_n(AFTER - BEFORE)}문항 이상이어야 하고, 10포인트 급락은 {min_n(DROP_PP)}문항이면 잡힌다", 11, SUB)
    out += t(20, h - 42, "좋아진 문항과 나빠진 문항이 섞이면 필요한 문항 수는 더 늘어난다. 매일 밤 비교라면 같은 세트를 쓰는 대응 비교가 맞다", 11, SUB)
    out += t(20, h - 24, "원문은 골든셋 문항 수를 밝히지 않았으므로 원문 결과의 유의성을 판정하는 그림이 아니다", 11, SUB)
    return svg(w, h, out, "골든셋 문항 수와 정답률 변화의 문항 수")


FIGS = [("errcut", fig_errcut), ("goldenn", fig_goldenn)]


def main(argv):
    global SHOW_NOTE
    if argv[:1] == ["--inline"]:
        out, prefix, SHOW_NOTE = Path(argv[1]), "", False
    else:
        out, prefix = OUT, "tossai_"
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in FIGS:
        (out / f"{prefix}{name}.svg").write_text(fn(), "utf-8")
        print("wrote", out / f"{prefix}{name}.svg")


if __name__ == "__main__":
    main(sys.argv[1:])
