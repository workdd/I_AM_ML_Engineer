"""hydra-db/selfgraph 노트의 그림을 생성한다.

    python scripts/figures/selfgraph_figures.py              readings/repos/assets/selfgraph_*.svg 를 다시 쓴다
    python scripts/figures/selfgraph_figures.py --inline DIR  HTML 보고서용 (접두어, 생성 메모 없이 DIR/<이름>.svg)

SVG 는 손으로 고치지 않는다. decay 식은 커밋 771f359 의 docs/maintenance.md, 병합 구간과 예시 유사도는
docs/resolution.md 의 값이다. 원문에는 이 곡선과 구간을 그린 그림이 없다.
"""
import sys
from pathlib import Path

FONT = "-apple-system,'Apple SD Gothic Neo','Noto Sans KR',sans-serif"
INK, SUB, GRID = "#1f2328", "#59636e", "#d8dee4"
GRAY, BLUE, ORANGE, GREEN, RED, PURPLE = "#8b949e", "#2563eb", "#ea580c", "#16a34a", "#dc2626", "#7c3aed"
SHOW_NOTE = True
NOTE = "생성: python scripts/figures/selfgraph_figures.py, 손으로 고치지 말 것"
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "readings" / "repos" / "assets"

HALF_LIFE = 30
THRESHOLD = 0.92            # DefaultResolver 기본
ALIAS_CONF = 0.95
BAND = (0.8, 0.92)          # LLMResolver 기본
# (이름 쌍, 원문 유사도, 결과, 색)
PAIRS = [
    ("Bill Gates / William Gates", 0.81, "구간 안이지만 단어 대응 없음 → 묻지 않고 분리", RED),
    ("Jon Smithers / Jonathan Smithers", 0.873, "기본은 분리, LLM 이 yes 면 병합", ORANGE),
    ("Jonathan Smithers / Jonathon Smithers", 0.976, "fuzzy 에서 바로 병합", GREEN),
]


def relevance(age_days, n, half_life=HALF_LIFE):
    """maintenance.decay 식. n = source_count (+ 엔티티면 degree)."""
    recency = 0.5 ** (max(0.0, age_days) / half_life)
    frequency = n / (n + 1)
    return round(recency * (1 + frequency) / 2, 6)


def svg(w, h, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'role="img" aria-label="{title}" style="max-width:100%;height:auto;font-family:{FONT}">'
            f'<title>{title}</title><rect width="{w}" height="{h}" fill="#ffffff"/>{body}'
            + (f'<text x="{w-8}" y="{h-6}" font-size="9" fill="{GRAY}" text-anchor="end">{NOTE}</text>' if SHOW_NOTE else "")
            + '</svg>')


def t(x, y, s, size=12, fill=INK, anchor="start", weight="normal"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}">{s}</text>')


CURVES = [(0, "출처 0개, 관계 0개", GRAY), (1, "n = 1", BLUE), (3, "n = 3", ORANGE), (10, "n = 10", GREEN)]


def fig_decay():
    W, H = 760, 400
    x0, x1, y0, y1 = 70, 560, 340, 70
    dmax = 120

    def X(d):
        return x0 + d / dmax * (x1 - x0)

    def Y(v):
        return y0 - v / 1.0 * (y0 - y1)

    b = [t(20, 28, "decay relevance: 나이와 출처 수 n 에 따른 점수 (반감기 30일)", 15, weight="bold"),
         t(20, 48, "relevance = 0.5^(나이/30) × (1 + n/(n+1)) / 2. 검색에서는 키워드 점수가 같은 seed 의 순서만 정한다", 11, SUB)]
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        b.append(f'<line x1="{x0}" y1="{Y(v):.1f}" x2="{x1}" y2="{Y(v):.1f}" stroke="{GRID}"/>')
        b.append(t(x0 - 8, Y(v) + 4, f"{v:g}", 10, SUB, "end"))
    for d in range(0, dmax + 1, 30):
        b.append(f'<line x1="{X(d):.1f}" y1="{y1}" x2="{X(d):.1f}" y2="{y0}" stroke="{GRID}"/>')
        b.append(t(X(d), y0 + 16, f"{d}일", 10, SUB, "middle"))
    b.append(t((x0 + x1) / 2, y0 + 34, "마지막 갱신 뒤 지난 날수", 11, SUB, "middle"))
    for i, (n, lab, col) in enumerate(CURVES):
        pts = " ".join(f"{X(d):.1f},{Y(relevance(d, n)):.1f}" for d in range(0, dmax + 1, 2))
        b.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.2"/>')
        ly = 90 + i * 26
        b.append(f'<line x1="{x1+24}" y1="{ly-4}" x2="{x1+44}" y2="{ly-4}" stroke="{col}" stroke-width="2.2"/>')
        b.append(t(x1 + 50, ly, f"{lab}: 새 항목 {relevance(0, n):g}", 11, col if n else SUB))
    b.append(t(x1 + 24, 210, "n 이 커져도 상한은 1 미만", 11, SUB))
    b.append(t(x1 + 24, 228, f"30일이면 점수가 절반: n=1 은", 11, SUB))
    b.append(t(x1 + 24, 246, f"{relevance(0, 1):g} → {relevance(30, 1):g}", 11, BLUE, weight="bold"))
    return svg(W, H, "".join(b), "SelfGraph decay relevance 곡선")


def fig_resolution():
    W, H = 760, 330
    x0, x1 = 60, 700
    lo, hi = 0.75, 1.0

    def X(v):
        return x0 + (v - lo) / (hi - lo) * (x1 - x0)

    b = [t(20, 28, "이름 유사도 구간별 엔티티 병합 규칙 (같은 타입, 기본값)", 15, weight="bold"),
         t(20, 48, "유사도 = max(Jaro-Winkler, 내용 토큰 Jaccard). 예시 점수는 원문 문서 값", 11, SUB)]
    top, bh = 70, 46
    zones = [(lo, BAND[0], "#f6f8fa", "LLM 에도 묻지 않음", SUB),
             (BAND[0], BAND[1], "#fff4e5", "llm 단계: yes/no 질문 (후보당 최대 3회)", ORANGE),
             (THRESHOLD, hi, "#e8f5ec", "fuzzy 단계: 바로 병합", GREEN)]
    for a, z, fill, lab, col in zones:
        b.append(f'<rect x="{X(a):.1f}" y="{top}" width="{X(z)-X(a):.1f}" height="{bh}" fill="{fill}" stroke="{GRID}"/>')
        b.append(t((X(a) + X(z)) / 2, top + 28, lab, 11, col, "middle", "bold"))
    for v in (0.75, 0.8, 0.85, 0.9, 0.92, 1.0):
        b.append(f'<line x1="{X(v):.1f}" y1="{top+bh}" x2="{X(v):.1f}" y2="{top+bh+6}" stroke="{SUB}"/>')
        b.append(t(X(v), top + bh + 20, f"{v:g}", 10, SUB, "middle"))
    for i, (pair, s, res, col) in enumerate(PAIRS):
        y = top + bh + 52 + i * 44
        b.append(f'<line x1="{X(s):.1f}" y1="{top+bh}" x2="{X(s):.1f}" y2="{y-12}" stroke="{col}" stroke-dasharray="2 3"/>')
        b.append(f'<circle cx="{X(s):.1f}" cy="{y-6}" r="5" fill="{col}"/>')
        anchor, xx = ("end", X(s) - 10) if s > 0.9 else ("start", X(s) + 10)
        b.append(t(xx, y - 8, f"{pair} ({s})", 11, INK, anchor, "bold"))
        b.append(t(xx, y + 8, res, 11, col, anchor))
    b.append(t(20, H - 22, "숫자열, 한 글자 단어, Jr/II 같은 접미사가 다르거나 단어 대응이 없으면 유사도와 무관하게 병합하지 않는다", 11, SUB))
    return svg(W, H, "".join(b), "SelfGraph 이름 유사도 병합 구간")


FIGS = [("decay", fig_decay), ("resolution", fig_resolution)]


def main(argv):
    global SHOW_NOTE
    if argv[:1] == ["--inline"]:
        out, prefix, SHOW_NOTE = Path(argv[1]), "", False
    else:
        out, prefix = OUT, "selfgraph_"
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in FIGS:
        (out / f"{prefix}{name}.svg").write_text(fn(), "utf-8")
        print("wrote", out / f"{prefix}{name}.svg")


if __name__ == "__main__":
    main(sys.argv[1:])
