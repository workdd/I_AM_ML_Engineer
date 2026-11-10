"""if(kakao)26 카카오페이증권 "지식그래프로 만드는 eXplainable 금융 에이전트" 노트의 그림을 생성한다.

    python scripts/figures/kpaykg_figures.py              readings/blogs/assets/kpaykg_*.svg 를 다시 쓴다
    python scripts/figures/kpaykg_figures.py --inline DIR  HTML 보고서용 (접두어·생성 메모 없이 DIR/<이름>.svg)

SVG 는 손으로 고치지 않는다. 발표 슬라이드의 그림(기술 지도, 온톨로지, 평가 추이)은 노트에서 PDF 쪽 링크로 인용하고,
여기서는 슬라이드에 없는 계산만 그린다.
  multihop: 발표에서 말한 "검색 한 번 80%면 세 번 연속은 절반 이하" 를 hop 수에 대해 검산 (hop 사이 독립 가정)
  scale:    32쪽 규모(기사 8만, 원시 사건 23만, 통합 사건 12만, 엣지 90만, IMPACTS 48.1만)를 기사 한 건당 값으로 환산
"""
import sys
from pathlib import Path

FONT = "-apple-system,'Apple SD Gothic Neo','Noto Sans KR',sans-serif"
INK, SUB, GRID = "#1f2328", "#59636e", "#d8dee4"
GRAY, BLUE, ORANGE, GREEN, RED = "#8b949e", "#2563eb", "#ea580c", "#16a34a", "#dc2626"
SHOW_NOTE = True
NOTE = "생성: python scripts/figures/kpaykg_figures.py · 손으로 고치지 말 것"
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "readings" / "blogs" / "assets"

# 32쪽 (만 단위)
ARTICLES, RAW_EVENTS, EVENTS, EDGES, IMPACTS = 8.0, 23.0, 12.0, 90.0, 48.1
SPOKEN_P, SPOKEN_HOPS = 0.8, 3  # 발표 06:54


def chain(p, hops):
    """hop 마다 필요한 문서가 상위 k 에 들 확률 p 가 독립이면, 모든 hop 이 성공할 확률."""
    return p ** hops


def per_article():
    return {
        "raw_events": RAW_EVENTS / ARTICLES,
        "events": EVENTS / ARTICLES,
        "edges": EDGES / ARTICLES,
        "edges_per_event": EDGES / EVENTS,
        "merge_cut": 1 - EVENTS / RAW_EVENTS,
        "impacts_share": IMPACTS / EDGES,
    }


def svg(w, h, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'role="img" aria-label="{title}" style="max-width:100%;height:auto;font-family:{FONT}">'
            f'<title>{title}</title><rect width="{w}" height="{h}" fill="#ffffff"/>{body}'
            + (f'<text x="{w-8}" y="{h-6}" font-size="9" fill="{GRAY}" text-anchor="end">{NOTE}</text>' if SHOW_NOTE else "")
            + '</svg>')


def t(x, y, s, size=12, fill=INK, anchor="start", weight="normal"):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}">{s}</text>')


def fig_multihop():
    w, h, L, R, T, B = 680, 420, 60, 600, 80, 300
    hops = list(range(1, 7))
    X = lambda n: L + (R - L) * (n - 1) / (hops[-1] - 1)
    Y = lambda v: B - (B - T) * v
    out = t(20, 28, "검색을 이어 붙일 때 모든 hop 이 성공할 확률 (노트에서 계산)", 14.5, INK, weight="bold")
    out += t(20, 46, "hop 마다 필요한 문서가 상위 k 에 들 확률을 p 로 두고, hop 사이가 독립이라고 가정한 p^n", 11.5, SUB)
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        out += f'<line x1="{L}" y1="{Y(v)}" x2="{R}" y2="{Y(v)}" stroke="{GRID}"/>' + t(L - 8, Y(v) + 4, f"{v:.0%}", 11, SUB, "end")
    out += f'<line x1="{L}" y1="{Y(0.5)}" x2="{R}" y2="{Y(0.5)}" stroke="{RED}" stroke-dasharray="4 4"/>'
    out += t(L + 6, Y(0.5) - 6, "절반", 11, RED)
    for n in hops:
        out += t(X(n), B + 18, f"{n}", 11, SUB, "middle")
    out += t((L + R) / 2, B + 38, "이어 붙인 검색 hop 수", 11, SUB, "middle")
    for p, col in ((0.9, GREEN), (SPOKEN_P, ORANGE), (0.7, BLUE)):
        pts = " ".join(f"{X(n):.1f},{Y(chain(p, n)):.1f}" for n in hops)
        out += f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.4"/>'
        for n in hops:
            out += f'<circle cx="{X(n):.1f}" cy="{Y(chain(p, n)):.1f}" r="3" fill="{col}"/>'
        out += t(X(hops[-1]) + 6, Y(chain(p, hops[-1])) + 4, f"p={p}", 11, col)
    for n in (SPOKEN_HOPS, SPOKEN_HOPS + 1):
        v = chain(SPOKEN_P, n)
        out += t(X(n) + 6, Y(v) - 8, f"{n} hop: {v:.3f}", 11, ORANGE, weight="bold")
    out += t(20, h - 42, f"발표는 p=0.8 이면 3번 연속 성공이 절반 이하라고 했다. 계산하면 0.8^3 = {chain(0.8, 3):.3f} 로 절반을 약간 넘고,", 11, SUB)
    out += t(20, h - 24, f"4 hop 부터 {chain(0.8, 4):.3f} 로 절반 아래다. 결론(이을수록 빠르게 줄어든다)은 그대로다", 11, SUB)
    return svg(w, h, out, "검색 hop 수와 연속 성공 확률")


def fig_scale():
    w, h = 680, 400
    r = per_article()
    rows = [
        ("뉴스 기사 (필터 후)", ARTICLES, 1.0, GRAY, "기준"),
        ("원시 사건 (추출 직후)", RAW_EVENTS, r["raw_events"], ORANGE, f"기사당 {r['raw_events']:.1f}"),
        ("사건 노드 (통합 후)", EVENTS, r["events"], BLUE, f"기사당 {r['events']:.1f} · 통합으로 {r['merge_cut']:.0%} 감소"),
        ("엣지 전체", EDGES, r["edges"], GREEN, f"기사당 {r['edges']:.2f} · 사건당 {r['edges_per_event']:.1f}"),
        ("그중 IMPACTS", IMPACTS, IMPACTS / ARTICLES, GREEN, f"엣지의 {r['impacts_share']:.0%}"),
    ]
    L, R, top, step = 190, 450, 80, 50
    vmax = EDGES
    out = t(20, 28, "넉 달 실측 규모를 기사 한 건당 값으로 바꾸면 (32쪽 값으로 계산)", 14.5, INK, weight="bold")
    out += t(20, 46, "2026.04~2026.08 실측. 막대는 만 단위 개수, 오른쪽은 기사 8만 건으로 나눈 값", 11.5, SUB)
    for i, (name, v, _, col, note) in enumerate(rows):
        y = top + i * step
        bw = (R - L) * v / vmax
        op = "0.55" if name.startswith("그중") else "1"
        out += t(L - 10, y + 18, name, 12, INK, "end")
        out += f'<rect x="{L}" y="{y}" width="{bw:.1f}" height="26" fill="{col}" opacity="{op}" rx="3"/>'
        out += t(L + bw + 6, y + 18, f"{v:g}만", 12, INK, weight="bold")
        out += t(L + bw + 60, y + 18, note, 11, SUB)
    out += t(20, h - 60, "사건 추출이 기사당 약 2.9건을 내고, 같은 사건 통합이 그 절반 가까이를 합친다. 통합 단계가 그래프 크기를 정하는 큰 요인이다", 11, SUB)
    out += t(20, h - 42, "발표의 12개월 270만 엣지 추정은 월 2만 건 유입의 선형 비례이고, 병합률이 오르면 그보다 적을 수 있다고 슬라이드 각주가 밝힌다", 11, SUB)
    out += t(20, h - 24, "2026-01~04 원시 113,704건은 백필 예정분으로 위 값에 들어 있지 않다", 11, SUB)
    return svg(w, h, out, "기사 한 건당 사건과 엣지 수")


FIGS = [("multihop", fig_multihop), ("scale", fig_scale)]


def main(argv):
    global SHOW_NOTE
    if argv[:1] == ["--inline"]:
        out, prefix, SHOW_NOTE = Path(argv[1]), "", False
    else:
        out, prefix = OUT, "kpaykg_"
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in FIGS:
        (out / f"{prefix}{name}.svg").write_text(fn(), "utf-8")
        print("wrote", out / f"{prefix}{name}.svg")


if __name__ == "__main__":
    main(sys.argv[1:])
