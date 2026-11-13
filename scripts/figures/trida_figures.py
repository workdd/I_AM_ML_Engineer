"""trida-stack 노트의 그림을 생성한다.

    python scripts/figures/trida_figures.py              readings/repos/assets/trida_*.svg 를 다시 쓴다
    python scripts/figures/trida_figures.py --inline DIR  HTML 보고서용 (접두어 · 생성 메모 없이 DIR/<이름>.svg)

SVG 는 손으로 고치지 않는다. 데이터는 Hugging Face trillionlabs/Zoro-1 모델 카드
(2026-11-06 판, 2026-11-13 확인)의 속도 표와 reasoned action escalation 표만 쓴다.
"""
import sys
from pathlib import Path

FONT = "-apple-system,'Apple SD Gothic Neo','Noto Sans KR',sans-serif"
INK, SUB, GRID = "#1f2328", "#59636e", "#d8dee4"
GRAY, BLUE, ORANGE, GREEN, YELLOW, RED, PURPLE = (
    "#8b949e", "#2563eb", "#ea580c", "#16a34a", "#ca8a04", "#dc2626", "#9333ea")
SHOW_NOTE = True
NOTE = "생성: python scripts/figures/trida_figures.py · 손으로 고치지 말 것"
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "readings" / "repos" / "assets"

# Zoro-1 카드: H100 1장, vLLM 0.27 + self-spec plugin, block 8, GSM8K, 설정당 200문항
# (동시 요청 수, AR tok/s per request, self-spec tok/s per request, 카드가 적은 speedup)
SPEED = [(1, 211, 312, 1.48), (4, 190, 257, 1.36), (8, 168, 222, 1.32), (16, 137, 156, 1.14)]
# 총 처리량 이득: 1.26× (C=1), 1.17–1.26× (C=4–8), 0.99× (C=16)
TOTAL = [((1,), 1.26, 1.26), ((4, 8), 1.17, 1.26), ((16,), 0.99, 0.99)]

# escalation 표: 이름 -> (direct 정확도, [(문턱, 정확도, 요청 비율 %)], always reason 정확도)
ESCALATE = {
    "JevBench public": (0.779, [(0.5, 0.835, 12), (0.7, 0.866, 29), (0.9, 0.879, 58)], 0.892),
    "hard-v1": (0.812, [(0.5, 0.852, 13), (0.7, 0.932, 42), (0.9, 0.960, 68)], 0.958),
    "transfer-v4": (0.803, [(0.5, 0.805, 6), (0.7, 0.843, 29), (0.9, 0.860, 63)], 0.861),
}
ESC_COLORS = {"JevBench public": BLUE, "hard-v1": ORANGE, "transfer-v4": GREEN}


def recovered(direct, acc, always):
    """always reason 대비 정확도 이득을 얼마나 회수했는가."""
    return (acc - direct) / (always - direct)


def svg(w, h, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'role="img" aria-label="{title}" style="max-width:100%;height:auto;font-family:{FONT}">'
            f'<title>{title}</title><rect width="{w}" height="{h}" fill="#ffffff"/>{body}'
            + (f'<text x="{w-8}" y="{h-6}" font-size="9" fill="{GRAY}" text-anchor="end">{NOTE}</text>' if SHOW_NOTE else "")
            + '</svg>')


def t(x, y, s, size=12, fill=INK, anchor="start", weight="normal"):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}">{s}</text>')


def fig_concurrency():
    w, h = 800, 490
    L, R, T, B = 80, 600, 80, 380
    vmin, vmax = 0.8, 1.6
    Y = lambda v: B - (B - T) * (v - vmin) / (vmax - vmin)
    n = len(SPEED)
    slot = (R - L) / n
    X = {c: L + slot * (i + 0.5) for i, (c, *_) in enumerate(SPEED)}
    out = t(20, 28, "동시 요청이 늘수록 self-spec 이득이 줄고, 16개에서는 총 처리량 이득이 사라진다", 15, INK, weight="bold")
    out += t(20, 46, "Zoro-1 모델 카드: H100 1장, vLLM 0.27 + self-spec plugin, block 8, GSM8K thinking on, 설정당 200문항", 11, SUB)
    for v in (0.8, 1.0, 1.2, 1.4, 1.6):
        out += f'<line x1="{L}" y1="{Y(v):.1f}" x2="{R}" y2="{Y(v):.1f}" stroke="{GRID if v != 1.0 else INK}" stroke-width="1"/>'
        out += t(L - 8, Y(v) + 4, f"{v:.1f}×", 10, SUB, "end")
    out += t(R + 4, Y(1.0) + 4, "AR 과 같음", 10, INK)
    bw = slot * 0.34
    for c, ar, ss, card in SPEED:
        x = X[c]
        out += (f'<rect x="{x-bw-2:.1f}" y="{Y(card):.1f}" width="{bw:.1f}" height="{Y(vmin)-Y(card):.1f}" '
                f'fill="{BLUE}" opacity="0.85"/>')
        out += t(x - bw / 2 - 2, Y(card) - 6, f"{card:.2f}×", 11, BLUE, "middle", "bold")
        out += t(x, B + 18, f"동시 {c}", 11, INK, "middle")
        out += t(x, B + 34, f"{ar} → {ss} tok/s", 10, SUB, "middle")
    for cs, lo, hi in TOTAL:
        for c in cs:
            x = X[c] + 2
            col = RED if hi < 1.0 else ORANGE
            if hi - lo < 1e-9:
                out += (f'<rect x="{x:.1f}" y="{Y(hi):.1f}" width="{bw:.1f}" height="{Y(vmin)-Y(hi):.1f}" '
                        f'fill="{col}" opacity="0.75"/>')
                out += t(x + bw / 2, Y(hi) - 6, f"{hi:.2f}×", 11, col, "middle", "bold")
            else:
                out += (f'<rect x="{x:.1f}" y="{Y(lo):.1f}" width="{bw:.1f}" height="{Y(vmin)-Y(lo):.1f}" '
                        f'fill="{col}" opacity="0.75"/>')
                out += (f'<rect x="{x:.1f}" y="{Y(hi):.1f}" width="{bw:.1f}" height="{Y(lo)-Y(hi):.1f}" '
                        f'fill="none" stroke="{col}" stroke-dasharray="3 2"/>')
                out += t(x + bw / 2, Y(hi) - 6, f"{lo:.2f}–{hi:.2f}×", 10, col, "middle", "bold")
    x0 = 625
    out += f'<rect x="{x0}" y="{T}" width="12" height="12" fill="{BLUE}"/>' + t(x0 + 18, T + 10, "요청당 속도 이득", 11, INK)
    out += t(x0 + 18, T + 26, "(self-spec ÷ AR tok/s)", 10, SUB)
    out += f'<rect x="{x0}" y="{T+44}" width="12" height="12" fill="{ORANGE}"/>' + t(x0 + 18, T + 54, "총 처리량 이득", 11, INK)
    out += t(x0 + 18, T + 70, "카드는 동시 4–8 을 범위로만 적음", 10, SUB)
    out += t(x0 + 18, T + 84, "(점선 = 범위 상단)", 10, SUB)
    out += f'<rect x="{x0}" y="{T+102}" width="12" height="12" fill="{RED}"/>' + t(x0 + 18, T + 112, "1× 미만 (손해)", 11, INK)
    out += t(20, h - 40, "요청당 속도는 16개에서도 1.14× 빠르지만, GPU 가 이미 바쁜 구간이라 전체 처리량은 0.99× 로 AR 과 같아진다.", 11, SUB)
    out += t(20, h - 24, "동시 4 의 257 ÷ 190 은 1.353 이다. 카드는 1.36× 로 적는다 (반올림 차).", 11, SUB)
    return svg(w, h, out, "Zoro-1 동시 요청 수별 self-spec 이득")


def fig_escalate():
    w, h = 800, 480
    L, R, T, B = 80, 560, 80, 390
    X = lambda p: L + (R - L) * p / 100
    Y = lambda r: B - (B - T) * r / 1.1
    out = t(20, 28, "확신도 0.7 미만만 reasoning 으로 넘기면, 요청 29–42% 비용으로 정확도 이득의 69–82% 를 회수한다", 15, INK, weight="bold")
    out += t(20, 46, "Zoro-1 모델 카드의 escalation 표를 다시 계산: 회수율 = (문턱 정확도 − direct) ÷ (항상 reasoning − direct)", 11, SUB)
    for r in (0, 0.25, 0.5, 0.75, 1.0):
        out += f'<line x1="{L}" y1="{Y(r):.1f}" x2="{R}" y2="{Y(r):.1f}" stroke="{GRID if r < 1 else INK}" stroke-width="1"/>'
        out += t(L - 8, Y(r) + 4, f"{r:.0%}", 10, SUB, "end")
    for p in (0, 20, 40, 60, 80, 100):
        out += t(X(p), B + 16, f"{p}%", 10, SUB, "middle")
    out += f'<line x1="{L}" y1="{Y(0):.1f}" x2="{X(100):.1f}" y2="{Y(1.0):.1f}" stroke="{GRAY}" stroke-dasharray="4 3"/>'
    out += t(X(70), Y(0.62), "비용에 비례하는 선", 10, GRAY, "middle")
    out += t((L + R) / 2, B + 34, "reasoning 으로 넘긴 요청 비율 (= 생성 비용)", 11, INK, "middle")
    out += (f'<text x="{L-50}" y="{(T+B)/2}" font-size="11" fill="{INK}" text-anchor="middle" '
            f'transform="rotate(-90 {L-50} {(T+B)/2})">항상 reasoning 대비 정확도 이득 회수율</text>')
    for name, (d, pts, a) in ESCALATE.items():
        c = ESC_COLORS[name]
        xy = [(0, 0.0)] + [(s, recovered(d, acc, a)) for _, acc, s in pts] + [(100, 1.0)]
        path = " ".join(f"{X(p):.1f},{Y(r):.1f}" for p, r in xy)
        out += f'<polyline points="{path}" fill="none" stroke="{c}" stroke-width="2"/>'
        for (th, acc, s) in pts:
            r = recovered(d, acc, a)
            out += f'<circle cx="{X(s):.1f}" cy="{Y(r):.1f}" r="{5 if th == 0.7 else 3.5}" fill="{c}"/>'
            if th == 0.7:
                if name == "hard-v1":
                    out += t(X(s) - 2, Y(r) - 10, f"{r:.0%}", 11, c, "end", "bold")
                else:
                    out += t(X(s) + 8, Y(r) + 4, f"{r:.0%}", 11, c, weight="bold")
    x0 = 590
    out += t(x0, T + 4, "문턱 (top 확률 미만이면 reasoning)", 12, INK, weight="bold")
    out += t(x0, T + 22, "점 3개 = 0.5 · 0.7 · 0.9 (큰 점 0.7)", 10, SUB)
    y = T + 50
    for name, (d, pts, a) in ESCALATE.items():
        c = ESC_COLORS[name]
        out += f'<rect x="{x0}" y="{y-10}" width="12" height="12" fill="{c}"/>' + t(x0 + 18, y, name, 11, INK, weight="bold")
        out += t(x0 + 18, y + 15, f"direct {d:.3f} → 항상 {a:.3f}", 10, SUB)
        y += 40
    out += t(x0, y + 4, "decision-v7 은 이득이 0.001 이라 제외", 10, SUB)
    out += t(x0, y + 18, "hard-v1 은 문턱 0.9 가 항상보다 높다", 10, SUB)
    out += t(20, h - 24, "점선 위에 있으면 비용보다 많은 이득을 회수한 것이다. 문턱 0.5 에서는 yes/no 질문이 걸리지 않는다 (top 확률이 0.5 미만일 수 없음, 카드).", 11, SUB)
    return svg(w, h, out, "Zoro-1 reasoned action 문턱별 비용과 회수율")


FIGS = [("concurrency", fig_concurrency), ("escalate", fig_escalate)]


def main(argv):
    global SHOW_NOTE
    if argv[:1] == ["--inline"]:
        out, prefix, SHOW_NOTE = Path(argv[1]), "", False
    else:
        out, prefix = OUT, "trida_"
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in FIGS:
        (out / f"{prefix}{name}.svg").write_text(fn(), "utf-8")
        print("wrote", out / f"{prefix}{name}.svg")


if __name__ == "__main__":
    main(sys.argv[1:])
