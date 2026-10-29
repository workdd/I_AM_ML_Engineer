"""SkipKV (MLSys 2026) 노트의 그림을 생성한다.

    python scripts/figures/skipkv_figures.py              readings/papers/assets/skipkv_*.svg 를 다시 쓴다
    python scripts/figures/skipkv_figures.py --inline DIR  HTML 보고서용 (접두어·생성 메모 없이 DIR/<이름>.svg)

SVG 는 손으로 고치지 않는다. 수치는 논문 표 1(처리량), 표 2(구성 요소), 표 3(배치 묶기)을 옮긴 아래 상수만 쓴다.
"""
import sys
from pathlib import Path

FONT = "-apple-system,'Apple SD Gothic Neo','Noto Sans KR',sans-serif"
INK, SUB, GRID = "#1f2328", "#59636e", "#d8dee4"
GRAY, BLUE, ORANGE, GREEN, PURPLE, LIGHT = "#8b949e", "#2563eb", "#ea580c", "#16a34a", "#7c3aed", "#93c5fd"
SHOW_NOTE = True
NOTE = "생성: python scripts/figures/skipkv_figures.py · 손으로 고치지 말 것"
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "readings" / "papers" / "assets"

# 표 1: GSM8K, R1-Qwen-7B, 예산 512, A100 40GB. 처리량(샘플/분), None = OOM
BATCHES = [10, 50, 100, 120, 140]
THROUGHPUT = {
    "FullKV": [4.07, 2.64, None, None, None],
    "SEAL": [7.41, 6.87, None, None, None],
    "R-KV": [5.81, 13.7, 20.0, 18.1, 18.8],
    "SkipKV": [9.70, 22.7, 25.4, 20.0, 19.4],
}
# 표 2: AIME24, R1-Qwen-7B. (단계, 정확도 3개, 토큰 3개), 예산 2220 / 3072 / 4096
BUDGETS = ["2220 (20%)", "3072 (27%)", "4096 (37%)"]
FULLKV_ACC, FULLKV_TOK = 46.7, 11132
ABLATION = [
    ("R-KV", (36.7, 33.3, 36.7), (12000, 12109, 11403)),
    ("+ 문장 점수", (40.0, 36.7, 40.0), (11332, 11342, 11819)),
    ("+ 적응형 steering", (40.0, 43.3, 40.0), (8860, 10101, 10041)),
    ("+ 배치 묶기 = SkipKV", (50.0, 53.3, 50.0), (9228, 8593, 7988)),
]
# 표 3: MATH-500, R1-Qwen-7B, SkipKV. (예산, 유효 예산 없음/있음, 정확도 없음/있음)
GROUPING = [
    (768, 630, 759, 77.8, 83.4),
    (1024, 886, 1015, 85.2, 86.4),
    (1260, 1122, 1251, 85.8, 86.0),
    (1536, 1398, 1527, 86.0, 88.0),
]
COLOR = {"FullKV": GRAY, "SEAL": GREEN, "R-KV": ORANGE, "SkipKV": BLUE}


def svg(w, h, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'role="img" aria-label="{title}" style="max-width:100%;height:auto;font-family:{FONT}">'
            f'<title>{title}</title><rect width="{w}" height="{h}" fill="#ffffff"/>{body}'
            + (f'<text x="{w-8}" y="{h-6}" font-size="9" fill="{GRAY}" text-anchor="end">{NOTE}</text>' if SHOW_NOTE else "")
            + '</svg>')


def t(x, y, s, size=12, fill=INK, anchor="start", weight="normal"):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}">{s}</text>')


def fig_throughput():
    w, h, L, R, T, B = 700, 420, 60, 520, 80, 320
    X = lambda i: L + 20 + (R - L - 40) * i / (len(BATCHES) - 1)
    Y = lambda v: B - (B - T) * v / 30
    out = t(20, 28, "배치 크기별 처리량 (GSM8K, R1-Qwen-7B, KV 예산 512, A100 40GB)", 14.5, INK, weight="bold")
    out += t(20, 46, "FullKV와 SEAL은 배치 100부터 메모리 부족. 고정 예산 방식만 큰 배치로 간다", 11.5, SUB)
    for v in (0, 10, 20, 30):
        out += f'<line x1="{L}" y1="{Y(v)}" x2="{R}" y2="{Y(v)}" stroke="{GRID}"/>' + t(L - 8, Y(v) + 4, v, 11, SUB, "end")
    for i, b in enumerate(BATCHES):
        out += t(X(i), B + 18, b, 11.5, SUB, "middle")
    out += t((L + R) / 2, B + 38, "배치 크기", 11, SUB, "middle")
    out += t(L - 40, T - 14, "샘플/분", 11, SUB)
    for name, vals in THROUGHPUT.items():
        c = COLOR[name]
        pts = [(X(i), Y(v)) for i, v in enumerate(vals) if v is not None]
        out += f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}" fill="none" stroke="{c}" stroke-width="2.2"/>'
        for i, v in enumerate(vals):
            if v is None:
                out += t(X(i), B - 6 - 12 * list(THROUGHPUT).index(name), "OOM", 9.5, c, "middle", "bold")
            else:
                out += f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="3.2" fill="{c}"/>'
        lv = [v for v in vals if v is not None][-1]
        li = max(i for i, v in enumerate(vals) if v is not None)
        dy = {"SkipKV": -8, "R-KV": 14}.get(name, 4)
        out += t(X(li) + 8, Y(lv) + dy, name, 11.5, c, weight="bold")
    # 9.6배의 기준점
    sk, fk = THROUGHPUT["SkipKV"][2], THROUGHPUT["FullKV"][1]
    out += t(R + 75, T + 40, "논문의 9.6배", 11.5, INK, weight="bold")
    out += t(R + 75, T + 58, f"{sk} ÷ {fk}", 11, SUB)
    out += t(R + 75, T + 74, "(FullKV 배치 50)", 11, SUB)
    out += t(R + 75, T + 110, "각자 최고점끼리", 11.5, INK, weight="bold")
    out += t(R + 75, T + 128, f"{sk} ÷ {THROUGHPUT['FullKV'][0]}", 11, SUB)
    out += t(R + 75, T + 144, f"≈ {sk / THROUGHPUT['FullKV'][0]:.1f}배", 11, SUB)
    out += t(20, h - 24, "같은 배치 10에서 SkipKV ÷ R-KV = 9.70 ÷ 5.81 ≈ 1.7배 (논문의 \"최대 1.7배\")", 11, SUB)
    return svg(w, h, out, "배치 크기별 처리량")


def fig_ablation():
    w, h, L, R, T, B = 700, 430, 60, 680, 90, 300
    Y = lambda v: B - (B - T) * (v - 30) / 30
    out = t(20, 28, "구성 요소를 하나씩 더할 때 정확도 (AIME24, R1-Qwen-7B, 배치 10)", 14.5, INK, weight="bold")
    out += t(20, 46, "가장 큰 증가는 배치 묶기 단계에서 나온다. AIME24는 한 문항이 약 3.3점이다", 11.5, SUB)
    for v in (30, 40, 50, 60):
        out += f'<line x1="{L}" y1="{Y(v)}" x2="{R}" y2="{Y(v)}" stroke="{GRID}"/>' + t(L - 8, Y(v) + 4, v, 11, SUB, "end")
    out += f'<line x1="{L}" y1="{Y(FULLKV_ACC)}" x2="{R}" y2="{Y(FULLKV_ACC)}" stroke="{GRAY}" stroke-dasharray="5 4"/>'
    out += t(L + 4, Y(FULLKV_ACC) - 5, f"FullKV {FULLKV_ACC}", 10.5, SUB)
    cols = [ORANGE, LIGHT, "#60a5fa", BLUE]
    gw, bw = (R - L) / len(BUDGETS), 34
    for g, bud in enumerate(BUDGETS):
        x0 = L + g * gw + (gw - bw * len(ABLATION)) / 2
        for s, (name, acc, tok) in enumerate(ABLATION):
            v, x = acc[g], x0 + s * bw
            out += f'<rect x="{x:.1f}" y="{Y(v):.1f}" width="{bw - 5}" height="{B - Y(v):.1f}" fill="{cols[s]}"/>'
            out += t(x + (bw - 5) / 2, Y(v) - 5, v, 10, INK, "middle")
            out += t(x + (bw - 5) / 2, B + 14, f"{tok[g] / 1000:.1f}k", 9.5, SUB, "middle")
        out += t(L + g * gw + gw / 2, B + 34, f"KV 예산 {bud}", 11.5, INK, "middle")
    out += t(L - 8, B + 14, "토큰", 9.5, SUB, "end")
    lx = L
    for s, (name, _, _) in enumerate(ABLATION):
        out += f'<rect x="{lx}" y="58" width="11" height="11" fill="{cols[s]}"/>' + t(lx + 15, 68, name, 11)
        lx += 15 + len(name) * 8.5 + 20
    out += t(20, h - 30, f"막대 아래 숫자는 총 생성 토큰 (FullKV {FULLKV_TOK:,}). 문장 점수 단계는 +3.3~3.4점, 한 문항 차이다", 11, SUB)
    return svg(w, h, out, "SkipKV 구성 요소별 정확도")


def fig_grouping():
    w, h = 760, 360
    L, T, rowh, maxb = 150, 86, 52, 1536
    sx = lambda v: 300 * v / maxb
    out = t(20, 28, "배치 묶기 전후의 유효 KV 예산과 정확도 (MATH-500, R1-Qwen-7B, 배치 10)", 14.5, INK, weight="bold")
    out += t(20, 46, "묶기 전에는 매 예산에서 패딩이 138토큰을 차지한다. prefill 길이로 정렬해 묶으면 9토큰으로 준다", 11.5, SUB)
    out += t(L, T - 14, "유효 예산 (회색: 패딩이 차지한 몫)", 11, SUB) + t(625, T - 14, "정확도 %", 11, SUB, "middle")
    for i, (b, v0, v1, a0, a1) in enumerate(GROUPING):
        y = T + i * rowh
        out += t(L - 10, y + 22, f"예산 {b}", 12.5, INK, "end", "bold")
        for k, (v, c, lab) in enumerate(((v0, ORANGE, "묶기 전"), (v1, BLUE, "묶기 후"))):
            yy = y + k * 20
            out += f'<rect x="{L}" y="{yy}" width="{sx(v):.1f}" height="15" fill="{c}"/>'
            out += f'<rect x="{L + sx(v):.1f}" y="{yy}" width="{sx(b - v):.1f}" height="15" fill="{GRID}"/>'
            out += t(L + sx(b) + 6, yy + 12, f"{v} {lab}", 10.5, c)
        out += t(605, y + 13, a0, 12, ORANGE, "end") + t(617, y + 13, "→", 12, SUB) + t(635, y + 13, a1, 12, BLUE, weight="bold")
        out += t(677, y + 13, f"(+{a1 - a0:.1f})", 11, SUB)
    out += t(20, h - 28, "연속 배칭·paged KV를 쓰는 서버(vLLM 등)는 prefill 패딩이 없어 이 이득이 생기지 않는다 (논문 부록 표 4는 묶기 없이 실험)", 11, SUB)
    return svg(w, h, out, "배치 묶기와 유효 KV 예산")


FIGS = [("throughput", fig_throughput), ("ablation", fig_ablation), ("grouping", fig_grouping)]


def main(argv):
    global SHOW_NOTE
    if argv[:1] == ["--inline"]:
        out, prefix, SHOW_NOTE = Path(argv[1]), "", False
    else:
        out, prefix = OUT, "skipkv_"
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in FIGS:
        (out / f"{prefix}{name}.svg").write_text(fn(), "utf-8")
        print("wrote", out / f"{prefix}{name}.svg")


if __name__ == "__main__":
    main(sys.argv[1:])
