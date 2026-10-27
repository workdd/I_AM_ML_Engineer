"""Dust: Pretraining Transformers Without Backpropagation (Q Labs, 2026-10) 노트의 그림을 생성한다.

    python scripts/figures/dust_figures.py              readings/papers/assets/dust_*.svg 를 다시 쓴다
    python scripts/figures/dust_figures.py --inline DIR  HTML 보고서용 (접두어·생성 메모 없이 DIR/<이름>.svg)

SVG 는 손으로 고치지 않는다. 수치는 원문 부록 Table 1·3·8 에서 옮긴 아래 데이터 절만 쓴다.
"""
import math
import sys
from pathlib import Path

FONT = "-apple-system,'Apple SD Gothic Neo','Noto Sans KR',sans-serif"
INK, SUB, GRID = "#1f2328", "#59636e", "#d8dee4"
VIOLET, GRAY, BLACK, BLUE, TEAL = "#7c3aed", "#8b949e", "#1f2328", "#2563eb", "#0f766e"
SHOW_NOTE = True
NOTE = "생성: python scripts/figures/dust_figures.py · 손으로 고치지 말 것"
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "readings" / "papers" / "assets"

# 부록 Table 1: 테스트 손실
BUDGETS = ["10만 토큰", "100만 토큰", "1,000만 토큰", "2,000만 토큰"]
POPS = [64, 256, 1024, 4096, 16384]
DUST = [[7.240, 7.165, 7.151, 7.164, 7.170], [6.290, 6.067, 5.934, 5.911, 5.916],
        [5.397, 5.231, 5.111, 5.072, 5.049], [5.265, 5.139, 4.960, 4.912, 4.802]]
EGG = [[7.961, 7.605, 7.386, 7.302, 7.263], [7.428, 7.200, 7.046, 6.892, 6.708],
       [7.056, 6.906, 6.692, 6.352, 6.033], [6.987, 6.811, 6.517, 6.200, 5.865]]
BP = [7.216, 5.959, 4.989, 4.633]

# 부록 Table 3: K = 16,384, SGD 배분
ALLOC = [("블록 출력 쓰기 쌍", 7040, "loss"), ("MLP 은닉층", 7040, "loss"),
         ("어텐션 출력", 1664, "loss"), ("토큰 임베딩", 640, "loss"),
         ("query (블록당)", 128, "attn"), ("value (블록당)", 1280, "attn"),
         ("gate (블록당)", 384, "attn"), ("key (블록당)", 768, "attn"),
         ("value embedding (블록당)", 512, "attn"), ("LM 헤드", 65536, "head"),
         ("잔차 스칼라", 8, "res")]

# 부록 Table 8: 1억 토큰 체크포인트 코사인 피팅
COS = [("LM head", 1.0, 759), ("Token emb", 0.7318, 19498), ("Attn out", 0.8729, 21627),
       ("MLP out", 0.8779, 31623), ("Value gate", 0.6657, 47863), ("MLP in", 0.8577, 48417),
       ("Attn Q", 1.0, 76736), ("Attn V", 0.8776, 173780), ("Value emb", 1.0, 495450),
       ("Attn K", 1.0, 616595)]
COS_COLORS = ["#0f766e", "#b45309", "#7c3aed", "#2563eb", "#be185d", "#0891b2",
              "#4d7c0f", "#9333ea", "#c2410c", "#475569"]


def svg(w, h, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'role="img" aria-label="{title}" style="max-width:100%;height:auto;font-family:{FONT}">'
            f'<title>{title}</title><rect width="{w}" height="{h}" fill="#ffffff"/>{body}'
            + (f'<text x="{w-8}" y="{h-6}" font-size="9" fill="{GRAY}" text-anchor="end">{NOTE}</text>' if SHOW_NOTE else "")
            + '</svg>')


def t(x, y, s, size=12, fill=INK, anchor="start", weight="normal"):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}">{s}</text>')


def box(x, y, w, h, lines, stroke, fill="#ffffff", size=12):
    out = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
    y0 = y + h / 2 - (len(lines) - 1) * (size + 3) / 2 + size * 0.35
    for i, s in enumerate(lines):
        out += t(x + w / 2, y0 + i * (size + 3), s, size, INK, "middle")
    return out


def arrow(x1, y1, x2, y2, color=SUB):
    ang = math.atan2(y2 - y1, x2 - x1)
    a1 = (x2 - 8 * math.cos(ang - 0.4), y2 - 8 * math.sin(ang - 0.4))
    a2 = (x2 - 8 * math.cos(ang + 0.4), y2 - 8 * math.sin(ang + 0.4))
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="1.6"/>'
            f'<polygon points="{x2},{y2} {a1[0]:.1f},{a1[1]:.1f} {a2[0]:.1f},{a2[1]:.1f}" fill="{color}"/>')


def fig_compare():
    W, H = 760, 400
    b = t(20, 28, "역전파와 Dust: 같은 공식, 다른 출력 오차", 15, INK, weight="bold")
    cols = [(20, "역전파", BLACK, "#f6f8fa",
             [["순전파 1회", "활성값 저장"], ["손실 계산"], ["역방향 계산", "연쇄 법칙으로 층마다", "출력 오차 δ = ∂L/∂y"]]),
            (400, "Dust", VIOLET, "#f5f0ff",
             [["clean 순전파 1회", "활성값 캐시"], ["층 출력에 토큰별 잡음 σa", "그 블록부터 다시 순전파", "K번 반복 (draw)"],
              ["토큰별 손실 감소 → 보상 r", "보상 가중 잡음 평균", "출력 오차 추정 ĝ"]])]
    for x, name, c, f, steps in cols:
        b += t(x + 170, 60, name, 14, c, "middle", "bold")
        y = 72
        for i, lines in enumerate(steps):
            h = 26 + 15 * len(lines)
            b += box(x, y, 340, h, lines, c, f)
            if i < len(steps) - 1:
                b += arrow(x + 170, y + h + 2, x + 170, y + h + 16, c)
            y += h + 18
        b += arrow(x + 170, y - 16, 380 if x < 380 else 380, 330, c)
    b += box(230, 330, 300, 48, ["가중치 기울기 = Σ (출력 오차) · 입력 xᵀ", "→ SGD / Adam 업데이트"], "#0f766e", "#ecfdf5", 13)
    return svg(W, H + 10, b, "역전파와 Dust 비교")


def fig_steps():
    steps = [("1", "clean 순전파", "잡음 없이 1회, 활성값 캐시"),
             ("2", "잡음 넣기", "층 출력 y → y + σa, 토큰마다 독립 a ~ N(0, I)"),
             ("3", "부분 순전파", "블록 l 의 draw 는 블록 l 부터만 재계산"),
             ("4", "토큰별 손실 감소", "c_s = (draw 평균 손실) - (이 draw 손실)"),
             ("5", "보상", "r_t = Σ_{s≥t} γ^(s-t) c_s"),
             ("6", "출력 오차 추정", "ĝ_t = -(1/Kσ) Σ_i r_t a_t"),
             ("7", "가중치 기울기", "Ĝ_W = Σ_t ĝ_t x_tᵀ"),
             ("8", "업데이트", "모멘텀 SGD 또는 Adam")]
    W, rowh = 760, 46
    H = 50 + rowh * len(steps)
    b = t(20, 28, "Dust 한 번의 업데이트 (3~6 단계를 층 종류·블록별로 K번 반복)", 15, INK, weight="bold")
    for i, (n, name, desc) in enumerate(steps):
        y = 44 + i * rowh
        c = VIOLET if 2 <= i <= 5 else TEAL
        b += f'<circle cx="38" cy="{y+18}" r="14" fill="{c}"/>' + t(38, y + 23, n, 13, "#ffffff", "middle", "bold")
        b += f'<rect x="62" y="{y}" width="678" height="36" rx="6" fill="#f6f8fa" stroke="{GRID}"/>'
        b += t(76, y + 23, name, 13, c, weight="bold") + t(220, y + 23, desc, 12.5, INK)
        if i < len(steps) - 1:
            b += f'<line x1="38" y1="{y+32}" x2="38" y2="{y+rowh+4}" stroke="{GRID}" stroke-width="2"/>'
    return svg(W, H + 10, b, "Dust 업데이트 단계")


def fig_population():
    W, H = 760, 330
    b = t(20, 28, "집단 구성원을 어디서 만드는가", 15, INK, weight="bold")
    b += t(20, 56, "가중치 공간 ES (EGGROLL): 구성원 1개 = 섭동한 가중치 사본 1벌 = 배치 원소 1개", 12.5, SUB)
    for i in range(4):
        x = 20 + i * 140
        b += box(x, 68, 120, 52, [f"W + ε{i+1}" if i < 3 else "…", "순전파 1회" if i < 3 else ""], GRAY, "#f6f8fa")
    b += t(600, 100, "구성원 수 ≈ 순전파 수", 13, GRAY, weight="bold")
    b += t(20, 160, "Dust: 구성원 1개 = 토큰 1개의 활성값 잡음. 가중치는 한 벌", 12.5, SUB)
    b += f'<rect x="20" y="172" width="720" height="40" rx="6" fill="#f5f0ff" stroke="{VIOLET}"/>'
    b += t(380, 197, "층 출력 활성값 (시퀀스 2,048 토큰)", 12.5, VIOLET, "middle", "bold")
    n = 16
    for i in range(n):
        x = 20 + i * 45
        lab = f"a{i+1}" if i < n - 1 else "a_T"
        b += f'<rect x="{x+4}" y="224" width="37" height="30" rx="4" fill="#ffffff" stroke="{VIOLET}"/>'
        b += t(x + 22, 244, lab if i != n - 2 else "…", 11, VIOLET, "middle")
    b += t(20, 284, "순전파 1회 안에서 토큰마다 다른 잡음 → 시퀀스당 구성원 수천 개", 13, VIOLET, weight="bold")
    b += t(20, 304, "토큰 축 집단은 배치 축과 직교하므로 배치 크기와 곱해진다", 12, SUB)
    return svg(W, H, b, "가상 집단")


def fig_alloc():
    W = 760
    rowh = 26
    H = 70 + rowh * len(ALLOC) + 40
    colors = {"loss": VIOLET, "attn": BLUE, "head": TEAL, "res": GRAY}
    names = {"loss": "토큰 손실로 보상 (합 = K)", "attn": "어텐션 내부 (어텐션 출력 정렬로 보상)",
             "head": "LM 헤드 (캐시 로짓에 직접 잡음)", "res": "잔차 스칼라 (가중치 공간 ES)"}
    b = t(20, 28, "업데이트 1회의 draw 배분 (K = 16,384, SGD, 100만 토큰)", 15, INK, weight="bold")
    x0, x1 = 230, 680
    lmax = math.log10(65536)
    for v in [10, 100, 1000, 10000]:
        x = x0 + (x1 - x0) * math.log10(v) / lmax
        b += f'<line x1="{x:.1f}" y1="50" x2="{x:.1f}" y2="{50+rowh*len(ALLOC)}" stroke="{GRID}"/>'
        b += t(x, 46, f"{v:,}", 10, SUB, "middle")
    for i, (name, v, g) in enumerate(ALLOC):
        y = 54 + i * rowh
        w = (x1 - x0) * math.log10(v) / lmax
        b += t(x0 - 8, y + 15, name, 12, INK, "end")
        b += f'<rect x="{x0}" y="{y+2}" width="{w:.1f}" height="17" rx="3" fill="{colors[g]}"/>'
        b += t(x0 + w + 6, y + 15, f"{v:,}", 11.5, INK)
    ly = 64 + rowh * len(ALLOC)
    for j, g in enumerate(["loss", "attn", "head", "res"]):
        x = 20 + j * 185
        b += f'<rect x="{x}" y="{ly}" width="12" height="12" fill="{colors[g]}"/>' + t(x + 16, ly + 10, names[g], 10.5, SUB)
    b += t(20, ly + 30, "가로축은 로그 눈금", 10.5, SUB)
    return svg(W, H + 10, b, "draw 배분")


def fig_cosine():
    W, H = 760, 400
    L, R, T, B = 60, 600, 46, 340
    kmin, kmax = 64, 131072
    b = t(20, 28, "역전파 기울기와의 코사인 (1억 토큰 체크포인트, 피팅 곡선)", 15, INK, weight="bold")
    def X(k): return L + (R - L) * (math.log2(k) - 6) / (math.log2(kmax) - 6)
    def Y(c): return B - (B - T) * c
    for c in [0, 0.25, 0.5, 0.75, 1.0]:
        b += f'<line x1="{L}" y1="{Y(c)}" x2="{R}" y2="{Y(c)}" stroke="{GRID}"/>' + t(L - 6, Y(c) + 4, f"{c:.2f}", 10, SUB, "end")
    for k, lab in [(64, "64"), (1024, "1k"), (16384, "16k"), (131072, "128k")]:
        b += f'<line x1="{X(k):.1f}" y1="{T}" x2="{X(k):.1f}" y2="{B}" stroke="{GRID}"/>' + t(X(k), B + 16, lab, 10.5, SUB, "middle")
    b += f'<line x1="{X(16384):.1f}" y1="{T}" x2="{X(16384):.1f}" y2="{B}" stroke="{INK}" stroke-dasharray="3 3"/>'
    b += t(X(16384) + 4, T + 12, "학습 실험 집단 16k", 10, SUB)
    b += t((L + R) / 2, B + 34, "집단 K (로그 눈금)", 11, SUB, "middle")
    ends = []
    for (name, cm, c), col in zip(COS, COS_COLORS):
        pts = []
        for i in range(61):
            k = 2 ** (6 + 11 * i / 60)
            pts.append(f"{X(k):.1f},{Y(cm / math.sqrt(1 + c / k)):.1f}")
        b += f'<polyline points="{" ".join(pts)}" fill="none" stroke="{col}" stroke-width="2"/>'
        ends.append([Y(cm / math.sqrt(1 + c / kmax)), name, col, cm / math.sqrt(1 + c / kmax)])
    ends.sort()
    last = -99
    for e in ends:
        y = max(e[0], last + 13)
        last = y
        b += t(R + 8, y + 4, f"{e[1]} {e[3]:.2f}", 10.5, e[2])
    b += t(L, B + 52, "cos(K) = c_max / √(1 + c/K), 부록 Table 8 의 층 종류별 c_max·c 로 그림. 오른쪽 숫자는 K = 128k 값", 10.5, SUB)
    return svg(W, H + 10, b, "코사인 곡선")


def fig_loss():
    W, H = 760, 572
    b = t(20, 28, "테스트 손실과 집단 크기 (부록 Table 1, 시드 3개 평균, 낮을수록 좋음)", 15, INK, weight="bold")
    pw, ph = 330, 210
    for p in range(4):
        ox = 50 + (p % 2) * 370
        oy = 60 + (p // 2) * 250
        vals = DUST[p] + EGG[p] + [BP[p]]
        lo, hi = min(vals), max(vals)
        pad = (hi - lo) * 0.08
        lo, hi = lo - pad, hi + pad
        def X(i): return ox + pw * i / 4
        def Y(v): return oy + ph - ph * (v - lo) / (hi - lo)
        b += t(ox, oy - 8, BUDGETS[p], 13, INK, weight="bold")
        b += f'<rect x="{ox}" y="{oy}" width="{pw}" height="{ph}" fill="none" stroke="{GRID}"/>'
        for k in range(5):
            b += t(X(k), oy + ph + 14, ["64", "256", "1k", "4k", "16k"][k], 10, SUB, "middle")
        for v in [lo + (hi - lo) * f for f in (0.1, 0.5, 0.9)]:
            b += t(ox - 4, Y(v) + 4, f"{v:.2f}", 9.5, SUB, "end")
        b += f'<line x1="{ox}" y1="{Y(BP[p]):.1f}" x2="{ox+pw}" y2="{Y(BP[p]):.1f}" stroke="{BLACK}" stroke-dasharray="5 4" stroke-width="1.5"/>'
        b += t(ox + 4, Y(BP[p]) - 5, f"역전파 {BP[p]:.3f}", 10.5, BLACK)
        for series, col in [(EGG[p], GRAY), (DUST[p], VIOLET)]:
            pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(series))
            b += f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2"/>'
            for i, v in enumerate(series):
                win = col == VIOLET and v < BP[p]
                b += f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="{4.5 if win else 3.2}" fill="{"#16a34a" if win else col}"/>'
        b += t(X(4) - 6, Y(DUST[p][4]) + (16 if DUST[p][4] < BP[p] else -9), f"Dust 16k {DUST[p][4]:.3f}", 10.5, VIOLET, "end")
    ly = 556
    b += f'<line x1="50" y1="{ly-4}" x2="72" y2="{ly-4}" stroke="{VIOLET}" stroke-width="2"/>' + t(78, ly, "Dust", 11, INK)
    b += f'<line x1="130" y1="{ly-4}" x2="152" y2="{ly-4}" stroke="{GRAY}" stroke-width="2"/>' + t(158, ly, "EGGROLL", 11, INK)
    b += f'<line x1="240" y1="{ly-4}" x2="262" y2="{ly-4}" stroke="{BLACK}" stroke-dasharray="5 4"/>' + t(268, ly, "역전파", 11, INK)
    b += f'<circle cx="340" cy="{ly-4}" r="4.5" fill="#16a34a"/>' + t(350, ly, "역전파보다 낮은 Dust 실측", 11, INK)
    b += t(530, ly, "패널마다 세로 범위가 다름", 10.5, SUB)
    return svg(W, H + 10, b, "손실 비교")


FIGS = [("compare", fig_compare), ("steps", fig_steps), ("population", fig_population),
        ("alloc", fig_alloc), ("cosine", fig_cosine), ("loss", fig_loss)]


def main(argv):
    global SHOW_NOTE
    if argv[:1] == ["--inline"]:
        out, prefix, SHOW_NOTE = Path(argv[1]), "", False
    else:
        out, prefix = OUT, "dust_"
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in FIGS:
        (out / f"{prefix}{name}.svg").write_text(fn(), "utf-8")
        print("wrote", out / f"{prefix}{name}.svg")


if __name__ == "__main__":
    main(sys.argv[1:])
