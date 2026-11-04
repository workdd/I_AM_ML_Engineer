"""edgextract 노트의 그림을 생성한다.

    python scripts/figures/edgex_figures.py              readings/repos/assets/edgex_*.svg 를 다시 쓴다
    python scripts/figures/edgex_figures.py --inline DIR  HTML 보고서용 (접두어·생성 메모 없이 DIR/<이름>.svg)

SVG 는 손으로 고치지 않는다. 데이터는 레포 커밋 3f08b1d 의 specs/0001-implementation/14-evaluation.md 와
src/edgextract/gate.py 에서 옮긴 아래 상수만 쓴다.
"""
import sys
from pathlib import Path

FONT = "-apple-system,'Apple SD Gothic Neo','Noto Sans KR',sans-serif"
INK, SUB, GRID = "#1f2328", "#59636e", "#d8dee4"
GRAY, BLUE, ORANGE, GREEN, YELLOW, RED, PURPLE = (
    "#8b949e", "#2563eb", "#ea580c", "#16a34a", "#ca8a04", "#dc2626", "#9333ea")
SHOW_NOTE = True
NOTE = "생성: python scripts/figures/edgex_figures.py · 손으로 고치지 말 것"
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "readings" / "repos" / "assets"

# CoNLL04 test 정답 개수
GOLD = {"names": 1079, "links": 422}
# (실행, names (맞음, 추가), links (맞음, 추가), 원문 F1 (names, links), 파생 여부)
# final · Mistral 은 원문 개수. first 는 원문 precision/recall 에서 역산 (names 맞음 634 · links 추가 177 은 원문)
COUNTS = [
    ("Tev1 final (출시 설정)", (683, 218), (154, 218), (0.690, 0.388), False),
    ("Tev1 first", (634, 212), (155, 177), (0.658, 0.411), True),
    ("Mistral Small JSON", (754, 514), (183, 504), (0.643, 0.330), False),
]

# span finder 를 small → larger 로 바꾼 효과: (구간, links F1 small, links F1 larger, 반전 small, 반전 larger)
TRANSFER = [
    ("dev", 0.427, 0.463, 4, 11),
    ("test", 0.411, 0.388, 9, 16),
]

# gate.py GateConfig 기본값 (noul)
NOUL_YES, NOUL_NO = 0.80, 0.20
# 원문에 적힌 실측 확률: (문장, P(yes), 정답과 맞는가)
PROBES = [
    ("EdgeQuake depends on PostgreSQL", 1.00, True),
    ("EdgeQuake uses PostgreSQL", 0.96, True),
    ("Ollama uses Nimble", 0.94, True),
    ("Jane Doe part of Acme Inc", 0.94, False),
    ("Nimble uses Ollama (방향 반대)", 0.85, False),
    ("Northwind: founded", 0.998, True),
    ("Northwind: acquired", 0.0038, True),
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


def f1(right, extra, gold):
    p, r = right / (right + extra), right / gold
    return 2 * p * r / (p + r)


def fig_error_counts():
    w, L, T, rh = 800, 190, 92, 30
    x0, x1 = L, w - 40
    xmax = 1600
    blocks = [("names", "이름 (정답 1,079개)"), ("links", "링크 (정답 422개)")]
    h = T + len(blocks) * (rh * len(COUNTS) + 46) + 70
    out = t(20, 28, "CoNLL04 test: F1 뒤의 맞음 · 추가 · 놓침 개수", 15, INK, weight="bold")
    out += t(20, 46, "288문장, exact span 채점. 막대 길이 = 개수. 놓침 = 정답 수에서 맞음을 뺀 값", 12, SUB)
    lx = 20
    for label, c in [("맞음", GREEN), ("추가(정답에 없음)", RED), ("놓침", GRAY)]:
        out += f'<rect x="{lx}" y="{60}" width="12" height="12" rx="2" fill="{c}"/>' + t(lx + 18, 70, label, 11, SUB)
        lx += 40 + len(label) * 9
    y = T
    for key, label in blocks:
        out += t(20, y + 4, label, 12, INK, weight="bold")
        y += 14
        for name, names, links, (fn, fl), derived in COUNTS:
            right, extra = names if key == "names" else links
            miss = GOLD[key] - right
            out += t(20, y + 18, name + (" *" if derived else ""), 11, INK)
            x = x0
            for v, c in [(right, GREEN), (extra, RED), (miss, GRAY)]:
                bw = (x1 - x0) * v / xmax
                out += f'<rect x="{x:.1f}" y="{y+4}" width="{bw:.1f}" height="{rh-10}" fill="{c}" opacity="0.85"/>'
                if bw > 28:
                    out += t(x + bw / 2, y + 19, str(v), 10, "#ffffff", "middle", "bold")
                x += bw
            out += t(x + 6, y + 19, f"F1 {fn if key == 'names' else fl:.3f}", 10, SUB)
            y += rh
        y += 32
    out += t(20, y + 4, "* first 는 원문 precision · recall 에서 역산한 개수 (이름 맞음 634 · 링크 추가 177 은 원문 값)", 11, SUB)
    out += t(20, y + 22, "Mistral 은 링크를 가장 많이 맞히지만 추가 링크가 504개로 Tev1 final(218)의 2배가 넘는다", 11, SUB)
    return svg(w, h, out, "edgextract CoNLL04 개수 분해")


def fig_dev_test():
    w, h = 800, 360
    L, R, T, B = 120, 420, 80, 290
    out = t(20, 28, "span finder 교체 효과는 dev 에서 test 로 옮겨 가지 않았다", 15, INK, weight="bold")
    out += t(20, 46, "small → larger GLiNER, cutoff 0.30. 왼쪽 = 링크 F1, 오른쪽 = 방향이 뒤집힌 링크 수", 12, SUB)

    def ypos(v, lo, hi):
        return B - (B - T) * (v - lo) / (hi - lo)

    # 왼쪽: F1 기울기 그림
    lo, hi = 0.37, 0.48
    for v in (0.38, 0.40, 0.42, 0.44, 0.46):
        yy = ypos(v, lo, hi)
        out += f'<line x1="{L-10}" y1="{yy:.1f}" x2="{R}" y2="{yy:.1f}" stroke="{GRID}"/>' + t(L - 16, yy + 4, f"{v:.2f}", 10, SUB, "end")
    out += t(L + 20, B + 22, "small", 11, SUB, "middle") + t(R - 40, B + 22, "larger", 11, SUB, "middle")
    for name, a, b, _, _ in TRANSFER:
        c = BLUE if name == "dev" else ORANGE
        ya, yb = ypos(a, lo, hi), ypos(b, lo, hi)
        out += f'<line x1="{L+20}" y1="{ya:.1f}" x2="{R-40}" y2="{yb:.1f}" stroke="{c}" stroke-width="3"/>'
        out += f'<circle cx="{L+20}" cy="{ya:.1f}" r="5" fill="{c}"/><circle cx="{R-40}" cy="{yb:.1f}" r="5" fill="{c}"/>'
        out += t(L + 10, ya + 4, f"{a:.3f}", 10, c, "end") + t(R - 30, yb + 4, f"{b:.3f} ({name}, {b-a:+.3f})", 10, c)
    # 오른쪽: 반전 개수 막대
    px, bw = 560, 34
    for i, (name, _, _, ra, rb) in enumerate(TRANSFER):
        c = BLUE if name == "dev" else ORANGE
        x = px + i * 110
        for j, v in enumerate((ra, rb)):
            bh = (B - T) * v / 20
            xx = x + j * (bw + 6)
            out += f'<rect x="{xx}" y="{B-bh:.1f}" width="{bw}" height="{bh:.1f}" rx="2" fill="{c}" opacity="{0.45 if j == 0 else 0.9}"/>'
            out += t(xx + bw / 2, B - bh - 6, str(v), 11, INK, "middle", "bold")
            out += t(xx + bw / 2, B + 14, "small" if j == 0 else "larger", 9, SUB, "middle")
        out += t(x + bw + 3, B + 30, name, 11, c, "middle", "bold")
    out += f'<line x1="{px-10}" y1="{B}" x2="{w-20}" y2="{B}" stroke="{GRID}"/>'
    out += t(20, h - 26, "dev 에서는 링크 F1 이 올랐지만 test 에서는 내려갔고, 두 구간 모두 방향 반전이 7개씩 늘었다", 11, SUB)
    return svg(w, h, out, "edgextract span finder 교체의 dev와 test 효과")


def fig_gate():
    w, L, R, T, rh = 800, 280, 760, 92, 30
    h = T + rh * len(PROBES) + 90

    def xpos(p):
        return L + (R - L) * p

    out = t(20, 28, "기본 noul cutoff 위에 놓은 실측 확률: 틀린 링크도 keep 구간에 들어간다", 15, INK, weight="bold")
    out += t(20, 46, f"GateConfig 기본값 (fitted=False): P(yes) ≥ {NOUL_YES:.2f} keep · ≤ {NOUL_NO:.2f} drop · 사이 review", 12, SUB)
    bands = [(0, NOUL_NO, RED, "drop"), (NOUL_NO, NOUL_YES, YELLOW, "review"), (NOUL_YES, 1.0, GREEN, "keep")]
    top, bot = T - 20, T + rh * len(PROBES)
    for a, b, c, lab in bands:
        out += f'<rect x="{xpos(a):.1f}" y="{top}" width="{xpos(b)-xpos(a):.1f}" height="{bot-top}" fill="{c}" opacity="0.10"/>'
        out += t((xpos(a) + xpos(b)) / 2, top - 6, lab, 11, c, "middle", "bold")
    for p in (0, 0.2, 0.4, 0.6, 0.8, 1.0):
        out += t(xpos(p), bot + 16, f"{p:.1f}", 10, SUB, "middle")
    for i, (s, p, ok) in enumerate(PROBES):
        y = T + rh * i + 12
        out += t(L - 12, y + 4, s, 11, INK if ok else RED, "end", "normal" if ok else "bold")
        out += f'<line x1="{L}" y1="{y}" x2="{R}" y2="{y}" stroke="{GRID}"/>'
        c = GREEN if ok else RED
        out += f'<circle cx="{xpos(p):.1f}" cy="{y}" r="6" fill="{c}"/>'
        lab = f"{p:.4f}" if p < 0.01 else f"{p:.3f}" if p not in (1.0,) and len(str(p)) > 4 else f"{p:.2f}"
        out += t(xpos(p) + (10 if p < 0.9 else -10), y - 8, lab, 10, SUB, "start" if p < 0.9 else "end")
    y = bot + 40
    out += t(20, y, "초록 = 정답과 맞는 판단, 빨강 = 정답에 없는 링크. 01_edgequake 노트와 Northwind 노트의 원문 기록값", 11, SUB)
    out += t(20, y + 18, "Nimble uses Ollama (0.85) 와 Ollama uses Nimble (0.94) 이 둘 다 keep. 확률이 높다고 방향이 맞는 것은 아니다", 11, SUB)
    return svg(w, h, out, "edgextract 기본 cutoff와 실측 확률")


FIGS = [("error_counts", fig_error_counts), ("dev_test", fig_dev_test), ("gate", fig_gate)]


def main(argv):
    global SHOW_NOTE
    if argv[:1] == ["--inline"]:
        out, prefix, SHOW_NOTE = Path(argv[1]), "", False
    else:
        out, prefix = OUT, "edgex_"
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in FIGS:
        (out / f"{prefix}{name}.svg").write_text(fn(), "utf-8")
        print("wrote", out / f"{prefix}{name}.svg")


if __name__ == "__main__":
    main(sys.argv[1:])
