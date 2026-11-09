"""OntoSmith 노트의 그림을 생성한다.

    python scripts/figures/ontosmith_figures.py              readings/repos/assets/ontosmith_*.svg 를 다시 쓴다
    python scripts/figures/ontosmith_figures.py --inline DIR  HTML 보고서용 (접두어·생성 메모 없이 DIR/<이름>.svg)

SVG 는 손으로 고치지 않는다. 데이터는 레포 커밋 9bc8981 의 docs/whitepaper.md §5,
benchmarks/appendix/raw_scores.json · sample_manifest.json, examples/demo/replay_scores.py 출력에서 옮긴
아래 상수만 쓴다.
"""
import sys
from pathlib import Path

FONT = "-apple-system,'Apple SD Gothic Neo','Noto Sans KR',sans-serif"
INK, SUB, GRID = "#1f2328", "#59636e", "#d8dee4"
GRAY, BLUE, ORANGE, GREEN, YELLOW, RED, PURPLE = (
    "#8b949e", "#2563eb", "#ea580c", "#16a34a", "#ca8a04", "#dc2626", "#9333ea")
SHOW_NOTE = True
NOTE = "생성: python scripts/figures/ontosmith_figures.py · 손으로 고치지 말 것"
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "readings" / "repos" / "assets"

# FULL-500 (Re-DocRED dev 500문서) 모집단
GOLD_TOTAL = 17284
TIER1 = 3902
MATCHED_EXTRACTED = 2233   # Tier-1 중 gold 정확 일치
UNMATCHED_POOL = 1669      # Tier-1 중 gold 정확 일치 없음
MATCHED_GOLD = 1838
RAW_P, TN_P = 0.5723, 0.5736
MATCHED_SUPPORT = 0.884    # 재검토한 matched stratum 의 지지율 (§5.3)

# 판정 계약 두 개: (이름, p_A, p_A CI, g_B, g_B CI, 보정 정밀도, CI, ceiling-adjusted recall, CI, controls)
CONTRACTS = [
    ("원래 프로토콜", 0.945, (0.9161, 0.9643), 0.841, (0.8008, 0.8743),
     0.9765, (0.9641, 0.9847), 0.1265, (0.1216, 0.1328), "40/40"),
    ("ruled 계약 (출하 수치)", 0.7919, (0.7472, 0.8305), 0.5244, (0.4754, 0.573),
     0.9110, (0.8919, 0.9275), 0.2028, (0.1856, 0.2237), "37/40"),
]

# gold fate (FULL-500 census)
FATES = {"matched": 1838, "emission_blocked": 1759, "none_judged": 3677,
         "never_enumerated": 6340, "roster_missed": 3670}

# 운영점: (이름, 후보 쌍, 판정 비용 USD, recall, recall 이 측정인가)
PAIR_RATE = 0.0006683
OPERATING = [
    ("sentence window (측정한 빌드)", 26938, 18.0029, 0.2028, True),
    ("max_dist=1 + REC (출하 기본)", 52463, 35.06, 0.1884, False),
]


def corrected_precision(p_a, matched_support=1.0):
    return (MATCHED_EXTRACTED * matched_support + p_a * UNMATCHED_POOL) / TIER1


def adjusted_recall(g_b):
    return MATCHED_GOLD / (GOLD_TOTAL * g_b)


def svg(w, h, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'role="img" aria-label="{title}" style="max-width:100%;height:auto;font-family:{FONT}">'
            f'<title>{title}</title><rect width="{w}" height="{h}" fill="#ffffff"/>{body}'
            + (f'<text x="{w-8}" y="{h-6}" font-size="9" fill="{GRAY}" text-anchor="end">{NOTE}</text>' if SHOW_NOTE else "")
            + '</svg>')


def t(x, y, s, size=12, fill=INK, anchor="start", weight="normal"):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}">{s}</text>')


def fig_adjudication():
    w, L, R, T, rh = 800, 250, 760, 104, 56
    rows = [("p_A: 불일치 Tier-1 지지율", 1, 2), ("g_B: gold 가 쓰인 그대로 지지되는 비율", 3, 4),
            ("보정 정밀도", 5, 6), ("ceiling-adjusted recall", 7, 8)]
    h = T + rh * len(rows) + 70

    def xpos(v):
        return L + (R - L) * v

    out = t(20, 28, "같은 빌드, 두 판정 계약: 정밀도는 내려가고 recall 은 올라갔다", 15, INK, weight="bold")
    out += t(20, 46, "Re-DocRED dev 500문서 · 같은 740개 표본을 seam ruling 전후로 판정 · 점 = 추정치, 선 = 95% 구간", 12, SUB)
    lx = 20
    for (name, *_), c in zip(CONTRACTS, (GRAY, BLUE)):
        out += f'<circle cx="{lx+6}" cy="66" r="6" fill="{c}"/>' + t(lx + 18, 70, name, 11, SUB)
        lx += 60 + len(name) * 10
    for v in (0, 0.2, 0.4, 0.6, 0.8, 1.0):
        out += f'<line x1="{xpos(v):.1f}" y1="{T-14}" x2="{xpos(v):.1f}" y2="{T+rh*len(rows)-20}" stroke="{GRID}"/>'
        out += t(xpos(v), T + rh * len(rows) - 4, f"{v:.1f}", 10, SUB, "middle")
    for i, (label, vi, ci) in enumerate(rows):
        y = T + rh * i
        out += t(L - 12, y + 10, label, 11, INK, "end")
        for j, (c_row, c) in enumerate(zip(CONTRACTS, (GRAY, BLUE))):
            yy = y + j * 16
            v, (lo, hi) = c_row[vi], c_row[ci]
            out += f'<line x1="{xpos(lo):.1f}" y1="{yy}" x2="{xpos(hi):.1f}" y2="{yy}" stroke="{c}" stroke-width="3"/>'
            out += f'<circle cx="{xpos(v):.1f}" cy="{yy}" r="5" fill="{c}"/>'
            out += t(xpos(hi) + 8, yy + 4, f"{v:.4f}", 10, c, "start", "bold")
    y = T + rh * len(rows) + 22
    out += t(20, y, "g_B 가 0.841 → 0.5244 로 내려가면 recall 분모(gold × g_B)가 줄어 ceiling-adjusted recall 이 오른다", 11, SUB)
    out += t(20, y + 18, f"blind control: 원래 {CONTRACTS[0][-1]} → ruled {CONTRACTS[1][-1]} supported. 출처 whitepaper §5.4, replay_scores.py", 11, SUB)
    return svg(w, h, out, "OntoSmith 판정 계약 두 개의 추정치 비교")


def fig_precision():
    w, h = 800, 400
    L, R = 40, 760
    out = t(20, 28, "보정 정밀도 0.911 의 구성: 정확 일치 2,233 + 불일치 1,669 × p_A", 15, INK, weight="bold")
    out += t(20, 46, "Tier-1 3,902개 (sentence-window 빌드). 불일치 풀은 표본 300개로 지지율을 추정해 곱한다", 12, SUB)
    sup = UNMATCHED_POOL * CONTRACTS[1][1]
    parts = [(MATCHED_EXTRACTED, GREEN, f"gold 정확 일치 {MATCHED_EXTRACTED:,}"),
             (sup, BLUE, f"불일치 중 지지 추정 {sup:,.0f}"),
             (UNMATCHED_POOL - sup, RED, f"불일치 중 미지지 추정 {UNMATCHED_POOL - sup:,.0f}")]
    x, y0 = L, 70
    for v, c, lab in parts:
        bw = (R - L) * v / TIER1
        out += f'<rect x="{x:.1f}" y="{y0}" width="{bw:.1f}" height="34" fill="{c}" opacity="0.85"/>'
        out += t(x + bw / 2, y0 + 22, f"{v:,.0f}", 11, "#ffffff", "middle", "bold")
        x += bw
    lx = L
    for v, c, lab in parts:
        out += f'<rect x="{lx}" y="{y0+44}" width="12" height="12" rx="2" fill="{c}"/>' + t(lx + 18, y0 + 54, lab, 11, SUB)
        lx += 40 + len(lab) * 9
    # 사다리
    ladder = [("raw (frozen scorer)", RAW_P, GRAY),
              ("TN-normalized", TN_P, GRAY),
              (f"보정, 정확 일치를 {MATCHED_SUPPORT} 로 가격", corrected_precision(CONTRACTS[1][1], MATCHED_SUPPORT), YELLOW),
              ("보정 (출하 수치)", corrected_precision(CONTRACTS[1][1]), BLUE)]
    T2, rh, LL = 180, 38, 300

    def xpos(v):
        return LL + (R - LL) * v

    out += t(20, T2 - 14, "같은 Tier-1 집합을 네 가지로 읽은 정밀도", 12, INK, weight="bold")
    for i, (lab, v, c) in enumerate(ladder):
        y = T2 + rh * i
        out += t(LL - 12, y + 16, lab, 11, INK, "end")
        out += f'<rect x="{LL}" y="{y+4}" width="{xpos(v)-LL:.1f}" height="18" rx="2" fill="{c}" opacity="0.85"/>'
        out += t(xpos(v) + 6, y + 18, f"{v:.4f}", 11, INK, "start", "bold")
    out += t(20, h - 26, "정확 일치를 1.0 으로 치는 것은 추정식의 약속이다. 재검토한 일치 표본의 지지율(0.884)로 바꾸면 ≈ 0.845", 11, SUB)
    return svg(w, h, out, "OntoSmith 보정 정밀도 분해")


def fig_operating():
    w, h = 800, 330
    out = t(20, 28, "운영점 두 개: 출하 기본은 판정 쌍과 비용이 1.95배이고 recall 은 서로 다른 계약의 값이다", 15, INK, weight="bold")
    out += t(20, 46, f"Re-DocRED dev 500문서 · 판정 비용은 쌍당 ${PAIR_RATE} 실현 단가 기준", 12, SUB)
    panels = [("후보 쌍 수", 1, 60000, lambda v: f"{v:,}"),
              ("판정 비용 (USD)", 2, 40, lambda v: f"${v:.2f}"),
              ("ceiling-adjusted recall", 3, 0.25, lambda v: f"{v:.4f}")]
    B, T, pw = 250, 80, 240
    for k, (title, idx, vmax, fmt) in enumerate(panels):
        x0 = 30 + k * (pw + 20)
        out += t(x0 + pw / 2, T - 6, title, 12, INK, "middle", "bold")
        out += f'<line x1="{x0}" y1="{B}" x2="{x0+pw}" y2="{B}" stroke="{GRID}"/>'
        for j, row in enumerate(OPERATING):
            v = row[idx]
            bh = (B - T - 30) * v / vmax
            bx = x0 + 30 + j * 100
            c = BLUE if j == 0 else ORANGE
            dashed = idx in (2, 3) and j == 1
            out += (f'<rect x="{bx}" y="{B-bh:.1f}" width="70" height="{bh:.1f}" rx="2" fill="{c}" '
                    f'opacity="{0.45 if dashed else 0.9}"' + (f' stroke="{c}" stroke-dasharray="4 3"' if dashed else "") + '/>')
            out += t(bx + 35, B - bh - 6, fmt(v), 11, INK, "middle", "bold")
            out += t(bx + 35, B + 16, "측정 빌드" if j == 0 else "출하 기본", 10, SUB, "middle")
    out += t(20, h - 44, "파랑 = sentence window (측정한 빌드), 주황 = max_dist=1 + REC (출하 기본). 흐린 막대 = 실측이 아닌 추정", 11, SUB)
    out += t(20, h - 26, "출하 기본의 recall 0.1884 는 채택 당시 판정 계약 기준 추정이다. ruled 재가격은 측정 빌드 쪽만 0.2028 로 올렸다", 11, SUB)
    return svg(w, h, out, "OntoSmith 운영점 비교")


FIGS = [("adjudication", fig_adjudication), ("precision", fig_precision), ("operating", fig_operating)]


def main(argv):
    global SHOW_NOTE
    if argv[:1] == ["--inline"]:
        out, prefix, SHOW_NOTE = Path(argv[1]), "", False
    else:
        out, prefix = OUT, "ontosmith_"
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in FIGS:
        (out / f"{prefix}{name}.svg").write_text(fn(), "utf-8")
        print("wrote", out / f"{prefix}{name}.svg")


if __name__ == "__main__":
    main(sys.argv[1:])
