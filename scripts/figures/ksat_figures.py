"""Sionic AI 3B 수능 수학 GRPO (2026-11-20) 노트의 그림을 생성한다.

    python scripts/figures/ksat_figures.py              readings/blogs/assets/ksat_*.svg 를 다시 쓴다
    python scripts/figures/ksat_figures.py --inline DIR  HTML 보고서용 (접두어, 생성 메모 없이 DIR/<이름>.svg)

SVG 는 손으로 고치지 않는다. 수치는 HF 모델 카드(sha 4066e88) 표에서 옮긴 아래 데이터 절만 쓴다.
"""
import sys
from pathlib import Path

FONT = "-apple-system,'Apple SD Gothic Neo','Noto Sans KR',sans-serif"
INK, SUB, GRID = "#1f2328", "#59636e", "#d8dee4"
GRAY, BLUE, LIGHT, RED, GREEN = "#8b949e", "#2563eb", "#93c5fd", "#dc2626", "#16a34a"
SHOW_NOTE = True
NOTE = "생성: python scripts/figures/ksat_figures.py, 손으로 고치지 말 것"
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "readings" / "blogs" / "assets"

# 모델 카드 결과 표: 2027학년도 수능 수학, (16회 평균, 다수결)
SUBJECTS = ["확률과 통계", "미적분", "기하"]
SCORES = [
    ("원본 (Nanbeige4.2-3B)", GRAY, [(94.0, 96), (93.5, 96), (92.6, 96)]),
    ("1차 RL step 40", BLUE, [(96.8, 100), (96.8, 100), (95.5, 100)]),
    ("2차 RL step 10 (공개 모델)", LIGHT, [(95.5, 96), (95.2, 96), (95.2, 96)]),
]
SINGLE_PERFECT = [4, 5, 2]  # 1차 RL, 16회 중 단일 회차 100점
RUNS = 16

# 모델 카드 학습 과정 표
STAGE1 = [("기출 22~26", 181), ("4점 17~21", 146), ("검증 합성 변형", 356)]
STAGE1_TOTAL = 683
STAGE2 = [("기출", 317), ("어려운 합성 변형", 562), ("디딤돌", 131), ("표기 변형", 25),
          ("용어", 43), ("힌트", 4)]
STAGE2_UNIQUE, STAGE2_ROWS, STAGE2_DOUBLED = 1078, 1145, 67
STAGE_COLORS = [BLUE, LIGHT, "#a78bfa", "#fbbf24", "#34d399", RED]


def svg(w, h, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'role="img" aria-label="{title}" style="max-width:100%;height:auto;font-family:{FONT}">'
            f'<title>{title}</title><rect width="{w}" height="{h}" fill="#ffffff"/>{body}'
            + (f'<text x="{w-8}" y="{h-6}" font-size="9" fill="{GRAY}" text-anchor="end">{NOTE}</text>' if SHOW_NOTE else "")
            + '</svg>')


def t(x, y, s, size=12, fill=INK, anchor="start", weight="normal"):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}">{s}</text>')


def fig_scores():
    w, h, L, R, T, B = 720, 420, 60, 700, 96, 320
    lo, hi = 90, 100
    out = t(20, 28, "평균은 세 모델 모두 오르지만 다수결 만점은 1차 RL 체크포인트뿐이다", 15, INK, weight="bold")
    out += t(20, 46, "2027학년도 수능 수학, 문항당 16회 생성. 막대 = 16회 평균, 점 = 문항별 다수결 점수 (30문항 100점 만점)", 12, SUB)
    lx = 20
    for name, color, _ in SCORES:
        out += f'<rect x="{lx}" y="62" width="12" height="12" rx="2" fill="{color}"/>'
        out += t(lx + 18, 72, name, 12)
        lx += 28 + len(name) * 9
    out += f'<circle cx="{lx + 6}" cy="68" r="5" fill="#ffffff" stroke="{INK}" stroke-width="2"/>'
    out += t(lx + 16, 72, "다수결", 12)
    for v in range(lo, hi + 1, 2):
        y = B - (B - T) * (v - lo) / (hi - lo)
        out += f'<line x1="{L}" y1="{y}" x2="{R}" y2="{y}" stroke="{GRID}"/>'
        out += t(L - 8, y + 4, v, 11, SUB, "end")
    out += t(L - 8, B + 16, "(90부터)", 10, SUB, "end")
    slot = (R - L) / len(SUBJECTS)
    bw = slot * 0.22
    for i, subj in enumerate(SUBJECTS):
        x0 = L + slot * i + slot * 0.12
        for j, (_, color, vals) in enumerate(SCORES):
            mean, maj = vals[i]
            x = x0 + j * (bw + 6)
            y = B - (B - T) * (mean - lo) / (hi - lo)
            out += f'<rect x="{x}" y="{y}" width="{bw}" height="{B - y}" rx="3" fill="{color}"/>'
            out += t(x + bw / 2, B - 8, f"{mean:.1f}", 11, "#ffffff" if color != LIGHT else INK, "middle", "bold")
            ym = B - (B - T) * (maj - lo) / (hi - lo)
            out += f'<circle cx="{x + bw / 2}" cy="{ym}" r="6" fill="#ffffff" stroke="{INK}" stroke-width="2"/>'
            out += t(x + bw / 2, ym - 10, maj, 11, INK, "middle", "bold")
        cx = L + slot * (i + 0.5)
        out += t(cx, B + 22, subj, 13, INK, "middle", "bold")
        out += t(cx, B + 40, f"1차 RL 단일 회차 100점: {SINGLE_PERFECT[i]}/{RUNS}회", 11, BLUE, "middle")
    out += t(20, B + 70, "2차 RL step 10은 공통 21번(4점)을 16회 모두 틀려 다수결이 96점에 머문다", 12, RED)
    return svg(w, h, out, "2027 수능 수학 체크포인트별 평균과 다수결 점수")


def stack(x0, y, width, items, total, bh=34):
    out, x = "", x0
    for (name, n), color in zip(items, STAGE_COLORS):
        ww = width * n / total
        out += f'<rect x="{x}" y="{y}" width="{ww}" height="{bh}" fill="{color}" stroke="#ffffff"/>'
        if ww > 46:
            out += t(x + ww / 2, y + 15, name, 11, INK if color in (LIGHT, "#fbbf24", "#34d399") else "#ffffff", "middle")
            out += t(x + ww / 2, y + 29, f"{n:,}", 11, INK if color in (LIGHT, "#fbbf24", "#34d399") else "#ffffff", "middle", "bold")
        x += ww
    return out, x


def fig_data():
    w, h, L, R = 760, 330, 170, 660
    scale = (R - L) / 1100
    out = t(20, 28, "2차 RL은 1차 모델 기준으로 다시 고른 문항으로 세트를 고유 문항 기준 약 1.6배 키웠다", 15, INK, weight="bold")
    out += t(20, 46, "모델 카드 학습 과정 표. 막대 길이는 문항 수 (같은 축)", 12, SUB)
    y1, y2 = 84, 170
    out += t(L - 10, y1 + 15, "1차 RL", 13, INK, "end", "bold")
    out += t(L - 10, y1 + 30, "원본 정답률로 선별", 11, SUB, "end")
    body, end1 = stack(L, y1, STAGE1_TOTAL * scale, STAGE1, STAGE1_TOTAL)
    out += body + t(end1 + 8, y1 + 22, f"{STAGE1_TOTAL:,}문항", 12, INK, weight="bold")
    s2sum = sum(n for _, n in STAGE2)
    out += t(L - 10, y2 + 15, "2차 RL", 13, INK, "end", "bold")
    out += t(L - 10, y2 + 30, "1차 step 40 정답률로 재선별", 11, SUB, "end")
    body, end2 = stack(L, y2, s2sum * scale, STAGE2, s2sum)
    out += body
    out += t(end2 + 8, y2 + 14, f"구성 합 {s2sum:,}", 12, INK, weight="bold")
    out += t(end2 + 8, y2 + 30, f"고유 {STAGE2_UNIQUE:,}", 12, RED, weight="bold")
    out += t(L, y2 + 56, f"정답률 25% 이하 {STAGE2_DOUBLED}문항을 2배로 넣어 학습 행은 {STAGE2_UNIQUE:,} + {STAGE2_DOUBLED} = {STAGE2_ROWS:,}행", 12, INK)
    out += t(L, y2 + 76, f"구성 항목을 더하면 {s2sum:,}로 고유 문항 수 {STAGE2_UNIQUE:,}와 {s2sum - STAGE2_UNIQUE}만큼 다르다. 모델 카드는 차이를 설명하지 않는다", 12, RED)
    lx = L
    for (name, n), color in list(zip(STAGE2, STAGE_COLORS))[3:]:
        out += f'<rect x="{lx}" y="{y2 + 88}" width="12" height="12" rx="2" fill="{color}"/>'
        out += t(lx + 18, y2 + 98, f"{name} {n}", 12)
        lx += 120
    return svg(w, h, out, "1차와 2차 RL 학습 세트 구성")


FIGS = [("scores", fig_scores), ("data", fig_data)]


def main(argv):
    global SHOW_NOTE
    if argv[:1] == ["--inline"]:
        out, prefix, SHOW_NOTE = Path(argv[1]), "", False
    else:
        out, prefix = OUT, "ksat_"
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in FIGS:
        (out / f"{prefix}{name}.svg").write_text(fn(), "utf-8")
        print("wrote", out / f"{prefix}{name}.svg")


if __name__ == "__main__":
    main(sys.argv[1:])
