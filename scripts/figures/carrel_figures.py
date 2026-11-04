"""Carrel 노트의 그림을 생성한다.

    python scripts/figures/carrel_figures.py              readings/repos/assets/carrel_*.svg 를 다시 쓴다
    python scripts/figures/carrel_figures.py --inline DIR  HTML 보고서용 (접두어·생성 메모 없이 DIR/<이름>.svg)

SVG 는 손으로 고치지 않는다. 데이터는 레포 커밋 7c449b7 의
docs/evaluations/capability-evaluations-0.4.0.md 표에서 옮긴 아래 상수만 쓴다.
"""
import sys
from pathlib import Path

FONT = "-apple-system,'Apple SD Gothic Neo','Noto Sans KR',sans-serif"
INK, SUB, GRID = "#1f2328", "#59636e", "#d8dee4"
GRAY, BLUE, ORANGE, GREEN, PURPLE = "#8b949e", "#2563eb", "#ea580c", "#16a34a", "#9333ea"
SHOW_NOTE = True
NOTE = "생성: python scripts/figures/carrel_figures.py · 손으로 고치지 말 것"
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "readings" / "repos" / "assets"

# Retrieval and answers: 조건 3개 (Full, Graph disabled, Text-only reranking)
CONDS = ["Full", "Graph off", "Text-only rerank"]
COND_COLORS = [BLUE, GRAY, ORANGE]
# (지표, 문항 수, 값 3개) — 백분율. 275문항 전체 + 원천 범위별 분할
ABLATION = [
    ("Evidence Hit@5", 275, (92.36, 93.45, 92.36)),
    ("Context coverage", 275, (92.13, 91.42, 92.24)),
    ("단일 문서 coverage", 237, (95.89, 95.78, None)),
    ("교차 문서 coverage", 38, (68.73, 64.21, None)),
    ("교차 문서 top10 완전", 38, (31.58, 36.84, None)),
]
# 같은 표의 MRR@10 (0~1)
MRR = (0.7979, 0.8129, 0.8182)

# Answer-level comparison: 조건 3개 (단일 검색, Skill follow-up, 원문 직접 제공)
ANSWER_CONDS = ["단일 검색", "Skill follow-up", "원문 직접 제공"]
ANSWER_COLORS = [GRAY, BLUE, GREEN]
# (그룹, 맞힌 수 3개, 문항 수)
COMPLETENESS = [
    ("실문서 단일", (29, 29, 29), 30),
    ("실문서 교차", (15, 19, 23), 25),
    ("합성", (16, 18, 20), 20),
    ("전체", (60, 66, 72), 75),
]
# 실문서 55문항 평균: 지연(s), P95(s), 총 토큰
COST = {"latency": (59.1, 71.0, 9.8), "p95": (100.8, 140.3, 17.1), "tokens": (96861, 131944, 12318)}

# Rebuild stability: Jaccard(%) — (집합, Base A 3개, Base E 3개). 시나리오 순서는 REBUILD_SCEN
REBUILD_SCEN = ["cached rebuild", "fresh (labels 고정)", "fresh (labels 재추출)"]
REBUILD = [
    ("엔티티", (100, 68.05, 44.02), (100, 59.66, 50.10)),
    ("관계", (100, 30.34, 11.15), (100, 25.68, 18.92)),
    ("fact", (100, 32.15, 22.97), (100, 46.42, 43.63)),
    ("fact 출처 청크", (100, 99.01, 97.04), (100, 100, 98.05)),
]
# 같은 30문항 × 6버전 Hit@5 (모두 90.00%)
REBUILD_HIT5 = 90.0


def svg(w, h, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'role="img" aria-label="{title}" style="max-width:100%;height:auto;font-family:{FONT}">'
            f'<title>{title}</title><rect width="{w}" height="{h}" fill="#ffffff"/>{body}'
            + (f'<text x="{w-8}" y="{h-6}" font-size="9" fill="{GRAY}" text-anchor="end">{NOTE}</text>' if SHOW_NOTE else "")
            + '</svg>')


def t(x, y, s, size=12, fill=INK, anchor="start", weight="normal"):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}">{s}</text>')


def pct(k, n):
    return 100.0 * k / n


def legend(x, y, names, colors):
    out = ""
    for n, c in zip(names, colors):
        out += f'<rect x="{x}" y="{y-10}" width="12" height="12" rx="2" fill="{c}"/>' + t(x + 18, y, n, 11, SUB)
        x += 34 + len(n) * 7
    return out


def fig_ablation():
    w, L, T, grp, bh = 800, 170, 92, 64, 16
    x0, x1 = L, w - 90  # 0% ~ 100%
    h = T + grp * len(ABLATION) + 70
    out = t(20, 28, "검색 ablation: 그래프는 교차 문서에서만 증거 폭을 늘린다", 15, INK, weight="bold")
    out += t(20, 46, "275문항, 같은 색인·모델·top_k=10. 단일/교차 분할 행은 원문이 Full 과 Graph off 만 보고", 12, SUB)
    out += legend(20, 70, CONDS, COND_COLORS)
    for k in range(0, 101, 20):
        x = x0 + (x1 - x0) * k / 100
        out += f'<line x1="{x}" y1="{T-6}" x2="{x}" y2="{T + grp*len(ABLATION) - 10}" stroke="{GRID}"/>'
        out += t(x, T + grp * len(ABLATION) + 6, f"{k}%", 10, SUB, "middle")
    for i, (name, n, vals) in enumerate(ABLATION):
        y = T + grp * i
        out += t(20, y + 18, name, 12, INK, weight="bold") + t(20, y + 34, f"{n}문항", 10, SUB)
        for j, v in enumerate(vals):
            if v is None:
                continue
            yy = y + j * (bh + 2)
            bw = (x1 - x0) * v / 100
            out += f'<rect x="{x0}" y="{yy}" width="{bw:.1f}" height="{bh}" rx="2" fill="{COND_COLORS[j]}" opacity="0.85"/>'
            out += t(x0 + bw + 6, yy + 12, f"{v:.2f}%", 10, SUB)
    y = T + grp * len(ABLATION) + 28
    out += t(20, y, f"MRR@10: Full {MRR[0]:.4f} · Graph off {MRR[1]:.4f} · Text-only {MRR[2]:.4f}. "
             "전체 순위 지표는 그래프·visual rerank 를 끈 쪽이 같거나 높다", 11, SUB)
    out += t(20, y + 18, "교차 문서 38문항: coverage 는 그래프로 +4.52pp, 상위 10개 안 완전 증거 비율은 -5.26pp", 11, SUB)
    return svg(w, h, out, "Carrel 검색 ablation 조건별 지표")


def fig_answer_cost():
    w, L, T, grp, bh = 800, 130, 92, 62, 15
    x0, x1 = L, 430
    h = T + grp * len(COMPLETENESS) + 80
    out = t(20, 28, "답변 완결률과 비용: Skill follow-up 의 이득은 교차 문서에 몰린다", 15, INK, weight="bold")
    out += t(20, 46, "required-point 완결률(모든 필수 항목을 치명 오류 없이 제시한 비율). 오른쪽은 실문서 55문항 평균 비용", 12, SUB)
    out += legend(20, 70, ANSWER_CONDS, ANSWER_COLORS)
    for i, (name, ks, n) in enumerate(COMPLETENESS):
        y = T + grp * i
        out += t(20, y + 18, name, 12, INK, weight="bold") + t(20, y + 34, f"{n}문항", 10, SUB)
        for j, k in enumerate(ks):
            v = pct(k, n)
            yy = y + j * (bh + 2)
            bw = (x1 - x0) * v / 100
            out += f'<rect x="{x0}" y="{yy}" width="{bw:.1f}" height="{bh}" rx="2" fill="{ANSWER_COLORS[j]}" opacity="0.85"/>'
            out += t(x0 + bw + 6, yy + 12, f"{k}/{n} ({v:.1f}%)", 10, SUB)
    # 비용 패널
    px = 560
    out += f'<line x1="{px-20}" y1="{T-8}" x2="{px-20}" y2="{T + grp*len(COMPLETENESS) - 10}" stroke="{GRID}"/>'
    rows = [("평균 지연", COST["latency"], "s", 150.0), ("P95 지연", COST["p95"], "s", 150.0),
            ("평균 총 토큰", COST["tokens"], "k", 140000.0)]
    for r, (label, vals, unit, vmax) in enumerate(rows):
        y = T + r * 74
        out += t(px, y + 4, label, 12, INK, weight="bold")
        for j, v in enumerate(vals):
            yy = y + 12 + j * 17
            bw = 150 * v / vmax
            out += f'<rect x="{px}" y="{yy}" width="{bw:.1f}" height="14" rx="2" fill="{ANSWER_COLORS[j]}" opacity="0.85"/>'
            s = f"{v/1000:.1f}k" if unit == "k" else f"{v:.1f} s"
            out += t(px + bw + 6, yy + 11, s, 10, SUB)
    y = T + grp * len(COMPLETENESS) + 26
    out += t(20, y, "교차 문서 +16pp (95% CI +4 ~ +32). 단일 문서는 세 조건 모두 29/30 으로 같다", 11, SUB)
    out += t(20, y + 18, "원문 직접 제공은 검색 실패가 없는 상한선이다. Skill 은 단일 검색보다 지연 +11.9 s, 토큰 +35.1k 를 더 쓴다", 11, SUB)
    return svg(w, h, out, "Carrel 답변 조건별 완결률과 비용")


def heat(v):
    # 0% = 빨강 계열, 100% = 초록 계열 (밝기만 조정)
    if v >= 90:
        return GREEN, "#ffffff"
    if v >= 50:
        return "#86efac", INK
    if v >= 30:
        return "#fde68a", INK
    return "#fca5a5", INK


def fig_rebuild():
    w, L, T, rh, cw = 800, 150, 96, 34, 100
    h = T + rh * len(REBUILD) + 96
    out = t(20, 28, "재빌드 안정성: 캐시 재사용은 동일, 새로 추출하면 관계가 가장 크게 바뀐다", 15, INK, weight="bold")
    out += t(20, 46, "두 버전의 집합 Jaccard (교집합 / 합집합). 실제 지식베이스 2개 (Base A · Base E)", 12, SUB)
    for b, base in enumerate(["Base A", "Base E"]):
        bx = L + b * (cw * 3 + 20)
        out += t(bx + cw * 1.5, T - 30, base, 12, INK, "middle", "bold")
        for k, s in enumerate(REBUILD_SCEN):
            out += t(bx + cw * k + cw / 2, T - 10, s, 10, SUB, "middle")
    for i, (name, a, e) in enumerate(REBUILD):
        y = T + rh * i
        out += t(20, y + 21, name, 12, INK, weight="bold")
        for b, vals in enumerate((a, e)):
            bx = L + b * (cw * 3 + 20)
            for k, v in enumerate(vals):
                fill, fg = heat(v)
                out += f'<rect x="{bx + cw*k + 2}" y="{y+2}" width="{cw-4}" height="{rh-4}" rx="3" fill="{fill}"/>'
                out += t(bx + cw * k + cw / 2, y + 21, f"{v:.2f}%" if v < 100 else "100%", 11, fg, "middle", "bold")
    y = T + rh * len(REBUILD) + 22
    lx = 20
    for label, v in [("90% 이상", 95), ("50~90%", 60), ("30~50%", 40), ("30% 미만", 10)]:
        out += f'<rect x="{lx}" y="{y-10}" width="12" height="12" rx="2" fill="{heat(v)[0]}"/>' + t(lx + 18, y, label, 11, SUB)
        lx += 40 + len(label) * 8
    out += t(20, y + 24, "fact 문장은 바뀌어도 그 fact 를 뽑은 출처 청크는 97% 이상 같다. 같은 30문항 Hit@5 는 6개 버전 모두 "
             f"{REBUILD_HIT5:.0f}%", 11, SUB)
    out += t(20, y + 42, "labels 재추출 = 엔티티 타입 라벨까지 다시 뽑은 뒤 재빌드. 관계 집합은 11~19% 만 겹친다", 11, SUB)
    return svg(w, h, out, "Carrel 그래프 재빌드 시나리오별 Jaccard")


FIGS = [("ablation", fig_ablation), ("answer_cost", fig_answer_cost), ("rebuild", fig_rebuild)]


def main(argv):
    global SHOW_NOTE
    if argv[:1] == ["--inline"]:
        out, prefix, SHOW_NOTE = Path(argv[1]), "", False
    else:
        out, prefix = OUT, "carrel_"
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in FIGS:
        (out / f"{prefix}{name}.svg").write_text(fn(), "utf-8")
        print("wrote", out / f"{prefix}{name}.svg")


if __name__ == "__main__":
    main(sys.argv[1:])
