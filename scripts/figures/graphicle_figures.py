"""Graphicle-AI/graphicle 노트의 그림을 생성한다.

    python scripts/figures/graphicle_figures.py              readings/repos/assets/graphicle_*.svg 를 다시 쓴다
    python scripts/figures/graphicle_figures.py --inline DIR  HTML 보고서용 (접두어·생성 메모 없이 DIR/<이름>.svg)

SVG 는 손으로 고치지 않는다. 수치는 레포 v0.1.9 (커밋 4bbe6cd) 의 docs/benchmarks/retrieval.md,
docs/benchmarks/whole-repo.md, CHANGELOG.md 0.1.9 절에서 옮긴 아래 상수만 쓴다.
"""
import sys
from pathlib import Path

FONT = "-apple-system,'Apple SD Gothic Neo','Noto Sans KR',sans-serif"
INK, SUB, GRID = "#1f2328", "#59636e", "#d8dee4"
GRAY, BLUE, ORANGE, LIGHT = "#8b949e", "#2563eb", "#ea580c", "#93c5fd"
SHOW_NOTE = True
NOTE = "생성: python scripts/figures/graphicle_figures.py · 손으로 고치지 말 것"
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "readings" / "repos" / "assets"

# retrieval.md 보류 세트: SWE-bench Verified 100개, (hit@1, hit@5, hit@10)
RETRIEVAL = [
    ("Graphicle 0.1.8", (43, 75, 81), BLUE),
    ("BM25 (SWE-bench 방식)", (35, 61, 77), ORANGE),
    ("grep (이슈의 이름)", (20, 54, 71), GRAY),
    ("Graphicle 0.1.2", (13, 30, 30), LIGHT),
]

# CHANGELOG 0.1.9 + whole-repo.md: 보류 저장소 90문항, 수정 파일 포함 비율 평균 (%)
BUDGETS = ["64K", "128K", "256K", "512K", "1M"]
PACK_AFTER = [58.1, 64.3, 64.6, 70.9, 77.9]
PACK_BEFORE = [55.8, 63.6, 61.4, 68.0, 76.8]
TRUNCATION = [1.1, 1.1, 1.1, 1.1, 2.2]

# whole-repo.md 1M 예산: (문맥, Muse Spark 1.3 후속 질문 입력 비용 $, 수정 파일 포함 비율 %, 정가 계산)
CACHE_1M = [
    ("standard", 1.27, 96.2),
    ("cache 50%", 0.610, 95.9),
    ("cache 70%", 0.433, 92.9),
    ("cache 90%", 0.252, 94.4),
    ("cache 95%", 0.204, 92.1),
    ("stable", 0.153, 18.2),
    ("자른 저장소", 0.153, 7.0),
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


def fig_retrieval():
    w, h, L, R, T, B = 680, 390, 60, 650, 90, 310
    Y = lambda v: B - (B - T) * v / 100
    out = t(20, 28, "고칠 파일을 상위 k개 안에 찾은 이슈 수 (SWE-bench Verified 100개, 튜닝에 안 쓴 보류 세트)", 14, INK, weight="bold")
    out += t(20, 46, "오프라인 랭킹만 잰 값이다. 에이전트가 더 많이 고친다는 뜻은 아니다", 12, SUB)
    for v in (0, 20, 40, 60, 80, 100):
        out += f'<line x1="{L}" y1="{Y(v)}" x2="{R}" y2="{Y(v)}" stroke="{GRID}"/>' + t(L - 8, Y(v) + 4, v, 11, SUB, "end")
    gw = (R - L) / 3
    bw = 30
    for gi, k in enumerate(("상위 1", "상위 5", "상위 10")):
        x0 = L + gi * gw + (gw - bw * len(RETRIEVAL)) / 2
        for si, (_, vals, c) in enumerate(RETRIEVAL):
            v, x = vals[gi], x0 + si * bw
            out += f'<rect x="{x:.1f}" y="{Y(v):.1f}" width="{bw - 4}" height="{B - Y(v):.1f}" fill="{c}"/>'
            out += t(x + (bw - 4) / 2, Y(v) - 4, v, 10.5, INK, "middle")
        out += t(L + gi * gw + gw / 2, B + 18, k, 12, SUB, "middle")
    lx = L
    for name, _, c in RETRIEVAL:
        out += f'<rect x="{lx}" y="62" width="11" height="11" fill="{c}"/>' + t(lx + 16, 72, name, 11.5)
        lx += 16 + len(name) * 7.2 + 18
    out += t(20, h - 30, "BM25 대비 유의한 차이는 상위 5개뿐 (McNemar exact p = 0.013). 0.1.2는 이전 노트가 읽은 판", 11, SUB)
    return svg(w, h, out, "고칠 파일 찾기 오프라인 랭킹")


def fig_packrecall():
    w, h, L, R, T, B = 680, 380, 60, 640, 80, 300
    X = lambda i: L + 30 + (R - L - 60) * i / (len(BUDGETS) - 1)
    Y = lambda v: B - (B - T) * v / 100
    out = t(20, 28, "토큰 예산별로 pack 안에 들어간 수정 파일 비율 (보류 저장소 3개, 90문항, 모델 없이 측정)", 14, INK, weight="bold")
    out += t(20, 46, "같은 예산에 파일을 앞에서부터 통째로 담으면 거의 들어가지 않는다", 12, SUB)
    for v in (0, 20, 40, 60, 80, 100):
        out += f'<line x1="{L}" y1="{Y(v)}" x2="{R}" y2="{Y(v)}" stroke="{GRID}"/>' + t(L - 8, Y(v) + 4, f"{v}%", 11, SUB, "end")
    for i, b in enumerate(BUDGETS):
        out += t(X(i), B + 18, b, 11.5, SUB, "middle")
    series = ((PACK_AFTER, BLUE, "pack 0.1.9 (대회 후)"), (PACK_BEFORE, LIGHT, "pack 대회 전 (2a46edb)"),
              (TRUNCATION, GRAY, "같은 예산으로 자른 저장소"))
    for vals, c, _ in series:
        pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(vals))
        out += f'<polyline points="{pts}" fill="none" stroke="{c}" stroke-width="2.2"/>'
        out += "".join(f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="3" fill="{c}"/>' for i, v in enumerate(vals))
    for i, v in enumerate(PACK_AFTER):
        out += t(X(i), Y(v) - 8, f"{v}", 10.5, BLUE, "middle")
    for i, v in enumerate(TRUNCATION):
        out += t(X(i), Y(v) - 7, f"{v}", 10, SUB, "middle")
    ly = T + 2
    for _, c, name in series:
        out += f'<rect x="{L + 10}" y="{ly}" width="12" height="3" fill="{c}"/>' + t(L + 28, ly + 5, name, 11.5)
        ly += 18
    out += t(20, h - 30, "대회 후 수치는 이슈 환경 절 처리 규칙을 마지막으로 바꾸기 전에 쟀다 (원문 단서)", 11, SUB)
    return svg(w, h, out, "토큰 예산별 pack 회수율")


def fig_cache():
    w, h, L, T = 680, 400, 110, 80
    rowh, maxw = 32, 330
    vmax = max(c for _, c, _ in CACHE_1M)
    out = t(20, 28, "1M 토큰 문맥의 후속 질문 입력 비용과 수정 파일 포함 비율 (Muse Spark 1.3, 정가 계산)", 14, INK, weight="bold")
    out += t(20, 46, "앞부분을 질문과 무관하게 고정할수록 캐시로 싸지지만, 질문에 쓸 몫이 줄어 포함 비율이 내려간다", 12, SUB)
    out += t(L, T - 10, "후속 질문 1회 입력 비용 ($)", 11, SUB) + t(L + maxw + 110, T - 10, "포함 비율", 11, SUB, "middle")
    for i, (name, cost, recall) in enumerate(CACHE_1M):
        y = T + i * rowh
        c = BLUE if name.startswith("cache") else (ORANGE if name == "standard" else GRAY)
        bw = maxw * cost / vmax
        out += t(L - 10, y + 17, name, 12, INK, "end")
        out += f'<rect x="{L}" y="{y + 4}" width="{bw:.1f}" height="{rowh - 12}" fill="{c}"/>'
        out += t(L + bw + 6, y + 17, f"${cost:g}", 11, INK)
        rc = INK if recall >= 90 else ORANGE
        out += t(L + maxw + 110, y + 17, f"{recall}%", 12, rc, "middle", "bold")
    y = T + len(CACHE_1M) * rowh + 16
    out += t(20, y, "cache N%: 저장소 부분이 예산의 N%. stable은 질문 부분 781토큰만 남는다. 개발 저장소(kubernetes, home-assistant) 평균", 11, SUB)
    out += t(20, y + 18, "청구 금액이 아니라 정가로 계산한 값이다. 캐시 적중 여부는 API가 정한다", 11, SUB)
    return svg(w, h, out, "프롬프트 캐시 레이아웃별 비용과 포함 비율")


FIGS = [("retrieval", fig_retrieval), ("packrecall", fig_packrecall), ("cache", fig_cache)]


def main(argv):
    global SHOW_NOTE
    if argv[:1] == ["--inline"]:
        out, prefix, SHOW_NOTE = Path(argv[1]), "", False
    else:
        out, prefix = OUT, "graphicle_"
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in FIGS:
        (out / f"{prefix}{name}.svg").write_text(fn(), "utf-8")
        print("wrote", out / f"{prefix}{name}.svg")


if __name__ == "__main__":
    main(sys.argv[1:])
