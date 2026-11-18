"""ModernRelay/omnigraph 노트의 그림을 생성한다.

    python scripts/figures/omnigraph_figures.py              readings/repos/assets/omnigraph_*.svg 를 다시 쓴다
    python scripts/figures/omnigraph_figures.py --inline DIR  HTML 보고서용 (접두어, 생성 메모 없이 DIR/<이름>.svg)

SVG 는 손으로 고치지 않는다. 수치는 v0.15.0 릴리즈 개요(docs/releases/v0.15.0-overview.md, 커밋 373834b)의
본문 문장에서 옮겼다. 원문에는 이 수치를 그린 그림이 없다.
"""
import math
import sys
from pathlib import Path

FONT = "-apple-system,'Apple SD Gothic Neo','Noto Sans KR',sans-serif"
INK, SUB, GRID = "#1f2328", "#59636e", "#d8dee4"
GRAY, BLUE, ORANGE = "#8b949e", "#2563eb", "#ea580c"
SHOW_NOTE = True
NOTE = "생성: python scripts/figures/omnigraph_figures.py, 손으로 고치지 말 것"
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "readings" / "repos" / "assets"

# (작업, 0.14 이전 요청 수, 0.15 요청 수) — 범위는 (하한, 상한)
REQUESTS = [
    ("diff: 10만 행 타입에서 10행 수정", (277, 277), (14, 14)),
    ("fast-forward 병합: 1행 커밋 1,024개 뒤", (4168, 4168), (54, 54)),
    ("3-way 병합: 1행 커밋 1,024개 뒤", (3216, 3216), (135, 135)),
    ("다른 프로세스 커밋 뒤 첫 읽기", (18, 20), (11, 12)),
]
# LDBC SNB SF30 새 그래프 CSV 적재 (초)
LOAD_014 = 261
LOAD_015 = (87, 110)


def ratio(before, after):
    """요청이 몇 분의 1로 줄었는지 (중간값 기준)."""
    return (sum(before) / 2) / (sum(after) / 2)


def svg(w, h, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'role="img" aria-label="{title}" style="max-width:100%;height:auto;font-family:{FONT}">'
            f'<title>{title}</title><rect width="{w}" height="{h}" fill="#ffffff"/>{body}'
            + (f'<text x="{w-8}" y="{h-6}" font-size="9" fill="{GRAY}" text-anchor="end">{NOTE}</text>' if SHOW_NOTE else "")
            + '</svg>')


def t(x, y, s, size=12, fill=INK, anchor="start", weight="normal"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}">{s}</text>')


def fig_requests():
    W, H = 760, 400
    x0, x1 = 250, 690
    lo, hi = 1, 10000

    def X(v):
        return x0 + (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * (x1 - x0)

    b = [t(20, 28, "v0.15 에서 저장소 요청 수 (로그 축)", 15, weight="bold"),
         t(20, 48, "diff, 병합은 시뮬레이션 인메모리 저장소 측정. S3 에서는 요청마다 네트워크 왕복", 11, SUB)]
    top = 84
    for v in (1, 10, 100, 1000, 10000):
        b.append(f'<line x1="{X(v):.1f}" y1="{top-6}" x2="{X(v):.1f}" y2="{top+4*68}" stroke="{GRID}"/>')
        b.append(t(X(v), top + 4 * 68 + 16, f"{v:,}", 10, SUB, "middle"))
    for i, (name, before, after) in enumerate(REQUESTS):
        y = top + i * 68
        b.append(t(20, y + 22, name, 12))
        for j, (vals, col, lab) in enumerate(((before, GRAY, "0.14"), (after, BLUE, "0.15"))):
            yy = y + 6 + j * 24
            b.append(f'<rect x="{x0}" y="{yy}" width="{X(vals[1])-x0:.1f}" height="18" fill="{col}" opacity="0.85"/>')
            if vals[0] != vals[1]:
                b.append(f'<rect x="{X(vals[0]):.1f}" y="{yy}" width="{X(vals[1])-X(vals[0]):.1f}" height="18" fill="#ffffff" opacity="0.5"/>')
            txt = f"{vals[0]:,}" if vals[0] == vals[1] else f"{vals[0]}~{vals[1]}"
            b.append(t(X(vals[1]) + 6, yy + 13, f"{lab}: {txt}", 11, col if j else SUB))
        r = ratio(before, after)
        b.append(t(20, y + 40, f"약 1/{r:.0f}" if r >= 10 else f"약 1/{r:.1f}", 11, BLUE, weight="bold"))
    lx = 560
    b.append(f'<rect x="{lx}" y="58" width="12" height="10" fill="{GRAY}"/>' + t(lx + 16, 67, "0.14 이전", 11, SUB))
    b.append(f'<rect x="{lx+90}" y="58" width="12" height="10" fill="{BLUE}"/>' + t(lx + 106, 67, "0.15", 11, SUB))
    return svg(W, H, "".join(b), "Omnigraph v0.15 저장소 요청 수 변화")


def fig_load():
    W, H = 760, 210
    x0, x1 = 170, 700
    vmax = 300

    def X(v):
        return x0 + v / vmax * (x1 - x0)

    b = [t(20, 28, "새 그래프 CSV 대량 적재 시간 (LDBC SNB SF30, CSV 34 GB, load 문 31개)", 15, weight="bold"),
         t(20, 48, "Apple silicon 워크스테이션, 로컬 디스크, release 빌드. S3 적재는 측정하지 않음", 11, SUB)]
    for v in range(0, vmax + 1, 50):
        b.append(f'<line x1="{X(v):.1f}" y1="66" x2="{X(v):.1f}" y2="146" stroke="{GRID}"/>')
        b.append(t(X(v), 162, f"{v}초", 10, SUB, "middle"))
    b.append(t(20, 90, "v0.14.0", 12))
    b.append(f'<rect x="{x0}" y="76" width="{X(LOAD_014)-x0:.1f}" height="22" fill="{GRAY}"/>')
    b.append(t(X(LOAD_014) + 6, 92, f"{LOAD_014}초", 11, SUB))
    b.append(t(20, 130, "v0.15", 12))
    b.append(f'<rect x="{x0}" y="116" width="{X(LOAD_015[0])-x0:.1f}" height="22" fill="{BLUE}"/>')
    b.append(f'<rect x="{X(LOAD_015[0]):.1f}" y="116" width="{X(LOAD_015[1])-X(LOAD_015[0]):.1f}" height="22" '
             f'fill="{BLUE}" opacity="0.4"/>')
    b.append(t(X(LOAD_015[1]) + 6, 132, f"{LOAD_015[0]}~{LOAD_015[1]}초", 11, BLUE))
    b.append(t(20, 190, f"0.14 대비 {LOAD_015[0]/LOAD_014*100:.0f}~{LOAD_015[1]/LOAD_014*100:.0f}% (원문: less than half)",
               11, SUB))
    return svg(W, H, "".join(b), "Omnigraph v0.15 CSV 대량 적재 시간")


FIGS = [("requests", fig_requests), ("load", fig_load)]


def main(argv):
    global SHOW_NOTE
    if argv[:1] == ["--inline"]:
        out, prefix, SHOW_NOTE = Path(argv[1]), "", False
    else:
        out, prefix = OUT, "omnigraph_"
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in FIGS:
        (out / f"{prefix}{name}.svg").write_text(fn(), "utf-8")
        print("wrote", out / f"{prefix}{name}.svg")


if __name__ == "__main__":
    main(sys.argv[1:])
