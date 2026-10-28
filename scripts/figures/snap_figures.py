"""토스 SNAP (손가락 튕기기로 LLM 비용 줄이기, 2026-10-27) 노트의 그림을 생성한다.

    python scripts/figures/snap_figures.py              readings/blogs/assets/snap_*.svg 를 다시 쓴다
    python scripts/figures/snap_figures.py --inline DIR  HTML 보고서용 (접두어·생성 메모 없이 DIR/<이름>.svg)

SVG 는 손으로 고치지 않는다. 수치는 원문 본문과 Figure 4 에서 옮긴 아래 데이터 절만 쓴다.
"""
import sys
from pathlib import Path

FONT = "-apple-system,'Apple SD Gothic Neo','Noto Sans KR',sans-serif"
INK, SUB, GRID = "#1f2328", "#59636e", "#d8dee4"
GRAY, BLUE, LIGHT, RED = "#8b949e", "#2563eb", "#93c5fd", "#dc2626"
SHOW_NOTE = True
NOTE = "생성: python scripts/figures/snap_figures.py · 손으로 고치지 말 것"
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "readings" / "blogs" / "assets"

# 본문 + Figure 4: 목표 일치율(%) → 오프라인 coverage(%). 첫 배포는 99.9%
COVERAGE = [(99.9, 34.74), (99.0, 51.59), (95.0, 72.35)]
DEPLOYED = 99.9

# 본문: 첫 배포 조건의 hold out 검증
TOTAL, ADOPTED, MISMATCH = 100_648, 34_963, 34
FALLBACK = TOTAL - ADOPTED


def svg(w, h, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'role="img" aria-label="{title}" style="max-width:100%;height:auto;font-family:{FONT}">'
            f'<title>{title}</title><rect width="{w}" height="{h}" fill="#ffffff"/>{body}'
            + (f'<text x="{w-8}" y="{h-6}" font-size="9" fill="{GRAY}" text-anchor="end">{NOTE}</text>' if SHOW_NOTE else "")
            + '</svg>')


def t(x, y, s, size=12, fill=INK, anchor="start", weight="normal"):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}">{s}</text>')


def fig_coverage():
    w, h, L, R, T, B = 640, 360, 70, 610, 60, 270
    out = t(20, 28, "목표 일치율을 낮출수록 작은 모델이 가져가는 요청이 늘어난다", 15, INK, weight="bold")
    out += t(20, 46, "동일상품 판별 모델, hold out 검증 데이터 기준 coverage", 12, SUB)
    for v in range(0, 101, 20):
        y = B - (B - T) * v / 100
        out += f'<line x1="{L}" y1="{y}" x2="{R}" y2="{y}" stroke="{GRID}"/>'
        out += t(L - 8, y + 4, v, 11, SUB, "end")
    out += t(18, (T + B) / 2, "coverage (%)", 11, SUB, "middle").replace("<text ", f'<text transform="rotate(-90 18 {(T + B) / 2})" ', 1)
    slot = (R - L) / len(COVERAGE)
    for i, (target, cov) in enumerate(COVERAGE):
        cx, bw = L + slot * (i + 0.5), slot * 0.5
        y = B - (B - T) * cov / 100
        color = BLUE if target == DEPLOYED else LIGHT
        out += f'<rect x="{cx - bw / 2}" y="{y}" width="{bw}" height="{B - y}" rx="4" fill="{color}"/>'
        out += t(cx, y - 8, f"{cov:.2f}%", 14, INK, "middle", "bold")
        out += t(cx, B + 20, f"목표 {target:g}%", 12, INK, "middle")
        if target == DEPLOYED:
            out += t(cx, B + 37, "첫 배포 기준", 11, BLUE, "middle")
    out += t((L + R) / 2, B + 58, "목표 selective accuracy (기존 LLM 판단과의 일치율)", 11, SUB, "middle")
    return svg(w, h, out, "목표 일치율별 오프라인 coverage")


def fig_split():
    w, h, L, R, y0, bh = 640, 230, 30, 610, 92, 46
    out = t(20, 28, "검증 요청 100,648건 중 약 3분의 1을 작은 모델이 처리한다", 15, INK, weight="bold")
    out += t(20, 46, "목표 일치율 99.9% threshold 적용 결과. 나머지는 기존 LLM으로 Fallback", 12, SUB)
    xa = L + (R - L) * ADOPTED / TOTAL
    out += f'<rect x="{L}" y="{y0}" width="{xa - L}" height="{bh}" fill="{BLUE}"/>'
    out += f'<rect x="{xa}" y="{y0}" width="{R - xa}" height="{bh}" fill="{GRID}"/>'
    xm = max(xa - (R - L) * MISMATCH / TOTAL, xa - 2)
    out += f'<rect x="{xm}" y="{y0}" width="{xa - xm}" height="{bh}" fill="{RED}"/>'
    out += t((L + xa) / 2, y0 + 29, f"채택 {ADOPTED:,}건 ({ADOPTED / TOTAL:.2%})", 13, "#ffffff", "middle", "bold")
    out += t((xa + R) / 2, y0 + 29, f"Fallback {FALLBACK:,}건", 13, INK, "middle", "bold")
    out += f'<line x1="{xa}" y1="{y0 + bh}" x2="{xa}" y2="{y0 + bh + 22}" stroke="{RED}"/>'
    out += t(xa, y0 + bh + 36, f"채택 중 LLM 판단과 불일치 {MISMATCH}건 (일치율 {(ADOPTED - MISMATCH) / ADOPTED:.4%})",
             12, RED, "middle")
    out += t(L, y0 - 10, "conf ≥ τ: 작은 모델 결과 사용", 11, SUB)
    out += t(R, y0 - 10, "conf &lt; τ: 기존 LLM 추가 호출", 11, SUB, "end")
    return svg(w, h, out, "검증 요청의 채택과 Fallback 분해")


FIGS = [("coverage", fig_coverage), ("split", fig_split)]


def main(argv):
    global SHOW_NOTE
    if argv[:1] == ["--inline"]:
        out, prefix, SHOW_NOTE = Path(argv[1]), "", False
    else:
        out, prefix = OUT, "snap_"
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in FIGS:
        (out / f"{prefix}{name}.svg").write_text(fn(), "utf-8")
        print("wrote", out / f"{prefix}{name}.svg")


if __name__ == "__main__":
    main(sys.argv[1:])
