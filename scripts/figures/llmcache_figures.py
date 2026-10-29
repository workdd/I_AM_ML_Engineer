"""로보코 "LLM 캐시는 어떻게 작동할까" 노트의 그림을 생성한다.

    python scripts/figures/llmcache_figures.py              readings/blogs/assets/llmcache_*.svg 를 다시 쓴다
    python scripts/figures/llmcache_figures.py --inline DIR  HTML 보고서용 (접두어·생성 메모 없이 DIR/<이름>.svg)

SVG 는 손으로 고치지 않는다. 원문 그림 6(토큰 비교)은 노트에서 원문 이미지로 인용하고,
여기서는 원문 그림에 없는 사용자별 격리 히트율만 그린다. 원문 7·8절의 가상 예시(문서 10,000토큰, 질문 500토큰, 요청 100개,
사용자 10명 × 10회, 만료·퇴거·라우팅 미스 없음)를 아래 함수로 계산한다. 실측값이 아니다.
"""
import sys
from pathlib import Path

FONT = "-apple-system,'Apple SD Gothic Neo','Noto Sans KR',sans-serif"
INK, SUB, GRID = "#1f2328", "#59636e", "#d8dee4"
GRAY, BLUE, ORANGE, GREEN = "#8b949e", "#2563eb", "#ea580c", "#16a34a"
SHOW_NOTE = True
NOTE = "생성: python scripts/figures/llmcache_figures.py · 손으로 고치지 말 것"
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "readings" / "blogs" / "assets"

DOC, Q = 10_000, 500      # 공통 문서, 요청마다 다른 질문
USERS, PER_USER = 10, 10  # 총 요청 100개


def total_input(n):
    return n * (DOC + Q)


def cached_tokens(groups, per_group):
    """격리 단위(group)마다 첫 요청만 문서를 계산하고 나머지는 캐시에서 읽는다."""
    return groups * (per_group - 1) * DOC


def hit_rate(groups, per_group):
    return cached_tokens(groups, per_group) / total_input(groups * per_group)


def scenarios():
    n = USERS * PER_USER
    return [
        ("캐시 없음", total_input(n), 0.0, GRAY),
        (f"사용자별 격리 ({USERS}명 × {PER_USER}회)", total_input(n) - cached_tokens(USERS, PER_USER),
         hit_rate(USERS, PER_USER), ORANGE),
        ("단일 공유 그룹 (100회)", total_input(n) - cached_tokens(1, n), hit_rate(1, n), BLUE),
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


def fig_hitrate():
    w, h, L, R, T, B = 680, 400, 60, 640, 80, 290
    ks = list(range(1, 21))
    X = lambda k: L + (R - L) * (k - 1) / (ks[-1] - 1)
    Y = lambda v: B - (B - T) * v / 100
    out = t(20, 28, "사용자별 격리일 때 사용자당 요청 수와 토큰 히트율 (노트에서 계산)", 14.5, INK, weight="bold")
    out += t(20, 46, "한 번 쓰고 떠나는 사용자는 이득이 없고, 반복이 많을수록 공유 그룹에 가까워진다", 11.5, SUB)
    for v in (0, 25, 50, 75, 100):
        out += f'<line x1="{L}" y1="{Y(v)}" x2="{R}" y2="{Y(v)}" stroke="{GRID}"/>' + t(L - 8, Y(v) + 4, f"{v}%", 11, SUB, "end")
    for k in (1, 5, 10, 15, 20):
        out += t(X(k), B + 18, k, 11, SUB, "middle")
    out += t((L + R) / 2, B + 38, "사용자당 요청 수 (문서 10,000 + 질문 500토큰)", 11, SUB, "middle")
    shared = hit_rate(1, USERS * PER_USER) * 100
    out += f'<line x1="{L}" y1="{Y(shared):.1f}" x2="{R}" y2="{Y(shared):.1f}" stroke="{BLUE}" stroke-dasharray="6 4"/>'
    out += t(L + 6, Y(shared) - 6, f"단일 공유 그룹, 요청 100개: {shared:.1f}%", 11, BLUE)
    cap = DOC / (DOC + Q) * 100
    pts = " ".join(f"{X(k):.1f},{Y(hit_rate(1, k) * 100):.1f}" for k in ks)
    out += f'<polyline points="{pts}" fill="none" stroke="{ORANGE}" stroke-width="2.4"/>'
    for k in ks:
        out += f'<circle cx="{X(k):.1f}" cy="{Y(hit_rate(1, k) * 100):.1f}" r="2.8" fill="{ORANGE}"/>'
    for k in (1, 2, PER_USER):
        v = hit_rate(1, k) * 100
        out += t(X(k) + 6, Y(v) + 16, f"{k}회: {v:.1f}%", 10.5, ORANGE)
    out += t(20, h - 42, "식: (k - 1) × 10,000 ÷ (k × 10,500). 원문 예시(사용자 10명 × 10회 = 85.7%)를 k에 대해 늘린 것이다", 11, SUB)
    out += t(20, h - 24, f"k가 커지면 {cap:.1f}%(문서 ÷ 요청 길이)에 다가간다. 질문 부분은 매번 새로 계산하기 때문이다", 11, SUB)
    return svg(w, h, out, "사용자당 요청 수와 토큰 히트율")


FIGS = [("hitrate", fig_hitrate)]


def main(argv):
    global SHOW_NOTE
    if argv[:1] == ["--inline"]:
        out, prefix, SHOW_NOTE = Path(argv[1]), "", False
    else:
        out, prefix = OUT, "llmcache_"
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in FIGS:
        (out / f"{prefix}{name}.svg").write_text(fn(), "utf-8")
        print("wrote", out / f"{prefix}{name}.svg")


if __name__ == "__main__":
    main(sys.argv[1:])
