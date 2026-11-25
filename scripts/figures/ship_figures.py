"""Anthropic Claude Ship 발표(2026-11-23) 노트의 그림을 생성한다.

    python scripts/figures/ship_figures.py              readings/blogs/assets/ship_*.svg 를 다시 쓴다
    python scripts/figures/ship_figures.py --inline DIR  HTML 보고서용 (접두어, 생성 메모 없이 DIR/<이름>.svg)

SVG 는 손으로 고치지 않는다. 수치는 발표 글의 approvals.yml 예시 그림과
함께 읽은 claude.dev 블로그(How the Claude Code team does code review)에서 옮긴 아래 데이터 절만 쓴다.
"""
import sys
from pathlib import Path

FONT = "-apple-system,'Apple SD Gothic Neo','Noto Sans KR',sans-serif"
INK, SUB, GRID = "#1f2328", "#59636e", "#d8dee4"
GRAY, BLUE, LIGHT, RED, GREEN, ORANGE = "#8b949e", "#2563eb", "#93c5fd", "#dc2626", "#16a34a", "#d97757"
SHOW_NOTE = True
NOTE = "생성: python scripts/figures/ship_figures.py, 손으로 고치지 말 것"
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "readings" / "blogs" / "assets"

# 발표 글 원문 그림 4 (.claude/approvals.yml 예시): (규칙, 경로, 기본 사람 seats, Code Review pass 시 seats, pass 시 Claude 역할)
RULES = [
    ("Most code", '"**", "!docs/**"', 2, 1, "approves"),
    ("Docs", '"docs/**"', 2, 0, "approves"),
    ("Billing", '"services/billing/**"', 2, None, "comments"),  # override 없음. 작성자 아닌 2명, 그중 1명 payments
]

# claude.dev 블로그 위험 등급 표: (등급, 예, 필요한 사람 승인 수)
TIERS = [
    ("Low", "문서, 문구, 내부 전용, flag 뒤 추가 변경", 0),
    ("Medium", "일반 제품 변경, 캐시, auto-updater, system prompt", 1),
    ("High", "최고 위험 코드, 항상 2명이 필요한 경로", 2),
]

# claude.dev 블로그 본문 수치
CHANGES_PER_DAY = (2500, 30000)   # Anthropic 모노레포, 5월 약 2,500 → 9월 30,000 이상
SUBSTANTIVE_PCT = (16, 54)        # research preview 출시 때 실질 리뷰 코멘트를 받은 내부 PR 비율


def svg(w, h, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'role="img" aria-label="{title}" style="max-width:100%;height:auto;font-family:{FONT}">'
            f'<title>{title}</title><rect width="{w}" height="{h}" fill="#ffffff"/>{body}'
            + (f'<text x="{w-8}" y="{h-6}" font-size="9" fill="{GRAY}" text-anchor="end">{NOTE}</text>' if SHOW_NOTE else "")
            + '</svg>')


def t(x, y, s, size=12, fill=INK, anchor="start", weight="normal"):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}">{s}</text>')


def seat_boxes(x, y, n, color, size=26, gap=8):
    out = ""
    for i in range(n):
        out += f'<rect x="{x + i * (size + gap)}" y="{y}" width="{size}" height="{size}" rx="5" fill="{color}"/>'
    if n == 0:
        out += f'<rect x="{x}" y="{y}" width="{size}" height="{size}" rx="5" fill="none" stroke="{GRAY}" stroke-dasharray="4 3"/>'
        out += t(x + size + 10, y + 18, "사람 승인 없이 Claude 승인으로 통과", 12, GREEN, weight="bold")
    return out


def fig_policy():
    w, h = 760, 380
    c1, c2, c3 = 30, 300, 520
    out = t(20, 28, "같은 정책 파일 안에서 경로마다 Code Review 통과가 줄여 주는 사람 승인 수가 다르다", 15, INK, weight="bold")
    out += t(20, 46, "발표 글의 .claude/approvals.yml 예시를 규칙별로 다시 셈. 칸 하나 = 필요한 사람 승인 1명", 12, SUB)
    out += t(c1, 82, "규칙 (paths)", 12, SUB, weight="bold")
    out += t(c2, 82, "기본 (Claude는 comments)", 12, SUB, weight="bold")
    out += t(c3, 82, "claude/code_review = pass 일 때", 12, SUB, weight="bold")
    y = 100
    for name, paths, base, passed, role in RULES:
        out += f'<line x1="20" y1="{y - 8}" x2="{w - 20}" y2="{y - 8}" stroke="{GRID}"/>'
        out += t(c1, y + 18, name, 14, INK, weight="bold")
        out += t(c1, y + 38, paths, 11, SUB)
        out += seat_boxes(c2, y + 4, base, GRAY)
        out += t(c2 + base * 34 + 6, y + 22, f"{base}명", 12, INK, weight="bold")
        if passed is None:
            out += seat_boxes(c3, y + 4, base, GRAY)
            out += t(c3 + base * 34 + 6, y + 22, "override 없음, 그대로", 12, RED, weight="bold")
            out += t(c3, y + 50, "작성자가 아닌 2명, 그중 1명은 payments 팀", 11, SUB)
        else:
            out += seat_boxes(c3, y + 4, passed, BLUE)
            if passed:
                out += t(c3 + passed * 34 + 6, y + 22, f"{passed}명 + Claude {role}", 12, BLUE, weight="bold")
            else:
                out += t(c3 + 36, y + 40, f"Claude {role}", 11, SUB)
        y += 80
    out += f'<line x1="20" y1="{y - 8}" x2="{w - 20}" y2="{y - 8}" stroke="{GRID}"/>'
    out += t(20, y + 16, "Billing 의 payments 1명이 앞의 2명 안에 포함되는지는 원문 그림의 seats 목록 문법만으로 확정할 수 없다 (purpose 문장 기준 2명으로 셈)", 11, SUB)
    return svg(w, h, out, "approvals.yml 예시 규칙별 필요한 사람 승인 수")


def fig_context():
    w, h = 760, 400
    out = t(20, 28, "변경량이 10배 넘게 늘어난 환경에서 Claude Code 팀은 사람 승인 수를 변경 위험으로 나눈다", 15, INK, weight="bold")
    out += t(20, 46, "함께 읽은 claude.dev 블로그 수치. 왼쪽 두 막대는 서로 다른 축", 12, SUB)
    # 왼쪽: 변경량
    L, B, T = 50, 330, 112
    a, b = CHANGES_PER_DAY
    out += t(L, 80, "모노레포 하루 변경 수", 12, INK, weight="bold")
    for i, (lab, v) in enumerate([("5월", a), ("9월", b)]):
        hh = (B - T) * v / b
        x = L + i * 70
        out += f'<rect x="{x}" y="{B - hh}" width="44" height="{hh}" rx="3" fill="{GRAY if i == 0 else ORANGE}"/>'
        out += t(x + 22, B - hh - 6, f"{v:,}" + ("+" if i else ""), 11, INK, "middle", "bold")
        out += t(x + 22, B + 16, lab, 12, SUB, "middle")
    out += t(L, B + 36, f"약 {b // a}배", 12, ORANGE, weight="bold")
    # 가운데: 실질 리뷰 코멘트 비율
    L2 = 250
    p, q = SUBSTANTIVE_PCT
    out += t(L2, 80, "실질 리뷰 코멘트를 받은 PR", 12, INK, weight="bold")
    for i, (lab, v) in enumerate([("이전", p), ("preview", q)]):
        hh = (B - T) * v / 100
        x = L2 + i * 70
        out += f'<rect x="{x}" y="{B - hh}" width="44" height="{hh}" rx="3" fill="{GRAY if i == 0 else BLUE}"/>'
        out += t(x + 22, B - hh - 6, f"{v}%", 11, INK, "middle", "bold")
        out += t(x + 22, B + 16, lab, 12, SUB, "middle")
    out += f'<line x1="{L2 - 6}" y1="{T}" x2="{L2 + 120}" y2="{T}" stroke="{GRID}" stroke-dasharray="3 3"/>'
    out += t(L2 + 124, T + 4, "100%", 10, SUB)
    # 오른쪽: 위험 등급
    X = 450
    out += t(X, 80, "위험 등급별 필요한 사람 승인", 12, INK, weight="bold")
    y = 104
    for name, ex, n in TIERS:
        out += t(X, y + 16, name, 13, INK, weight="bold")
        for k in range(max(n, 1)):
            if n == 0:
                out += f'<rect x="{X + 70}" y="{y}" width="24" height="24" rx="5" fill="none" stroke="{GRAY}" stroke-dasharray="4 3"/>'
            else:
                out += f'<rect x="{X + 70 + k * 30}" y="{y}" width="24" height="24" rx="5" fill="{BLUE}"/>'
        out += t(X + 140, y + 17, f"{n}명", 12, INK, weight="bold")
        out += t(X, y + 42, ex, 11, SUB)
        y += 72
    out += t(X, y + 4, "등급은 작성자가 아니라 변경을 읽은 분류기가 정한다", 11, INK)
    out += t(X, y + 20, "에이전트 승인은 사람 승인 칸을 채우지 못한다", 11, RED)
    return svg(w, h, out, "Claude Code 팀의 변경량, 리뷰 코멘트 비율, 위험 등급")


FIGS = [("policy", fig_policy), ("context", fig_context)]


def main(argv):
    global SHOW_NOTE
    if argv[:1] == ["--inline"]:
        out, prefix, SHOW_NOTE = Path(argv[1]), "", False
    else:
        out, prefix = OUT, "ship_"
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in FIGS:
        (out / f"{prefix}{name}.svg").write_text(fn(), "utf-8")
        print("wrote", out / f"{prefix}{name}.svg")


if __name__ == "__main__":
    main(sys.argv[1:])
