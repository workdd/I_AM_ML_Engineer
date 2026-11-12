"""NVIDIA Auto Ontology 노트의 그림을 생성한다.

    python scripts/figures/aonto_figures.py              readings/repos/assets/aonto_*.svg 를 다시 쓴다
    python scripts/figures/aonto_figures.py --inline DIR  HTML 보고서용 (접두어·생성 메모 없이 DIR/<이름>.svg)

SVG 는 손으로 고치지 않는다. 수치는 아래 데이터 절만 쓴다.
  CORRECTNESS   레포 skills/*/BENCHMARK.md (2026-10-08 평가)
  STARS         GitHub API (10-28 · 11-04 · 11-12) 와 andrew.ooo 리뷰 "Last verified" (11-09 · 11-11)
  OPEN_ISSUES   gh issue list --state open, 2026-11-12 (PR 제외 62건). 분류는 제목을 읽고 정했다
"""
import sys
from pathlib import Path

FONT = "-apple-system,'Apple SD Gothic Neo','Noto Sans KR',sans-serif"
INK, SUB, GRID = "#1f2328", "#59636e", "#d8dee4"
GRAY, BLUE, ORANGE = "#8b949e", "#2563eb", "#ea580c"
SHOW_NOTE = True
NOTE = "생성: python scripts/figures/aonto_figures.py · 손으로 고치지 말 것"
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "readings" / "repos" / "assets"

# BENCHMARK.md Correctness: (스킬, 과제 수, 에이전트) → (스킬 없음 %, 스킬 있음 %)
CORRECTNESS = [
    ("setup", 10, "Claude Code", 36.0, 100.0), ("setup", 10, "Codex", 68.0, 100.0),
    ("management", 13, "Claude Code", 29.2, 100.0), ("management", 13, "Codex", 47.7, 89.2),
    ("query", 13, "Claude Code", 47.7, 100.0), ("query", 13, "Codex", 50.8, 70.8),
]
AGENT_COLOR = {"Claude Code": BLUE, "Codex": ORANGE}

# 별 수 추이: (날짜, 별, 출처)
STARS = [("10-28", 344, "API"), ("11-04", 2867, "API"), ("11-09", 10746, "리뷰"),
         ("11-11", 11593, "리뷰"), ("11-12", 11996, "API")]
HN = ("11-03", 292, 139)  # Hacker News item 50348962: 점수, 댓글

# 열린 이슈 유형: (유형, 색, 이슈 번호들). 2026-11-12 열린 이슈 62건 전부를 한 번씩 담는다
RED, YELLOW, PURPLE, GREEN = "#c0392b", "#a86b00", "#7048b8", "#1f8a4c"
OPEN_ISSUES = [
    ("거버넌스 · 인증 로드맵 (미구현)", RED,
     [250, 251, 253, 254, 256, 257, 258, 259, 260, 264, 268, 269, 427, 475]),
    ("질의 응답 경로 버그", RED, [252, 261, 263, 440, 461, 462, 486, 496, 497]),
    ("커넥터 · 메타데이터 · 컴파일 버그", YELLOW, [271, 272, 333, 408, 443, 453, 459, 463]),
    ("다국어 · 지역화 (한중일 · 비 ASCII)", RED, [468, 469, 470, 471, 472, 473, 474]),
    ("시맨틱 레이어 상태 일관성", YELLOW, [267, 444, 445, 499]),
    ("예측 (KumoRFM)", PURPLE, [334, 335, 361, 498]),
    ("문서 · 설정 · 화면", GRAY, [403, 428, 432, 446, 465, 466, 467]),
    ("신규 커넥터 · 통합 요청", PURPLE, [406, 407, 410, 431, 434, 435, 460, 464, 477]),
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


def fig_skill_correctness():
    w, L, R, T, row = 680, 210, 640, 92, 34
    h = T + row * len(CORRECTNESS) + 70
    out = t(20, 28, "스킬을 읽힌 에이전트의 정답률 (Correctness)", 15, INK, weight="bold")
    out += t(20, 46, "같은 과제를 스킬 없이(회색 점) · 스킬과 함께(색 점) 1회씩 수행. 레포 BENCHMARK.md, 2026-10-08", 12, SUB)
    for v in range(0, 101, 25):
        x = L + (R - L) * v / 100
        out += f'<line x1="{x}" y1="{T - 14}" x2="{x}" y2="{T + row * len(CORRECTNESS) - 10}" stroke="{GRID}"/>'
        out += t(x, T + row * len(CORRECTNESS) + 6, f"{v}%", 11, SUB, "middle")
    for i, (skill, n, agent, base, with_skill) in enumerate(CORRECTNESS):
        y = T + row * i
        if i % 2 == 0:
            out += t(20, y + 4 + row / 2, f"{skill} ({n}개 과제)", 12, INK, weight="bold")
        out += t(L - 10, y + 4, agent, 11, SUB, "end")
        x0, x1, c = L + (R - L) * base / 100, L + (R - L) * with_skill / 100, AGENT_COLOR[agent]
        out += f'<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" stroke="{c}" stroke-width="3" opacity="0.35"/>'
        out += f'<circle cx="{x0}" cy="{y}" r="6" fill="{GRAY}"/><circle cx="{x1}" cy="{y}" r="6" fill="{c}"/>'
        out += t(x0 - 9, y - 9, f"{base:.1f}", 10, SUB, "middle")
        out += t(x1 + (0 if with_skill > 95 else 9), y - 9, f"{with_skill:.1f}", 10, c, "middle", "bold")
    out += t(20, h - 30, "평가 대상은 스킬 문서이지 text-to-SQL 정확도가 아니다. 과제 수가 10~13개, 시도 1회라 한 과제가 7~10%p를 움직인다", 11, SUB)
    return svg(w, h, out, "스킬 유무별 에이전트 정답률")


def fig_star_growth():
    w, L, R, T, B = 680, 70, 640, 70, 250
    top = 12000
    out = t(20, 28, "GitHub 별 수 추이 (2026년)", 15, INK, weight="bold")
    out += t(20, 46, "API = GitHub API 직접 조회 · 리뷰 = andrew.ooo 리뷰의 확인 시점 값", 12, SUB)
    days = {"10-28": 0, "11-03": 6, "11-04": 7, "11-09": 12, "11-11": 14, "11-12": 15}
    X = lambda d: L + (R - L) * days[d] / 15
    Y = lambda v: B - (B - T) * v / top
    for v in range(0, top + 1, 3000):
        out += f'<line x1="{L}" y1="{Y(v)}" x2="{R}" y2="{Y(v)}" stroke="{GRID}"/>'
        out += t(L - 8, Y(v) + 4, f"{v:,}", 11, SUB, "end")
    hx = X(HN[0])
    out += f'<line x1="{hx}" y1="{T}" x2="{hx}" y2="{B}" stroke="{ORANGE}" stroke-dasharray="4 3"/>'
    out += t(hx + 5, T + 12, f"Hacker News {HN[1]}점 · 댓글 {HN[2]}", 11, ORANGE)
    pts = " ".join(f"{X(d)},{Y(v)}" for d, v, _ in STARS)
    out += f'<polyline points="{pts}" fill="none" stroke="{BLUE}" stroke-width="2.5"/>'
    for d, v, src in STARS:
        x, y = X(d), Y(v)
        out += f'<circle cx="{x}" cy="{y}" r="5" fill="{BLUE if src == "API" else "#ffffff"}" stroke="{BLUE}" stroke-width="2"/>'
        out += t(x, B + 18, d, 11, SUB, "middle")
        dx, dy, anc = {"11-11": (-8, 18, "end"), "11-12": (8, 18, "start")}.get(d, (0, -10, "middle"))
        out += t(x + dx, y + dy, f"{v:,}", 11, INK, anc, "bold")
    ratio = STARS[-1][1] / STARS[1][1]
    out += t(20, B + 44, f"11-04에서 11-12까지 8일 동안 {ratio:.1f}배. 관심도 지표이지 품질 지표가 아니다", 11, SUB)
    return svg(w, B + 64, out, "GitHub 별 수 추이")


def fig_open_issues():
    total = sum(len(n) for _, _, n in OPEN_ISSUES)
    w, L, R, T, row = 680, 250, 600, 70, 30
    h = T + row * len(OPEN_ISSUES) + 40
    mx = max(len(n) for _, _, n in OPEN_ISSUES)
    out = t(20, 28, f"열린 이슈 {total}건의 유형 (2026-11-12)", 15, INK, weight="bold")
    out += t(20, 46, "빨강 = README 주장 또는 한국어 데이터에 직접 걸림 · 노랑 = 정확도 버그 · 보라 = 범위 확장 요청", 12, SUB)
    for i, (name, c, nums) in enumerate(OPEN_ISSUES):
        y = T + row * i
        bw = (R - L) * len(nums) / mx
        out += t(L - 10, y + 15, name, 12, INK, "end")
        out += f'<rect x="{L}" y="{y + 3}" width="{bw}" height="{row - 10}" rx="3" fill="{c}" opacity="0.85"/>'
        out += t(L + bw + 6, y + 15, str(len(nums)), 12, c, weight="bold")
    out += t(20, h - 14, "거버넌스 14건 중 13건은 메인테이너가 2026-09에 직접 올린 로드맵 항목이다", 11, SUB)
    return svg(w, h, out, "열린 이슈 유형")


FIGS = [("skill_correctness", fig_skill_correctness), ("star_growth", fig_star_growth),
        ("open_issues", fig_open_issues)]


def main(argv):
    global SHOW_NOTE
    if argv[:1] == ["--inline"]:
        out, prefix, SHOW_NOTE = Path(argv[1]), "", False
    else:
        out, prefix = OUT, "aonto_"
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in FIGS:
        (out / f"{prefix}{name}.svg").write_text(fn(), "utf-8")
        print("wrote", out / f"{prefix}{name}.svg")


if __name__ == "__main__":
    main(sys.argv[1:])
