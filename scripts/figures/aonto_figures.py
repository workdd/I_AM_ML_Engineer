"""NVIDIA Auto Ontology 노트의 그림을 생성한다.

    python scripts/figures/aonto_figures.py              readings/repos/assets/aonto_*.svg 를 다시 쓴다
    python scripts/figures/aonto_figures.py --inline DIR  HTML 보고서용 (접두어·생성 메모 없이 DIR/<이름>.svg)

SVG 는 손으로 고치지 않는다. 수치는 레포 skills/*/BENCHMARK.md (2026-10-08 평가) 에서 옮긴 아래 데이터 절만 쓴다.
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


FIGS = [("skill_correctness", fig_skill_correctness)]


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
