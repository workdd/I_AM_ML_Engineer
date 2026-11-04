"""Elephant-Goldfish Model(EGM) 노트의 그림을 생성한다.

    python scripts/figures/egm_figures.py              readings/repos/assets/egm_*.svg 를 다시 쓴다
    python scripts/figures/egm_figures.py --inline DIR  HTML 보고서용 (접두어·생성 메모 없이 DIR/<이름>.svg)

SVG 는 손으로 고치지 않는다. 데이터는 레포 커밋 113a3af 의 skills/egm-review/SKILL.md (Uninterrupted Full Flow),
skills/egm-brainstorm/SKILL.md, TESTING.md (Harness Test Record) 에서 옮긴 아래 상수만 쓴다.
"""
import sys
from pathlib import Path

FONT = "-apple-system,'Apple SD Gothic Neo','Noto Sans KR',sans-serif"
INK, SUB, GRID = "#1f2328", "#59636e", "#d8dee4"
GRAY, BLUE, ORANGE = "#8b949e", "#2563eb", "#ea580c"
GREEN, YELLOW, RED, PURPLE = "#16a34a", "#ca8a04", "#dc2626", "#9333ea"
SHOW_NOTE = True
NOTE = "생성: python scripts/figures/egm_figures.py · 손으로 고치지 말 것"
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "readings" / "repos" / "assets"

# egm-review Uninterrupted Full Flow: (stage, 이름, 범위, reviewer persona 목록)
REVIEW_STAGES = [
    ("5", "Initial Zoom Out", "full file", ["zoom-out"]),
    ("6", "Comprehension Test", "Feature + Detailed", ["comprehension"]),
    ("7", "Critic Review", "FeatureRange", ["design", "structural", "style"]),
    ("8", "Intermediate Zoom Out", "full file", ["zoom-out"]),
    ("9", "Implementation Readiness", "DetailedRange", ["implementation-readiness", "codebase-details", "style"]),
]
SHORT = {"implementation-readiness": "impl-readiness", "codebase-details": "codebase"}
# egm-brainstorm: synthesizer 가 병렬로 띄우는 lens 수
BRAINSTORM_LENSES = 6
# TESTING.md 에 적힌 Codex · Muse 전체 실행의 reviewer 수
RECORDED_TOTAL = 15

# TESTING.md Harness Test Record: (하네스, 버전, 날짜, Install, Brainstorm, Draft, Review)
# 상태 코드: P = Pass, C = Completed(부분 복구 포함), A = Pass after fix, H = Partial, N = Not run
HARNESS = [
    ("Antigravity CLI", "1.4.5", "10-22", "P", "P", "P", "P"),
    ("Claude Code", "2.1.308", "10-22", "P", "P", "P", "A"),
    ("Claude Cowork", "Opus 5.5", "10-22", "P", "C", "C", "H"),
    ("OpenAI Codex", "0.163.0", "10-22", "P", "P", "P", "P"),
    ("Codex CLI (v0.1.0 ZIP)", "0.164.0-a", "10-29", "P", "P", "P", "P"),
    ("ChatGPT Work", "GPT-6.1 Sol", "10-22", "P", "C", "C", "C"),
    ("omp", "v18.12.6", "10-27", "P", "N", "N", "N"),
    ("Muse", "1.4.14", "10-30", "P", "P", "P", "P"),
]
STATUS = {
    "P": ("Pass", GREEN), "A": ("Pass after fix", GREEN), "C": ("Completed (검증 일부 미완)", YELLOW),
    "H": ("Partial", RED), "N": ("Not run", PURPLE),
}


def svg(w, h, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'role="img" aria-label="{title}" style="max-width:100%;height:auto;font-family:{FONT}">'
            f'<title>{title}</title><rect width="{w}" height="{h}" fill="#ffffff"/>{body}'
            + (f'<text x="{w-8}" y="{h-6}" font-size="9" fill="{GRAY}" text-anchor="end">{NOTE}</text>' if SHOW_NOTE else "")
            + '</svg>')


def t(x, y, s, size=12, fill=INK, anchor="start", weight="normal"):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}">{s}</text>')


def review_total():
    return sum(len(r) for *_, r in REVIEW_STAGES)


def fig_reviewer_calls():
    w, L, T, row, unit = 800, 190, 84, 38, 82
    rows = [(f"brainstorm", "6 lens 병렬", "concept", ["lens"] * BRAINSTORM_LENSES, ORANGE)]
    rows += [(f"Stage {s}", n, rng, r, BLUE) for s, n, rng, r in REVIEW_STAGES]
    h = T + row * len(rows) + 92
    out = t(20, 28, "전체 흐름 1회에 뜨는 fresh-context reviewer 수", 15, INK, weight="bold")
    out += t(20, 46, "egm-brainstorm 6 lens + egm-review stage 5~9. 칸 하나 = subagent 1개 (A/B 이중 실행을 켜면 stage 칸이 두 배)", 12, SUB)
    for i, (label, name, rng, reviewers, color) in enumerate(rows):
        y = T + row * i
        out += t(20, y + 15, label, 12, INK, weight="bold")
        out += t(20, y + 29, name, 10, SUB)
        for j, rv in enumerate(reviewers):
            x = L + unit * j
            out += f'<rect x="{x}" y="{y}" width="{unit-4}" height="{row-8}" rx="4" fill="{color}" opacity="0.85"/>'
            short = SHORT.get(rv, rv)
            out += t(x + (unit - 4) / 2, y + 19, short, 9, "#ffffff", "middle")
        out += t(L + unit * len(reviewers) + 8, y + 19, f"{len(reviewers)}개 · {rng}", 11, SUB)
    total = BRAINSTORM_LENSES + review_total()
    y = T + row * len(rows) + 14
    out += t(20, y + 4, f"합계 {BRAINSTORM_LENSES} + {review_total()} = {total}개", 13, INK, weight="bold")
    out += t(20, y + 24, f"TESTING.md 의 Codex · Muse 전체 실행 기록 \"{RECORDED_TOTAL} native reviewers\" 와 같다. "
             f"A/B 이중 실행 시 review 쪽은 최대 {review_total() * 2}개", 11, SUB)
    out += t(20, y + 42, "stage 순서는 고정(5→9)이고 같은 stage 안의 reviewer 만 병렬로 돈다. Stage 6 은 synthesizer 없이 직접 띄운다", 11, SUB)
    return svg(w, h, out, "EGM 전체 흐름의 reviewer subagent 수")


def fig_harness_matrix():
    cols = ["Install", "Brainstorm", "Draft", "Review"]
    w, L, T, row, cw = 720, 300, 88, 30, 96
    h = T + row * len(HARNESS) + 96
    out = t(20, 28, "하네스별 end-to-end 검증 기록", 15, INK, weight="bold")
    out += t(20, 46, "TESTING.md Harness Test Record (fixture 설계 1건 기준). 리뷰 품질이 아니라 실행 성공 여부만 기록한다", 12, SUB)
    for k, c in enumerate(cols):
        out += t(L + cw * k + cw / 2, T - 12, c, 12, INK, "middle", "bold")
    for i, (name, ver, date, *cells) in enumerate(HARNESS):
        y = T + row * i
        if i % 2 == 0:
            out += f'<rect x="16" y="{y-2}" width="{w-32}" height="{row}" fill="#f6f8fa"/>'
        out += t(20, y + 17, name, 12, INK, weight="bold")
        out += t(L - 10, y + 17, f"{ver} · {date}", 10, SUB, "end")
        for k, code in enumerate(cells):
            label, color = STATUS[code]
            x = L + cw * k + 6
            out += f'<rect x="{x}" y="{y+3}" width="{cw-12}" height="{row-10}" rx="4" fill="{color}" opacity="0.85"/>'
            short = {"P": "Pass", "A": "Pass(수정 후)", "C": "Completed", "H": "Partial 5–7", "N": "Not run"}[code]
            out += t(x + (cw - 12) / 2, y + 17, short, 10, "#ffffff", "middle", "bold")
    y = T + row * len(HARNESS) + 16
    lx = 20
    for code in ["P", "C", "H", "N"]:
        label, color = STATUS[code]
        out += f'<rect x="{lx}" y="{y}" width="12" height="12" rx="2" fill="{color}"/>'
        out += t(lx + 18, y + 10, label, 11, SUB)
        lx += 40 + len(label) * 7
    out += t(20, y + 36, "Completed: 산출물은 나왔지만 child 도구 기록 · 격리 확인이 남음. Partial: 사용량 한도로 stage 8 에서 중단", 11, SUB)
    out += t(20, y + 54, "Antigravity · Codex 는 지시 파일(AGENTS.md · GEMINI.md) 주입 이슈 #237 이 열려 있어 instruction-file 격리는 미검증", 11, SUB)
    return svg(w, h, out, "EGM 하네스별 검증 상태")


FIGS = [("reviewer_calls", fig_reviewer_calls), ("harness_matrix", fig_harness_matrix)]


def main(argv):
    global SHOW_NOTE
    if argv[:1] == ["--inline"]:
        out, prefix, SHOW_NOTE = Path(argv[1]), "", False
    else:
        out, prefix = OUT, "egm_"
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in FIGS:
        (out / f"{prefix}{name}.svg").write_text(fn(), "utf-8")
        print("wrote", out / f"{prefix}{name}.svg")


if __name__ == "__main__":
    main(sys.argv[1:])
