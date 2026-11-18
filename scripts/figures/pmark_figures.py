"""Palantir Foundry Markings 문서 노트의 그림을 생성한다.

    python scripts/figures/pmark_figures.py              readings/blogs/assets/pmark_*.svg 를 다시 쓴다
    python scripts/figures/pmark_figures.py --inline DIR  HTML 보고서용 (접두어, 생성 메모 없이 DIR/<이름>.svg)

SVG 는 손으로 고치지 않는다. 판정 규칙은 원문 문서의 문장을 옮긴 decide() 하나이고,
사용자와 자원은 규칙을 보여 주려고 만든 가상 사례다.
"""
import sys
from pathlib import Path

FONT = "-apple-system,'Apple SD Gothic Neo','Noto Sans KR',sans-serif"
INK, SUB, GRID = "#1f2328", "#59636e", "#d8dee4"
GRAY = "#8b949e"
SHOW_NOTE = True
NOTE = "생성: python scripts/figures/pmark_figures.py, 손으로 고치지 말 것"
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "readings" / "blogs" / "assets"

# 판정 결과 4가지와 색 (green 정상, yellow 부분, red 차단, gray 역할 없음)
HIDDEN, NOROLE, META, READ = "발견 불가", "접근 불가", "메타데이터만", "데이터 열람"
COLOR = {READ: ("#dcfce7", "#166534"), META: ("#fef9c3", "#854d0e"),
         HIDDEN: ("#fee2e2", "#991b1b"), NOROLE: ("#eef0f2", "#424a53")}

# 가상 자원: (이름, Project, 파일 마킹, 데이터 마킹, 설명)
# salary_raw 는 raw 폴더의 Raw Data 마킹을 계층으로 상속하고 PII 를 직접 단다.
# salary_clean 은 transform 에서 Raw Data 전파를 끊고 PII 만 data marking 으로 이어받는다.
# salary_masked 는 PII 를 해싱한 뒤 PII 전파도 끊는다.
RESOURCES = [
    ("salary_raw", "재무", {"Raw Data", "PII"}, set(), "Raw Data(폴더 상속) + PII"),
    ("salary_clean", "재무", set(), {"PII"}, "상위에서 PII 상속"),
    ("salary_masked", "재무", set(), set(), "PII 해싱 후 전파 중단"),
    ("hr_notes", "인사", {"PII"}, set(), "PII 직접 적용"),
]

# 가상 사용자: (이름, 역할을 가진 Project, 자격을 가진 마킹)
USERS = [
    ("재무 분석가", {"재무"}, {"PII"}),
    ("재무 신입 (교육 전)", {"재무"}, set()),
    ("인사 담당", {"인사"}, {"PII"}),
    ("데이터 엔지니어", {"재무"}, {"PII", "Raw Data"}),
]


def decide(projects, eligible, res):
    """원문 규칙: 파일 마킹 미충족이면 자원을 발견하지 못하고, 역할이 없으면 접근하지 못하며,
    파일 요건만 채우고 상속된 데이터 마킹을 못 채우면 메타데이터만 본다. 마킹은 AND 다."""
    _, project, file_m, data_m, _ = res
    if not file_m <= eligible:
        return HIDDEN
    if project not in projects:
        return NOROLE
    if not data_m <= eligible:
        return META
    return READ


def matrix():
    return [[decide(p, e, r) for r in RESOURCES] for _, p, e in USERS]


def svg(w, h, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'role="img" aria-label="{title}" style="max-width:100%;height:auto;font-family:{FONT}">'
            f'<title>{title}</title><rect width="{w}" height="{h}" fill="#ffffff"/>{body}'
            + (f'<text x="{w-8}" y="{h-6}" font-size="9" fill="{GRAY}" text-anchor="end">{NOTE}</text>' if SHOW_NOTE else "")
            + '</svg>')


def t(x, y, s, size=12, fill=INK, anchor="start", weight="normal"):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}">{s}</text>')


def fig_decision():
    w, h = 760, 430
    L, T, cw, ch, lw = 170, 120, 140, 50, 150
    out = t(20, 28, "같은 데이터 계보에서도 사용자마다 결과가 네 가지로 갈린다", 15, INK, weight="bold")
    out += t(20, 46, "원문 규칙(마킹 AND, 파일 마킹 → 역할 → 데이터 마킹 순)을 가상 사용자 4명과 자원 4개에 적용", 12, SUB)
    for j, (name, project, file_m, data_m, desc) in enumerate(RESOURCES):
        cx = L + cw * j + cw / 2
        out += t(cx, T - 38, name, 13, INK, "middle", "bold")
        out += t(cx, T - 22, f"{project} Project", 11, SUB, "middle")
        out += t(cx, T - 8, desc, 10.5, SUB, "middle")
    for i, (uname, projects, elig) in enumerate(USERS):
        y = T + ch * i
        out += t(20, y + 22, uname, 12.5, INK, weight="bold")
        tag = "역할: " + ", ".join(sorted(projects)) + " / 자격: " + (", ".join(sorted(elig)) or "없음")
        out += t(20, y + 38, tag, 10.5, SUB)
        for j, res in enumerate(RESOURCES):
            r = decide(projects, elig, res)
            bg, fg = COLOR[r]
            x = L + cw * j
            out += f'<rect x="{x + 4}" y="{y + 4}" width="{cw - 8}" height="{ch - 8}" rx="6" fill="{bg}"/>'
            out += t(x + cw / 2, y + 30, r, 12.5, fg, "middle", "bold")
    ly = T + ch * len(USERS) + 28
    legend = [(HIDDEN, "파일 마킹 미충족: 검색, 폴더 보기에서 숨김"),
              (NOROLE, "마킹은 충족, 역할 없음: 자격과 접근은 별개"),
              (META, "파일 요건 충족, 상속된 데이터 마킹 미충족"),
              (READ, "역할과 모든 마킹 충족")]
    for k, (r, d) in enumerate(legend):
        x, y = 20 + (k % 2) * 370, ly + (k // 2) * 24
        bg, fg = COLOR[r]
        out += f'<rect x="{x}" y="{y - 12}" width="86" height="18" rx="4" fill="{bg}"/>'
        out += t(x + 43, y + 1, r, 11, fg, "middle", "bold")
        out += t(x + 94, y + 1, d, 11, SUB)
    out += t(20, h - 22, "사용자와 자원은 규칙 설명용 가상 사례이며 원문에 없다", 10.5, GRAY)
    return svg(w, h, out, "마킹과 역할에 따른 접근 판정 매트릭스")


FIGS = [("decision", fig_decision)]


def main(argv):
    global SHOW_NOTE
    if argv[:1] == ["--inline"]:
        out, prefix, SHOW_NOTE = Path(argv[1]), "", False
    else:
        out, prefix = OUT, "pmark_"
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in FIGS:
        (out / f"{prefix}{name}.svg").write_text(fn(), "utf-8")
        print("wrote", out / f"{prefix}{name}.svg")


if __name__ == "__main__":
    main(sys.argv[1:])
