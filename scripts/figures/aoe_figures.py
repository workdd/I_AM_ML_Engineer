"""auto-ontology-eval 노트의 그림을 생성한다.

    python scripts/figures/aoe_figures.py              readings/repos/assets/aoe_*.svg 를 다시 쓴다
    python scripts/figures/aoe_figures.py --inline DIR  HTML 보고서용 (접두어·생성 메모 없이 DIR/<이름>.svg)

SVG 는 손으로 고치지 않는다. 데이터는 레포 커밋 ec41b2c 의 resources_servers/README.md 와
커밋 e53dbc0 · 13205a2 · 2a46224 본문에서 옮긴 아래 상수만 쓴다.
"""
import sys
from pathlib import Path

FONT = "-apple-system,'Apple SD Gothic Neo','Noto Sans KR',sans-serif"
INK, SUB, GRID = "#1f2328", "#59636e", "#d8dee4"
GRAY, BLUE, GREEN, YELLOW, RED = "#8b949e", "#2563eb", "#16a34a", "#ca8a04", "#dc2626"
SHOW_NOTE = True
NOTE = "생성: python scripts/figures/aoe_figures.py · 손으로 고치지 말 것"
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "readings" / "repos" / "assets"

# 60문항 BIRD, 8 workers 사고 (README · 커밋 e53dbc0)
CALLS, CALLS_429 = 1126, 747
QUESTIONS, EMPTY_SQL = 60, 37
REPORTED_EX = 13.3  # %
HEALTH_THRESHOLD = 0.05  # compare.py

# 숨은 설정 결함: (라벨, 전, 후, 단위, 기준선 또는 None, 기준선 라벨)
HIDDEN = [
    ("nemo-retriever 버전 고정 (BIRD Dev 1,534문항)", 64.3, 76.7, "%", 76.4, "eval_chatbot 76.4%"),
    ("calculation_only · validate_sql_values 추가 (20문항)", 14 / 20 * 100, 16 / 20 * 100, "%", None, ""),
]
HIDDEN_RAW = {"slice": (14, 16, 20), "failed_vector_searches": 10471}


def rate(a, b):
    return a / b


def svg(w, h, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'role="img" aria-label="{title}" style="max-width:100%;height:auto;font-family:{FONT}">'
            f'<title>{title}</title><rect width="{w}" height="{h}" fill="#ffffff"/>{body}'
            + (f'<text x="{w-8}" y="{h-6}" font-size="9" fill="{GRAY}" text-anchor="end">{NOTE}</text>' if SHOW_NOTE else "")
            + '</svg>')


def t(x, y, s, size=12, fill=INK, anchor="start", weight="normal"):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}">{s}</text>')


def fig_ratelimit():
    W, H = 760, 330
    L, R = 200, 700
    b = [t(20, 28, "rate limit 에 막힌 60문항 실행이 '약한 모델' 처럼 보인 과정", 15, weight="bold"),
         t(20, 48, "BIRD 60문항 · 8 workers. 빨강 = 실패 몫, 노란 선 = health gate 5%", 11, SUB)]
    rows = [
        ("HTTP 호출", CALLS_429, CALLS, f"429 응답 {CALLS_429} / {CALLS} ({rate(CALLS_429, CALLS)*100:.0f}%)"),
        ("문항", EMPTY_SQL, QUESTIONS, f"빈 SQL {EMPTY_SQL} / {QUESTIONS} ({rate(EMPTY_SQL, QUESTIONS)*100:.0f}%)"),
    ]
    y = 80
    for name, bad, tot, lab in rows:
        b.append(t(L - 10, y + 17, name, 13, INK, "end"))
        bw = (R - L) * bad / tot
        b.append(f'<rect x="{L}" y="{y}" width="{R-L}" height="26" fill="{GRID}" rx="3"/>')
        b.append(f'<rect x="{L}" y="{y}" width="{bw:.1f}" height="26" fill="{RED}" rx="3"/>')
        b.append(t(L + 34, y + 17, lab, 12, "#ffffff", weight="bold"))
        y += 50
    gx = L + (R - L) * HEALTH_THRESHOLD
    b.append(f'<line x1="{gx:.1f}" y1="{y-50-6}" x2="{gx:.1f}" y2="{y-50+32}" stroke="{YELLOW}" stroke-width="3"/>')
    b.append(t(gx + 4, y - 50 + 44, "health gate 5%: 이 선을 넘으면 compare 가 대표 수치 출력을 거부", 11, YELLOW, weight="bold"))
    y += 30
    b.append(t(L - 10, y + 17, "보고된 EX", 13, INK, "end"))
    bw = (R - L) * REPORTED_EX / 100
    b.append(f'<rect x="{L}" y="{y}" width="{R-L}" height="26" fill="{GRID}" rx="3"/>')
    b.append(f'<rect x="{L}" y="{y}" width="{bw:.1f}" height="26" fill="{GRAY}" rx="3"/>')
    b.append(t(L + bw + 8, y + 17, f"{REPORTED_EX}%: 오류 칸이 비어 있어 모델 성능처럼 읽힘", 12, INK))
    b.append(t(20, H - 28, "재시도 3회 뒤 유효한 종료 상태(unconstructable_sql_response)로 끝나 오류로 집계되지 않았다", 11, SUB))
    return svg(W, H, "".join(b), "rate limit 실행의 실패 분해와 health gate")


def fig_hidden_loss():
    W, H = 760, 300
    L, R, TOP, ROW = 300, 700, 80, 90
    lo, hi = 50, 90

    def xpos(v):
        return L + (R - L) * (v - lo) / (hi - lo)

    b = [t(20, 28, "결과에 흔적 없이 점수를 깎은 설정 결함 두 건", 15, weight="bold"),
         t(20, 48, "회색 = 결함 상태, 초록 = 고친 뒤. 둘 다 오류 없이 점수만 낮게 나왔다", 11, SUB)]
    for v in range(lo, hi + 1, 10):
        x = xpos(v)
        b.append(f'<line x1="{x:.1f}" y1="{TOP-10}" x2="{x:.1f}" y2="{TOP+ROW*len(HIDDEN)-20}" stroke="{GRID}"/>')
        b.append(t(x, TOP + ROW * len(HIDDEN) - 4, f"{v}%", 10, SUB, "middle"))
    for i, (name, before, after, _, ref, reflab) in enumerate(HIDDEN):
        y = TOP + i * ROW + 20
        words = name.split(" (")
        b.append(t(L - 12, y - 2, words[0], 12, INK, "end"))
        if len(words) > 1:
            b.append(t(L - 12, y + 14, "(" + words[1], 11, SUB, "end"))
        b.append(f'<line x1="{xpos(before):.1f}" y1="{y}" x2="{xpos(after):.1f}" y2="{y}" stroke="{GRAY}" stroke-width="2"/>')
        b.append(f'<circle cx="{xpos(before):.1f}" cy="{y}" r="7" fill="{GRAY}"/>')
        b.append(f'<circle cx="{xpos(after):.1f}" cy="{y}" r="7" fill="{GREEN}"/>')
        b.append(t(xpos(before), y - 14, f"{before:.1f}", 11, SUB, "middle"))
        b.append(t(xpos(after), y - 14, f"{after:.1f}", 11, GREEN, "middle", "bold"))
        if ref is not None:
            b.append(f'<line x1="{xpos(ref):.1f}" y1="{y+10}" x2="{xpos(ref):.1f}" y2="{y+22}" stroke="{BLUE}" stroke-width="2"/>')
            b.append(t(xpos(ref), y + 34, reflab, 10, BLUE, "middle"))
    return svg(W, H, "".join(b), "숨은 설정 결함 두 건의 전후 점수")


FIGS = [("ratelimit", fig_ratelimit), ("hidden_loss", fig_hidden_loss)]


def main(argv):
    global SHOW_NOTE
    if argv[:1] == ["--inline"]:
        out, prefix, SHOW_NOTE = Path(argv[1]), "", False
    else:
        out, prefix = OUT, "aoe_"
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in FIGS:
        (out / f"{prefix}{name}.svg").write_text(fn(), "utf-8")
        print("wrote", out / f"{prefix}{name}.svg")


if __name__ == "__main__":
    main(sys.argv[1:])
