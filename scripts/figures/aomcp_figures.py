"""auto-ontology-mcp 노트의 그림을 생성한다.

    python scripts/figures/aomcp_figures.py              readings/repos/assets/aomcp_*.svg 를 다시 쓴다
    python scripts/figures/aomcp_figures.py --inline DIR  HTML 보고서용 (접두어 · 생성 메모 없이 DIR/<이름>.svg)

SVG 는 손으로 고치지 않는다. 데이터는 gogoymh/auto-ontology-mcp 커밋 b6e1f92 의
inference.py 상수와 server.py TOOL_GROUPS, README 의 도구 정의 문자 수(근사값)만 쓴다.
"""
import sys
from pathlib import Path

FONT = "-apple-system,'Apple SD Gothic Neo','Noto Sans KR',sans-serif"
INK, SUB, GRID = "#1f2328", "#59636e", "#d8dee4"
GRAY, BLUE, ORANGE, GREEN, YELLOW, RED, PURPLE = (
    "#8b949e", "#2563eb", "#ea580c", "#16a34a", "#ca8a04", "#dc2626", "#9333ea")
SHOW_NOTE = True
NOTE = "생성: python scripts/figures/aomcp_figures.py · 손으로 고치지 말 것"
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "readings" / "repos" / "assets"

# inference.py
AUTO_ACCEPT_CONTAINMENT = 0.98
AUTO_ACCEPT_NAME = 0.75
MIN_CONTAINMENT = 0.6

# server.py TOOL_GROUPS
TOOL_GROUPS = [("answer (항상 켜짐)", 13, BLUE), ("model", 10, GREEN), ("knowledge", 10, ORANGE), ("share", 2, PURPLE)]
# README: "from about 52,000 characters to about 12,000"
TOOL_CHARS = [("전체 도구 (all)", 52000), ("answer 만", 12000)]


def decide(c, name):
    """inference.decide 와 같은 판정 (probe 성공 시)."""
    if c < MIN_CONTAINMENT:
        return "drop", None
    conf = round(0.5 * name + 0.5 * c, 3)
    if c >= AUTO_ACCEPT_CONTAINMENT and name >= AUTO_ACCEPT_NAME:
        return "accepted", conf
    return "proposed", conf


def svg(w, h, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'role="img" aria-label="{title}" style="max-width:100%;height:auto;font-family:{FONT}">'
            f'<title>{title}</title><rect width="{w}" height="{h}" fill="#ffffff"/>{body}'
            + (f'<text x="{w-8}" y="{h-6}" font-size="9" fill="{GRAY}" text-anchor="end">{NOTE}</text>' if SHOW_NOTE else "")
            + '</svg>')


def t(x, y, s, size=12, fill=INK, anchor="start", weight="normal"):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}">{s}</text>')


def fig_fkzones():
    w, h = 800, 520
    L, R, T, B = 90, 560, 80, 440          # 플롯 영역: x = containment 0~1, y = name_score 0~1
    X = lambda c: L + (R - L) * c
    Y = lambda n: B - (B - T) * n
    out = t(20, 28, "미선언 FK 후보 판정: 값 containment 와 이름 점수로 버림 · 검토 제안 · 자동 수락을 가른다", 15, INK, weight="bold")
    out += t(20, 46, f"inference.py 상수 그대로. containment &lt; {MIN_CONTAINMENT} 은 버리고, {AUTO_ACCEPT_CONTAINMENT} 이상이면서 이름 점수 {AUTO_ACCEPT_NAME} 이상만 자동 수락", 11, SUB)
    # 영역
    out += f'<rect x="{X(0)}" y="{T}" width="{X(MIN_CONTAINMENT)-X(0)}" height="{B-T}" fill="{GRAY}" opacity="0.25"/>'
    out += f'<rect x="{X(MIN_CONTAINMENT)}" y="{T}" width="{X(1)-X(MIN_CONTAINMENT)}" height="{B-T}" fill="{YELLOW}" opacity="0.22"/>'
    out += (f'<rect x="{X(AUTO_ACCEPT_CONTAINMENT)}" y="{Y(1)}" width="{X(1)-X(AUTO_ACCEPT_CONTAINMENT)}" '
            f'height="{Y(AUTO_ACCEPT_NAME)-Y(1)}" fill="{GREEN}" opacity="0.85"/>')
    # confidence 등고선: 0.5 n + 0.5 c = k
    for k in (0.7, 0.8):
        c0, c1 = max(MIN_CONTAINMENT, 2 * k - 1), min(1.0, 2 * k)
        if c0 < c1:
            out += (f'<line x1="{X(c0):.1f}" y1="{Y(2*k-c0):.1f}" x2="{X(c1):.1f}" y2="{Y(2*k-c1):.1f}" '
                    f'stroke="{SUB}" stroke-dasharray="4 3" stroke-width="1"/>')
            out += t(X(c0) + 4, Y(2 * k - c0) + 14, f"confidence {k}", 10, SUB)
    # 축
    out += f'<rect x="{L}" y="{T}" width="{R-L}" height="{B-T}" fill="none" stroke="{INK}" stroke-width="1"/>'
    for v in (0, 0.2, 0.4, 0.6, 0.8, 1.0):
        out += t(X(v), B + 16, f"{v:g}", 10, SUB, "middle") + t(L - 8, Y(v) + 4, f"{v:g}", 10, SUB, "end")
    out += t((L + R) / 2, B + 34, "containment (소스 distinct 값 중 대상 키에 있는 비율, 최대 500개 샘플)", 11, INK, "middle")
    out += f'<text x="{L-46}" y="{(T+B)/2}" font-size="11" fill="{INK}" text-anchor="middle" transform="rotate(-90 {L-46} {(T+B)/2})">이름 점수 (name_score)</text>'
    out += t(X(MIN_CONTAINMENT / 2), (T + B) / 2, "버림", 14, INK, "middle", "bold")
    out += t((X(MIN_CONTAINMENT) + X(AUTO_ACCEPT_CONTAINMENT)) / 2, (T + B) / 2 + 40, "검토 제안 (proposed)", 14, INK, "middle", "bold")
    out += t(X(AUTO_ACCEPT_CONTAINMENT) - 6, Y(AUTO_ACCEPT_NAME) + 2, "자동 수락 ↗", 11, "#14532d", "end", "bold")
    # 오른쪽 설명
    x0 = 590
    lines = [("판정 규칙", True),
             (f"containment &lt; {MIN_CONTAINMENT}: 후보 삭제", False),
             (f"c ≥ {AUTO_ACCEPT_CONTAINMENT} 이고 이름 ≥ {AUTO_ACCEPT_NAME}: accepted", False),
             ("그 밖: proposed (모델이 검토)", False),
             ("", False),
             ("confidence", True),
             ("= 0.5 × 이름 점수 + 0.5 × c", False),
             ("probe 실패: 이름 점수 × 0.5", False),
             ("", False),
             ("README 데모 (미선언 FK 9개)", True),
             ("자동 수락 7 · 검토 1 · 연결 못 함 1", False)]
    y = T + 10
    for s, b in lines:
        out += t(x0, y, s, 12 if b else 11, INK if b else SUB, weight="bold" if b else "normal")
        y += 20
    out += t(20, h - 24, "자동 수락 영역은 전체의 한 귀퉁이다. 값이 거의 다 맞고 이름도 맞아야 하므로, 우연히 겹치는 정수 키는 대부분 검토 제안으로 떨어진다", 11, SUB)
    return svg(w, h, out, "auto-ontology-mcp FK 후보 판정 영역")


def fig_tools():
    w = 800
    L, R = 200, 740
    out = t(20, 28, "도구 35개를 4개 그룹으로 나누고, 질의만 하는 클라이언트는 answer 13개만 받는다", 15, INK, weight="bold")
    out += t(20, 46, "server.py TOOL_GROUPS 를 센 값과 README 의 도구 정의 문자 수(근사값). 도구 정의는 클라이언트가 요청마다 보낸다", 11, SUB)
    total = sum(v for _, v, _ in TOOL_GROUPS)
    y = 74
    out += t(20, y + 16, f"그룹별 도구 수 (합 {total})", 12, INK, weight="bold")
    x = L
    for lab, v, c in TOOL_GROUPS:
        bw = (R - L) * v / total
        out += f'<rect x="{x:.1f}" y="{y+2}" width="{bw:.1f}" height="22" fill="{c}" opacity="0.85"/>'
        out += t(x + bw / 2, y + 17, v, 11, "#ffffff", "middle", "bold")
        x += bw
    yy = y + 44
    x = L
    for lab, v, c in TOOL_GROUPS:
        out += f'<rect x="{x}" y="{yy-10}" width="12" height="12" rx="2" fill="{c}"/>' + t(x + 18, yy, lab, 11, SUB)
        x += 40 + len(lab) * 8
    y2 = yy + 34
    out += t(20, y2, "도구 정의 문자 수", 12, INK, weight="bold")
    vmax = TOOL_CHARS[0][1]
    for i, (lab, v) in enumerate(TOOL_CHARS):
        yb = y2 + 14 + i * 32
        bw = (R - 100 - L) * v / vmax
        out += t(L - 10, yb + 15, lab, 11, INK, "end")
        out += f'<rect x="{L}" y="{yb}" width="{bw:.1f}" height="22" fill="{BLUE if i else GRAY}" opacity="0.85"/>'
        out += t(L + bw + 6, yb + 15, f"약 {v:,}자", 11, INK, weight="bold")
    yn = y2 + 14 + len(TOOL_CHARS) * 32 + 22
    out += t(20, yn, f"answer 만 켜면 정의가 약 {TOOL_CHARS[1][1]/TOOL_CHARS[0][1]:.0%} 로 준다. 모델링 · 지식 저장 단계의 handshake 지시문도 함께 빠진다", 11, SUB)
    return svg(w, yn + 26, out, "auto-ontology-mcp 도구 그룹과 도구 정의 크기")


FIGS = [("fkzones", fig_fkzones), ("tools", fig_tools)]


def main(argv):
    global SHOW_NOTE
    if argv[:1] == ["--inline"]:
        out, prefix, SHOW_NOTE = Path(argv[1]), "", False
    else:
        out, prefix = OUT, "aomcp_"
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in FIGS:
        (out / f"{prefix}{name}.svg").write_text(fn(), "utf-8")
        print("wrote", out / f"{prefix}{name}.svg")


if __name__ == "__main__":
    main(sys.argv[1:])
