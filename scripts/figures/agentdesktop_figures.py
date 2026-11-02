"""vercel-labs/agent-desktop 노트의 그림을 생성한다.

    python scripts/figures/agentdesktop_figures.py              readings/repos/assets/agentdesktop_*.svg 를 다시 쓴다
    python scripts/figures/agentdesktop_figures.py --inline DIR  HTML 보고서용 (접두어·생성 메모 없이 DIR/<이름>.svg)

SVG 는 손으로 고치지 않는다. 크레이트 줄 수는 커밋 9eb9b29 에서 crates/<이름> 아래 .rs 파일을 센 값이고
(테스트 = 경로에 tests/ 또는 파일명에 test 포함), 게이트 임계값은 skills/jev-desktop/SKILL.md 의 문장 규칙이다.
원문에는 두 내용을 그린 그림이 없다.
"""
import sys
from pathlib import Path

FONT = "-apple-system,'Apple SD Gothic Neo','Noto Sans KR',sans-serif"
INK, SUB, GRID = "#1f2328", "#59636e", "#d8dee4"
GRAY, BLUE, ORANGE, GREEN, RED, YELLOW = "#8b949e", "#2563eb", "#ea580c", "#16a34a", "#dc2626", "#ca8a04"
SHOW_NOTE = True
NOTE = "생성: python scripts/figures/agentdesktop_figures.py · 손으로 고치지 말 것"
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "readings" / "repos" / "assets"

# (크레이트, 역할, 전체 줄, 테스트 줄, .rs 파일 수)
CRATES = [
    ("linux", "AT-SPI 어댑터", 123_534, 64_513, 1027),
    ("windows", "UI Automation 어댑터", 82_470, 41_458, 446),
    ("core", "PlatformAdapter trait · 명령", 79_297, 47_490, 638),
    ("macos", "AX API 어댑터", 43_144, 8_991, 277),
    ("ffi", "C ABI 라이브러리", 15_107, 5_014, 190),
]

# jev-desktop 게이트: 실행 없음 하한, 일반 단계 기준, destructive 기준, destructive 판정 위험 점수
NOTHING_BELOW, ORDINARY_BAR, DESTRUCTIVE_BAR, DESTRUCTIVE_RISK = 0.55, 0.70, 0.90, 0.50


def test_share(total, test):
    return test / total * 100


def gate(confidence, risk):
    """run.mjs 가 한 단계를 어떻게 다루는지. risk 는 평가될 때만 의미가 있다."""
    if confidence < NOTHING_BELOW:
        return "none"
    if confidence < ORDINARY_BAR:
        return "stop"
    if confidence >= DESTRUCTIVE_BAR:
        return "run_unrated"             # 위험 평가 호출 없이 실행
    return "stop" if risk >= DESTRUCTIVE_RISK else "run"


def svg(w, h, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'role="img" aria-label="{title}" style="max-width:100%;height:auto;font-family:{FONT}">'
            f'<title>{title}</title><rect width="{w}" height="{h}" fill="#ffffff"/>{body}'
            + (f'<text x="{w-8}" y="{h-6}" font-size="9" fill="{GRAY}" text-anchor="end">{NOTE}</text>' if SHOW_NOTE else "")
            + '</svg>')


def t(x, y, s, size=12, fill=INK, anchor="start", weight="normal"):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}">{s}</text>')


def fig_crates():
    w, h = 760, 410
    out = t(20, 28, "크레이트별 Rust 코드 줄 수: Linux 어댑터가 가장 크고, 절반 이상이 테스트다", 14.5, INK, weight="bold")
    out += t(20, 46, "커밋 9eb9b29 의 .rs 파일 집계. 진한 막대 = 본체, 옅은 막대 = 테스트(경로에 tests/ 또는 파일명에 test)", 11.5, SUB)
    L, R, T, row = 210, 640, 76, 48
    vmax = 130_000
    X = lambda v: L + (R - L) * v / vmax
    bottom = T + row * len(CRATES)
    for v in range(0, vmax + 1, 25_000):
        out += f'<line x1="{X(v):.1f}" y1="{T-8}" x2="{X(v):.1f}" y2="{bottom}" stroke="{GRID}"/>'
        out += t(X(v), bottom + 16, f"{v // 1000}k", 11, SUB, "middle")
    for i, (name, role, total, test, files) in enumerate(CRATES):
        y = T + row * i
        body = total - test
        out += t(L - 10, y + 13, name, 12.5, INK, "end", "bold")
        out += t(L - 10, y + 28, role, 10.5, SUB, "end")
        out += f'<rect x="{L}" y="{y+4}" width="{X(body)-L:.1f}" height="22" fill="{BLUE}"/>'
        out += f'<rect x="{X(body):.1f}" y="{y+4}" width="{X(total)-X(body):.1f}" height="22" fill="{BLUE}" fill-opacity="0.3"/>'
        out += t(X(total) + 6, y + 14, f"{total:,}줄", 11, INK, weight="bold")
        out += t(X(total) + 6, y + 27, f"테스트 {test_share(total, test):.0f}% · 파일 {files}", 10.5, SUB)
    all_total = sum(c[2] for c in CRATES)
    adapters = sum(c[2] for c in CRATES if c[0] in ("linux", "windows", "macos"))
    out += t(20, h - 40, f"다섯 크레이트 합 {all_total:,}줄 중 OS 어댑터 셋이 {adapters / all_total * 100:.0f}%. "
             "core 는 어느 플랫폼도 import 하지 않는다(CI 가 cargo tree 로 확인)", 11, SUB)
    out += t(20, h - 22, "macOS 의 crate 안 테스트 비율이 낮은 것은 접근성 권한이 필요한 통합 테스트가 따로 돌기 때문일 수 있다(미확인)", 11, SUB)
    return svg(w, h, out, "agent-desktop 크레이트별 코드 줄 수")


def fig_gate():
    w, h = 760, 430
    out = t(20, 28, "JEV 도우미(run.mjs)가 한 단계를 실행할지 정하는 규칙", 14.5, INK, weight="bold")
    out += t(20, 46, "가로 = 모델이 고른 대상의 신뢰도, 세로 = 두 번째 모델 호출이 매긴 위험 점수. 원문 문장 규칙을 평면에 펼침", 11.5, SUB)
    L, R, T, B = 90, 520, 74, 334
    X = lambda v: L + (R - L) * (v - 0.4) / 0.6
    Y = lambda v: B - (B - T) * v
    fills = {"none": ("#f3f4f6", GRAY), "stop": ("#fee2e2", RED), "run": ("#dcfce7", GREEN), "run_unrated": ("#fef3c7", YELLOW)}
    # 경계가 축에 평행하므로 영역은 직사각형 다섯 개다. 색은 gate() 가 영역 중심에서 정한다
    regions = [(0.4, NOTHING_BELOW, 0, 1), (NOTHING_BELOW, ORDINARY_BAR, 0, 1),
               (ORDINARY_BAR, DESTRUCTIVE_BAR, 0, DESTRUCTIVE_RISK), (ORDINARY_BAR, DESTRUCTIVE_BAR, DESTRUCTIVE_RISK, 1),
               (DESTRUCTIVE_BAR, 1.0, 0, 1)]
    for c0, c1, r0, r1 in regions:
        fill = fills[gate((c0 + c1) / 2, (r0 + r1) / 2)][0]
        out += f'<rect x="{X(c0):.1f}" y="{Y(r1):.1f}" width="{X(c1)-X(c0):.1f}" height="{Y(r0)-Y(r1):.1f}" fill="{fill}"/>'
    for v in (0.4, 0.55, 0.7, 0.9, 1.0):
        out += f'<line x1="{X(v):.1f}" y1="{T}" x2="{X(v):.1f}" y2="{B}" stroke="{GRAY if v in (0.4, 1.0) else INK}" stroke-dasharray="{"" if v in (0.4, 1.0) else "4 3"}"/>'
        out += t(X(v), B + 16, f"{v:.2f}", 11, SUB, "middle")
    for v in (0, 0.5, 1.0):
        out += t(L - 8, Y(v) + 4, f"{v:.1f}", 11, SUB, "end")
    out += f'<line x1="{X(ORDINARY_BAR):.1f}" y1="{Y(DESTRUCTIVE_RISK):.1f}" x2="{X(DESTRUCTIVE_BAR):.1f}" y2="{Y(DESTRUCTIVE_RISK):.1f}" stroke="{INK}" stroke-dasharray="4 3"/>'
    out += f'<rect x="{L}" y="{T}" width="{R-L}" height="{B-T}" fill="none" stroke="{GRAY}"/>'
    out += t((L + R) / 2, B + 34, "대상 신뢰도", 11, SUB, "middle")
    out += t(L - 60, T - 10, "위험 점수", 11, SUB)
    out += t(X(0.475), Y(0.5), "실행 없음", 11.5, GRAY, "middle", "bold")
    out += t(X(0.625), Y(0.5), "멈춤", 11.5, RED, "middle", "bold")
    out += t(X(0.625), Y(0.5) + 15, "후보 알림", 10, RED, "middle")
    out += t(X(0.8), Y(0.25), "실행", 11.5, GREEN, "middle", "bold")
    out += t(X(0.8), Y(0.75), "destructive", 11, RED, "middle", "bold")
    out += t(X(0.8), Y(0.75) + 15, "→ 멈춤", 10.5, RED, "middle")
    out += t(X(0.95), Y(0.55), "평가 없이", 10.5, YELLOW, "middle", "bold")
    out += t(X(0.95), Y(0.55) + 14, "실행", 10.5, YELLOW, "middle", "bold")
    # 범례와 해설
    lx = 560
    items = [("run", "실행"), ("run_unrated", "위험 평가 호출 없이 실행"), ("stop", "멈추고 사람에게 넘김"), ("none", "실행하지 않음")]
    for i, (k, label) in enumerate(items):
        y = T + 6 + 24 * i
        out += f'<rect x="{lx}" y="{y}" width="14" height="14" fill="{fills[k][0]}" stroke="{fills[k][1]}"/>'
        out += t(lx + 22, y + 11, label, 11, INK)
    notes = [
        f"일반 단계 기준 {ORDINARY_BAR:.2f}",
        f"위험 {DESTRUCTIVE_RISK:.2f} 이상 = destructive,",
        f"  기준 {DESTRUCTIVE_BAR:.2f}",
        f"{NOTHING_BELOW:.2f} 미만은 실행 없음",
        "위험 점수가 오지 않으면",
        "  destructive 로 간주",
    ]
    for i, s in enumerate(notes):
        out += t(lx, T + 130 + 17 * i, s, 10.5, SUB)
    out += t(20, h - 40, f"노란 띠: 신뢰도 {DESTRUCTIVE_BAR:.2f} 이상이면 위험을 묻지 않으므로, 위험한 단계도 그대로 실행된다(원문이 밝힌 동작)", 11, SUB)
    out += t(20, h - 22, "--confirm-destructive 를 주면 destructive 로 평가된 모든 단계에서 확인을 받는다. act.mjs 는 한 요청 안에서 위험을 함께 받는다", 11, SUB)
    return svg(w, h, out, "JEV 도우미 신뢰도와 위험 게이트")


FIGS = [("crates", fig_crates), ("gate", fig_gate)]


def main(argv):
    global SHOW_NOTE
    if argv[:1] == ["--inline"]:
        out, prefix, SHOW_NOTE = Path(argv[1]), "", False
    else:
        out, prefix = OUT, "agentdesktop_"
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in FIGS:
        (out / f"{prefix}{name}.svg").write_text(fn(), "utf-8")
        print("wrote", out / f"{prefix}{name}.svg")


if __name__ == "__main__":
    main(sys.argv[1:])
