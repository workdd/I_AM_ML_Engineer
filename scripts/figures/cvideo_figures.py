"""bradautomates/claude-video (/watch) 노트의 그림을 생성한다.

    python scripts/figures/cvideo_figures.py              readings/repos/assets/cvideo_*.svg 를 다시 쓴다
    python scripts/figures/cvideo_figures.py --inline DIR  HTML 보고서용 (접두어·생성 메모 없이 DIR/<이름>.svg)

SVG 는 손으로 고치지 않는다. 규칙은 레포 skills/watch/scripts/frames.py 의 auto_fps · auto_fps_focus ·
_clamp_fps (v0.3.2, 커밋 03ceb42) 를 그대로 옮긴 아래 함수만 쓴다.
"""
import sys
from pathlib import Path

FONT = "-apple-system,'Apple SD Gothic Neo','Noto Sans KR',sans-serif"
INK, SUB, GRID = "#1f2328", "#59636e", "#d8dee4"
GRAY, BLUE, ORANGE = "#8b949e", "#2563eb", "#ea580c"
SHOW_NOTE = True
NOTE = "생성: python scripts/figures/cvideo_figures.py · 손으로 고치지 말 것"
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "readings" / "repos" / "assets"

MAX_FPS = 2.0
CAPS = {"efficient": 50, "balanced": 100}


def _clamp(fps, d, cap):
    fps = min(fps, MAX_FPS)
    return min(cap, max(1, int(round(fps * d))))


def budget_full(d, cap=100):
    if d <= 30:
        t = min(cap, max(12, int(round(d))))
    elif d <= 60:
        t = min(cap, 40)
    elif d <= 180:
        t = min(cap, 60)
    elif d <= 600:
        t = min(cap, 80)
    else:
        t = cap
    return _clamp(t / d, d, cap)


def budget_focus(d, cap=100):
    if d <= 5:
        t = min(cap, max(10, int(round(d * 6))))
    elif d <= 15:
        t = min(cap, max(30, int(round(d * 4))))
    elif d <= 30:
        t = min(cap, 60)
    elif d <= 60:
        t = min(cap, 80)
    else:
        t = cap
    return _clamp(t / d, d, cap)


DURATIONS = [3, 5, 10, 15, 20, 30, 45, 60, 120, 180, 300, 600, 900, 1800, 3600]


def svg(w, h, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'role="img" aria-label="{title}" style="max-width:100%;height:auto;font-family:{FONT}">'
            f'<title>{title}</title><rect width="{w}" height="{h}" fill="#ffffff"/>{body}'
            + (f'<text x="{w-8}" y="{h-6}" font-size="9" fill="{GRAY}" text-anchor="end">{NOTE}</text>' if SHOW_NOTE else "")
            + '</svg>')


def t(x, y, s, size=12, fill=INK, anchor="start", weight="normal"):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}">{s}</text>')


def fig_budget():
    import math
    w, h, L, R, T, B = 680, 380, 60, 640, 70, 300
    lo, hi = math.log10(DURATIONS[0]), math.log10(DURATIONS[-1])
    X = lambda d: L + (R - L) * (math.log10(d) - lo) / (hi - lo)
    Y = lambda v: B - (B - T) * v / 110
    out = t(20, 28, "영상 길이별 프레임 예산 (balanced, 상한 100장)", 15, INK, weight="bold")
    out += t(20, 46, "전체 영상은 길수록 장당 간격이 벌어지고, --start/--end 구간은 같은 길이에서 더 촘촘하다", 12, SUB)
    for v in (0, 20, 40, 60, 80, 100):
        out += f'<line x1="{L}" y1="{Y(v)}" x2="{R}" y2="{Y(v)}" stroke="{GRID}"/>' + t(L - 8, Y(v) + 4, v, 11, SUB, "end")
    for d, lab in ((5, "5초"), (30, "30초"), (60, "1분"), (180, "3분"), (600, "10분"), (3600, "60분")):
        out += t(X(d), B + 18, lab, 11, SUB, "middle")
    out += f'<line x1="{X(600)}" y1="{T - 6}" x2="{X(600)}" y2="{B}" stroke="{GRAY}" stroke-dasharray="4 3"/>'
    out += t(X(600) + 5, T + 6, "10분 초과: 범위 지정 권고 경고", 10.5, SUB)
    for fn, c, name in ((budget_full, BLUE, "전체 영상"), (budget_focus, ORANGE, "구간 지정")):
        pts = " ".join(f"{X(d):.1f},{Y(fn(d)):.1f}" for d in DURATIONS)
        out += f'<polyline points="{pts}" fill="none" stroke="{c}" stroke-width="2.2"/>'
        out += "".join(f'<circle cx="{X(d):.1f}" cy="{Y(fn(d)):.1f}" r="3" fill="{c}"/>' for d in DURATIONS)
    out += f'<rect x="{L + 10}" y="{T + 4}" width="12" height="3" fill="{BLUE}"/>' + t(L + 28, T + 9, "전체 영상 (auto_fps)", 11.5, INK)
    out += f'<rect x="{L + 10}" y="{T + 22}" width="12" height="3" fill="{ORANGE}"/>' + t(L + 28, T + 27, "구간 지정 (auto_fps_focus)", 11.5, INK)
    out += t((L + R) / 2, B + 40, "영상 또는 구간 길이 (로그 축)", 11, SUB, "middle")
    out += t(20, h - 26, "y는 고를 프레임 수. 상한 2fps가 짧은 구간을 깎는다 (5초 구간은 10장)", 11, SUB)
    return svg(w, h, out, "영상 길이별 프레임 예산")


FIGS = [("budget", fig_budget)]


def main(argv):
    global SHOW_NOTE
    if argv[:1] == ["--inline"]:
        out, prefix, SHOW_NOTE = Path(argv[1]), "", False
    else:
        out, prefix = OUT, "cvideo_"
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in FIGS:
        (out / f"{prefix}{name}.svg").write_text(fn(), "utf-8")
        print("wrote", out / f"{prefix}{name}.svg")


if __name__ == "__main__":
    main(sys.argv[1:])
