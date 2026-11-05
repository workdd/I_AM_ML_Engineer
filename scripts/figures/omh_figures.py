"""oh-my-hermes 노트의 그림을 생성한다.

    python scripts/figures/omh_figures.py              readings/repos/assets/omh_*.svg 를 다시 쓴다
    python scripts/figures/omh_figures.py --inline DIR  HTML 보고서용 (접두어·생성 메모 없이 DIR/<이름>.svg)

SVG 는 손으로 고치지 않는다. 데이터는 레포 커밋 83c2f61 의 benchmarks/live-model-tools/v1/README.md 와
src/coding/request_complexity.py 에서 옮긴 아래 상수만 쓴다.
"""
import math
import sys
from pathlib import Path

FONT = "-apple-system,'Apple SD Gothic Neo','Noto Sans KR',sans-serif"
INK, SUB, GRID = "#1f2328", "#59636e", "#d8dee4"
GRAY, BLUE, ORANGE, GREEN, YELLOW, RED, PURPLE = (
    "#8b949e", "#2563eb", "#ea580c", "#16a34a", "#ca8a04", "#dc2626", "#9333ea")
SHOW_NOTE = True
NOTE = "생성: python scripts/figures/omh_figures.py · 손으로 고치지 말 것"
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "readings" / "repos" / "assets"

N_TASKS = 30
CEILING = 18  # read · lsp 템플릿은 어떤 모델도 0 이라 이 코퍼스의 상한

# 2026-09-05 라우팅 레인: (라벨, 종류, 통과, list-price 비용 $, 총 분)
# 종류: single = 단일 모델 실행, routed = 단일 모델 행을 문항별로 조합한 결과
LANES = [
    ("Hermes 단독 · Sol medium", "single", 18, 0.96, 24.8),
    ("Astra xhigh", "single", 18, 4.29, 23.3),
    ("Astra xhigh + 보정", "single", 18, 4.30, 22.5),
    ("Astra low", "single", 18, 4.64, 22.3),
    ("GLM ultrafast low (quick)", "single", 14, 0.06, 2.1),
    ("Kimi ultrafast medium", "single", 11, 0.36, 7.6),
    ("라우팅 · 신호 추가 후 · 출시 체인", "routed", 12, 0.10, 2.5),
    ("라우팅 · 신호 추가 후 · Sol 첫 모델", "routed", 18, 0.18, 5.2),
]
# 신호 추가 전 라우팅(30개 모두 quick)은 GLM 단독 행과 같은 값이라 점을 따로 찍지 않는다
ROUTED_BEFORE_SAME_AS = "GLM ultrafast low (quick)"

# request_complexity.py 신호 가중치 (이름, 가중치, 설명)
SIGNALS = [
    ("exhaustive_search", 4, "every · exhaustive"),
    ("architecture_keywords", 3, "refactor · migrate · redesign"),
    ("impact_system_wide", 3, "everywhere"),
    ("risk_keywords", 2, "security · concurrency · rollback"),
    ("debugging_keywords", 2, "regression · flaky · bisect"),
    ("fanout_intent", 2, "parallelize · fanout"),
    ("routed_skill_class", 2, "coding 위임 +2 · 계획·운영·학습 +1"),
    ("message_length", 2, "200자 +1 · 600자 +2"),
    ("simple_request", -2, "one-liner · typo · rename"),
]
TIERS = [("light", 0, "quick · low"), ("standard", 4, "unspecified-high · medium"), ("deep", 8, "deep · high")]

# gpt-6-astra 보정 4-arm paired bootstrap: (쌍, 평균 Δ 토큰, CI 하한, CI 상한)
CALIB = [
    ("family − baseline", -1251, -3763, 1300),
    ("optimized − baseline", 4168, -324, 9100),
    ("optimized − family", 5419, 1444, 10150),
    ("revised − family", 629, -1109, 2445),
    ("revised − optimized", -4790, -9336, -1075),
]
CALIB_PASSED = {"baseline": 18, "family": 18, "optimized": 18, "revised": 18}


def svg(w, h, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'role="img" aria-label="{title}" style="max-width:100%;height:auto;font-family:{FONT}">'
            f'<title>{title}</title><rect width="{w}" height="{h}" fill="#ffffff"/>{body}'
            + (f'<text x="{w-8}" y="{h-6}" font-size="9" fill="{GRAY}" text-anchor="end">{NOTE}</text>' if SHOW_NOTE else "")
            + '</svg>')


def t(x, y, s, size=12, fill=INK, anchor="start", weight="normal"):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}">{s}</text>')


def cost_ratio(a, b):
    """레인 a 비용이 레인 b 의 몇 배인지."""
    d = {name: c for name, _, _, c, _ in LANES}
    return d[a] / d[b]


def fig_routing_lanes():
    w, h = 820, 500
    L, R, T, B = 80, w - 40, 80, h - 110
    lo, hi = math.log10(0.03), math.log10(10)
    pmin, pmax = 8, 20

    def xpos(c):
        return L + (math.log10(c) - lo) / (hi - lo) * (R - L)

    def ypos(p):
        return B - (p - pmin) / (pmax - pmin) * (B - T)

    out = t(20, 28, "같은 30문항에서 레인별 비용과 통과 수", 15, INK, weight="bold")
    out += t(20, 46, "2026-09-05 저자 측정. 가로 = list-price 비용(로그), 세로 = 통과 문항 수. 빨간 점선 = 천장 18 (read · lsp 템플릿은 모든 모델 0)", 12, SUB)
    for c in (0.03, 0.1, 0.3, 1, 3, 10):
        x = xpos(c)
        out += f'<line x1="{x:.1f}" y1="{T}" x2="{x:.1f}" y2="{B}" stroke="{GRID}"/>'
        out += t(x, B + 18, f"${c:g}", 11, SUB, "middle")
    for p in range(pmin, pmax + 1, 2):
        y = ypos(p)
        out += f'<line x1="{L}" y1="{y:.1f}" x2="{R}" y2="{y:.1f}" stroke="{GRID}"/>'
        out += t(L - 8, y + 4, str(p), 11, SUB, "end")
    yc = ypos(CEILING)
    out += f'<line x1="{L}" y1="{yc:.1f}" x2="{R}" y2="{yc:.1f}" stroke="{RED}" stroke-dasharray="5 4"/>'
    out += t(L + 4, yc - 6, f"천장 {CEILING}", 11, RED)
    out += t(L - 50, T - 10, "통과", 11, SUB)
    out += t((L + R) / 2, B + 38, "list-price 비용 ($, 로그 눈금)", 11, SUB, "middle")
    # 라벨 위치 조정 (dx, dy, anchor)
    place = {
        "Hermes 단독 · Sol medium": (0, -14, "end"),
        "Astra xhigh": (-12, -36, "end"),
        "Astra xhigh + 보정": (-12, -20, "end"),
        "Astra low": (-12, 22, "end"),
        "GLM ultrafast low (quick)": (10, 4, "start"),
        "Kimi ultrafast medium": (10, 4, "start"),
        "라우팅 · 신호 추가 후 · 출시 체인": (10, 4, "start"),
        "라우팅 · 신호 추가 후 · Sol 첫 모델": (0, 22, "middle"),
    }
    for name, kind, p, c, mins in LANES:
        x, y = xpos(c), ypos(p)
        if kind == "routed":
            out += f'<rect x="{x-7:.1f}" y="{y-7:.1f}" width="14" height="14" fill="{ORANGE}" transform="rotate(45 {x:.1f} {y:.1f})"/>'
        else:
            out += f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{BLUE}"/>'
        dx, dy, anc = place[name]
        out += t(x + dx, y + dy, f"{name} · {mins:g}분", 11, INK, anc)
    ly = h - 52
    out += f'<circle cx="26" cy="{ly-4}" r="6" fill="{BLUE}"/>' + t(38, ly, "단일 모델 실행", 11, SUB)
    out += f'<rect x="150" y="{ly-10}" width="11" height="11" fill="{ORANGE}" transform="rotate(45 155.5 {ly-4.5})"/>'
    out += t(168, ly, "라우팅 결과 (단일 모델 행을 문항별로 조합. 실제로 라우팅해 돌린 실행 아님)", 11, SUB)
    out += t(20, ly + 20, f"신호 추가 전 라우팅(30개 모두 quick)은 {ROUTED_BEFORE_SAME_AS} 점과 같은 값", 11, SUB)
    return svg(w, h, out, "oh-my-hermes 라우팅 레인 비용과 통과 수")


def fig_complexity():
    w = 820
    L, rh, T = 330, 28, 76
    h = T + rh * len(SIGNALS) + 150
    zero = L + 60
    unit = 70
    out = t(20, 28, "요청 복잡도 점수기: 신호 가중치와 tier 경계", 15, INK, weight="bold")
    out += t(20, 46, "LLM 없이 요청 문장에서 신호를 찾아 가중치를 더한다. 점수 합으로 tier 와 모델 카테고리 · effort 가 정해진다", 12, SUB)
    for k in range(-2, 5):
        x = zero + k * unit
        out += f'<line x1="{x}" y1="{T-8}" x2="{x}" y2="{T + rh*len(SIGNALS)}" stroke="{GRID if k else GRAY}"/>'
        out += t(x, T - 14, f"{k:+d}" if k else "0", 10, SUB, "middle")
    for i, (name, wgt, desc) in enumerate(SIGNALS):
        y = T + rh * i
        c = RED if wgt < 0 else (ORANGE if wgt >= 4 else BLUE)
        x0, x1 = (zero + wgt * unit, zero) if wgt < 0 else (zero, zero + wgt * unit)
        out += f'<rect x="{x0}" y="{y+4}" width="{x1-x0}" height="{rh-10}" rx="3" fill="{c}"/>'
        out += t(L - 10, y + rh / 2 + 4, name, 12, INK, "end", "bold")
        out += t(max(x1, zero) + 8, y + rh / 2 + 4, desc, 11, SUB)
    # tier 띠
    ty = T + rh * len(SIGNALS) + 40
    out += t(20, ty - 12, "점수 합 → tier → 모델 카테고리 · effort", 12, INK, weight="bold")
    bx0, bw = 20, (w - 40) / 3
    cols = [GREEN, YELLOW, RED]
    for i, (tier, lo_, cat) in enumerate(TIERS):
        x = bx0 + bw * i
        rng = f"점수 &lt; {TIERS[1][1]}" if i == 0 else (f"{lo_} ≤ 점수 &lt; {TIERS[2][1]}" if i == 1 else f"점수 ≥ {lo_}")
        out += f'<rect x="{x+2:.1f}" y="{ty}" width="{bw-4:.1f}" height="54" rx="6" fill="#ffffff" stroke="{cols[i]}" stroke-width="2"/>'
        out += t(x + bw / 2, ty + 22, f"{tier} ({rng})", 13, INK, "middle", "bold")
        out += t(x + bw / 2, ty + 42, cat, 11, SUB, "middle")
    out += t(20, ty + 78, "주황 = exhaustive_search (2026-09-05 측정 뒤 추가, 혼자서 standard 로 올린다). 빨강 = 유일한 감점 신호", 11, SUB)
    return svg(w, h, out, "oh-my-hermes 요청 복잡도 신호 가중치")


def fig_calib_ci():
    w, L, R, T, rh = 820, 200, 780, 80, 40
    h = T + rh * len(CALIB) + 80
    lo, hi = -10000, 11000

    def xpos(v):
        return L + (v - lo) / (hi - lo) * (R - L)

    out = t(20, 28, "모델별 보정 프롬프트 4-arm: 통과는 모두 18/30, 토큰 차이만 남는다", 15, INK, weight="bold")
    out += t(20, 46, "gpt-6-astra xhigh, 문항별 paired bootstrap 10,000회. 점 = 평균 Δ 토큰, 선 = CI95. 0을 걸치면 차이 판단 불가", 12, SUB)
    for v in range(-10000, 11001, 5000):
        x = xpos(v)
        out += f'<line x1="{x:.1f}" y1="{T-10}" x2="{x:.1f}" y2="{T + rh*len(CALIB)}" stroke="{GRAY if v == 0 else GRID}"/>'
        out += t(x, T - 16, f"{v:+,}" if v else "0", 10, SUB, "middle")
    for i, (pair, m, a, b) in enumerate(CALIB):
        y = T + rh * i + rh / 2
        sig = a > 0 or b < 0
        c = (RED if m > 0 else GREEN) if sig else GRAY
        out += t(L - 12, y + 4, pair, 12, INK, "end", "bold" if sig else "normal")
        out += f'<line x1="{xpos(a):.1f}" y1="{y}" x2="{xpos(b):.1f}" y2="{y}" stroke="{c}" stroke-width="3"/>'
        out += f'<circle cx="{xpos(m):.1f}" cy="{y}" r="6" fill="{c}"/>'
        out += t(xpos(b) + 8, y + 4, f"{m:+,}", 11, SUB)
    y = T + rh * len(CALIB) + 26
    out += t(20, y, "빨강 = 확실히 더 씀, 초록 = 확실히 덜 씀, 회색 = CI 가 0을 걸침", 11, SUB)
    out += t(20, y + 18, "원래 override 가 family 보다 더 썼고(빨강), 고친 revised 가 그 차이를 되돌렸다(초록). revised 와 family 는 구분되지 않는다", 11, SUB)
    return svg(w, h, out, "oh-my-hermes 보정 프롬프트 토큰 차이 CI")


FIGS = [("routing_lanes", fig_routing_lanes), ("complexity", fig_complexity), ("calib_ci", fig_calib_ci)]


def main(argv):
    global SHOW_NOTE
    if argv[:1] == ["--inline"]:
        out, prefix, SHOW_NOTE = Path(argv[1]), "", False
    else:
        out, prefix = OUT, "omh_"
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in FIGS:
        (out / f"{prefix}{name}.svg").write_text(fn(), "utf-8")
        print("wrote", out / f"{prefix}{name}.svg")


if __name__ == "__main__":
    main(sys.argv[1:])
