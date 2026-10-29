"""Physical-Intelligence/openpi 노트의 그림을 생성한다.

    python scripts/figures/openpi_figures.py              readings/repos/assets/openpi_*.svg 를 다시 쓴다
    python scripts/figures/openpi_figures.py --inline DIR  HTML 보고서용 (접두어·생성 메모 없이 DIR/<이름>.svg)

SVG 는 손으로 고치지 않는다. 규칙은 레포 커밋 215abfb 의 src/openpi/models/pi0.py
(make_attn_mask, embed_suffix, compute_loss, sample_actions) 와 examples/libero/README.md 결과 표를 옮긴 것이다.
"""
import sys
from pathlib import Path

FONT = "-apple-system,'Apple SD Gothic Neo','Noto Sans KR',sans-serif"
INK, SUB, GRID = "#1f2328", "#59636e", "#d8dee4"
GRAY, BLUE, ORANGE, GREEN, PURPLE = "#8b949e", "#2563eb", "#ea580c", "#16a34a", "#7c3aed"
SHOW_NOTE = True
NOTE = "생성: python scripts/figures/openpi_figures.py · 손으로 고치지 말 것"
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "readings" / "repos" / "assets"

# 작은 예시 길이: 이미지 3, 언어 2, 상태 1(π0만), 행동 4 (실제는 이미지 수백, 행동 action_horizon 개)
TOY = {"img": 3, "txt": 2, "act": 4}
BETA_A, BETA_B = 1.5, 1.0     # time = Beta(1.5, 1) * 0.999 + 0.001
NUM_STEPS = 10               # sample_actions 기본값
LIBERO = [("Spatial", 98.8), ("Object", 98.2), ("Goal", 98.0), ("Libero 10", 92.4)]
LIBERO_AVG = 96.85


def tokens(pi05):
    """(종류, ar_mask) 목록. embed_prefix / embed_suffix 의 ar_mask 규칙 그대로."""
    seq = [("img", False)] * TOY["img"] + [("txt", False)] * TOY["txt"]
    if not pi05:
        seq.append(("state", True))
    seq += [("act", True)] + [("act", False)] * (TOY["act"] - 1)
    return seq


def attn_mask(seq):
    """make_attn_mask: 토큰 i 는 누적 ar 이 자기 이하인 토큰 j 를 본다."""
    cum, c = [], 0
    for _, ar in seq:
        c += int(ar)
        cum.append(c)
    return [[cum[j] <= cum[i] for j in range(len(seq))] for i in range(len(seq))]


def beta_pdf(x, a=BETA_A, b=BETA_B):
    # Beta(1.5, 1) 정규화 상수 B(1.5,1) = 1/1.5
    return x ** (a - 1) * (1 - x) ** (b - 1) * a


def euler_times(n=NUM_STEPS):
    dt, t, out = -1.0 / n, 1.0, []
    while t >= -dt / 2:
        out.append(round(t, 6))
        t += dt
    return out


def svg(w, h, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'role="img" aria-label="{title}" style="max-width:100%;height:auto;font-family:{FONT}">'
            f'<title>{title}</title><rect width="{w}" height="{h}" fill="#ffffff"/>{body}'
            + (f'<text x="{w-8}" y="{h-6}" font-size="9" fill="{GRAY}" text-anchor="end">{NOTE}</text>' if SHOW_NOTE else "")
            + '</svg>')


def t(x, y, s, size=12, fill=INK, anchor="start", weight="normal"):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}">{s}</text>')


KIND = {"img": ("이미지", BLUE), "txt": ("언어", GREEN), "state": ("상태", ORANGE), "act": ("행동", PURPLE)}


def _matrix(x0, y0, seq, cell, title):
    m = attn_mask(seq)
    n = len(seq)
    out = t(x0 + n * cell / 2, y0 - 34, title, 13, INK, "middle", "bold")
    out += t(x0 - 6, y0 - 8, "질의 ↓ / 키 →", 9.5, SUB, "end")
    for j, (k, _) in enumerate(seq):
        out += f'<rect x="{x0 + j * cell + 1}" y="{y0 - 12}" width="{cell - 2}" height="8" fill="{KIND[k][1]}"/>'
    for i, (k, _) in enumerate(seq):
        out += f'<rect x="{x0 - 12}" y="{y0 + i * cell + 1}" width="8" height="{cell - 2}" fill="{KIND[k][1]}"/>'
        for j in range(n):
            fill = "#1f2328" if m[i][j] else "#f3f4f6"
            out += f'<rect x="{x0 + j * cell + 1}" y="{y0 + i * cell + 1}" width="{cell - 2}" height="{cell - 2}" fill="{fill}"/>'
    return out


def fig_attn():
    w, h, cell = 720, 450, 22
    out = t(20, 28, "π0 / π0.5 블록 어텐션 마스크 (작은 예: 이미지 3, 언어 2, 상태 1, 행동 4)", 14.5, INK, weight="bold")
    out += t(20, 46, "검은 칸은 볼 수 있는 자리. 이미지와 언어는 행동을 보지 않고, 행동 토큰끼리는 서로 본다", 11.5, SUB)
    out += _matrix(90, 112, tokens(False), cell, "π0: 상태 토큰이 따로 한 블록")
    out += _matrix(450, 112, tokens(True), cell, "π0.5: 상태는 suffix에서 빠짐")
    ly = 112 + 10 * cell + 30
    lx = 90
    for k in ("img", "txt", "state", "act"):
        name, c = KIND[k]
        out += f'<rect x="{lx}" y="{ly}" width="11" height="11" fill="{c}"/>' + t(lx + 16, ly + 10, name, 11.5)
        lx += 90
    out += t(20, ly + 34, "앞 블록(이미지·언어)은 VLM 2B 가중치, 뒤 블록(상태·행동)은 행동 전문가 300M 가중치로 같은 18층을 지난다", 11, SUB)
    out += t(20, ly + 52, "π0.5는 상태를 언어 토큰으로 이산화해 앞 블록에 넣을 수 있고(discrete_state_input), 시점은 adaRMS 조건으로 넣는다", 11, SUB)
    return svg(w, h, out, "π0 블록 어텐션 마스크")


def fig_flow():
    w, h, L, R, T, B = 680, 380, 60, 640, 80, 290
    X = lambda v: L + (R - L) * v
    ymax = 1.6
    Y = lambda v: B - (B - T) * v / ymax
    out = t(20, 28, "flow matching 시점: 학습 때 뽑는 분포와 추론 때 밟는 10단계", 14.5, INK, weight="bold")
    out += t(20, 46, "t=1이 순수 잡음, t=0이 행동. 학습은 잡음 쪽을 더 자주 보고, 추론은 1에서 0까지 같은 간격으로 내려간다", 11.5, SUB)
    for v in (0, 0.5, 1.0, 1.5):
        out += f'<line x1="{L}" y1="{Y(v)}" x2="{R}" y2="{Y(v)}" stroke="{GRID}"/>' + t(L - 8, Y(v) + 4, v, 11, SUB, "end")
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        out += t(X(v), B + 18, v, 11, SUB, "middle")
    xs = [0.001 + 0.999 * i / 200 for i in range(201)]
    # time = 0.001 + 0.999 * s, s ~ Beta → 밀도는 beta_pdf(s) / 0.999
    pts = " ".join(f"{X(x):.1f},{Y(beta_pdf((x - 0.001) / 0.999) / 0.999):.1f}" for x in xs)
    out += f'<polyline points="{X(0.001):.1f},{B} {pts} {X(1.0):.1f},{B}" fill="#dbeafe" stroke="{BLUE}" stroke-width="2"/>'
    out += t(X(0.62), Y(1.35), "학습 시점 밀도 Beta(1.5, 1)", 11.5, BLUE)
    for tt in euler_times():
        out += f'<line x1="{X(tt):.1f}" y1="{B + 26}" x2="{X(tt):.1f}" y2="{B + 36}" stroke="{ORANGE}" stroke-width="2"/>'
    out += t(L, B + 52, f"주황 눈금: 추론 시 평가 시점 {NUM_STEPS}개 (dt = -1/{NUM_STEPS}, Euler). 이미지·언어 KV 캐시는 한 번만 계산한다", 11, ORANGE)
    out += t((L + R) / 2, B + 70, "시점 t", 11, SUB, "middle")
    return svg(w, h, out, "flow matching 학습 시점 분포와 추론 단계")


def fig_libero():
    w, h, L, R, T, B = 620, 330, 70, 600, 80, 250
    lo = 85
    Y = lambda v: B - (B - T) * (v - lo) / (100 - lo)
    out = t(20, 28, "π0.5 LIBERO 미세조정 결과 (pi05_libero, 30k step)", 14.5, INK, weight="bold")
    out += t(20, 46, "y축은 85에서 시작한다. 긴 과제 묶음(Libero 10)만 90점대 초반이다", 11.5, SUB)
    for v in (85, 90, 95, 100):
        out += f'<line x1="{L}" y1="{Y(v)}" x2="{R}" y2="{Y(v)}" stroke="{GRID}"/>' + t(L - 8, Y(v) + 4, v, 11, SUB, "end")
    gw = (R - L) / len(LIBERO)
    for i, (name, v) in enumerate(LIBERO):
        x = L + i * gw + gw * 0.25
        c = ORANGE if v < 95 else BLUE
        out += f'<rect x="{x:.1f}" y="{Y(v):.1f}" width="{gw * 0.5:.1f}" height="{B - Y(v):.1f}" fill="{c}"/>'
        out += t(x + gw * 0.25, Y(v) - 6, v, 12, INK, "middle", "bold")
        out += t(x + gw * 0.25, B + 18, name, 12, SUB, "middle")
    out += f'<line x1="{L}" y1="{Y(LIBERO_AVG):.1f}" x2="{R}" y2="{Y(LIBERO_AVG):.1f}" stroke="{GRAY}" stroke-dasharray="5 4"/>'
    out += t(R, Y(LIBERO_AVG) - 6, f"평균 {LIBERO_AVG}", 11, SUB, "end")
    out += t(20, h - 26, "README 표에 단위·시행 수·시드 표기가 없다. 실제 로봇 수치는 저장소에 없다", 11, SUB)
    return svg(w, h, out, "π0.5 LIBERO 결과")


FIGS = [("attn", fig_attn), ("flow", fig_flow), ("libero", fig_libero)]


def main(argv):
    global SHOW_NOTE
    if argv[:1] == ["--inline"]:
        out, prefix, SHOW_NOTE = Path(argv[1]), "", False
    else:
        out, prefix = OUT, "openpi_"
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in FIGS:
        (out / f"{prefix}{name}.svg").write_text(fn(), "utf-8")
        print("wrote", out / f"{prefix}{name}.svg")


if __name__ == "__main__":
    main(sys.argv[1:])
