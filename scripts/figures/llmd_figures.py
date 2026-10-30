"""llm-d/llm-d 노트의 그림을 생성한다.

    python scripts/figures/llmd_figures.py              readings/repos/assets/llmd_*.svg 를 다시 쓴다
    python scripts/figures/llmd_figures.py --inline DIR  HTML 보고서용 (접두어·생성 메모 없이 DIR/<이름>.svg)

SVG 는 손으로 고치지 않는다. 수치는 레포 커밋 abffd56 의 README.md "Performance and Adoption Highlights" 와
guides/models/nemotron-3-ultra/README.md "Benchmark Results" 표에서 옮긴 아래 상수만 쓴다.
원문은 성능 주장을 문장으로만 나열하므로, 배수 한 축에 모으고 Nemotron 표를 검산한다.
"""
import math
import sys
from pathlib import Path

FONT = "-apple-system,'Apple SD Gothic Neo','Noto Sans KR',sans-serif"
INK, SUB, GRID = "#1f2328", "#59636e", "#d8dee4"
GRAY, BLUE, ORANGE, GREEN, PURPLE = "#8b949e", "#2563eb", "#ea580c", "#16a34a", "#7c3aed"
SHOW_NOTE = True
NOTE = "생성: python scripts/figures/llmd_figures.py · 손으로 고치지 말 것"
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "readings" / "repos" / "assets"

# README 성능 주장: (기법, 지표, 배수 하한, 배수 상한, 비교 대상 · 환경)
# "40% reduction" 은 1/(1-0.4) 배, "10–30%" 는 1.1~1.3 배, "up to 70%" 는 1.7 배로 바꾼다
CLAIMS = [
    ("계층 KV 오프로딩", "처리량", 13.9, 13.9, "GPU only 대비 · H100 4장 · 동시 사용자 250"),
    ("prefix-cache 라우팅", "출력 처리량", 3.0, 3.0, "round-robin 대비 · Llama 3.1 70B · MI300X 4장"),
    ("prefix-cache 라우팅", "TTFT", 2.0, 2.0, "round-robin 대비 · 같은 환경"),
    ("P/D 분리 (AWS)", "tokens/sec", 1.7, 1.7, "표준 vLLM 대비 최대 · GPT-OSS · B200"),
    ("예측 지연 스케줄링", "TTFT · ITL", 1 / 0.6, 1 / 0.6, "휴리스틱 대비 40% 감소 · NVIDIA GPU"),
    ("P/D 분리 (Oracle)", "처리량", 1.1, 1.3, "동일 인프라 대비 · GPT-OSS-120B 등 · MI300X"),
]
WIDE_EP = (50_000, 3_100)            # 클러스터 tok/s, GPU당 tok/s ("16×16 NVIDIA B200")

# Nemotron 3 Ultra, H200 8노드, 동시 코딩 세션 60: (지표, k8s Service, llm-d, 원문 Δ %)
NEMOTRON = [
    ("TTFT P50 (ms)", 49462, 20543, -58),
    ("Total tok/s", 64958, 67645, 4),
    ("Input tok/s", 64474, 66716, 3),
    ("Output tok/s", 484, 929, 92),
]


def change_pct(old, new):
    return (new / old - 1) * 100


def input_share(row_in, row_total):
    return row_in / row_total * 100


def svg(w, h, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'role="img" aria-label="{title}" style="max-width:100%;height:auto;font-family:{FONT}">'
            f'<title>{title}</title><rect width="{w}" height="{h}" fill="#ffffff"/>{body}'
            + (f'<text x="{w-8}" y="{h-6}" font-size="9" fill="{GRAY}" text-anchor="end">{NOTE}</text>' if SHOW_NOTE else "")
            + '</svg>')


def t(x, y, s, size=12, fill=INK, anchor="start", weight="normal"):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}">{s}</text>')


def fig_claims():
    w, h = 760, 450
    out = t(20, 28, "README 성능 주장 6개를 배수 한 축에 모은 것 (비교 대상·하드웨어가 모두 다르다)", 14.5, INK, weight="bold")
    out += t(20, 46, "가로축은 로그 배수. 감소율은 1/(1−감소율) 배로 바꿨다. 막대 오른쪽 글자가 비교 대상과 환경", 11.5, SUB)
    L, R, T, row = 170, 420, 74, 44
    X = lambda v: L + (R - L) * math.log(v) / math.log(16)
    bottom = T + row * len(CLAIMS)
    for v in (1, 2, 4, 8, 16):
        out += f'<line x1="{X(v):.1f}" y1="{T-8}" x2="{X(v):.1f}" y2="{bottom}" stroke="{GRID}"/>'
        out += t(X(v), bottom + 16, f"{v}배", 11, SUB, "middle")
    for i, (tech, metric, lo, hi, ctx) in enumerate(CLAIMS):
        y = T + row * i
        color = PURPLE if "KV" in tech else BLUE if "prefix" in tech else ORANGE if "P/D" in tech else GREEN
        out += t(L - 10, y + 12, tech, 12, INK, "end", "bold")
        out += t(L - 10, y + 27, metric, 10.5, SUB, "end")
        out += f'<rect x="{L}" y="{y+4}" width="{X(lo)-L:.1f}" height="18" rx="2" fill="{color}" fill-opacity="0.85"/>'
        if hi > lo:
            out += f'<rect x="{X(lo):.1f}" y="{y+4}" width="{X(hi)-X(lo):.1f}" height="18" rx="2" fill="{color}" fill-opacity="0.35"/>'
        label = f"{lo:.1f}배" if hi == lo else f"{lo:.1f}~{hi:.1f}배"
        out += t(X(hi) + 6, y + 17, label, 11, color, "start", "bold")
        out += t(R + 50, y + 17, ctx, 10.5, SUB)
    cluster, per_gpu = WIDE_EP
    out += t(20, h - 58, f"표에 없는 wide EP: 클러스터 {cluster:,} tok/s, GPU당 약 {per_gpu:,} tok/s → {cluster / per_gpu:.1f}장 분량. "
             f"\"16×16 B200\" 을 256장으로 읽으면 GPU당 {cluster / 256:.0f} tok/s 라 맞지 않는다", 11, SUB)
    out += t(20, h - 40, "출처는 README 가 링크한 각 블로그이며 저장소 안에 재현 자료가 없다. 막대 길이를 서로 비교하지 않는다", 11, SUB)
    out += t(20, h - 22, "색: 보라 = KV 계층, 파랑 = prefix 라우팅, 주황 = P/D 분리, 초록 = 지연 예측", 11, SUB)
    return svg(w, h, out, "llm-d README 성능 주장 배수 비교")


def fig_nemotron():
    w, h = 760, 400
    out = t(20, 28, "Nemotron 3 Ultra (H200 8노드, 동시 코딩 세션 60): k8s Service 대비 llm-d", 14.5, INK, weight="bold")
    out += t(20, 46, "왼쪽은 원문 표의 변화율 검산, 오른쪽은 처리량 안에서 입력·출력 토큰이 차지하는 몫", 11.5, SUB)
    # 왼쪽: 변화율 막대
    L, R, T, row = 130, 380, 84, 50
    X = lambda v: L + (R - L) * (v + 70) / 170
    zero = X(0)
    for v in (-60, -30, 0, 30, 60, 90):
        out += f'<line x1="{X(v):.1f}" y1="{T-8}" x2="{X(v):.1f}" y2="{T + row * 4}" stroke="{GRID if v else GRAY}"/>'
        out += t(X(v), T + row * 4 + 16, f"{v:+d}%" if v else "0", 10.5, SUB, "middle")
    for i, (label, old, new, stated) in enumerate(NEMOTRON):
        y = T + row * i
        pct = change_pct(old, new)
        good = pct < 0 if "TTFT" in label else pct > 0
        x0, x1 = sorted((zero, X(pct)))
        out += t(L - 10, y + 16, label, 11.5, INK, "end")
        out += f'<rect x="{x0:.1f}" y="{y+4}" width="{x1-x0:.1f}" height="18" rx="2" fill="{GREEN if good else ORANGE}"/>'
        txt = f"{pct:+.1f}% (원문 {stated:+d}%)"
        if pct < 0:
            out += t(zero + 6, y + 17, txt, 10.5, INK)
        elif X(pct) > R - 40:
            out += t(X(pct) - 6, y + 17, txt, 10.5, "#ffffff", "end", "bold")
        else:
            out += t(X(pct) + 6, y + 17, txt, 10.5, INK)
    # 오른쪽: 입력·출력 비중 막대
    L2, R2 = 470, 740
    k8s_total, llmd_total = NEMOTRON[1][1], NEMOTRON[1][2]
    k8s_out, llmd_out = NEMOTRON[3][1], NEMOTRON[3][2]
    out += t(L2, T - 14, "총 처리량 중 출력 토큰 몫", 11, SUB)
    for j, (name, total, outp) in enumerate((("k8s Service", k8s_total, k8s_out), ("llm-d", llmd_total, llmd_out))):
        y = T + 10 + 60 * j
        out += t(L2, y - 4, f"{name}: 총 {total:,} tok/s", 11, INK)
        share_out = outp / total * 100
        wout = max((R2 - L2) * share_out / 100 * 10, 2)        # 출력 몫은 10배 확대해 그린다
        out += f'<rect x="{L2}" y="{y}" width="{R2-L2}" height="16" rx="2" fill="{GRID}"/>'
        out += f'<rect x="{L2}" y="{y}" width="{wout:.1f}" height="16" rx="2" fill="{BLUE}"/>'
        out += t(L2 + wout + 6, y + 12, f"출력 {share_out:.1f}%", 10.5, BLUE)
    gain_total = llmd_total - k8s_total
    gain_out = llmd_out - k8s_out
    out += t(L2, T + 150, "출력 막대는 10배 확대 (회색 = 전체)", 10.5, SUB)
    out += t(L2, T + 172, f"총 증가 {gain_total:,} tok/s 중 출력 증가 {gain_out:,}", 11, INK)
    out += t(L2, T + 190, f"= {gain_out / gain_total * 100:.0f}%. 나머지는 입력(prefill)", 11, INK)
    out += t(20, h - 40, "워크로드는 동적 시스템 프롬프트 평균 160,000 토큰, 턴당 입력 1,500 · 출력 425 토큰. prefill 이 처리량을 정한다", 11, SUB)
    out += t(20, h - 22, "출력 92% 증가는 절대량이 작다. 주된 변화는 TTFT 58% 감소다. Input 원문 3% 는 3.5% 를 내린 값", 11, SUB)
    return svg(w, h, out, "Nemotron 3 Ultra 벤치마크 검산")


FIGS = [("claims", fig_claims), ("nemotron", fig_nemotron)]


def main(argv):
    global SHOW_NOTE
    if argv[:1] == ["--inline"]:
        out, prefix, SHOW_NOTE = Path(argv[1]), "", False
    else:
        out, prefix = OUT, "llmd_"
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in FIGS:
        (out / f"{prefix}{name}.svg").write_text(fn(), "utf-8")
        print("wrote", out / f"{prefix}{name}.svg")


if __name__ == "__main__":
    main(sys.argv[1:])
