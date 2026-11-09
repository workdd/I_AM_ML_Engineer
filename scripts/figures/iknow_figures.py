"""iKnow 노트의 그림을 생성한다.

    python scripts/figures/iknow_figures.py              readings/repos/assets/iknow_*.svg 를 다시 쓴다
    python scripts/figures/iknow_figures.py --inline DIR  HTML 보고서용 (접두어·생성 메모 없이 DIR/<이름>.svg)

SVG 는 손으로 고치지 않는다. 데이터는 저자 write-up "Anatomy of Apple's Personal Knowledge Graph"
(the Eighth Rank, 2026-10-28) 의 두 표(class.json 181행, predicate.json 278행)를 행 단위로 센 아래 상수만 쓴다.
"""
import sys
from pathlib import Path

FONT = "-apple-system,'Apple SD Gothic Neo','Noto Sans KR',sans-serif"
INK, SUB, GRID = "#1f2328", "#59636e", "#d8dee4"
GRAY, BLUE, ORANGE, GREEN, YELLOW, RED, PURPLE = (
    "#8b949e", "#2563eb", "#ea580c", "#16a34a", "#ca8a04", "#dc2626", "#9333ea")
SHOW_NOTE = True
NOTE = "생성: python scripts/figures/iknow_figures.py · 손으로 고치지 말 것"
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "readings" / "repos" / "assets"

N_CLASSES, N_PREDICATES = 181, 278

# (획득 방식, 메커니즘 확인, 미확인 추정, 찾지 못함) — predicate 표 "How it is obtained" × "Mechanism found?"
PREDICATES = [
    ("Copied (원천에서 복사)", 102, 15, 44),
    ("Apple-shipped (세계 지식)", 9, 2, 29),
    ("Rule (규칙)", 14, 0, 6),
    ("Joined (다른 레코드와 조인)", 13, 3, 2),
    ("Not established (불명)", 2, 3, 6),
    ("모델 산출 (아래 8종 합)", 21, 1, 6),
]
# 모델 산출 predicate 28개의 세부 방식 (표에서 굵게 표시된 행)
MODEL_KINDS = [("Unknown model", 7), ("Scored model", 6), ("Embedding model", 4),
               ("XGBoost + rules + copied", 3), ("Local LLM", 3), ("Face clustering", 2),
               ("XGBoost + copied", 2), ("XGBoost", 1)]

# (클래스 그룹, 확인, 미확인 추정, 찾지 못함) — class 표 "Group" × "Mechanism found?"
CLASSES = [
    ("edge (관계 타입)", 30, 4, 12),
    ("activity", 21, 7, 3),
    ("activity type", 0, 1, 25),
    ("core", 16, 0, 3),
    ("attribute", 6, 1, 4),
    ("person", 1, 0, 8),
    ("transport", 2, 0, 6),
    ("media", 4, 0, 3),
    ("animal", 0, 0, 7),
    ("commerce", 5, 0, 2),
    ("place", 2, 3, 0),
    ("event", 3, 0, 2),
]
MECH = [("메커니즘 확인", GREEN), ("있을 수 있음 (미확인)", YELLOW), ("메커니즘 못 찾음", GRAY)]


def svg(w, h, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'role="img" aria-label="{title}" style="max-width:100%;height:auto;font-family:{FONT}">'
            f'<title>{title}</title><rect width="{w}" height="{h}" fill="#ffffff"/>{body}'
            + (f'<text x="{w-8}" y="{h-6}" font-size="9" fill="{GRAY}" text-anchor="end">{NOTE}</text>' if SHOW_NOTE else "")
            + '</svg>')


def t(x, y, s, size=12, fill=INK, anchor="start", weight="normal"):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}">{s}</text>')


def legend(x, y):
    out = ""
    for lab, c in MECH:
        out += f'<rect x="{x}" y="{y-10}" width="12" height="12" rx="2" fill="{c}"/>' + t(x + 18, y, lab, 11, SUB)
        x += 40 + len(lab) * 9
    return out


def stacked(rows, L, R, T, rh, vmax):
    out = ""
    for i, (lab, *vals) in enumerate(rows):
        y = T + rh * i
        out += t(L - 10, y + 15, lab, 11, INK, "end")
        x = L
        for v, (_, c) in zip(vals, MECH):
            if v:
                bw = (R - L) * v / vmax
                out += f'<rect x="{x:.1f}" y="{y+3}" width="{bw:.1f}" height="18" fill="{c}" opacity="0.88"/>'
                if bw >= 18:
                    out += t(x + bw / 2, y + 16, v, 10, "#ffffff", "middle", "bold")
                x += bw
        out += t(x + 6, y + 16, sum(vals), 11, INK, "start", "bold")
    return out


def fig_predicates():
    w, L, R, T, rh = 800, 230, 740, 92, 30
    out = t(20, 28, f"predicate {N_PREDICATES}개 중 모델이 만드는 것은 28개, 나머지는 복사 · 규칙 · 조인 · 출하 지식", 15, INK, weight="bold")
    out += t(20, 46, "저자 write-up 의 predicate 표를 획득 방식 × 메커니즘 확인 여부로 센 값. 메커니즘 확인은 저자 추정이지 ground truth 가 아니다", 11, SUB)
    out += legend(20, 70)
    vmax = max(sum(r[1:]) for r in PREDICATES)
    out += stacked(PREDICATES, L, R, T, rh, vmax)
    y = T + rh * len(PREDICATES) + 26
    out += t(20, y, "모델 산출 28개의 세부", 12, INK, weight="bold")
    x = L
    mmax = sum(v for _, v in MODEL_KINDS)
    for k, (lab, v) in enumerate(MODEL_KINDS):
        bw = (R - L) * v / mmax
        c = PURPLE if k % 2 == 0 else BLUE
        out += f'<rect x="{x:.1f}" y="{y+10}" width="{bw:.1f}" height="20" fill="{c}" opacity="0.8"/>'
        out += t(x + bw / 2, y + 24, v, 10, "#ffffff", "middle", "bold")
        x += bw
    yy = y + 50
    for k, (lab, v) in enumerate(MODEL_KINDS):
        col, row = k % 4, k // 4
        out += t(20 + col * 190, yy + row * 18, f"{lab} {v}", 11, SUB)
    out += t(20, yy + 48, "모든 행은 같은 9컬럼에 confidence 하나를 달고 들어간다. 복사한 사실과 모델이 추론한 사실을 가르는 전용 컬럼은 없다", 11, SUB)
    h = yy + 72
    return svg(w, h, out, "iKnow 저장소 predicate 획득 방식")


def fig_classes():
    w, L, R, T, rh = 800, 150, 720, 92, 28
    out = t(20, 28, f"클래스 {N_CLASSES}개: 관계 타입 46 · 활동 57, 활동 타입 26개 중 25개는 채우는 메커니즘을 못 찾음", 15, INK, weight="bold")
    out += t(20, 46, "저자 write-up 의 class 표를 그룹(저자 분류) × 메커니즘 확인 여부로 센 값", 11, SUB)
    out += legend(20, 70)
    vmax = max(sum(r[1:]) for r in CLASSES)
    out += stacked(CLASSES, L, R, T, rh, vmax)
    y = T + rh * len(CLASSES) + 20
    impl = sum(r[1] for r in CLASSES)
    unv = sum(r[2] for r in CLASSES)
    nf = sum(r[3] for r in CLASSES)
    out += t(20, y, f"합계: 확인 {impl} · 미확인 {unv} · 못 찾음 {nf}. edge 46개는 엣지를 reify 해 엣지마다 confidence 와 철회를 달기 위한 클래스다", 11, SUB)
    out += t(20, y + 18, "선언만 되고 채워지지 않는 placeholder 가 있을 수 있다고 원문도 밝힌다 (동물 · 반려동물 7개 전부 못 찾음)", 11, SUB)
    return svg(w, y + 42, out, "iKnow 저장소 클래스 그룹 분포")


FIGS = [("predicates", fig_predicates), ("classes", fig_classes)]


def main(argv):
    global SHOW_NOTE
    if argv[:1] == ["--inline"]:
        out, prefix, SHOW_NOTE = Path(argv[1]), "", False
    else:
        out, prefix = OUT, "iknow_"
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in FIGS:
        (out / f"{prefix}{name}.svg").write_text(fn(), "utf-8")
        print("wrote", out / f"{prefix}{name}.svg")


if __name__ == "__main__":
    main(sys.argv[1:])
