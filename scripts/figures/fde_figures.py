"""forward-deployed-engineer-roadmap 노트의 그림을 생성한다.

    python scripts/figures/fde_figures.py              readings/repos/assets/fde_*.svg 를 다시 쓴다
    python scripts/figures/fde_figures.py --inline DIR  HTML 보고서용 (접두어·생성 메모 없이 DIR/<이름>.svg)

SVG 는 손으로 고치지 않는다. 데이터는 레포 커밋 6868f0f 의 README.md 와,
거기 걸린 블로그 140편을 2026-11-06 에 받아 센 아래 상수만 쓴다.
"""
import sys
from datetime import date
from pathlib import Path

FONT = "-apple-system,'Apple SD Gothic Neo','Noto Sans KR',sans-serif"
INK, SUB, GRID = "#1f2328", "#59636e", "#d8dee4"
GRAY, BLUE, ORANGE, GREEN, YELLOW, RED, PURPLE = (
    "#8b949e", "#2563eb", "#ea580c", "#16a34a", "#ca8a04", "#dc2626", "#9333ea")
SHOW_NOTE = True
NOTE = "생성: python scripts/figures/fde_figures.py · 손으로 고치지 말 것"
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "readings" / "repos" / "assets"

REPO_CREATED = date(2026, 10, 30)

# README 의 engagement lifecycle 표: (단계, Step 목록, 색)
PHASES = [
    ("Learn the toolbox", [1, 2, 3], GRAY),
    ("Prototype", [4, 5, 6, 7], BLUE),
    ("Prove value", [8], GREEN),
    ("Harden", [9, 10, 11], YELLOW),
    ("Deploy", [12], PURPLE),
    ("Ship a real use case", [13], ORANGE),
]

# Step: (짧은 제목, 블로그 수, 영상 수, 블로그 단어 합)
STEPS = {
    1: ("ML 기초", 14, 4, 38996),
    2: ("LLM 동작", 18, 3, 66616),
    3: ("모델 고르기", 6, 0, 21304),
    4: ("프롬프트 · 컨텍스트", 6, 0, 21066),
    5: ("고객 데이터 RAG", 16, 1, 64678),
    6: ("에이전트 · 도구", 17, 0, 60969),
    7: ("코딩 에이전트 · 하네스", 7, 0, 28571),
    8: ("평가 · 관측", 4, 0, 16931),
    9: ("안전 · 보안", 4, 0, 18409),
    10: ("지연 · 비용", 14, 2, 62551),
    11: ("파인튜닝", 7, 0, 25278),
    12: ("시스템 설계 · 인프라", 6, 0, 27557),
    13: ("실시간 음성 에이전트", 21, 0, 80298),
}

# 블로그 발행일 (YYMMDD, 글 머리 "Published on"). 2026 이전 글은 OLD 로 따로 센다
PUB = {
    1: ["190802", "251209", "190802", "261030", "261027", "261103", "261101", "260801", "261020",
        "261027", "261028", "261030", "260417", "190802"],
    2: ["260805", "261029", "260513", "261003", "260331", "260731", "261031", "260407", "260529",
        "260921", "260915", "260919", "261004", "261105", "260610", "261104", "260819", "261014"],
    3: ["260511", "260512", "260409", "260510", "261030", "260509"],
    4: ["260711", "260710", "261007", "260508", "260721", "260609"],
    5: ["260606", "260629", "260723", "260707", "260608", "260910", "261006", "260706", "261008",
        "260812", "260626", "260621", "260708", "260501", "260505", "260910"],
    6: ["260427", "260613", "260528", "260430", "260504", "260507", "260428", "261012", "260728",
        "260730", "260726", "260429", "260525", "260526", "260522", "260624", "261011"],
    7: ["260402", "260820", "260821", "260701", "260702", "260720", "260722"],
    8: ["260524", "260521", "260523", "260527"],
    9: ["260810", "260803", "261105", "260818"],
    10: ["260614", "260705", "260327", "260329", "260511", "260506", "260913", "260715", "260612",
         "260716", "261005", "260617", "260630", "260904"],
    11: ["260727", "260424", "260824", "260611", "260601", "260514", "260517"],
    12: ["261101", "260703", "260929", "260628", "260903", "260807"],
    13: ["260916", "261021", "261102", "261019", "261026", "261026", "261028", "261101", "261029",
         "261031", "261020", "261103", "261105", "261021", "261022", "261022", "261023", "261023",
         "261024", "261024", "261025"],
}

# voice-agent-latency-budget 글의 예시 예산과 예시 측정 출력 (ms). 측정 출력에 없는 단계는 None
VOICE = [
    ("Network 상행", 50, None),
    ("Endpointing", 200, 210),
    ("STT 최종 텍스트", 100, 85),
    ("LLM 첫 토큰", 300, 520),
    ("TTS 첫 오디오", 100, 95),
    ("Network 하행", 50, None),
]
VOICE_TARGET = 800


def d(s):
    return date(2000 + int(s[:2]), int(s[2:4]), int(s[4:]))


def phase_of(step):
    for name, steps, color in PHASES:
        if step in steps:
            return name, color
    raise KeyError(step)


def svg(w, h, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'role="img" aria-label="{title}" style="max-width:100%;height:auto;font-family:{FONT}">'
            f'<title>{title}</title><rect width="{w}" height="{h}" fill="#ffffff"/>{body}'
            + (f'<text x="{w-8}" y="{h-6}" font-size="9" fill="{GRAY}" text-anchor="end">{NOTE}</text>' if SHOW_NOTE else "")
            + '</svg>')


def t(x, y, s, size=12, fill=INK, anchor="start", weight="normal"):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}">{s}</text>')


def fig_step_volume():
    W, H = 760, 545
    L, R, TOP, ROW = 210, 640, 70, 29
    maxw = 90000
    b = [t(20, 28, "Step 별 블로그 분량: 단어 합(막대)과 편수(오른쪽)", 15, weight="bold"),
         t(20, 48, "색은 원문 lifecycle 단계. Step 0 과 14 는 블로그 목록이 없어 뺐다", 11, SUB)]
    for k in range(0, maxw + 1, 20000):
        x = L + (R - L) * k / maxw
        b.append(f'<line x1="{x:.1f}" y1="{TOP-6}" x2="{x:.1f}" y2="{TOP+ROW*13}" stroke="{GRID}"/>')
        b.append(t(x, TOP + ROW * 13 + 16, f"{k//1000}k", 10, SUB, "middle"))
    for i, (st, (name, n, v, words)) in enumerate(STEPS.items()):
        y = TOP + i * ROW
        _, color = phase_of(st)
        b.append(t(L - 8, y + 15, f"Step {st} {name}", 12, INK, "end"))
        bw = (R - L) * words / maxw
        b.append(f'<rect x="{L}" y="{y+4}" width="{bw:.1f}" height="{ROW-10}" fill="{color}" rx="2"/>')
        b.append(t(L + bw + 6, y + 15, f"{words/1000:.1f}k", 10, SUB))
        b.append(t(R + 70, y + 15, f"{n}편" + (f" + 영상 {v}" if v else ""), 11, INK, "end"))
    b.append(t((L + R) / 2, TOP + ROW * 13 + 32, "블로그 단어 합 (article 텍스트, 저자 소개 포함 근사)", 11, SUB, "middle"))
    ly = H - 36
    x = 20
    for name, steps, color in PHASES:
        b.append(f'<rect x="{x}" y="{ly}" width="11" height="11" fill="{color}" rx="2"/>')
        b.append(t(x + 15, ly + 10, name, 11))
        x += 30 + len(name) * 6.2
    return svg(W, H, "".join(b), "Step 별 블로그 편수와 단어 합")


def fig_pub_timeline():
    W, H = 760, 500
    L, R, TOP, ROW = 215, 730, 70, 27
    lo, hi = date(2026, 3, 15), date(2026, 11, 15)
    span = (hi - lo).days

    def xpos(dd):
        return L + (R - L) * (dd - lo).days / span

    b = [t(20, 28, "블로그 발행일: 로드맵은 이미 쓴 글을 엮고, 음성 단계만 공개 무렵에 몰려 있다", 15, weight="bold"),
         t(20, 48, "점 하나가 글 한 편. 세로 선은 저장소 생성일 2026-10-30. (+이전 N) 은 2026 이전 글로 축 밖", 11, SUB)]
    for m in range(4, 12):
        x = xpos(date(2026, m, 1))
        b.append(f'<line x1="{x:.1f}" y1="{TOP-6}" x2="{x:.1f}" y2="{TOP+ROW*13}" stroke="{GRID}"/>')
        b.append(t(x, TOP + ROW * 13 + 16, f"2026-{m:02d}", 10, SUB, "middle"))
    xc = xpos(REPO_CREATED)
    b.append(f'<line x1="{xc:.1f}" y1="{TOP-10}" x2="{xc:.1f}" y2="{TOP+ROW*13}" stroke="{RED}" stroke-dasharray="4 3"/>')
    for i, (st, dates) in enumerate(PUB.items()):
        y = TOP + i * ROW + 13
        _, color = phase_of(st)
        old = sum(1 for s in dates if d(s) < lo)
        label = f"Step {st} {STEPS[st][0]}" + (f" (+이전 {old})" if old else "")
        b.append(t(L - 8, y + 4, label, 12, INK, "end"))
        seen = {}
        for s in dates:
            dd = d(s)
            if dd < lo:
                continue
            k = seen.get(s, 0)
            seen[s] = k + 1
            b.append(f'<circle cx="{xpos(dd):.1f}" cy="{y - k*6:.1f}" r="3.6" fill="{color}" fill-opacity="0.8"/>')
    return svg(W, H, "".join(b), "Step 별 블로그 발행일 분포")


def fig_voice_budget():
    W, H = 760, 340
    L, R, TOP, ROW = 150, 700, 80, 36
    maxv = 600
    b = [t(20, 28, "음성 에이전트 지연 예산 800 ms 의 단계별 몫과 예시 측정", 15, weight="bold"),
         t(20, 48, "원문 블로그의 예시 표와 예시 출력을 한 축에 겹쳤다. 회색 = 예산, 색 = 예시 측정", 11, SUB)]
    for k in range(0, maxv + 1, 100):
        x = L + (R - L) * k / maxv
        b.append(f'<line x1="{x:.1f}" y1="{TOP-6}" x2="{x:.1f}" y2="{TOP+ROW*len(VOICE)}" stroke="{GRID}"/>')
        b.append(t(x, TOP + ROW * len(VOICE) + 16, f"{k}", 10, SUB, "middle"))
    for i, (name, bud, meas) in enumerate(VOICE):
        y = TOP + i * ROW
        b.append(t(L - 8, y + 20, name, 12, INK, "end"))
        bw = (R - L) * bud / maxv
        b.append(f'<rect x="{L}" y="{y+5}" width="{bw:.1f}" height="12" fill="{GRID}" rx="2"/>')
        b.append(t(L + bw + 5, y + 15, f"예산 {bud}", 10, SUB))
        if meas is None:
            b.append(t(L + 4, y + 31, "예시 출력에 없음", 9, GRAY))
            continue
        over = meas > bud
        mw = (R - L) * meas / maxv
        b.append(f'<rect x="{L}" y="{y+19}" width="{mw:.1f}" height="12" fill="{RED if over else GREEN}" rx="2"/>')
        b.append(t(L + mw + 5, y + 29, f"측정 {meas}" + (f" (+{meas-bud} 초과)" if over else ""), 10,
                   RED if over else GREEN, weight="bold" if over else "normal"))
    b.append(t((L + R) / 2, TOP + ROW * len(VOICE) + 32, "ms", 11, SUB, "middle"))
    return svg(W, H, "".join(b), "음성 에이전트 지연 예산과 예시 측정")


FIGS = [("step_volume", fig_step_volume), ("pub_timeline", fig_pub_timeline), ("voice_budget", fig_voice_budget)]


def main(argv):
    global SHOW_NOTE
    if argv[:1] == ["--inline"]:
        out, prefix, SHOW_NOTE = Path(argv[1]), "", False
    else:
        out, prefix = OUT, "fde_"
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in FIGS:
        (out / f"{prefix}{name}.svg").write_text(fn(), "utf-8")
        print("wrote", out / f"{prefix}{name}.svg")


if __name__ == "__main__":
    main(sys.argv[1:])
