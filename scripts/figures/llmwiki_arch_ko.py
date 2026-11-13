"""llm_wiki 원본 그림 assets/llm_wiki_arch.jpg(중국어)를 같은 배치로 한국어로 다시 그린다.

    python scripts/figures/llmwiki_arch_ko.py   # books/llm-wiki/assets/llm_wiki_arch_ko.png

원본: https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/assets/llm_wiki_arch.jpg (@nash_su, GPL-3.0)
글은 원본 그림의 문구를 옮긴 것이며 내용을 더하지 않는다. PNG 는 손으로 고치지 않는다.
"""
import html
import pathlib
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "books" / "llm-wiki" / "assets" / "llm_wiki_arch_ko.png"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

TITLE = "LLM Wiki 방법론"
SUBTITLE = "범용 LLM 지식 베이스 구축 패턴 · Markdown + Obsidian · 사람과 LLM의 협업"
LAYERS = [  # (제목, 설명, 색)
    ("Raw Sources 원본 자료", "직접 고른 원본 문서 · 글/논문/데이터 · 변경 불가 · LLM은 읽기만", "purple"),
    ("Wiki 층 LLM이 생성한 Markdown", "개요 / 개체 페이지 / 개념 페이지 / 비교 · LLM이 전부 관리 · 사람은 읽기만", "green"),
    ("Schema 층 CLAUDE.md / AGENTS.md", "규칙/구조/작업 흐름 · 사람과 LLM이 함께 발전시킴", "orange"),
]
OPS = [
    ("Ingest 수집", "원본 문서 투입 → LLM 읽기/논의/요약 작성/인덱스 갱신<br>원본 하나가 위키 페이지 10~15개에 영향", "purple"),
    ("Query 질의", "위키에 질문 → 검색/종합/인용<br>좋은 답변은 새 페이지로 위키에 보관", "green"),
    ("Lint 점검", "정기 상태 점검: 모순/낡은 내용/고아 페이지/누락 인용<br>LLM이 새 질문과 새 원본 제안", "orange"),
]
INDEX = [
    ("index.md", "내용 중심<br>전체 페이지 목록<br>LLM이 인덱스를 먼저 읽고 페이지 조회", "mint"),
    ("log.md", "시간 중심<br>작업 이력 추가 기록<br>## [날짜] 유형 | 제목", "yellow"),
]
ROLES = [
    ("사람", "원본 선별<br>질문 제기 · 방향 제시", "purple"),
    ("LLM (AI)", "요약 / 교차 인용<br>정리 / 일관성 유지", "blue"),
]
FLOW = [("새 원본 문서", "orange"), ("LLM 읽기 · 분석", "purple"), ("요점 논의", "blue")]
FLOW_END = ("위키 갱신", "요약 작성 + 개체 페이지 갱신")
OUTPUTS = [("요약 페이지", "mint"), ("개체 페이지 × N", "mint"), ("개념 페이지", "mint"), ("index 갱신", "yellow")]
FLOW_NOTE = "원본 하나가 위키 페이지 10~15개에 영향"
USES = [
    ("자기 계발", "일기/건강/심리", "purple"), ("심층 연구", "논문/보고서/논지 변화", "mint"),
    ("독서 동반자", "인물/주제/줄거리 페이지", "blue"), ("팀 지식 베이스", "Slack/회의/문서", "orange"),
    ("경쟁 분석", "실사/비교", "red"), ("여행 계획", "일정/명소/공략", "pink"),
    ("강의 노트", "지식 체계 구축", "yellow"), ("취미 심화", "모든 지식 축적", "mint"),
]
INSIGHT = [
    "지식 베이스 유지의 병목은 읽기나 사고가 아니라 기록 관리다. 인용 갱신, 일관성 유지, 모순 표시.",
    "사람이 위키를 포기하는 이유는 유지 부담이 가치보다 빠르게 늘기 때문이다. "
    "LLM은 지루해하지 않고 잊지 않으며 한 번에 파일 15개를 다룬다. 유지 비용이 0에 가까워진다.",
]
COLORS = {
    "purple": ("#ede7fb", "#8b6cdb"), "green": ("#e2f5e4", "#3f9d55"), "orange": ("#fdebd6", "#d98a2b"),
    "mint": ("#dff6ee", "#3aa985"), "yellow": ("#fdf3cf", "#c49a12"), "blue": ("#e1edfb", "#3b7bd8"),
    "red": ("#fbe2e2", "#cf4b4b"), "pink": ("#f8e2f4", "#bf56a8"),
}


def box(title, body, color, cls="box"):
    bg, bd = COLORS[color]
    b = f"<p>{body}</p>" if body else ""
    return f'<div class="{cls}" style="background:{bg};border-color:{bd}"><h4 style="color:{bd}">{html.escape(title)}</h4>{b}</div>'


def page():
    e = html.escape
    layers = "<div class='arrow'>↓</div>".join(box(t, e(d), c) for t, d, c in LAYERS)
    ops = "".join(box(t, d, c, "box op") for t, d, c in OPS)
    index = "".join(box(t, d.replace("|", "&#124;"), c) for t, d, c in INDEX)
    roles = (box(*ROLES[0][:2], ROLES[0][2]) + "<div class='collab'>협업</div>" + box(*ROLES[1][:2], ROLES[1][2]))
    flow = "<span class='to'>→</span>".join(box(t, "", c, "box sm") for t, c in FLOW)
    flow += "<span class='to'>→</span>" + box(FLOW_END[0], e(FLOW_END[1]), "green", "box sm")
    outs = "".join(box(t, "", c, "box sm") for t, c in OUTPUTS)
    uses = "".join(box(t, e(d), c, "box use") for t, d, c in USES)
    insight = "".join(f"<p>{e(s)}</p>" for s in INSIGHT)
    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><style>
:root{{color-scheme:light}}
body{{margin:0;width:1600px;background:#fff;font-family:-apple-system,"Apple SD Gothic Neo","Noto Sans KR",sans-serif;color:#1d2330}}
h1{{text-align:center;margin:26px 0 4px;font-size:38px}} .sub{{text-align:center;color:#5b6475;font-size:17px;margin-bottom:22px}}
.row{{display:grid;gap:18px;padding:0 22px;margin-bottom:18px}} .r1{{grid-template-columns:1fr 1fr 1fr}} .r2{{grid-template-columns:1fr 1fr}}
.panel{{background:#f7f8fb;border-radius:22px;padding:14px 18px 18px}} .panel>h3{{text-align:center;margin:2px 0 12px;font-size:21px;color:#3b4ec8}}
.box{{border:2.5px solid;border-radius:14px;padding:9px 14px;margin:0}} .box h4{{margin:0 0 4px;font-size:17px}} .box p{{margin:0;font-size:13.5px;line-height:1.5;color:#2d3442}}
.arrow{{text-align:center;color:#8b6cdb;font-size:18px;line-height:20px}} .op{{margin-bottom:10px}}
.pair{{display:grid;grid-template-columns:1fr 1fr;gap:12px}} .roles{{display:grid;grid-template-columns:1fr auto 1fr;gap:6px;align-items:center}} .collab{{font-size:13px;color:#5b6475}}
.flow{{display:flex;align-items:center;gap:6px;flex-wrap:nowrap}} .sm{{padding:7px 10px}} .sm h4{{font-size:14.5px;margin:0}} .sm p{{font-size:12.5px}} .to{{font-size:20px;color:#5b6475}}
.outs{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:14px 0 6px 120px}} .note{{text-align:center;color:#5b6475;font-size:13.5px}}
.uses{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}
.insight{{margin:4px 22px 0;border:2.5px solid #c49a12;background:#fdf3cf;border-radius:14px;padding:12px 22px}} .insight p{{margin:4px 0;font-size:15px}}
.ih{{text-align:center;font-size:21px;font-weight:700;margin:12px 0 6px}}
.foot{{display:flex;justify-content:space-between;color:#8a92a3;font-size:12.5px;padding:10px 26px 18px}}
</style></head><body>
<h1>{e(TITLE)}</h1><div class="sub">{e(SUBTITLE)}</div>
<div class="row r1">
 <div class="panel"><h3>3층 구조</h3>{layers}</div>
 <div class="panel"><h3>3대 작업</h3>{ops}</div>
 <div class="panel"><h3>인덱스와 로그</h3><div class="pair">{index}</div><h3 style="margin-top:16px">역할 분담</h3><div class="roles">{roles}</div></div>
</div>
<div class="row r2">
 <div class="panel"><h3>Ingest 수집 흐름</h3><div class="flow">{flow}</div><div class="outs">{outs}</div><div class="note">{e(FLOW_NOTE)}</div></div>
 <div class="panel"><h3>활용 분야</h3><div class="uses">{uses}</div></div>
</div>
<div class="ih">핵심 통찰</div><div class="insight">{insight}</div>
<div class="foot"><span>원본: nashsu/llm_wiki 아키텍처 그림을 한국어로 다시 그림</span><span>@nash_su</span></div>
</body></html>"""


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as d:
        h = pathlib.Path(d) / "arch.html"
        h.write_text(page(), encoding="utf-8")
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=1.5",
                        "--window-size=1600,872", f"--screenshot={OUT}", h.as_uri()], check=True, capture_output=True)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
