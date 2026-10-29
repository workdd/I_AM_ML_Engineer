"""AWS 블로그 Deep Insight Part 1 노트의 그림을 생성한다.

    python scripts/figures/deepinsight_figures.py              readings/blogs/assets/deepinsight_*.svg 를 다시 쓴다
    python scripts/figures/deepinsight_figures.py --inline DIR  HTML 보고서용 (접두어·생성 메모 없이 DIR/<이름>.svg)

SVG 는 손으로 고치지 않는다. 구성은 원문(2026-04-01)의 에이전트 표·모델 표와
aws-samples/sample-deep-insight self-hosted/src/graph/builder.py (커밋 d9bc7b0) 의 엣지를 옮긴 아래 상수만 쓴다.
"""
import sys
from pathlib import Path

FONT = "-apple-system,'Apple SD Gothic Neo','Noto Sans KR',sans-serif"
INK, SUB, GRID = "#1f2328", "#59636e", "#d8dee4"
GRAY, BLUE, ORANGE, GREEN, PURPLE = "#8b949e", "#2563eb", "#ea580c", "#16a34a", "#7c3aed"
SHOW_NOTE = True
NOTE = "생성: python scripts/figures/deepinsight_figures.py · 손으로 고치지 말 것"
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "readings" / "blogs" / "assets"

# builder.py 의 그래프 노드와 엣지 (조건 이름)
GRAPH_NODES = ["coordinator", "planner", "plan_reviewer", "supervisor"]
GRAPH_EDGES = [
    ("coordinator", "planner", "should_handoff_to_planner"),
    ("planner", "plan_reviewer", None),
    ("plan_reviewer", "planner", "should_revise_plan"),
    ("plan_reviewer", "supervisor", "should_proceed_to_supervisor"),
]
MAX_NODE_EXECUTIONS = 25
MAX_PLAN_REVISIONS = 10  # nodes.py 기본값, 환경 변수로 변경
REVIEW_TIMEOUT_S = 300   # Web 버전, 무응답이면 자동 승인

# Supervisor 시스템 프롬프트의 필수 실행 순서와 각 단계 산출물
SUPERVISOR_ORDER = [
    ("Coder", "*.png · all_results.txt · calculation_metadata.json"),
    ("Tracker", "full_plan [x] 갱신"),
    ("Validator", "citations.json · validation_report.txt"),
    ("Tracker", "full_plan [x] 갱신"),
    ("Reporter", "final_report(_with_citations).docx"),
    ("Tracker", "full_plan [x] 갱신"),
]

# 원문 모델 표: (에이전트, 모델, Extended Thinking, Prompt Caching)
MODELS = [
    ("Coordinator", "Haiku", False, False),
    ("Planner", "Opus", True, False),
    ("Supervisor", "Sonnet", False, True),
    ("Coder", "Sonnet", False, True),
    ("Validator", "Sonnet", False, False),
    ("Reporter", "Sonnet", False, True),
    ("Tracker", "Sonnet", False, False),
]
MODEL_COLOR = {"Haiku": GREEN, "Opus": PURPLE, "Sonnet": BLUE}


def svg(w, h, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'role="img" aria-label="{title}" style="max-width:100%;height:auto;font-family:{FONT}">'
            f'<title>{title}</title><rect width="{w}" height="{h}" fill="#ffffff"/>'
            '<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            f'orient="auto-start-reverse"><path d="M0,0L10,5L0,10z" fill="{SUB}"/></marker></defs>{body}'
            + (f'<text x="{w-8}" y="{h-6}" font-size="9" fill="{GRAY}" text-anchor="end">{NOTE}</text>' if SHOW_NOTE else "")
            + '</svg>')


def t(x, y, s, size=12, fill=INK, anchor="start", weight="normal"):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}">{s}</text>')


def box(x, y, w, h, label, color, sub=None):
    out = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="#ffffff" stroke="{color}" stroke-width="1.8"/>'
    out += t(x + w / 2, y + (h / 2 + 4 if sub is None else h / 2 - 2), label, 12.5, INK, "middle", "bold")
    if sub:
        out += t(x + w / 2, y + h / 2 + 13, sub, 10, SUB, "middle")
    return out


def arrow(x1, y1, x2, y2, dash=False):
    d = ' stroke-dasharray="4 3"' if dash else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{SUB}" stroke-width="1.5"{d} marker-end="url(#a)"/>'


def fig_flow():
    w, h = 720, 490
    out = t(20, 28, "Deep Insight 실행 흐름: 그래프로 고정한 앞단과 Supervisor가 도구로 부르는 뒷단", 14.5, INK, weight="bold")
    out += t(20, 46, "위는 Strands Graph 노드와 조건부 엣지, 아래는 Supervisor 프롬프트가 강제하는 호출 순서와 각 단계의 산출 파일", 11.5, SUB)
    # 1계층 + 2계층: 그래프
    labels = {"coordinator": ("Coordinator", "라우팅 (Haiku)"), "planner": ("Planner", "체크리스트 계획 (Opus)"),
              "plan_reviewer": ("Plan Reviewer", "사람 검토 (HITL)"), "supervisor": ("Supervisor", "오케스트레이션 (Sonnet)")}
    bw, bh, y0, gap = 145, 50, 90, 32
    xs = {n: 20 + i * (bw + gap) for i, n in enumerate(GRAPH_NODES)}
    for n in GRAPH_NODES:
        c = ORANGE if n == "plan_reviewer" else BLUE
        out += box(xs[n], y0, bw, bh, *labels[n][:1], c, labels[n][1])
    for a, b, cond in GRAPH_EDGES:
        if xs[b] > xs[a]:
            out += arrow(xs[a] + bw, y0 + bh / 2, xs[b] - 2, y0 + bh / 2)
        else:  # 되돌아가는 수정 엣지
            x1, x2 = xs[a] + bw / 2, xs[b] + bw / 2
            out += (f'<path d="M{x1},{y0 + bh} C{x1},{y0 + bh + 40} {x2},{y0 + bh + 40} {x2},{y0 + bh + 3}" '
                    f'fill="none" stroke="{ORANGE}" stroke-width="1.5" stroke-dasharray="4 3" marker-end="url(#a)"/>')
            out += t((x1 + x2) / 2, y0 + bh + 44, f"수정 요청 (최대 {MAX_PLAN_REVISIONS}회, 기본값)", 10.5, ORANGE, "middle")
    out += t(xs["plan_reviewer"] + bw / 2, y0 - 8, f"{REVIEW_TIMEOUT_S}초 무응답이면 자동 승인", 10.5, ORANGE, "middle")
    out += t(20, y0 + bh + 62, f"그래프 전체 노드 실행 상한 {MAX_NODE_EXECUTIONS}회 (set_max_node_executions)", 10.5, SUB)
    # 3계층: Supervisor 호출 순서
    y1 = 272
    out += t(20, y1 - 46, "Supervisor의 필수 호출 순서 (agents-as-tools, 각 에이전트는 독립 context)", 12, INK, weight="bold")
    sx = xs["supervisor"] + bw / 2
    cx = 20 + 104 / 2
    out += (f'<path d="M{sx},{y0 + bh} L{sx},{y1 - 22} L{cx},{y1 - 22} L{cx},{y1 - 3}" fill="none" '
            f'stroke="{SUB}" stroke-width="1.5" stroke-dasharray="4 3" marker-end="url(#a)"/>')
    out += t(sx - 8, y1 - 28, "도구 호출", 10.5, SUB, "end")
    cw, ch, cg = 104, 46, 8
    for i, (name, artifact) in enumerate(SUPERVISOR_ORDER):
        x = 20 + i * (cw + cg)
        c = GRAY if name == "Tracker" else (GREEN if name == "Validator" else BLUE)
        out += box(x, y1, cw, ch, name, c)
        if i < len(SUPERVISOR_ORDER) - 1:
            out += arrow(x + cw, y1 + ch / 2, x + cw + cg - 1, y1 + ch / 2)
        parts = artifact.split(" · ")
        for j, p in enumerate(parts):
            out += t(x + cw / 2, y1 + ch + 16 + j * 13, p, 9.5, SUB, "middle")
    yl = y1 + ch + 70
    out += t(20, yl, "Tracker: 도구 없이 LLM 추론만으로 체크리스트를 [ ] → [x]로 갱신. 모든 항목이 [x]가 되기 전에는 다음 에이전트로 넘어가지 않는다", 10.5, SUB)
    out += t(20, yl + 18, "Validator: calculation_metadata.json의 계산을 원천 CSV에서 다시 해 0.01 이내면 통과, 통과한 수치만 citations.json에 올라 보고서 인용이 된다", 10.5, SUB)
    for i, (c, lab) in enumerate(((BLUE, "실행"), (ORANGE, "사람 검토"), (GREEN, "검증"), (GRAY, "상태 추적"))):
        out += f'<rect x="{20 + i * 110}" y="{h - 40}" width="11" height="11" rx="2" fill="{c}"/>' + t(36 + i * 110, h - 30, lab, 11)
    return svg(w, h, out, "Deep Insight 실행 흐름")


def fig_models():
    w, h = 600, 380
    L, T, rowh = 130, 86, 30
    cols = [("모델", 230), ("Extended Thinking", 360), ("Prompt Caching", 500)]
    out = t(20, 28, "에이전트별 모델과 설정 (원문 표)", 14.5, INK, weight="bold")
    out += t(20, 46, "계획 한 곳에만 큰 모델과 확장 사고를, 반복 호출되는 세 곳에만 프롬프트 캐시를 쓴다", 11.5, SUB)
    for name, x in cols:
        out += t(x, T - 12, name, 11.5, SUB, "middle")
    for i, (agent, model, think, cache) in enumerate(MODELS):
        y = T + i * rowh
        if i % 2 == 0:
            out += f'<rect x="20" y="{y}" width="{w - 40}" height="{rowh}" fill="#f6f8fa"/>'
        out += t(L - 10, y + 20, agent, 12.5, INK, "end", "bold")
        c = MODEL_COLOR[model]
        out += f'<rect x="{cols[0][1] - 34}" y="{y + 6}" width="68" height="{rowh - 12}" rx="9" fill="{c}"/>'
        out += t(cols[0][1], y + 20, model, 11.5, "#ffffff", "middle", "bold")
        for flag, (_, x) in ((think, cols[1]), (cache, cols[2])):
            out += (f'<circle cx="{x}" cy="{y + rowh / 2}" r="7" fill="{INK}"/>' if flag
                    else f'<circle cx="{x}" cy="{y + rowh / 2}" r="6" fill="none" stroke="{GRID}" stroke-width="1.5"/>')
    yb = T + len(MODELS) * rowh + 22
    out += t(20, yb, "Planner의 확장 사고 예산은 약 8,192토큰. 캐시는 시스템 프롬프트와 도구 정의를 대상으로 한다", 10.5, SUB)
    out += t(20, yb + 17, "원문의 \"입력 토큰 비용 최대 90% 절감\"은 캐시 일반 상한이고 이 시스템에서 잰 값이 아니다", 10.5, SUB)
    return svg(w, h, out, "에이전트별 모델과 설정")


FIGS = [("flow", fig_flow), ("models", fig_models)]


def main(argv):
    global SHOW_NOTE
    if argv[:1] == ["--inline"]:
        out, prefix, SHOW_NOTE = Path(argv[1]), "", False
    else:
        out, prefix = OUT, "deepinsight_"
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in FIGS:
        (out / f"{prefix}{name}.svg").write_text(fn(), "utf-8")
        print("wrote", out / f"{prefix}{name}.svg")


if __name__ == "__main__":
    main(sys.argv[1:])
