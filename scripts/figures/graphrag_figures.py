#!/usr/bin/env python3
"""RAGSearch · PathRAG · GraphRAG-Bench 노트의 그림을 논문 표 수치에서 생성한다.

    python scripts/figures/graphrag_figures.py

readings/papers/assets/ 아래 SVG 를 다시 쓴다. SVG 는 손으로 고치지 않는다.
표 수치는 아래 데이터 절에 원문 그대로 옮겼고, 파생 수치(격차, 평균)는 여기서 계산한다.
표준 라이브러리만 쓴다.
"""
from __future__ import annotations

from pathlib import Path
from statistics import mean, median
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "readings" / "papers" / "assets"
GEN_NOTE = "생성: python scripts/figures/graphrag_figures.py · 손으로 고치지 말 것"

# ---------------------------------------------------------------- 데이터: RAGSearch (arXiv:2604.09666)

DATASETS = ["NQ", "PopQA", "TriviaQA", "HotpotQA", "2Wiki", "Musique"]
MULTIHOP = ["HotpotQA", "2Wiki", "Musique"]

# Table 1. Contain-EM, 7B. GraphRAG 행은 5종 중 데이터셋별 최고값
RS_T1 = {
    "단일 검색 Dense":        [46.62, 32.14, 58.60, 19.00, 35.53, 20.99],
    "단일 검색 GraphRAG":     [48.31, 32.82, 57.65, 46.70, 62.56, 47.95],
    "Search-o1 Dense":        [38.20, 25.57, 58.74, 33.76, 29.64, 12.62],
    "Search-o1 GraphRAG":     [38.34, 28.01, 59.50, 42.75, 65.56, 32.44],
    "GraphSearch Dense":      [58.27, 36.29, 68.70, 38.22, 47.43, 13.33],
    "GraphSearch GraphRAG":   [61.22, 44.77, 72.47, 58.64, 79.88, 55.26],
    "Search-R1":              [48.72, 33.10, 63.96, 35.76, 33.56, 14.42],
    "Graph-R1":               [46.71, 36.23, 66.21, 53.42, 66.25, 40.82],
}

# Table 7. F1 (NQ, TriviaQA, HotpotQA, Musique 만 보고됨)
F1_DATASETS = ["NQ", "TriviaQA", "HotpotQA", "Musique"]
RS_T7 = {
    "단일 검색 Dense":        [39.30, 61.12, 20.12, 29.08],
    "단일 검색 GraphRAG":     [27.92, 49.25, 33.72, 41.13],
    "Search-o1 Dense":        [36.53, 59.73, 39.08, 17.46],
    "Search-o1 GraphRAG":     [36.62, 58.18, 41.57, 37.01],
    "GraphSearch Dense":      [8.70, 13.12, 6.02, 2.36],
    "GraphSearch GraphRAG":   [4.61, 14.18, 5.48, 4.21],
    "Search-R1":              [47.26, 59.21, 37.13, 16.02],
    "Graph-R1":               [44.21, 62.10, 43.25, 35.12],
}

# Table 2. 학습 없는 에이전트 x 백엔드, Contain-EM, 7B
BACKENDS = ["Dense", "HyperGraphRAG", "HippoRAG2", "LinearRAG", "RAPTOR", "MS GraphRAG"]
RS_T2 = {
    "Search-o1": {
        "Dense":         [38.20, 25.78, 58.74, 33.76, 29.64, 12.62],
        "HyperGraphRAG": [33.02, 25.57, 56.72, 33.90, 50.58, 28.05],
        "HippoRAG2":     [38.34, 28.01, 59.50, 42.75, 65.56, 32.44],
        "LinearRAG":     [34.32, 25.69, 57.03, 35.76, 58.94, 29.46],
        "RAPTOR":        [34.82, 23.20, 52.52, 29.51, 29.87, 29.50],
        "MS GraphRAG":   [35.10, 26.10, 56.89, 32.73, 54.25, 26.48],
    },
    "GraphSearch": {
        "Dense":         [58.27, 36.29, 68.70, 38.22, 47.43, 13.33],
        "HyperGraphRAG": [51.04, 44.72, 69.97, 46.83, 73.62, 54.80],
        "HippoRAG2":     [61.22, 43.65, 72.47, 58.64, 79.88, 55.10],
        "LinearRAG":     [52.12, 44.77, 68.52, 41.65, 70.26, 49.35],
        "RAPTOR":        [53.80, 42.38, 68.56, 40.14, 71.24, 55.26],
        "MS GraphRAG":   [52.78, 43.60, 69.64, 42.25, 72.41, 46.73],
    },
}

# Table 8. NQ 기준, 100만 토큰당 구축 비용($)과 평균 컨텍스트 토큰
RS_T8 = {
    "HyperGraphRAG": (3.93, 1680),
    "HippoRAG2":     (2.85, 3229),
    "LinearRAG":     (0.00, 4600),
    "RAPTOR":        (6.38, 814),
    "MS GraphRAG":   (13.19, 22160),
}

# ---------------------------------------------------------------- 데이터: PathRAG (arXiv:2502.14902)

PR_DATASETS = ["Legal", "History", "Biology", "Mix", "SQuALITY", "SummScreen"]
PR_DIMS = ["포괄성", "다양성", "논리성", "관련성", "일관성"]

# Table 1. PathRAG 의 승률(%). [평가 축][데이터셋]
PR_T1 = {
    "NaiveRAG": [
        [68.4, 66.8, 70.2, 73.8, 64.8, 70.0], [75.6, 61.6, 64.8, 66.8, 70.8, 75.8],
        [64.8, 59.6, 65.6, 63.8, 65.6, 69.8], [72.8, 62.8, 58.0, 61.6, 68.6, 67.0],
        [66.0, 57.6, 61.6, 58.0, 62.8, 69.8]],
    "HyDE": [
        [61.6, 65.2, 66.8, 57.2, 62.8, 69.8], [78.4, 64.8, 64.0, 66.2, 66.8, 70.0],
        [69.8, 61.6, 54.8, 54.4, 64.4, 64.8], [64.4, 64.4, 53.6, 56.6, 59.6, 68.6],
        [58.0, 59.6, 57.6, 54.4, 60.0, 64.4]],
    "G-retriever": [
        [66.2, 58.8, 56.4, 72.6, 64.8, 55.8], [64.8, 56.4, 68.0, 75.6, 61.6, 70.0],
        [65.6, 58.0, 60.0, 69.8, 59.8, 55.2], [64.4, 56.0, 61.6, 63.8, 60.0, 58.8],
        [62.0, 53.4, 64.8, 65.6, 62.8, 56.4]],
    "HippoRAG": [
        [65.6, 56.4, 54.0, 64.2, 57.0, 70.0], [62.0, 61.6, 75.6, 62.8, 72.8, 73.6],
        [65.6, 54.4, 58.2, 59.8, 52.8, 67.0], [59.8, 56.8, 60.0, 56.0, 53.4, 65.6],
        [58.8, 55.6, 56.4, 54.6, 57.4, 68.2]],
    "MS GraphRAG": [
        [66.2, 59.0, 60.4, 58.8, 58.0, 63.0], [70.2, 63.4, 61.8, 63.8, 61.6, 58.8],
        [58.4, 56.4, 65.6, 58.0, 58.0, 60.0], [59.4, 56.0, 57.6, 59.6, 57.6, 55.6],
        [61.8, 59.2, 56.4, 58.4, 58.4, 56.4]],
    "LightRAG": [
        [63.4, 56.0, 57.4, 59.6, 56.0, 53.6], [61.8, 56.8, 56.4, 58.0, 56.8, 53.6],
        [62.8, 58.4, 54.8, 56.4, 55.2, 55.2], [60.0, 56.0, 55.2, 56.0, 54.4, 55.6],
        [61.2, 55.6, 55.6, 61.6, 55.6, 53.6]],
}

# Table 2 · 3. 변형 대비 PathRAG 승률(%)
PR_ABL = {
    "무작위 정렬 대비": [
        [56.0, 54.0, 56.0, 57.2, 54.4, 53.4], [54.8, 68.6, 70.2, 54.0, 55.6, 57.2],
        [53.4, 56.0, 58.0, 53.6, 58.2, 56.4], [55.2, 54.0, 54.2, 54.4, 54.4, 55.2],
        [55.4, 59.0, 58.4, 56.0, 56.4, 53.6]],
    "홉 수 우선 정렬 대비": [
        [55.6, 54.2, 51.2, 56.8, 54.8, 53.6], [64.0, 50.4, 54.0, 52.4, 56.0, 54.6],
        [54.8, 58.8, 55.2, 56.4, 54.0, 54.0], [56.4, 54.0, 62.6, 58.6, 55.2, 53.2],
        [59.0, 60.0, 57.4, 55.2, 53.2, 53.6]],
    "평평한 프롬프트 대비": [
        [60.0, 51.2, 54.4, 50.4, 52.8, 53.2], [58.0, 60.4, 55.6, 56.8, 55.6, 54.8],
        [62.8, 54.4, 52.0, 58.0, 52.4, 56.8], [55.2, 51.0, 52.6, 55.2, 54.6, 54.0],
        [60.8, 54.4, 55.4, 57.6, 52.0, 53.2]],
}
PR_LT_VS_LIGHTRAG = 50.56  # 본문 수치. 세부 표는 부록 I 에 있다는데 arXiv 판에 부록이 없다

# Table 6. 질의당 토큰
PR_T6 = {"LightRAG": 16728, "PathRAG-lt": 9968, "PathRAG": 14438}

# ---------------------------------------------------------------- 파생 수치


def multihop_gap(dense: list[float], graph: list[float]) -> float:
    idx = [DATASETS.index(d) for d in MULTIHOP]
    return mean(graph[i] - dense[i] for i in idx)


def rs_gaps() -> dict[str, float]:
    t = RS_T1
    return {
        "단일 검색": multihop_gap(t["단일 검색 Dense"], t["단일 검색 GraphRAG"]),
        "Search-o1": multihop_gap(t["Search-o1 Dense"], t["Search-o1 GraphRAG"]),
        "GraphSearch": multihop_gap(t["GraphSearch Dense"], t["GraphSearch GraphRAG"]),
        "RL (Search-R1 · Graph-R1)": multihop_gap(t["Search-R1"], t["Graph-R1"]),
    }


def em_f1_pairs() -> dict[str, tuple[float, float]]:
    idx = [DATASETS.index(d) for d in F1_DATASETS]
    return {k: (mean(RS_T1[k][i] for i in idx), mean(RS_T7[k])) for k in RS_T1}


def multihop_avg(row: list[float]) -> float:
    return mean(row[DATASETS.index(d)] for d in MULTIHOP)


def pr_avg(table: list[list[float]]) -> float:
    return mean(v for dim in table for v in dim)


def pr_by_dataset(table: list[list[float]]) -> list[float]:
    return [mean(dim[j] for dim in table) for j in range(len(PR_DATASETS))]


# ---------------------------------------------------------------- PathRAG 흐름 전파 (공개 코드 이식)
# BUPT-GAMMA/PathRAG PathRAG/operate.py 의 find_paths_and_edges_with_stats, bfs_weighted_paths
# 를 그대로 옮겼다. 논문 본문(α=0.7, 노드 자원 합)과 다르다: α=0.8, θ=0.3, 3홉 이하, 엣지 가중 평균.

FLOW_ALPHA, FLOW_THETA = 0.8, 0.3
FLOW_EDGES = [("A", "H"), ("H", "T"), ("H", "D"), ("D", "T"), ("A", "B"), ("B", "C"),
              ("C", "T"), ("H", "x1"), ("H", "x2")]


def candidate_paths(edges, source, target, max_hops=3):
    adj: dict[str, list[str]] = {}
    for u, v in edges:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
    found = []

    def dfs(cur, path, depth):
        if depth > max_hops:
            return
        if cur == target:
            found.append(list(path))
            return
        for nb in adj[cur]:
            if nb not in path:
                dfs(nb, path + [nb], depth + 1)

    dfs(source, [source], 0)
    return found


def flow_weights(paths, source, target, alpha=FLOW_ALPHA, theta=FLOW_THETA):
    follow: dict[str, set[str]] = {}
    for p in paths:
        for a, b in zip(p, p[1:]):
            follow.setdefault(a, set()).add(b)
    w: dict[tuple[str, str], float] = {}
    for n1 in sorted(follow[source]):
        w[(source, n1)] = w.get((source, n1), 0) + 1 / len(follow[source])
        if n1 == target or w[(source, n1)] <= theta:
            continue
        for n2 in sorted(follow[n1]):
            w[(n1, n2)] = w.get((n1, n2), 0) + w[(source, n1)] * alpha / len(follow[n1])
            if n2 == target or w[(n1, n2)] <= theta:
                continue
            for n3 in sorted(follow[n2]):
                w[(n2, n3)] = w.get((n2, n3), 0) + w[(n1, n2)] * alpha / len(follow[n2])
    scores = {tuple(p): mean(w.get((a, b), 0.0) for a, b in zip(p, p[1:])) for p in paths}
    return w, follow, scores


# ---------------------------------------------------------------- SVG 공통

FONT = "'Noto Sans KR','Apple SD Gothic Neo','Malgun Gothic',sans-serif"
STYLE = f"""<style>
svg{{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#8a8984;--grid:#e6e5e1;--rule:#c9c8c3;--s1:#2a78d6;--s2:#eb6834;--faint:#d9d8d3}}
@media (prefers-color-scheme:dark){{svg{{--surface:#1a1a19;--ink:#ffffff;--ink2:#c3c2b7;--muted:#8f8e88;--grid:#343431;--rule:#4a4945;--s1:#3987e5;--s2:#d95926;--faint:#4a4945}}}}
text{{font-family:{FONT};fill:var(--ink);font-size:13px}}
.bg{{fill:var(--surface)}} .t2{{fill:var(--ink2)}} .mu{{fill:var(--muted);font-size:11px}}
.ttl{{font-size:16px;font-weight:700}} .sub{{fill:var(--ink2);font-size:12.5px}}
.grid{{stroke:var(--grid);stroke-width:1}} .rule{{stroke:var(--rule);stroke-width:1}}
.s1f{{fill:var(--s1)}} .s2f{{fill:var(--s2)}} .s1s{{stroke:var(--s1)}} .s2s{{stroke:var(--s2)}}
.ring{{stroke:var(--surface);stroke-width:2}} .ref{{stroke:var(--ink2);stroke-width:1.5}}
.faintf{{fill:var(--faint)}} .fainte{{stroke:var(--faint)}} .inkS{{stroke:var(--ink2)}}
.b{{font-weight:700}}
</style>"""


def t(x, y, s, cls="", anchor="start", size=None, weight=None, extra=""):
    a = f' class="{cls}"' if cls else ""
    st = []
    if size:
        st.append(f"font-size:{size}px")
    if weight:
        st.append(f"font-weight:{weight}")
    sa = f' style="{";".join(st)}"' if st else ""
    return f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}"{a}{sa}{extra}>{escape(str(s))}</text>'


def doc(w, h, title, subtitle, body, source):
    head = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
        f'role="img" aria-label="{escape(title)}">',
        f"<title>{escape(title)}</title>",
        STYLE,
        f'<rect class="bg" x="0" y="0" width="{w}" height="{h}" rx="10"/>',
        t(24, 34, title, "ttl"),
        t(24, 55, subtitle, "sub"),
    ]
    foot = [t(24, h - 26, source, "mu"), t(24, h - 11, GEN_NOTE, "mu")]
    return "\n".join(head + body + foot) + "\n</svg>\n"


def fmt(v, d=1):
    return f"{v:.{d}f}"


# 순차 파랑 (dataviz 기준 팔레트 100..700)
BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
        "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]


def ramp(frac):
    frac = max(0.0, min(1.0, frac))
    i = round(frac * (len(BLUE) - 1))
    c = BLUE[i]
    ink = "#ffffff" if i >= 6 else "#0b0b0b"
    return c, ink


# ---------------------------------------------------------------- RAGSearch 그림


def fig_rs_gap():
    gaps = rs_gaps()
    w, h = 780, 350
    x0, x1 = 230, 660
    top, rowh = 92, 40
    vmax = 35
    sx = lambda v: x0 + (x1 - x0) * v / vmax
    body = []
    for v in range(0, vmax + 1, 5):
        body.append(f'<line class="grid" x1="{sx(v):.1f}" y1="{top - 12}" x2="{sx(v):.1f}" y2="{top + rowh * len(gaps) - 8}"/>')
        body.append(t(sx(v), top + rowh * len(gaps) + 8, v, "mu", "middle"))
    base = gaps["단일 검색"]
    for i, (k, v) in enumerate(gaps.items()):
        y = top + i * rowh
        body.append(t(x0 - 12, y + 14, k, "", "end"))
        cls = "faintf" if k == "단일 검색" else "s1f"
        body.append(f'<rect class="{cls}" x="{x0}" y="{y}" width="{sx(v) - x0:.1f}" height="20" rx="4"/>')
        body.append(t(sx(v) + 8, y + 15, fmt(v, 2), "b"))
        if k != "단일 검색":
            body.append(t(sx(v) + 58, y + 15, f"기준 대비 {v - base:+.2f}", "t2", size=12))
    authors = mean([gaps["Search-o1"], gaps["GraphSearch"]])
    body.append(t(24, top + rowh * len(gaps) + 40,
                  f"저자가 보고한 '에이전트 적용 후 26.59'는 Search-o1과 GraphSearch 격차의 평균({fmt(authors, 2)})입니다. "
                  "GraphSearch만 보면 격차는 더 커졌습니다.", "t2", size=12))
    return doc(w, h, "멀티홉 QA에서 GraphRAG가 dense RAG보다 높은 폭",
               "HotpotQA · 2Wiki · Musique 평균 Contain-EM 차. 회색 = 에이전트 없는 단일 검색 기준",
               body, "출처: RAGSearch (arXiv:2604.09666) Table 1에서 계산")


def fig_rs_heatmap():
    w = 800
    lab_w, cw, ch = 150, 96, 30
    x0 = 24 + lab_w
    body = []
    y = 84
    for agent, table in RS_T2.items():
        body.append(t(24, y, agent, "b", size=14))
        y += 12
        body.append(t(x0 + cw * 1.5, y + 10, "일반 QA", "t2", "middle", size=12))
        body.append(t(x0 + cw * 4.5, y + 10, "멀티홉 QA", "t2", "middle", size=12))
        y += 18
        for j, d in enumerate(DATASETS):
            body.append(t(x0 + cw * j + cw / 2, y + 10, d, "t2", "middle", size=12))
        y += 18
        cols = list(zip(*table.values()))
        for r, name in enumerate(BACKENDS):
            row = table[name]
            yy = y + r * ch
            body.append(t(x0 - 12, yy + ch / 2 + 5, name, "b" if name == "HippoRAG2" else "", "end"))
            for j, v in enumerate(row):
                lo, hi = min(cols[j]), max(cols[j])
                c, ink = ramp((v - lo) / (hi - lo) if hi > lo else 0)
                xx = x0 + cw * j
                body.append(f'<rect x="{xx + 1:.1f}" y="{yy + 1:.1f}" width="{cw - 2}" height="{ch - 2}" rx="3" fill="{c}"/>')
                best = v == hi
                body.append(f'<text x="{xx + cw / 2:.1f}" y="{yy + ch / 2 + 5:.1f}" text-anchor="middle" '
                            f'style="fill:{ink};font-size:12.5px{";font-weight:700" if best else ""}">'
                            f'{fmt(v, 2)}{" ▲" if best else ""}</text>')
        body.append(f'<line class="rule" x1="{x0 + cw * 3:.1f}" y1="{y - 20}" x2="{x0 + cw * 3:.1f}" y2="{y + ch * 6}"/>')
        y += ch * 6 + 36
    # 범례
    ly = y - 10
    body.append(t(24, ly + 11, "열 안 상대 위치:", "t2", size=12))
    for i, c in enumerate(BLUE):
        body.append(f'<rect x="{130 + i * 16}" y="{ly}" width="16" height="14" fill="{c}"/>')
    body.append(t(130, ly + 30, "열 최저", "mu"))
    body.append(t(130 + 16 * len(BLUE), ly + 30, "열 최고", "mu", "end"))
    body.append(t(360, ly + 11, "▲ 굵은 숫자 = 열 1위. 색은 데이터셋마다 따로 맞췄으므로 열끼리 비교하지 않습니다.", "t2", size=12))
    h = ly + 76
    return doc(w, h, "백엔드별 Contain-EM: 에이전트는 고정, 검색 백엔드만 교체",
               "Qwen2.5-7B. 멀티홉 세 열에서 dense는 두 에이전트 모두 하위권, HippoRAG2는 대부분 1위",
               body, "출처: RAGSearch (arXiv:2604.09666) Table 2")


def fig_rs_cost():
    w, h = 780, 440
    x0, x1, y0, y1 = 90, 720, 360, 90
    xmax, ymin, ymax = 14, 20, 70
    sx = lambda v: x0 + (x1 - x0) * v / xmax
    sy = lambda v: y0 - (y0 - y1) * (v - ymin) / (ymax - ymin)
    body = []
    for v in range(ymin, ymax + 1, 10):
        body.append(f'<line class="grid" x1="{x0}" y1="{sy(v):.1f}" x2="{x1}" y2="{sy(v):.1f}"/>')
        body.append(t(x0 - 10, sy(v) + 4, v, "mu", "end"))
    for v in range(0, xmax + 1, 2):
        body.append(t(sx(v), y0 + 18, f"${v}", "mu", "middle"))
    body.append(t((x0 + x1) / 2, y0 + 40, "그래프 구축 비용 (NQ, 100만 토큰당, GPT-4o-mini)", "t2", "middle", size=12))
    body.append(f'<text x="{x0 - 52}" y="{(y0 + y1) / 2:.1f}" text-anchor="middle" class="t2" style="font-size:12px" '
                f'transform="rotate(-90 {x0 - 52} {(y0 + y1) / 2:.1f})">멀티홉 평균 Contain-EM</text>')
    for agent, cls, off in (("Search-o1", "s2f", 0), ("GraphSearch", "s1f", 0)):
        dense = multihop_avg(RS_T2[agent]["Dense"])
        body.append(f'<line class="{cls.replace("f", "s")}" x1="{x0}" y1="{sy(dense):.1f}" x2="{x1}" y2="{sy(dense):.1f}" '
                    f'style="stroke-width:1.5;opacity:.55"/>')
        body.append(t(x1, sy(dense) - 6, f"{agent} + dense {fmt(dense)}", "t2", "end", size=11.5))
    for name, (cost, ctx) in RS_T8.items():
        gs = multihop_avg(RS_T2["GraphSearch"][name])
        so = multihop_avg(RS_T2["Search-o1"][name])
        x = sx(cost)
        body.append(f'<line class="inkS" x1="{x:.1f}" y1="{sy(so):.1f}" x2="{x:.1f}" y2="{sy(gs):.1f}" style="stroke-width:1;opacity:.4"/>')
        body.append(f'<circle class="s2f ring" cx="{x:.1f}" cy="{sy(so):.1f}" r="6"/>')
        body.append(f'<circle class="s1f ring" cx="{x:.1f}" cy="{sy(gs):.1f}" r="7"/>')
        anchor = "end" if cost > 12 else "start"
        dx = -12 if anchor == "end" else 12
        body.append(t(x + dx, sy(gs) - 4, name, "b" if name == "HippoRAG2" else "", anchor))
        body.append(t(x + dx, sy(gs) + 12, f"${fmt(cost, 2)} · {ctx:,} 토큰", "mu", anchor))
    # 범례
    body.append(f'<circle class="s1f" cx="{x0 + 10}" cy="72" r="6"/>')
    body.append(t(x0 + 22, 76, "GraphSearch", size=12))
    body.append(f'<circle class="s2f" cx="{x0 + 130}" cy="72" r="6"/>')
    body.append(t(x0 + 142, 76, "Search-o1", size=12))
    body.append(t(x0 + 230, 76, "토큰 = 검색 시 평균 컨텍스트 · 가로선 = 같은 에이전트의 dense RAG", "mu"))
    return doc(w, h, "구축 비용과 멀티홉 성능은 비례하지 않는다",
               "가장 비싼 MS GraphRAG($13.19)보다 HippoRAG2($2.85)가 두 에이전트 모두에서 높다",
               body, "출처: RAGSearch (arXiv:2604.09666) Table 2, Table 8")


def fig_rs_em_f1():
    pairs = em_f1_pairs()
    w = 800
    x0, x1 = 220, 640
    top, rowh = 124, 32
    vmax = 70
    sx = lambda v: x0 + (x1 - x0) * v / vmax
    h = top + rowh * len(pairs) + 86
    body = []
    for v in range(0, vmax + 1, 10):
        body.append(f'<line class="grid" x1="{sx(v):.1f}" y1="{top - 14}" x2="{sx(v):.1f}" y2="{top + rowh * len(pairs) - 12}"/>')
        body.append(t(sx(v), top + rowh * len(pairs) + 4, v, "mu", "middle"))
    body.append(t(x1 + 60, top - 20, "차이", "t2", "middle", size=12))
    for i, (k, (em, f1)) in enumerate(pairs.items()):
        y = top + i * rowh
        hot = k.startswith("GraphSearch")
        body.append(t(x0 - 14, y + 4, k, "b" if hot else "", "end"))
        body.append(f'<line class="inkS" x1="{sx(min(em, f1)):.1f}" y1="{y}" x2="{sx(max(em, f1)):.1f}" y2="{y}" '
                    f'style="stroke-width:{3 if hot else 2};opacity:{.9 if hot else .35}"/>')
        body.append(f'<circle class="s2f ring" cx="{sx(f1):.1f}" cy="{y}" r="6"/>')
        body.append(f'<circle class="s1f ring" cx="{sx(em):.1f}" cy="{y}" r="6"/>')
        body.append(t(x1 + 60, y + 4, f"{em - f1:+.1f}", "b" if hot else "t2", "middle"))
    ly = 84
    body.append(f'<circle class="s1f" cx="{x0}" cy="{ly - 4}" r="6"/>')
    body.append(t(x0 + 12, ly, "Contain-EM (정답 문자열이 응답 안에 있으면 정답)", size=12))
    body.append(f'<circle class="s2f" cx="{x0 + 330}" cy="{ly - 4}" r="6"/>')
    body.append(t(x0 + 342, ly, "토큰 F1", size=12))
    body.append(t(24, top + rowh * len(pairs) + 30,
                  "GraphSearch는 Contain-EM으로 1위지만 F1은 한 자릿수입니다. 답을 길게 늘어놓아 정답이 우연히 포함됐을 가능성이 큽니다.",
                  "t2", size=12))
    return doc(w, h, "같은 답, 다른 지표: Contain-EM과 F1",
               "NQ · TriviaQA · HotpotQA · Musique 평균 (F1이 보고된 네 데이터셋)",
               body, "출처: RAGSearch (arXiv:2604.09666) Table 1, Table 7에서 계산")


# ---------------------------------------------------------------- PathRAG 그림


def fig_pr_winrate():
    groups = [
        ("기준선 대비", [(k, pr_avg(v)) for k, v in PR_T1.items()]),
        ("변형 비교 (같은 PathRAG의 다른 설정 대비)", [(k, pr_avg(v)) for k, v in PR_ABL.items()]
         + [("PathRAG-lt의 LightRAG 대비", PR_LT_VS_LIGHTRAG)]),
    ]
    w = 760
    x0, x1 = 250, 700
    vmin, vmax = 45, 70
    sx = lambda v: x0 + (x1 - x0) * (v - vmin) / (vmax - vmin)
    rowh = 28
    body = []
    y = 96
    rows_total = sum(len(g[1]) for g in groups)
    y_end = y + rows_total * rowh + len(groups) * 30
    for v in range(vmin, vmax + 1, 5):
        body.append(f'<line class="grid" x1="{sx(v):.1f}" y1="{y - 8}" x2="{sx(v):.1f}" y2="{y_end - 16}"/>')
        body.append(t(sx(v), y_end, f"{v}%", "mu", "middle"))
    body.append(f'<line class="ref" x1="{sx(50):.1f}" y1="{y - 12}" x2="{sx(50):.1f}" y2="{y_end - 16}"/>')
    body.append(t(sx(50), y - 16, "50% = 동률", "t2", "middle", size=12))
    for gname, rows in groups:
        body.append(t(24, y + 12, gname, "b", size=13.5))
        y += 30
        for name, v in rows:
            body.append(t(x0 - 14, y + 4, name, "", "end"))
            body.append(f'<line class="s1s" x1="{sx(50):.1f}" y1="{y}" x2="{sx(v):.1f}" y2="{y}" style="stroke-width:2;opacity:.45"/>')
            body.append(f'<circle class="s1f ring" cx="{sx(v):.1f}" cy="{y}" r="6.5"/>')
            body.append(t(sx(v) + 12, y + 4, f"{v:.2f}%", "b"))
            y += rowh
    h = y_end + 52
    return doc(w, h, "PathRAG 승률: 모두 50%를 넘지만 폭은 좁다",
               "GPT-4o-mini가 두 답을 쌍으로 비교한 승률. 6개 데이터셋 × 5개 평가 축 평균",
               body, "출처: PathRAG (arXiv:2502.14902) Table 1-3에서 계산, PathRAG-lt는 본문 수치")


def fig_pr_heatmap():
    w = 800
    lab_w, cw, ch = 130, 100, 32
    x0 = 24 + lab_w
    body = []
    y = 88
    for j, d in enumerate(PR_DATASETS):
        body.append(t(x0 + cw * j + cw / 2, y, d, "t2", "middle", size=12))
    y += 10
    lo, hi = 50, 72
    for r, (name, table) in enumerate(PR_T1.items()):
        yy = y + r * ch
        body.append(t(x0 - 12, yy + ch / 2 + 5, name, "", "end"))
        for j, v in enumerate(pr_by_dataset(table)):
            c, ink = ramp((v - lo) / (hi - lo))
            xx = x0 + cw * j
            body.append(f'<rect x="{xx + 1:.1f}" y="{yy + 1:.1f}" width="{cw - 2}" height="{ch - 2}" rx="3" fill="{c}"/>')
            body.append(f'<text x="{xx + cw / 2:.1f}" y="{yy + ch / 2 + 5:.1f}" text-anchor="middle" '
                        f'style="fill:{ink};font-size:12.5px">{v:.1f}%</text>')
    y += ch * len(PR_T1) + 22
    body.append(t(24, y + 11, "승률:", "t2", size=12))
    for i, c in enumerate(BLUE):
        body.append(f'<rect x="{70 + i * 16}" y="{y}" width="16" height="14" fill="{c}"/>')
    body.append(t(70, y + 30, f"{lo}%", "mu"))
    body.append(t(70 + 16 * len(BLUE), y + 30, f"{hi}%", "mu", "end"))
    lr = pr_by_dataset(PR_T1["LightRAG"])
    body.append(t(300, y + 11, f"아래로 갈수록 강한 기준선입니다. LightRAG 행은 {min(lr):.1f}~{max(lr):.1f}%입니다.", "t2", size=12))
    h = y + 76
    return doc(w, h, "기준선 × 데이터셋별 PathRAG 승률",
               "평가 축 5개(포괄성·다양성·논리성·관련성·일관성) 평균",
               body, "출처: PathRAG (arXiv:2502.14902) Table 1에서 계산")


def fig_pr_tokens():
    w, h = 760, 280
    x0, x1 = 150, 620
    top, rowh = 96, 44
    vmax = 18000
    sx = lambda v: x0 + (x1 - x0) * v / vmax
    body = []
    for v in range(0, vmax + 1, 3000):
        body.append(f'<line class="grid" x1="{sx(v):.1f}" y1="{top - 10}" x2="{sx(v):.1f}" y2="{top + rowh * 3 - 16}"/>')
        body.append(t(sx(v), top + rowh * 3, f"{v // 1000}k", "mu", "middle"))
    note = {"LightRAG": "기준",
            "PathRAG-lt": f"토큰 {1 - PR_T6['PathRAG-lt'] / PR_T6['LightRAG']:.2%} 절감 · 승률 {PR_LT_VS_LIGHTRAG}%",
            "PathRAG": f"토큰 {1 - PR_T6['PathRAG'] / PR_T6['LightRAG']:.2%} 절감 · 승률 {pr_avg(PR_T1['LightRAG']):.2f}%"}
    for i, (k, v) in enumerate(PR_T6.items()):
        y = top + i * rowh
        cls = "faintf" if k == "LightRAG" else "s1f"
        body.append(t(x0 - 12, y + 15, k, "b" if k == "PathRAG" else "", "end"))
        body.append(f'<rect class="{cls}" x="{x0}" y="{y}" width="{sx(v) - x0:.1f}" height="22" rx="4"/>')
        body.append(t(sx(v) + 8, y + 16, f"{v:,}", "b"))
        body.append(t(sx(v) + 60, y + 16, note[k], "t2", size=12))
    return doc(w, h, "질의당 토큰: LightRAG 대비",
               "PathRAG는 N=40, K=15, PathRAG-lt는 N=20, K=5. 승률은 각각의 LightRAG 대비 값",
               body, "출처: PathRAG (arXiv:2502.14902) Table 6, Table 1")


def fig_pr_flow():
    paths = candidate_paths(FLOW_EDGES, "A", "T")
    w_, follow, scores = flow_weights(paths, "A", "T")
    best = max(scores, key=scores.get)
    best_edges = set(zip(best, best[1:]))
    on_path = {(a, b) for p in paths for a, b in zip(p, p[1:])}
    pos = {"A": (90, 300), "H": (290, 210), "T": (560, 300), "D": (450, 160),
           "B": (230, 400), "C": (420, 400), "x1": (200, 130), "x2": (310, 115)}
    W, H = 860, 570
    body = []
    body.append('<defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
                '<path d="M0,0 L10,5 L0,10 z" style="fill:var(--ink2)"/></marker>'
                '<marker id="arb" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
                '<path d="M0,0 L10,5 L0,10 z" style="fill:var(--s1)"/></marker></defs>')
    R = 22

    def seg(a, b):
        (xa, ya), (xb, yb) = pos[a], pos[b]
        dx, dy = xb - xa, yb - ya
        L = (dx * dx + dy * dy) ** 0.5
        return xa + dx / L * R, ya + dy / L * R, xb - dx / L * (R + 3), yb - dy / L * (R + 3)

    for u, v in FLOW_EDGES:
        a, b = (u, v) if (u, v) in on_path else (v, u) if (v, u) in on_path else (u, v)
        x_1, y_1, x_2, y_2 = seg(a, b)
        if (a, b) not in on_path:
            body.append(f'<line class="fainte" x1="{x_1:.1f}" y1="{y_1:.1f}" x2="{x_2:.1f}" y2="{y_2:.1f}" style="stroke-width:1.5"/>')
            continue
        hot = (a, b) in best_edges
        mk = "arb" if hot else "ar"
        cls = "s1s" if hot else "inkS"
        body.append(f'<line class="{cls}" x1="{x_1:.1f}" y1="{y_1:.1f}" x2="{x_2:.1f}" y2="{y_2:.1f}" '
                    f'style="stroke-width:{3 if hot else 1.6}" marker-end="url(#{mk})"/>')
        wv = w_.get((a, b))
        mx, my = (pos[a][0] + pos[b][0]) / 2, (pos[a][1] + pos[b][1]) / 2
        label = fmt(wv, 2) if wv is not None else "0 · 중단"
        tw = 8 * len(label) + 10
        body.append(f'<rect class="bg" x="{mx - tw / 2:.1f}" y="{my - 11:.1f}" width="{tw}" height="20" rx="4"/>')
        body.append(t(mx, my + 4, label, "b" if hot else "t2", "middle", size=12.5))
    for n, (x, y) in pos.items():
        faint = n.startswith("x")
        endpoint = n in ("A", "T")
        body.append(f'<circle cx="{x}" cy="{y}" r="{R}" class="{"faintf" if faint else "bg"}" '
                    f'style="stroke:var({"--faint" if faint else "--s1" if endpoint else "--ink2"});stroke-width:{2.5 if endpoint else 1.5}"/>')
        body.append(t(x, y + 5, n, "b" if not faint else "mu", "middle", size=14 if not faint else 12))
    body.append(t(pos["x2"][0] + 32, pos["x2"][1] + 4, "← H의 다른 이웃은 후보 경로 밖이라 나눗셈에서 빠짐", "mu"))
    body.append(t(pos["H"][0] - 58, pos["H"][1] + 8, f"갈래 {len(follow['H'])}", "mu", "end"))
    # 오른쪽 점수표
    px = 630
    body.append(t(px, 170, "후보 경로 (3홉 이하)", "b"))
    body.append(t(px, 190, "점수 = 경로 엣지 가중치의 평균", "mu"))
    for i, (p, s) in enumerate(sorted(scores.items(), key=lambda kv: -kv[1])):
        yy = 222 + i * 30
        hot = p == best
        body.append(t(px, yy, " → ".join(p), "b" if hot else "", size=13))
        body.append(t(px + 200, yy, fmt(s, 3), "b" if hot else "t2", "end", size=13))
    body.append(t(px, 222 + len(scores) * 30 + 6, "채택 = 파란 경로", "t2", size=12))
    # 계산 규칙
    ry = 470
    body.append(t(24, ry, f"A에서 나가는 엣지는 1 ÷ (A의 갈래 수)로 시작하고, 한 홉마다 × α({FLOW_ALPHA}) ÷ (다음 노드의 갈래 수)로 줄어듭니다.", "t2", size=12.5))
    body.append(t(24, ry + 20, f"가중치가 θ({FLOW_THETA}) 이하인 엣지에서는 더 전파하지 않습니다. 갈래 수는 A와 T 사이 후보 경로에서만 셉니다.", "t2", size=12.5))
    body.append(t(24, ry + 40, "2홉인 A→H→T가 H의 갈림 때문에 약해지고, 3홉이지만 갈림 없는 A→B→C→T가 채택됩니다.", "t2", size=12.5))
    return doc(W, H, "흐름 전파 예시: 짧은 경로보다 갈림 없는 경로",
               "PathRAG 공개 코드(operate.py)의 경로 점수 계산을 그대로 옮겨 계산한 값",
               body, "출처: BUPT-GAMMA/PathRAG PathRAG/operate.py bfs_weighted_paths 이식, 예시 그래프는 임의 구성")


# ---------------------------------------------------------------- 데이터: GraphRAG-Bench (arXiv:2506.05690)

GB_TASKS = ["사실 검색", "복합 추론", "맥락 요약", "창작 생성"]
GB_RAG = "RAG (rerank)"
GB_RAG0 = "RAG (rerank 없음)"
GB_GRAPH = ["MS-GraphRAG local", "MS-GraphRAG global", "HippoRAG", "HippoRAG2", "LightRAG",
            "Fast-GraphRAG", "RAPTOR", "Lazy-GraphRAG", "KGP", "StructRAG", "KET-RAG"]

# Table 9. 생성 정확도(ACC), GPT-4o-mini. [사실 검색, 복합 추론, 맥락 요약, 창작 생성]
GB_ACC = {
    "Novel": {
        GB_RAG0: [58.76, 41.35, 50.08, 41.52], GB_RAG: [60.92, 42.93, 51.30, 38.26],
        "MS-GraphRAG local": [49.29, 50.93, 64.40, 39.10], "MS-GraphRAG global": [36.92, 43.17, 56.87, 41.11],
        "HippoRAG": [52.93, 38.52, 48.70, 38.85], "HippoRAG2": [60.14, 53.38, 64.10, 48.28],
        "LightRAG": [58.62, 49.07, 48.85, 23.80], "Fast-GraphRAG": [56.95, 48.55, 56.41, 46.18],
        "RAPTOR": [49.25, 38.59, 47.10, 38.01], "Lazy-GraphRAG": [51.65, 49.22, 58.29, 43.23],
        "KGP": [54.15, 46.31, 51.21, 40.37], "StructRAG": [53.84, 46.27, 54.28, 42.16],
        "KET-RAG": [55.39, 36.59, 52.47, 46.03],
    },
    "Medical": {
        GB_RAG0: [63.72, 57.61, 63.72, 58.94], GB_RAG: [64.73, 58.64, 65.75, 60.61],
        "MS-GraphRAG local": [38.63, 47.04, 41.87, 53.11], "MS-GraphRAG global": [16.42, 15.61, 19.82, 20.81],
        "HippoRAG": [56.14, 55.87, 59.86, 64.43], "HippoRAG2": [66.28, 61.98, 63.08, 68.05],
        "LightRAG": [63.32, 61.32, 63.14, 67.91], "Fast-GraphRAG": [60.93, 61.73, 67.88, 65.93],
        "RAPTOR": [54.07, 53.20, 58.73, 62.38], "Lazy-GraphRAG": [60.25, 47.82, 57.28, 62.22],
        "KGP": [52.34, 51.53, 54.51, 63.77], "StructRAG": [55.38, 56.17, 62.48, 60.21],
        "KET-RAG": [60.35, 39.56, 45.27, 43.04],
    },
}

# Table 10. 검색 [재현율, 관련도] x 4 과제
GB_RET = {
    "Novel": {
        GB_RAG0: [(61.37, 74.66), (59.80, 80.82), (69.08, 80.05), (32.48, 82.84)],
        GB_RAG: [(83.21, 77.77), (64.47, 82.08), (73.38, 83.10), (39.59, 78.73)],
        "MS-GraphRAG local": [(61.04, 27.30), (73.03, 39.09), (82.02, 43.13), (53.55, 35.07)],
        "MS-GraphRAG global": [(42.27, 9.37), (86.68, 14.36), (89.69, 15.35), (83.14, 19.40)],
        "HippoRAG": [(80.44, 56.34), (87.91, 58.75), (90.95, 59.46), (65.51, 46.64)],
        "HippoRAG2": [(70.29, 79.25), (69.77, 85.75), (82.50, 87.82), (42.18, 79.10)],
        "LightRAG": [(73.69, 33.08), (85.52, 37.46), (87.59, 38.02), (71.72, 38.06)],
        "Fast-GraphRAG": [(64.48, 47.86), (73.51, 55.21), (78.58, 49.74), (56.31, 46.27)],
        "RAPTOR": [(62.14, 54.08), (67.80, 61.26), (75.79, 63.00), (58.66, 58.46)],
        "Lazy-GraphRAG": [(59.25, 30.76), (57.73, 42.98), (77.38, 43.62), (55.24, 31.94)],
        "KGP": [(55.71, 23.71), (63.51, 31.96), (61.54, 64.20), (67.57, 35.52)],
        "StructRAG": [(55.38, 27.53), (56.17, 34.79), (62.48, 65.66), (60.21, 42.35)],
        "KET-RAG": [(63.55, 39.11), (56.93, 32.59), (67.35, 39.05), (53.40, 36.74)],
    },
    "Medical": {
        GB_RAG0: [(86.24, 63.71), (84.97, 84.11), (84.14, 89.94), (44.88, 58.73)],
        GB_RAG: [(87.83, 64.73), (86.49, 85.56), (85.87, 91.35), (45.23, 60.50)],
        "MS-GraphRAG local": [(38.06, 5.67), (61.32, 4.25), (59.66, 5.24), (66.59, 2.76)],
        "MS-GraphRAG global": [(65.98, 7.46), (78.46, 11.72), (89.06, 11.72), (85.28, 2.73)],
        "HippoRAG": [(87.25, 52.44), (83.80, 42.19), (83.46, 49.13), (81.66, 45.03)],
        "HippoRAG2": [(78.70, 87.96), (77.00, 80.94), (77.40, 86.85), (61.12, 78.64)],
        "LightRAG": [(80.32, 41.27), (82.91, 42.79), (85.71, 43.11), (81.34, 45.17)],
        "Fast-GraphRAG": [(66.82, 45.86), (74.93, 38.80), (77.27, 47.58), (62.99, 25.15)],
        "RAPTOR": [(85.40, 69.38), (89.70, 53.20), (88.86, 58.73), (72.70, 52.71)],
        "Lazy-GraphRAG": [(74.29, 19.90), (78.65, 17.50), (78.72, 21.35), (83.41, 15.09)],
        "KGP": [(57.51, 27.34), (53.51, 26.59), (59.38, 56.20), (68.42, 43.85)],
        "StructRAG": [(63.25, 37.26), (61.75, 35.68), (62.55, 32.01), (62.76, 46.75)],
        "KET-RAG": [(86.44, 57.07), (80.62, 30.86), (89.07, 44.59), (44.06, 32.38)],
    },
}

# Table 6 · 7. 질의당 평균 프롬프트 토큰 [Novel, Medical]. V-RAG 는 rerank 없는 기본 RAG 로 본다
GB_TOKENS = {
    GB_RAG0: (879, 954), "MS-GraphRAG local": (38707, 39821), "MS-GraphRAG global": (331375, 332881),
    "HippoRAG2": (1008, 1020), "LightRAG": (100832, 100310), "Fast-GraphRAG": (4204, 4298),
    "RAPTOR": (3441, 3510), "HippoRAG": (7208, 7342),
}

# Table 15. 소설 한 권(약 5.6만 토큰) 색인: (초, 전체 토큰)
GB_INDEX = {
    "MS-GraphRAG": (292.45, 654673), "LightRAG": (710.32, 474172), "Fast-GraphRAG": (281.74, 251817),
    "HippoRAG": (77.42, 177961), "HippoRAG2": (96.71, 329993), "KGP": (32.01, 89215),
    "RAPTOR": (135.21, 115541), "KET-RAG": (350.43, 517551), "Lazy-GraphRAG": (253.59, 591150),
}

# Table 17. Novel 코퍼스 크기별 ACC
GB_SIZE = ["56k", "603k", "1,132k"]
GB_SCALE = {
    "HippoRAG2": [[60.14, 53.38, 64.10, 48.28], [59.99, 56.67, 65.77, 50.06], [59.19, 54.29, 62.63, 51.18]],
    "RAG": [[64.73, 58.64, 65.75, 60.61], [58.43, 41.33, 62.06, 52.54], [58.04, 43.20, 62.43, 47.19]],
}

# Table 13. 코퍼스 그래프 밀도: (평균 차수, 고립되지 않은 엔티티 비율)
GB_DENSITY = {
    "UltraDomain": (0.86, 0.40), "MultiHop-RAG": (0.76, 0.41), "HotpotQA": (0.65, 0.41),
    "MuSiQue": (0.60, 0.39), "2WikiMultihopQA": (0.64, 0.40),
    "GraphRAG-Bench Novel": (2.27, 0.66), "GraphRAG-Bench Medical": (1.05, 0.48),
}


def gb_avg_acc(ds: str, name: str) -> float:
    return mean(GB_ACC[ds][name])


def gb_avg_ret(ds: str, name: str) -> tuple[float, float]:
    rows = GB_RET[ds][name]
    return mean(r for r, _ in rows), mean(v for _, v in rows)


def gb_graph_band(ds: str) -> list[tuple[float, float, float]]:
    """과제별 GraphRAG 11종의 (최소, 중앙값, 최대)."""
    out = []
    for j in range(len(GB_TASKS)):
        vals = sorted(GB_ACC[ds][m][j] for m in GB_GRAPH)
        out.append((vals[0], median(vals), vals[-1]))
    return out


def gb_beats_rag(ds: str) -> list[int]:
    """과제별로 rerank RAG 보다 ACC 가 높은 GraphRAG 수."""
    rag = GB_ACC[ds][GB_RAG]
    return [sum(GB_ACC[ds][m][j] > rag[j] for m in GB_GRAPH) for j in range(len(GB_TASKS))]


def place_labels(points, h=15, gap=2):
    """(x, y, text) 목록을 받아 겹치지 않게 y 를 아래로 민다. 오른쪽 배치 가정."""
    placed = []
    out = []
    for x, y, s in sorted(points, key=lambda p: p[1]):
        w = 7.2 * len(s) + 4
        yy = y
        while any(abs(yy - py) < h + gap and x < px + pw and px < x + w for px, py, pw in placed):
            yy += 4
        placed.append((x, yy, w))
        out.append((x, y, yy, s))
    return out


def fig_gb_levels():
    W = 880
    pw, ph = 340, 260
    body = []
    for pi, ds in enumerate(("Novel", "Medical")):
        ox, oy = 70 + pi * (pw + 90), 132
        vmin, vmax = (10, 70)
        sx = lambda j: ox + 30 + j * (pw - 60) / 3
        sy = lambda v: oy + ph - (v - vmin) / (vmax - vmin) * ph
        body.append(t(ox, oy - 18, "소설 (느슨한 서사)" if ds == "Novel" else "의료 지침 (위계가 뚜렷함)", "b", size=13.5))
        for v in range(vmin, vmax + 1, 10):
            body.append(f'<line class="grid" x1="{ox}" y1="{sy(v):.1f}" x2="{ox + pw}" y2="{sy(v):.1f}"/>')
            body.append(t(ox - 8, sy(v) + 4, v, "mu", "end"))
        for j, task in enumerate(GB_TASKS):
            body.append(t(sx(j), oy + ph + 20, f"L{j + 1} {task}", "t2", "middle", size=11.5))
        band = gb_graph_band(ds)
        pts_hi = " ".join(f"{sx(j):.1f},{sy(b[2]):.1f}" for j, b in enumerate(band))
        pts_lo = " ".join(f"{sx(j):.1f},{sy(b[0]):.1f}" for j, b in reversed(list(enumerate(band))))
        body.append(f'<polygon class="faintf" points="{pts_hi} {pts_lo}" style="opacity:.55"/>')
        med = " ".join(f"{sx(j):.1f},{sy(b[1]):.1f}" for j, b in enumerate(band))
        body.append(f'<polyline points="{med}" style="fill:none;stroke:var(--muted);stroke-width:1.5"/>')
        for name, cls in ((GB_RAG, "s2"), ("HippoRAG2", "s1")):
            vals = GB_ACC[ds][name]
            pts = " ".join(f"{sx(j):.1f},{sy(v):.1f}" for j, v in enumerate(vals))
            body.append(f'<polyline points="{pts}" class="{cls}s" style="fill:none;stroke-width:2"/>')
            for j, v in enumerate(vals):
                body.append(f'<circle class="{cls}f ring" cx="{sx(j):.1f}" cy="{sy(v):.1f}" r="5"><title>{escape(name)} {GB_TASKS[j]} {v}</title></circle>')
        beats = gb_beats_rag(ds)
        for j, n in enumerate(beats):
            body.append(t(sx(j), oy + ph + 38, f"RAG 초과 {n}/11", "mu", "middle"))
    ly = 80
    items = [("s2f", "RAG (rerank)"), ("s1f", "HippoRAG2"), ("faintf", "GraphRAG 11종 범위"), (None, "11종 중앙값")]
    x = 70
    for cls, label in items:
        if cls:
            body.append(f'<rect class="{cls}" x="{x}" y="{ly - 10}" width="14" height="12" rx="3"/>')
        else:
            body.append(f'<line x1="{x}" y1="{ly - 4}" x2="{x + 14}" y2="{ly - 4}" style="stroke:var(--muted);stroke-width:2"/>')
        body.append(t(x + 20, ly, label, size=12))
        x += 20 + 8 * len(label) + 30
    H = 132 + ph + 90
    return doc(W, H, "그래프의 이득은 과제 난도와 코퍼스에 달려 있다",
               "소설에서는 L2부터 대부분 RAG를 넘지만, 의료 지침에서는 L4 말고는 소수만 넘는다. 아래 숫자 = RAG보다 높은 GraphRAG 수",
               body, "출처: GraphRAG-Bench (arXiv:2506.05690) Table 9")


def fig_gb_retrieval():
    W = 860
    pw, ph = 350, 320
    body = []
    hl = {GB_RAG: "s2f", GB_RAG0: "s2f", "HippoRAG2": "s1f"}
    for pi, ds in enumerate(("Novel", "Medical")):
        ox, oy = 80 + pi * (pw + 90), 100
        sx = lambda v: ox + (v - 40) / 60 * pw
        sy = lambda v: oy + ph - v / 100 * ph
        body.append(t(ox, oy - 14, "소설" if ds == "Novel" else "의료 지침", "b", size=13.5))
        for v in range(0, 101, 20):
            body.append(f'<line class="grid" x1="{ox}" y1="{sy(v):.1f}" x2="{ox + pw}" y2="{sy(v):.1f}"/>')
            body.append(t(ox - 8, sy(v) + 4, v, "mu", "end"))
        for v in range(40, 101, 20):
            body.append(t(sx(v), oy + ph + 18, v, "mu", "middle"))
        body.append(t(ox + pw / 2, oy + ph + 38, "근거 재현율 (4개 과제 평균)", "t2", "middle", size=12))
        if pi == 0:
            body.append(f'<text x="{ox - 44}" y="{oy + ph / 2:.1f}" text-anchor="middle" class="t2" style="font-size:12px" '
                        f'transform="rotate(-90 {ox - 44} {oy + ph / 2:.1f})">문맥 관련도 (4개 과제 평균)</text>')
        pts = []
        for name in [GB_RAG0, GB_RAG] + GB_GRAPH:
            r, rel = gb_avg_ret(ds, name)
            cls = hl.get(name, "inkF")
            fill = f'class="{cls} ring"' if cls != "inkF" else 'class="ring" style="fill:var(--muted)"'
            body.append(f'<circle {fill} cx="{sx(r):.1f}" cy="{sy(rel):.1f}" r="{6 if name in hl else 4.5}">'
                        f'<title>{escape(name)} 재현율 {r:.1f} 관련도 {rel:.1f}</title></circle>')
            pts.append((sx(r) + 9, sy(rel) + 4, name))
        for x, y0, y, s in place_labels(pts):
            if abs(y - y0) > 2:
                body.append(f'<line x1="{x - 6:.1f}" y1="{y0 - 4:.1f}" x2="{x - 1:.1f}" y2="{y - 4:.1f}" style="stroke:var(--rule);stroke-width:1"/>')
            bold = s in hl
            body.append(t(x, y, s, "b" if bold else "t2", size=11))
    H = 100 + ph + 80
    return doc(W, H, "넓게 가져오면 재현율은 오르고 관련도는 무너진다",
               "오른쪽 위일수록 좋음. 그래프 방식 대부분은 재현율을 얻는 대신 관련도를 잃고, HippoRAG2만 RAG 수준의 관련도를 유지한다",
               body, "출처: GraphRAG-Bench (arXiv:2506.05690) Table 10에서 과제 평균 계산")


def fig_gb_tokens():
    W, H = 820, 440
    x0, x1, y0, y1 = 90, 760, 350, 100
    import math
    lx = lambda v: math.log10(v)
    xmin, xmax = 2.7, 5.7
    ymin, ymax = 20, 65
    sx = lambda v: x0 + (lx(v) - xmin) / (xmax - xmin) * (x1 - x0)
    sy = lambda v: y0 - (v - ymin) / (ymax - ymin) * (y0 - y1)
    body = []
    for v in range(ymin, ymax + 1, 10):
        body.append(f'<line class="grid" x1="{x0}" y1="{sy(v):.1f}" x2="{x1}" y2="{sy(v):.1f}"/>')
        body.append(t(x0 - 8, sy(v) + 4, v, "mu", "end"))
    for v, lab in ((1000, "1천"), (10000, "1만"), (100000, "10만")):
        body.append(f'<line class="grid" x1="{sx(v):.1f}" y1="{y1}" x2="{sx(v):.1f}" y2="{y0}"/>')
        body.append(t(sx(v), y0 + 18, lab, "mu", "middle"))
    body.append(t((x0 + x1) / 2, y0 + 40, "질의당 평균 프롬프트 토큰 (로그 눈금, 두 데이터셋 평균)", "t2", "middle", size=12))
    body.append(f'<text x="{x0 - 50}" y="{(y0 + y1) / 2:.1f}" text-anchor="middle" class="t2" style="font-size:12px" '
                f'transform="rotate(-90 {x0 - 50} {(y0 + y1) / 2:.1f})">평균 ACC (2개 데이터셋 × 4개 과제)</text>')
    pts = []
    for name, (a, b) in GB_TOKENS.items():
        tok = (a + b) / 2
        acc = mean([gb_avg_acc("Novel", name), gb_avg_acc("Medical", name)])
        cls = "s2f" if name == GB_RAG0 else "s1f" if name == "HippoRAG2" else None
        fill = f'class="{cls} ring"' if cls else 'class="ring" style="fill:var(--muted)"'
        body.append(f'<circle {fill} cx="{sx(tok):.1f}" cy="{sy(acc):.1f}" r="6"><title>{escape(name)} {tok:,.0f} 토큰, ACC {acc:.1f}</title></circle>')
        label = f"{name} · {tok:,.0f}"
        anchor_end = sx(tok) > x1 - 150
        below = name == "RAPTOR"
        pts.append((sx(tok), sy(acc), label, anchor_end, cls is not None, below))
    for x, y, label, end, bold, below in pts:
        body.append(t(x - 10 if end else x + 10, y + 20 if below else y - 8, label, "b" if bold else "t2", "end" if end else "start", size=11.5))
    body.append(t(24, 78, "HippoRAG2(파랑)는 RAG(주황)와 같은 1천 토큰대에서 가장 높다. 수만~수십만 토큰을 쓰는 방식은 RAG와 비슷하거나 낮다", "t2", size=12))
    ratio = mean(GB_TOKENS["MS-GraphRAG global"]) / mean(GB_TOKENS[GB_RAG0])
    return doc(W, H, "토큰을 더 쓴다고 더 맞히지 않는다",
               f"MS-GraphRAG global은 RAG보다 약 {ratio:.0f}배 긴 프롬프트를 쓰고 평균 ACC는 가장 낮다",
               body, "출처: GraphRAG-Bench (arXiv:2506.05690) Table 6, 7, 9에서 계산")


def fig_gb_index():
    rows = sorted(GB_INDEX.items(), key=lambda kv: -kv[1][1])
    W = 780
    x0, x1 = 170, 600
    top, rowh = 92, 30
    vmax = 700000
    sx = lambda v: x0 + (x1 - x0) * v / vmax
    body = []
    for v in range(0, vmax + 1, 100000):
        body.append(f'<line class="grid" x1="{sx(v):.1f}" y1="{top - 8}" x2="{sx(v):.1f}" y2="{top + rowh * len(rows) - 6}"/>')
        body.append(t(sx(v), top + rowh * len(rows) + 10, f"{v // 1000}k", "mu", "middle"))
    for i, (name, (sec, tok)) in enumerate(rows):
        y = top + i * rowh
        cls = "s1f" if name == "HippoRAG2" else "faintf" if name != "MS-GraphRAG" else "s2f"
        body.append(t(x0 - 12, y + 14, name, "b" if name in ("HippoRAG2", "MS-GraphRAG") else "", "end"))
        body.append(f'<rect class="{cls}" x="{x0}" y="{y}" width="{sx(tok) - x0:.1f}" height="20" rx="4"/>')
        body.append(t(sx(tok) + 8, y + 15, f"{tok:,} 토큰 · {sec:.0f}초", "t2", size=12))
    H = top + rowh * len(rows) + 64
    return doc(W, H, "책 한 권 색인 비용",
               "소설 한 권(약 5.6만 토큰)의 그래프 구축에 쓴 LLM 입출력 토큰과 시간",
               body, "출처: GraphRAG-Bench (arXiv:2506.05690) Table 15")


def fig_gb_scale():
    W = 860
    pw, ph = 170, 170
    body = []
    novel_rag = GB_ACC["Novel"][GB_RAG]
    for j, task in enumerate(GB_TASKS):
        ox, oy = 60 + j * (pw + 36), 110
        vmin, vmax = 35, 70
        sx = lambda k: ox + 16 + k * (pw - 32) / 2
        sy = lambda v: oy + ph - (v - vmin) / (vmax - vmin) * ph
        body.append(t(ox, oy - 14, task, "b", size=13))
        for v in range(40, vmax + 1, 10):
            body.append(f'<line class="grid" x1="{ox}" y1="{sy(v):.1f}" x2="{ox + pw}" y2="{sy(v):.1f}"/>')
            if j == 0:
                body.append(t(ox - 8, sy(v) + 4, v, "mu", "end"))
        for k, lab in enumerate(GB_SIZE):
            body.append(t(sx(k), oy + ph + 18, lab, "mu", "middle"))
        for name, cls in (("RAG", "s2"), ("HippoRAG2", "s1")):
            vals = [GB_SCALE[name][k][j] for k in range(3)]
            pts = " ".join(f"{sx(k):.1f},{sy(v):.1f}" for k, v in enumerate(vals))
            body.append(f'<polyline points="{pts}" class="{cls}s" style="fill:none;stroke-width:2"/>')
            for k, v in enumerate(vals):
                body.append(f'<circle class="{cls}f ring" cx="{sx(k):.1f}" cy="{sy(v):.1f}" r="4.5"><title>{name} {GB_SIZE[k]} {v}</title></circle>')
        v = novel_rag[j]
        body.append(f'<circle cx="{sx(0):.1f}" cy="{sy(v):.1f}" r="6" class="s2s" style="fill:var(--surface);stroke-width:2"><title>Table 9 Novel RAG {v}</title></circle>')
    ly = 76
    body.append(f'<rect class="s2f" x="60" y="{ly - 10}" width="14" height="12" rx="3"/>')
    body.append(t(80, ly, "RAG (Table 17 보고값)", size=12))
    body.append(f'<circle cx="260" cy="{ly - 4}" r="6" class="s2s" style="fill:var(--surface);stroke-width:2"/>')
    body.append(t(272, ly, "같은 소설 데이터셋에서 Table 9가 보고한 RAG 값", size=12))
    body.append(f'<rect class="s1f" x="600" y="{ly - 10}" width="14" height="12" rx="3"/>')
    body.append(t(620, ly, "HippoRAG2", size=12))
    ny = 110 + ph + 46
    body.append(t(60, ny, "Table 17의 RAG 56k 값(64.73 · 58.64 · 65.75 · 60.61)은 의료 데이터셋 Table 9 값과 네 자리 모두 같습니다.", "t2", size=12.5))
    body.append(t(60, ny + 20, "빈 원(소설 Table 9 값)으로 바꾸면 복합 추론의 RAG는 42.93 → 41.33 → 43.20으로 거의 평평합니다.", "t2", size=12.5))
    H = ny + 70
    return doc(W, H, "코퍼스가 커지면 RAG만 무너지는가",
               "소설 데이터셋 코퍼스 크기별 ACC. 저자 결론은 'RAG는 크기에 따라 떨어지고 HippoRAG2는 안정적'",
               body, "출처: GraphRAG-Bench (arXiv:2506.05690) Table 17, Table 9")


def fig_gb_density():
    W = 860
    rows = list(GB_DENSITY.items())
    lab_w = 190
    top, rowh = 110, 28
    panels = [("평균 차수", 0, 2.5, [0, 0.5, 1, 1.5, 2, 2.5]), ("고립되지 않은 엔티티 비율", 1, 0.8, [0, 0.2, 0.4, 0.6, 0.8])]
    body = []
    pw = 270
    for pi, (title, idx, vmax, ticks) in enumerate(panels):
        ox = 24 + lab_w + pi * (pw + 60)
        sx = lambda v: ox + v / vmax * pw
        body.append(t(ox, top - 22, title, "b", size=13))
        for v in ticks:
            body.append(f'<line class="grid" x1="{sx(v):.1f}" y1="{top - 8}" x2="{sx(v):.1f}" y2="{top + rowh * len(rows) - 10}"/>')
            body.append(t(sx(v), top + rowh * len(rows) + 6, f"{v:g}", "mu", "middle"))
        for i, (name, vals) in enumerate(rows):
            y = top + i * rowh
            ours = name.startswith("GraphRAG-Bench")
            if pi == 0:
                body.append(t(ox - 14, y + 4, name, "b" if ours else "", "end"))
            v = vals[idx]
            cls = "s1" if ours else None
            style = "" if ours else ' style="fill:var(--muted)"'
            body.append(f'<line x1="{ox}" y1="{y}" x2="{sx(v):.1f}" y2="{y}" style="stroke:var({"--s1" if ours else "--rule"});stroke-width:2"/>')
            body.append(f'<circle {"class=" + chr(34) + cls + "f ring" + chr(34) if ours else "class=" + chr(34) + "ring" + chr(34)}{style} cx="{sx(v):.1f}" cy="{y}" r="6"/>')
            body.append(t(sx(v) + 10, y + 4, f"{v:g}", "b" if ours else "t2", size=12))
    H = top + rowh * len(rows) + 60
    others = [v[0] for k, v in GB_DENSITY.items() if not k.startswith("GraphRAG-Bench")]
    nov = GB_DENSITY["GraphRAG-Bench Novel"][0]
    return doc(W, H, "코퍼스에서 뽑은 그래프가 얼마나 이어져 있는가",
               f"소설 코퍼스의 평균 차수는 기존 벤치마크의 {nov / max(others):.1f}~{nov / min(others):.1f}배. 의료 코퍼스는 UltraDomain보다 조금 높은 수준",
               body, "출처: GraphRAG-Bench (arXiv:2506.05690) Table 13")


# ----------------------------------------------------------------

FIGURES = {
    "ragsearch_gap.svg": fig_rs_gap,
    "ragsearch_backend_heatmap.svg": fig_rs_heatmap,
    "ragsearch_cost_vs_score.svg": fig_rs_cost,
    "ragsearch_em_vs_f1.svg": fig_rs_em_f1,
    "pathrag_winrate.svg": fig_pr_winrate,
    "pathrag_winrate_heatmap.svg": fig_pr_heatmap,
    "pathrag_tokens.svg": fig_pr_tokens,
    "pathrag_flow_example.svg": fig_pr_flow,
    "graphragbench_levels.svg": fig_gb_levels,
    "graphragbench_retrieval.svg": fig_gb_retrieval,
    "graphragbench_tokens.svg": fig_gb_tokens,
    "graphragbench_index_cost.svg": fig_gb_index,
    "graphragbench_scale_check.svg": fig_gb_scale,
    "graphragbench_density.svg": fig_gb_density,
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in FIGURES.items():
        (OUT / name).write_text(fn(), encoding="utf-8")
        print("wrote", (OUT / name).relative_to(ROOT))


if __name__ == "__main__":
    main()
