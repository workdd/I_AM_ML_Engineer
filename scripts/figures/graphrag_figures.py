#!/usr/bin/env python3
"""RAGSearch · PathRAG 노트의 그림을 논문 표 수치에서 생성한다.

    python scripts/figures/graphrag_figures.py

readings/papers/assets/ 아래 SVG 를 다시 쓴다. SVG 는 손으로 고치지 않는다.
표 수치는 아래 데이터 절에 원문 그대로 옮겼고, 파생 수치(격차, 평균)는 여기서 계산한다.
표준 라이브러리만 쓴다.
"""
from __future__ import annotations

from pathlib import Path
from statistics import mean
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
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in FIGURES.items():
        (OUT / name).write_text(fn(), encoding="utf-8")
        print("wrote", (OUT / name).relative_to(ROOT))


if __name__ == "__main__":
    main()
