"""ralforion/orionbelt-ontology-builder 노트의 그림을 생성한다.

    python scripts/figures/orionbelt_figures.py              readings/repos/assets/orionbelt_*.svg 를 다시 쓴다
    python scripts/figures/orionbelt_figures.py --inline DIR  HTML 보고서용 (접두어, 생성 메모 없이 DIR/<이름>.svg)

SVG 는 손으로 고치지 않는다. 값은 커밋 6929408 기준이다.
SKOS 검사 등급과 자동 수정 목록은 README 표 3개, 품질 규칙은 quality/registry.RULES 를 실행해 센 값,
SPARQL 엔진 수치는 sparql.py 의 OXIGRAPH_MIN_TRIPLES 와 그 주석이다. 원문에는 이 값을 모은 그림이 없다.
"""
import math
import sys
from pathlib import Path

FONT = "-apple-system,'Apple SD Gothic Neo','Noto Sans KR',sans-serif"
INK, SUB, GRID = "#1f2328", "#59636e", "#d8dee4"
GRAY, BLUE, ORANGE, GREEN, RED, PURPLE = "#8b949e", "#2563eb", "#ea580c", "#16a34a", "#dc2626", "#7c3aed"
SHOW_NOTE = True
NOTE = "생성: python scripts/figures/orionbelt_figures.py, 손으로 고치지 말 것"
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "readings" / "repos" / "assets"

# README "What SKOS validation checks": 등급 → 검사 id
SKOS_TIERS = [
    ("오류 (항상 검사)", RED, ["missing_prefLabel", "multi_prefLabel_per_lang", "empty_label", "self_relation",
                          "dangling_relation", "relation_clash", "broader_cycle", "xl_literal_form",
                          "xl_label_disjoint"]),
    ("관례 (check_conventions)", ORANGE, ["missing_lang", "label_overlap", "duplicate_prefLabel",
                                        "ambiguous_prefLabel", "orphan", "top_with_broader",
                                        "hierarchy_redundancy", "mapping_within_scheme", "notation_lang_tagged"]),
    ("편집 (check_editorial)", PURPLE, ["no_scheme", "scheme_untitled", "undocumented", "valueless_association",
                                      "disconnected_components"]),
]
# README "Fixing what it finds"
AUTOFIX = {"missing_lang", "self_relation", "dangling_relation", "top_with_broader", "label_overlap",
           "hierarchy_redundancy", "broader_cycle"}
# quality/registry.RULES: 범주 → (규칙 수, 기본 켬 수)
QUALITY = [("metadata", 6, 4), ("naming", 4, 3), ("structure", 10, 6), ("skos", 2, 1)]

# sparql.py
OXIGRAPH_MIN_TRIPLES = 20_000
EVEN_AT = 8_500           # 주석: about even at 8.5k triples
FAST_AT = 43_000          # 주석: Oxigraph 2 to 8 times faster at 43k
FAST_RANGE = (2, 8)
DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_MAX_ROWS = 1000


def svg(w, h, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'role="img" aria-label="{title}" style="max-width:100%;height:auto;font-family:{FONT}">'
            f'<title>{title}</title><rect width="{w}" height="{h}" fill="#ffffff"/>{body}'
            + (f'<text x="{w-8}" y="{h-6}" font-size="9" fill="{GRAY}" text-anchor="end">{NOTE}</text>' if SHOW_NOTE else "")
            + '</svg>')


def t(x, y, s, size=12, fill=INK, anchor="start", weight="normal"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}">{s}</text>')


def fig_checks():
    W, H = 760, 440
    unit = 26
    b = [t(20, 28, "검사 두 벌: SKOS 검증 23개와 모델 품질 규칙 22개", 15, weight="bold"),
         t(20, 48, "칸 하나가 검사 하나. 진한 칸은 버튼 하나로 자동 수정되는 SKOS 검사, 품질 규칙은 진한 칸이 기본으로 켜진 규칙", 11, SUB)]
    x0 = 200
    y = 84
    b.append(t(20, y - 8, "SKOS 검증 (validate_skos)", 13, weight="bold"))
    for lab, col, ids in SKOS_TIERS:
        y += 34
        fixed = sum(i in AUTOFIX for i in ids)
        b.append(t(x0 - 12, y + 16, lab, 11, INK, "end"))
        for k, i in enumerate(sorted(ids, key=lambda i: i not in AUTOFIX)):
            on = i in AUTOFIX
            b.append(f'<rect x="{x0 + k*unit}" y="{y}" width="{unit-3}" height="22" rx="3" '
                     f'fill="{col if on else "#ffffff"}" stroke="{col}" stroke-width="1.4"/>')
        b.append(t(x0 + len(ids) * unit + 8, y + 16, f"{len(ids)}개, 자동 수정 {fixed}", 11, col, weight="bold"))
    y += 70
    b.append(t(20, y - 8, "모델 품질 규칙 (Validation &amp; Reasoning 페이지, 권고)", 13, weight="bold"))
    for cat, n, on in QUALITY:
        y += 34
        b.append(t(x0 - 12, y + 16, cat, 11, INK, "end"))
        for k in range(n):
            b.append(f'<rect x="{x0 + k*unit}" y="{y}" width="{unit-3}" height="22" rx="3" '
                     f'fill="{BLUE if k < on else "#ffffff"}" stroke="{BLUE}" stroke-width="1.4"/>')
        b.append(t(x0 + n * unit + 8, y + 16, f"{n}개, 기본 켬 {on}", 11, BLUE, weight="bold"))
    return svg(W, H, "".join(b), "OrionBelt SKOS 검증과 모델 품질 규칙 구성")


def fig_sparql():
    W, H = 760, 330
    x0, x1 = 60, 700
    lo, hi = 1_000, 200_000

    def X(v):
        return x0 + (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * (x1 - x0)

    b = [t(20, 28, "SPARQL 엔진 자동 선택: 온톨로지 트리플 수 기준", 15, weight="bold"),
         t(20, 48, f"기본 제한은 두 엔진 모두 {DEFAULT_TIMEOUT_SECONDS:g}초, {DEFAULT_MAX_ROWS:,}행. 측정값은 원문 코드 주석(페이지 예제 질의 기준)", 11, SUB)]
    top, bh = 76, 46
    b.append(f'<rect x="{x0}" y="{top}" width="{X(OXIGRAPH_MIN_TRIPLES)-x0:.1f}" height="{bh}" fill="#eef4ff" stroke="{GRID}"/>')
    b.append(t((x0 + X(OXIGRAPH_MIN_TRIPLES)) / 2, top + 28, "rdflib (앱 프로세스 안, store proxy 가 deadline 확인)", 12, BLUE, "middle", "bold"))
    b.append(f'<rect x="{X(OXIGRAPH_MIN_TRIPLES):.1f}" y="{top}" width="{x1-X(OXIGRAPH_MIN_TRIPLES):.1f}" height="{bh}" fill="#e8f5ec" stroke="{GRID}"/>')
    b.append(t((X(OXIGRAPH_MIN_TRIPLES) + x1) / 2, top + 28, "Oxigraph (worker 프로세스, 제한 시간에 종료)", 12, GREEN, "middle", "bold"))
    for v in (1_000, 10_000, 100_000, 200_000):
        b.append(f'<line x1="{X(v):.1f}" y1="{top+bh}" x2="{X(v):.1f}" y2="{top+bh+6}" stroke="{SUB}"/>')
        b.append(t(X(v), top + bh + 20, f"{v:,}", 10, SUB, "middle"))
    marks = [(EVEN_AT, "8,500: 두 엔진 속도 비슷", SUB, "end"),
             (OXIGRAPH_MIN_TRIPLES, f"{OXIGRAPH_MIN_TRIPLES:,}: 전환 기준", INK, "start"),
             (FAST_AT, f"43,000: Oxigraph {FAST_RANGE[0]}~{FAST_RANGE[1]}배 빠름", GREEN, "start")]
    for i, (v, lab, col, anchor) in enumerate(marks):
        y = top + bh + 50 + i * 30
        b.append(f'<line x1="{X(v):.1f}" y1="{top+bh}" x2="{X(v):.1f}" y2="{y-10}" stroke="{col}" stroke-dasharray="3 3"/>')
        b.append(f'<circle cx="{X(v):.1f}" cy="{y-5}" r="4.5" fill="{col}"/>')
        b.append(t(X(v) + (-10 if anchor == "end" else 10), y - 1, lab, 11, col, anchor, "bold"))
    b.append(t(20, H - 44, "크기와 무관하게 rdflib 로 보내는 질의: 역참조, lookaround 가 있는 REGEX, 대소문자 섞인 언어 태그의 lang()", 11, SUB))
    b.append(t(20, H - 26, "Oxigraph 답이 달라지는 지점: 1.50 → 1.5, \"a\" 와 \"a\"^^xsd:string 동일, decimal 18자리, 다국어 ORDER BY", 11, SUB))
    return svg(W, H, "".join(b), "OrionBelt SPARQL 엔진 선택 기준")


FIGS = [("checks", fig_checks), ("sparql", fig_sparql)]


def main(argv):
    global SHOW_NOTE
    if argv[:1] == ["--inline"]:
        out, prefix, SHOW_NOTE = Path(argv[1]), "", False
    else:
        out, prefix = OUT, "orionbelt_"
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in FIGS:
        (out / f"{prefix}{name}.svg").write_text(fn(), "utf-8")
        print("wrote", out / f"{prefix}{name}.svg")


if __name__ == "__main__":
    main(sys.argv[1:])
