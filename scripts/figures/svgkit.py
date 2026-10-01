"""그림 생성기들이 함께 쓰는 SVG 조각. 라이트·다크 모두 불투명 배경을 깐다."""
from __future__ import annotations

from xml.sax.saxutils import escape

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


def make_doc(gen_note):
    def doc(w, h, title, subtitle, body, source):
        return _doc(w, h, title, subtitle, body, source, gen_note)
    return doc


def _doc(w, h, title, subtitle, body, source, gen_note):
    head = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
        f'role="img" aria-label="{escape(title)}">',
        f"<title>{escape(title)}</title>",
        STYLE,
        f'<rect class="bg" x="0" y="0" width="{w}" height="{h}" rx="10"/>',
        t(24, 34, title, "ttl"),
        t(24, 55, subtitle, "sub"),
    ]
    foot = [t(24, h - 26, source, "mu"), t(24, h - 11, gen_note, "mu")]
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


