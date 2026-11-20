"""decay 식이 원문 문서의 기준값을 내는지, 병합 구간이 원문 기본값인지, SVG 가 깨지지 않았는지 확인한다."""
import xml.etree.ElementTree as ET

import pytest

import selfgraph_figures as m


def test_decay_documented_values():
    assert m.relevance(0, 1) == 0.75          # 원문: 출처 1개 새 relation
    assert m.relevance(0, 0) == 0.5           # 원문: 출처, 관계 없는 새 항목
    assert m.relevance(30, 1) == 0.375        # 반감기
    assert all(m.relevance(0, n) < 1 for n in (1, 10, 1000))   # 범위 [0, 1)


def test_resolution_defaults_and_examples():
    assert m.THRESHOLD == 0.92 and m.BAND == (0.8, 0.92) and m.ALIAS_CONF == 0.95
    s = {p.split(" / ")[0]: v for p, v, _, _ in m.PAIRS}
    assert s == {"Bill Gates": 0.81, "Jon Smithers": 0.873, "Jonathan Smithers": 0.976}
    assert m.BAND[0] <= s["Bill Gates"] < m.BAND[1]
    assert m.BAND[0] <= s["Jon Smithers"] < m.THRESHOLD <= s["Jonathan Smithers"]


@pytest.mark.parametrize("name,fn", m.FIGS)
def test_svg_is_well_formed(name, fn):
    root = ET.fromstring(fn())
    assert root.get("viewBox")
    assert root.find("{http://www.w3.org/2000/svg}title").text
