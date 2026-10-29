"""원문 가상 예시의 수치를 계산식이 그대로 내는지, SVG 가 깨지지 않았는지 확인한다."""
import xml.etree.ElementTree as ET

import pytest

import llmcache_figures as c


def test_article_numbers():
    rows = {name: (v, hr) for name, v, hr, _ in c.scenarios()}
    no_cache = rows["캐시 없음"][0]
    assert no_cache == 1_050_000
    shared = [v for n, (v, _) in rows.items() if n.startswith("단일")][0]
    assert shared == 60_000                                    # 10,000 + 500 × 100
    assert round((1 - shared / no_cache) * 100, 1) == 94.3
    assert c.cached_tokens(10, 10) == 900_000
    assert round(c.hit_rate(10, 10) * 100, 1) == 85.7
    assert round(c.hit_rate(1, 100) * 100, 1) == 94.3


def test_single_visit_has_no_hit():
    assert c.hit_rate(1, 1) == 0
    assert c.hit_rate(1, 20) < c.DOC / (c.DOC + c.Q)


@pytest.mark.parametrize("name,fn", c.FIGS)
def test_svg_is_well_formed(name, fn):
    root = ET.fromstring(fn())
    assert root.get("viewBox")
    assert root.find("{http://www.w3.org/2000/svg}title").text
