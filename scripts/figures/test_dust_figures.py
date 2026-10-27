"""옮겨 적은 수치가 원문 본문 주장과 맞는지, 생성한 SVG 가 깨지지 않았는지 확인한다."""
import xml.etree.ElementTree as ET

import pytest

import dust_figures as d


def test_dust_beats_backprop_only_at_small_budgets():
    # 부록 Table 1: 16k 집단 기준 10만·100만 토큰만 역전파보다 낮다
    wins = [row[-1] < bp for row, bp in zip(d.DUST, d.BP)]
    assert wins == [True, True, False, False]


def test_best_population_per_budget():
    # 본문: 10만 토큰은 1k, 100만 토큰은 4k 에서 가장 낮다
    best = [d.POPS[row.index(min(row))] for row in d.DUST]
    assert best[:2] == [1024, 4096]
    assert min(d.DUST[0]) == pytest.approx(7.151)
    assert min(d.DUST[1]) == pytest.approx(5.911)


def test_dust_below_eggroll_everywhere():
    assert all(a < b for ra, rb in zip(d.DUST, d.EGG) for a, b in zip(ra, rb))


@pytest.mark.parametrize("name,fn", d.FIGS)
def test_svg_is_well_formed(name, fn):
    root = ET.fromstring(fn())
    assert root.get("viewBox")
    assert root.find("{http://www.w3.org/2000/svg}title") is not None
