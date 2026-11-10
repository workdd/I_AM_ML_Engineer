"""원문 수치(90% · 94% → 98% · 두 자릿수 포인트)를 옮긴 상수와 계산, 생성한 SVG 형식을 확인한다."""
import xml.etree.ElementTree as ET

import pytest

import tossai_figures as k


def test_constants_match_article():
    assert (k.FLOOR, k.BEFORE, k.AFTER, k.DROP_PP) == (90, 94, 98, 10)


def test_error_cut_is_two_thirds():
    e0, e1, cut = k.error_cut()
    assert (e0, e1) == (6, 2)
    assert cut == pytest.approx(2 / 3)


def test_mcnemar_threshold_is_six_flips():
    assert k.mcnemar_p(5) == pytest.approx(0.0625)
    assert k.mcnemar_p(6) == pytest.approx(0.03125)
    assert k.min_flips() == 6


def test_min_golden_set_sizes():
    assert k.min_n(4) == 150
    assert k.min_n(10) == 60
    assert k.min_n(4) * 4 / 100 >= k.min_flips() > (k.min_n(4) - 1) * 4 / 100


@pytest.mark.parametrize("name,fn", k.FIGS)
def test_svg_is_well_formed(name, fn):
    root = ET.fromstring(fn())
    assert root.get("viewBox")
    assert root.find("{http://www.w3.org/2000/svg}title").text
