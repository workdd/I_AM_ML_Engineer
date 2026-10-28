"""옮겨 적은 프레임 예산 규칙이 원 레포 동작과 맞는지, SVG 가 깨지지 않았는지 확인한다."""
import xml.etree.ElementTree as ET

import pytest

import cvideo_figures as c


@pytest.mark.parametrize("d,expected", [(10, 12), (30, 30), (60, 40), (180, 60), (600, 80), (3600, 100)])
def test_full_budget(d, expected):
    assert c.budget_full(d) == expected


@pytest.mark.parametrize("d,expected", [(5, 10), (10, 20), (15, 30), (30, 60), (60, 80), (120, 100)])
def test_focus_budget_respects_2fps(d, expected):
    # 5초 구간은 목표 30장이지만 2fps 상한으로 10장, 10초 구간은 40장 목표가 20장으로 깎인다
    assert c.budget_focus(d) == expected


def test_efficient_cap():
    assert c.budget_full(3600, cap=c.CAPS["efficient"]) == 50


@pytest.mark.parametrize("name,fn", c.FIGS)
def test_svg_is_well_formed(name, fn):
    root = ET.fromstring(fn())
    assert root.get("viewBox")
    assert root.find("{http://www.w3.org/2000/svg}title").text
