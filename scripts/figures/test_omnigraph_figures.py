"""v0.15.0 릴리즈 개요의 수치를 그대로 옮겼는지, SVG 가 깨지지 않았는지 확인한다."""
import xml.etree.ElementTree as ET

import pytest

import omnigraph_figures as m


def test_release_numbers():
    d = {name.split(":")[0]: (b, a) for name, b, a in m.REQUESTS}
    assert d["diff"] == ((277, 277), (14, 14))
    assert d["fast-forward 병합"] == ((4168, 4168), (54, 54))
    assert d["3-way 병합"] == ((3216, 3216), (135, 135))
    assert d["다른 프로세스 커밋 뒤 첫 읽기"] == ((18, 20), (11, 12))
    assert m.LOAD_014 == 261 and m.LOAD_015 == (87, 110)


def test_every_case_drops_and_load_is_under_half():
    for _, b, a in m.REQUESTS:
        assert a[1] < b[0]
    assert m.LOAD_015[1] < m.LOAD_014 / 2          # 원문 "less than half"
    assert round(m.ratio((4168, 4168), (54, 54))) == 77


@pytest.mark.parametrize("name,fn", m.FIGS)
def test_svg_is_well_formed(name, fn):
    root = ET.fromstring(fn())
    assert root.get("viewBox")
    assert root.find("{http://www.w3.org/2000/svg}title").text
