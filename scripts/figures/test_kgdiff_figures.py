"""combineEvidence 옮김이 원문 규칙과 같은 값을 내는지, SVG 가 깨지지 않았는지 확인한다."""
import xml.etree.ElementTree as ET

import pytest

import kgdiff_figures as m


def test_label_only_score_is_label():
    assert m.score(0.9) == 0.9
    assert m.min_label(None) == pytest.approx(0.82)


def test_floor_caps_unrelated_names():
    assert m.score(0.4, neighbors=1.0, types=1.0, attrs=1.0) == 0.5


def test_neighbor_evidence_shifts_threshold():
    assert m.min_label(1.0) == pytest.approx(0.738182, abs=1e-6)
    assert m.min_label(0.0) is None                      # 이웃이 하나도 안 맞으면 라벨만으로 못 닿음
    assert m.score(1.0, neighbors=0.0) == pytest.approx(0.6875)
    assert m.SUGGEST <= m.score(1.0, neighbors=0.0) < m.THRESHOLD


@pytest.mark.parametrize("name,fn", m.FIGS)
def test_svg_is_well_formed(name, fn):
    root = ET.fromstring(fn())
    assert root.get("viewBox")
    assert root.find("{http://www.w3.org/2000/svg}title").text
