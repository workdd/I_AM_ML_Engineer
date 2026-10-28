"""옮겨 적은 수치가 BENCHMARK.md 와 맞는지, 생성한 SVG 가 깨지지 않았는지 확인한다."""
import xml.etree.ElementTree as ET

import pytest

import aonto_figures as a


def test_uplift_points_match_benchmark():
    # BENCHMARK.md 의 "(+N points)" 표기
    uplift = {(s, ag): round(w - b, 1) for s, _, ag, b, w in a.CORRECTNESS}
    assert uplift == {("setup", "Claude Code"): 64.0, ("setup", "Codex"): 32.0,
                      ("management", "Claude Code"): 70.8, ("management", "Codex"): 41.5,
                      ("query", "Claude Code"): 52.3, ("query", "Codex"): 20.0}


def test_skill_never_lowers_correctness():
    assert all(w >= b for *_, b, w in a.CORRECTNESS)


@pytest.mark.parametrize("name,fn", a.FIGS)
def test_svg_is_well_formed(name, fn):
    root = ET.fromstring(fn())
    assert root.get("viewBox")
    assert root.find("{http://www.w3.org/2000/svg}title").text
