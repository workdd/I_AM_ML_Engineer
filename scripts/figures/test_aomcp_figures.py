"""옮겨 적은 상수가 원 코드 · README 와 맞는지, 생성한 SVG 가 깨지지 않았는지 확인한다."""
import xml.etree.ElementTree as ET

import pytest

import aomcp_figures as a


def test_tool_groups_total_35_and_answer_13():
    assert sum(v for _, v, _ in a.TOOL_GROUPS) == 35
    assert a.TOOL_GROUPS[0][1] == 13


def test_answer_only_cuts_tool_definitions():
    full, answer = (v for _, v in a.TOOL_CHARS)
    assert (full, answer) == (52000, 12000) and answer < full


@pytest.mark.parametrize("c,name,status", [
    (0.59, 1.0, "drop"),        # containment 미달은 이름이 완벽해도 버린다
    (0.98, 0.75, "accepted"),   # 경계값은 포함
    (0.97, 1.0, "proposed"),
    (1.0, 0.74, "proposed"),
])
def test_decide_matches_inference_rules(c, name, status):
    assert a.decide(c, name)[0] == status


def test_confidence_is_mean_of_name_and_containment():
    assert a.decide(0.8, 0.6) == ("proposed", 0.7)


@pytest.mark.parametrize("name,fn", a.FIGS)
def test_svg_is_well_formed(name, fn):
    root = ET.fromstring(fn())
    assert root.get("viewBox")
    assert root.find("{http://www.w3.org/2000/svg}title").text
