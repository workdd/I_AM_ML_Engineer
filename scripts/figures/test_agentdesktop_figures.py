"""원문 수치와 게이트 규칙을 상수와 함수가 그대로 내는지, SVG 가 깨지지 않았는지 확인한다."""
import xml.etree.ElementTree as ET

import pytest

import agentdesktop_figures as m


def test_linux_is_largest_and_macos_tests_least():
    totals = [c[2] for c in m.CRATES]
    assert totals == sorted(totals, reverse=True)
    shares = {name: m.test_share(total, test) for name, _, total, test, _ in m.CRATES}
    assert min(shares, key=shares.get) == "macos"
    assert round(shares["core"]) == 60


def test_gate_matches_skill_text():
    assert m.gate(0.50, 0.0) == "none"           # below 0.55 nothing runs
    assert m.gate(0.65, 0.0) == "stop"           # ordinary step needs 0.70
    assert m.gate(0.75, 0.2) == "run"
    assert m.gate(0.75, 0.6) == "stop"           # destructive (>= 0.50) needs 0.90
    assert m.gate(0.95, 0.9) == "run_unrated"    # run.mjs rates risk only for 0.70 up to 0.90


@pytest.mark.parametrize("name,fn", m.FIGS)
def test_svg_is_well_formed(name, fn):
    root = ET.fromstring(fn())
    assert root.get("viewBox")
    assert root.find("{http://www.w3.org/2000/svg}title").text
