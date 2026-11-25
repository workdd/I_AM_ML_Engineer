"""옮겨 적은 수치가 원문과 맞는지, 생성한 SVG 가 깨지지 않았는지 확인한다."""
import xml.etree.ElementTree as ET

import pytest

import ship_figures as s


def test_policy_rules_match_example_file():
    rules = {n: (base, passed, role) for n, _, base, passed, role in s.RULES}
    assert rules["Most code"] == (2, 1, "approves")
    assert rules["Docs"] == (2, 0, "approves")
    assert rules["Billing"] == (2, None, "comments")


def test_review_pass_never_raises_seats():
    for _, _, base, passed, _ in s.RULES:
        assert passed is None or passed < base


def test_tiers_increase():
    assert [n for _, _, n in s.TIERS] == [0, 1, 2]


def test_blog_numbers():
    a, b = s.CHANGES_PER_DAY
    assert (a, b) == (2500, 30000) and b // a == 12
    assert s.SUBSTANTIVE_PCT == (16, 54)


@pytest.mark.parametrize("name,fn", s.FIGS)
def test_svg_is_well_formed(name, fn):
    root = ET.fromstring(fn())
    assert root.get("viewBox")
    assert root.find("{http://www.w3.org/2000/svg}title").text
