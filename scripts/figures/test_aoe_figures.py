"""옮겨 적은 사고 · 결함 수치가 원문 해석과 맞는지, 생성한 SVG 가 깨지지 않았는지 확인한다."""
import xml.etree.ElementTree as ET

import pytest

import aoe_figures as a


def test_ratelimit_shares_match_readme():
    assert round(a.rate(a.CALLS_429, a.CALLS) * 100) == 66  # "66% rate-limited run"
    assert a.rate(a.EMPTY_SQL, a.QUESTIONS) > a.HEALTH_THRESHOLD
    # 13.3% 는 60 문항 중 8 문항 정답
    assert round(8 / a.QUESTIONS * 100, 1) == a.REPORTED_EX


def test_hidden_loss_values():
    retr, flags = a.HIDDEN
    assert (retr[1], retr[2], retr[4]) == (64.3, 76.7, 76.4)
    assert abs(retr[2] - retr[4]) < 1  # 고친 뒤 기준선과 맞는다
    s0, s1, n = a.HIDDEN_RAW["slice"]
    assert (flags[1], flags[2]) == (s0 / n * 100, s1 / n * 100)
    assert flags[2] - flags[1] == pytest.approx(10.0)


@pytest.mark.parametrize("name,fn", a.FIGS)
def test_svg_is_well_formed(name, fn):
    root = ET.fromstring(fn())
    assert root.get("viewBox")
    assert root.find("{http://www.w3.org/2000/svg}title").text
