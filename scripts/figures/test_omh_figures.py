"""옮겨 적은 수치가 원문 해석(비용 비율·천장·CI 판정)과 맞는지, 생성한 SVG 가 깨지지 않았는지 확인한다."""
import xml.etree.ElementTree as ET

import pytest

import omh_figures as o


def test_no_lane_exceeds_ceiling():
    assert max(p for _, _, p, _, _ in o.LANES) == o.CEILING


def test_routed_sol_head_cost_ratios_match_readme():
    # 원문: Sol medium 대비 비용 19%, flagship xhigh 대비 4%
    r = o.cost_ratio("라우팅 · 신호 추가 후 · Sol 첫 모델", "Hermes 단독 · Sol medium")
    assert round(r * 100) == 19
    r = o.cost_ratio("라우팅 · 신호 추가 후 · Sol 첫 모델", "Astra xhigh")
    assert round(r * 100) == 4


def test_routed_sol_head_time_ratio_matches_readme():
    d = {n: m for n, _, _, _, m in o.LANES}
    assert round(d["라우팅 · 신호 추가 후 · Sol 첫 모델"] / d["Hermes 단독 · Sol medium"] * 100) == 21


def test_tier_thresholds_and_signal_sign():
    assert [lo for _, lo, _ in o.TIERS] == [0, 4, 8]
    neg = [n for n, w, _ in o.SIGNALS if w < 0]
    assert neg == ["simple_request"]
    # exhaustive_search 혼자서 standard 경계에 닿는다
    assert dict((n, w) for n, w, _ in o.SIGNALS)["exhaustive_search"] == o.TIERS[1][1]


def test_calibration_ci_verdicts():
    sig = {p: (a > 0 or b < 0) for p, _, a, b in o.CALIB}
    assert sig == {
        "family − baseline": False,
        "optimized − baseline": False,
        "optimized − family": True,
        "revised − family": False,
        "revised − optimized": True,
    }
    assert set(o.CALIB_PASSED.values()) == {18}
    for _, m, a, b in o.CALIB:
        assert a <= m <= b


@pytest.mark.parametrize("name,fn", o.FIGS)
def test_svg_is_well_formed(name, fn):
    root = ET.fromstring(fn())
    assert root.get("viewBox")
    assert root.find("{http://www.w3.org/2000/svg}title").text
