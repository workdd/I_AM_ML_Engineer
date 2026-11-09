"""옮겨 적은 값이 whitepaper §5 의 헤드라인을 다시 만드는지, 생성한 SVG 가 깨지지 않았는지 확인한다."""
import xml.etree.ElementTree as ET

import pytest

import ontosmith_figures as o


@pytest.mark.parametrize("row", o.CONTRACTS, ids=lambda r: r[0])
def test_estimators_reproduce_published_headlines(row):
    _, p_a, _, g_b, _, prec, _, rec, _, _ = row
    assert o.corrected_precision(p_a) == pytest.approx(prec, abs=0.0006)
    assert o.adjusted_recall(g_b) == pytest.approx(rec, abs=0.0006)


def test_ruling_moves_precision_down_and_recall_up():
    orig, ruled = o.CONTRACTS
    assert ruled[5] < orig[5] and ruled[7] > orig[7]
    assert ruled[3] < orig[3]


def test_matched_stratum_repricing_reads_about_0_845():
    assert o.corrected_precision(o.CONTRACTS[1][1], o.MATCHED_SUPPORT) == pytest.approx(0.845, abs=0.001)


def test_population_counts_are_consistent():
    assert o.MATCHED_EXTRACTED + o.UNMATCHED_POOL == o.TIER1
    assert sum(o.FATES.values()) == o.GOLD_TOTAL
    never = o.FATES["never_enumerated"] + o.FATES["roster_missed"]
    assert never == 10010 and round(never / o.GOLD_TOTAL, 3) == 0.579
    assert round(o.MATCHED_EXTRACTED / o.TIER1, 4) == o.RAW_P


def test_operating_costs_follow_pair_rate():
    for _, pairs, cost, _, _ in o.OPERATING:
        assert pairs * o.PAIR_RATE == pytest.approx(cost, abs=0.01)
    a, b = o.OPERATING
    assert round(b[1] / a[1], 2) == 1.95 and b[3] < a[3]


@pytest.mark.parametrize("name,fn", o.FIGS)
def test_svg_is_well_formed(name, fn):
    root = ET.fromstring(fn())
    assert root.get("viewBox")
    assert root.find("{http://www.w3.org/2000/svg}title").text
