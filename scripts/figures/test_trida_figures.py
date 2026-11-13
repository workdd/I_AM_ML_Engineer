"""옮겨 적은 카드 수치가 원문과 맞는지, 재계산이 카드 서술과 맞는지, SVG 가 깨지지 않았는지 확인한다."""
import xml.etree.ElementTree as ET

import pytest

import trida_figures as f


def test_speed_ratios_match_card_within_rounding():
    for c, ar, ss, card in f.SPEED:
        assert abs(ss / ar - card) < 0.01, c


def test_speedup_shrinks_with_concurrency():
    cards = [card for *_, card in f.SPEED]
    assert cards == sorted(cards, reverse=True)


def test_total_throughput_below_one_only_at_16():
    below = [cs for cs, lo, hi in f.TOTAL if hi < 1.0]
    assert below == [(16,)]


def test_threshold_07_recovers_about_70_to_80_percent():
    # 카드: "recovers about 70–80 % of the accuracy gain" (JevBench, hard-v1, transfer-v4)
    rs = [f.recovered(d, acc, a) for d, pts, a in f.ESCALATE.values() for th, acc, _ in pts if th == 0.7]
    assert [round(r, 2) for r in rs] == [0.77, 0.82, 0.69]


def test_hard_v1_threshold_09_beats_always():
    d, pts, a = f.ESCALATE["hard-v1"]
    assert pts[-1][1] > a


@pytest.mark.parametrize("name,fn", f.FIGS)
def test_svg_is_well_formed(name, fn):
    root = ET.fromstring(fn())
    assert root.get("viewBox")
    assert root.find("{http://www.w3.org/2000/svg}title").text
