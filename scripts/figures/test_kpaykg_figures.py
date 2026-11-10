"""발표 슬라이드 32쪽 값과 발표 발언(80%, 3 hop)을 옮긴 상수가 맞는지, 생성한 SVG 가 깨지지 않았는지 확인한다."""
import xml.etree.ElementTree as ET

import pytest

import kpaykg_figures as k


def test_spoken_claim_is_slightly_above_half_at_three_hops():
    assert k.chain(0.8, 3) == pytest.approx(0.512)
    assert k.chain(0.8, 3) > 0.5 > k.chain(0.8, 4)


def test_chain_decreases_with_hops():
    vals = [k.chain(0.8, n) for n in range(1, 7)]
    assert all(a > b for a, b in zip(vals, vals[1:]))


def test_per_article_ratios_match_slide_32():
    r = k.per_article()
    assert round(r["raw_events"], 1) == 2.9
    assert r["events"] == 1.5
    assert round(r["edges"], 1) == 11.2 or round(r["edges"], 2) == 11.25
    assert r["edges_per_event"] == 7.5
    assert round(r["merge_cut"], 2) == 0.48
    assert round(r["impacts_share"], 2) == 0.53
    assert k.IMPACTS < k.EDGES and k.EVENTS < k.RAW_EVENTS


@pytest.mark.parametrize("name,fn", k.FIGS)
def test_svg_is_well_formed(name, fn):
    root = ET.fromstring(fn())
    assert root.get("viewBox")
    assert root.find("{http://www.w3.org/2000/svg}title").text
