"""옮겨 적은 개수가 평가 기록의 F1 과 맞는지, 생성한 SVG 가 깨지지 않았는지 확인한다."""
import xml.etree.ElementTree as ET

import pytest

import edgex_figures as e


@pytest.mark.parametrize("row", e.COUNTS, ids=lambda r: r[0])
def test_counts_reproduce_published_f1(row):
    name, names, links, (fn, fl), _ = row
    assert round(e.f1(*names, e.GOLD["names"]), 3) == pytest.approx(fn, abs=0.002)
    assert round(e.f1(*links, e.GOLD["links"]), 3) == pytest.approx(fl, abs=0.002)


def test_mistral_keeps_more_than_twice_the_extra_links():
    final, _, mistral = e.COUNTS
    assert mistral[2][1] > 2 * final[2][1]
    assert mistral[2][0] > final[2][0]  # 맞힌 링크는 Mistral 이 더 많다


def test_span_finder_gain_did_not_transfer():
    dev, test = e.TRANSFER
    assert dev[2] > dev[1] and test[2] < test[1]
    assert dev[4] - dev[3] == test[4] - test[3] == 7


def test_wrong_links_sit_above_keep_cutoff():
    wrong = [p for _, p, ok in e.PROBES if not ok]
    assert wrong and all(p >= e.NOUL_YES for p in wrong)


@pytest.mark.parametrize("name,fn", e.FIGS)
def test_svg_is_well_formed(name, fn):
    root = ET.fromstring(fn())
    assert root.get("viewBox")
    assert root.find("{http://www.w3.org/2000/svg}title").text
