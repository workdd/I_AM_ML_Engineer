"""옮겨 적은 수치가 원문 본문 주장과 맞는지, 생성한 SVG 가 깨지지 않았는지 확인한다."""
import xml.etree.ElementTree as ET

import pytest

import snap_figures as s


def test_coverage_matches_counts():
    # 본문: 100,648건 중 34,963건 채택, coverage 34.74%
    assert round(100 * s.ADOPTED / s.TOTAL, 2) == dict(s.COVERAGE)[s.DEPLOYED] == 34.74


def test_actual_agreement_rate():
    # 본문: 불일치 34건, 실제 일치율 99.9028%
    assert round(100 * (s.ADOPTED - s.MISMATCH) / s.ADOPTED, 4) == 99.9028


def test_coverage_grows_as_target_drops():
    targets = [a for a, _ in s.COVERAGE]
    covs = [c for _, c in s.COVERAGE]
    assert targets == sorted(targets, reverse=True)
    assert covs == sorted(covs)


@pytest.mark.parametrize("name,fn", s.FIGS)
def test_svg_is_well_formed(name, fn):
    root = ET.fromstring(fn())
    assert root.get("viewBox")
    assert root.find("{http://www.w3.org/2000/svg}title").text
